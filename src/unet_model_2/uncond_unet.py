import torch
import torch.nn as nn
import torch.nn.functional as F

def space_to_depth(x, size=2):
    """
    Downsacle method that use the depth dimension to
    downscale the spatial dimensions
    Args:
        x (torch.Tensor): a tensor to downscale
        size (float): the scaling factor

    Returns:
        (torch.Tensor): new spatial downscale tensor
    """
    b, c, h, w = x.shape
    out_h = h // size
    out_w = w // size
    out_c = c * (size * size)
    x = x.reshape((-1, c, out_h, size, out_w, size))
    x = x.permute((0, 1, 3, 5, 2, 4))
    x = x.reshape((-1, out_c, out_h, out_w))
    return x


class SpaceToDepth(nn.Module):
  def __init__(self, size):
    super().__init__()
    self.size = size

  def forward(self, x):
    return space_to_depth(x, self.size)


class Residual(nn.Module):
  """
  Apply residual connection using an input function
  Args:
    func (function): a function to apply over the input
  """
  def __init__(self, func):
    super().__init__()
    self.func = func

  def forward(self, x, *args, **kwargs):
    return x + self.func(x, *args, **kwargs)

def upsample(in_channels, out_channels=None):
  out_channels = in_channels if out_channels is None else out_channels
  seq = nn.Sequential(
      nn.Upsample(scale_factor=2, mode='nearest'),
      nn.Conv2d(in_channels, out_channels, 3, padding=1)
  )
  return seq

def downsample(in_channels, out_channels=None):
  out_channels = in_channels if out_channels is None else out_channels
  seq = nn.Sequential(
      SpaceToDepth(2),
      nn.Conv2d(4 * in_channels, out_channels, 1)
  )
  return seq

class SinusodialPositionEmbedding(nn.Module):
  def __init__(self, embedding_dim):
    super().__init__()
    self.embedding_dim = embedding_dim

  def forward(self, time_steps):
    positions = torch.unsqueeze(time_steps, 1)
    half_dim = self.embedding_dim // 2
    embeddings = torch.zeros((time_steps.shape[0], self.embedding_dim), device=time_steps.device)
    denominators = 10_000 ** (2 * torch.arange(self.embedding_dim // 2, device=time_steps.device) / self.embedding_dim)
    embeddings[:, 0::2] = torch.sin(positions/denominators)
    embeddings[:, 1::2] = torch.cos(positions/denominators)
    return embeddings
  
class WeightStandardizedConv2d(nn.Conv2d):
  """
  https://arxiv.org/abs/1903.10520
  weight standardization purportedly works synergistically with group normalization
  """

  def forward(self, x):
    eps = 1e-5 if x.dtype == torch.float32 else 1e-3

    weight = self.weight
    mean = weight.mean(dim=[1,2,3], keepdim=True)
    variance = weight.var(dim=[1,2,3], keepdim=True, correction=0)
    normalized_weight = (weight - mean) / torch.sqrt(variance)

    return F.conv2d(
        x,
        normalized_weight,
        self.bias,
        self.stride,
        self.padding,
        self.dilation,
        self.groups
    )


class Block(nn.Module):
  def __init__(self, in_channels, out_channels, groups=8):
    super().__init__()
    self.proj = WeightStandardizedConv2d(in_channels, out_channels, 3, padding=1)
    self.norm = nn.GroupNorm(groups, out_channels)
    self.act = nn.SiLU()

  def forward(self, x, scale_shift=None):
    x = self.proj(x)
    x = self.norm(x)

    if scale_shift is not None:
      scale, shift = scale_shift
      x = x * (scale + 1) + shift

    x = self.act(x)
    return x

class ResnetBlock(nn.Module):
  def __init__(self, in_channels, out_channels, time_emb_dim=None, groups=8):
    super().__init__()
    if time_emb_dim is not None:
      self.mlp = nn.Sequential(
          nn.SiLU(),
          nn.Linear(time_emb_dim, 2 * out_channels)
      )
    else:
      self.mlp = None

    self.block1 = Block(in_channels, out_channels, groups)
    self.block2 = Block(out_channels, out_channels, groups)
    if in_channels == out_channels:
      self.res_conv = nn.Identity()
    else:
      self.res_conv = nn.Conv2d(in_channels, out_channels, 1)

  def forward(self, x, time_emb=None):
    scale_shift = None
    if self.mlp is not None and time_emb is not None:
      time_emb = self.mlp(time_emb)
      time_emb = time_emb.view(*time_emb.shape, 1, 1)
      scale_shift = time_emb.chunk(2, dim=1) ########

    h = self.block1(x, scale_shift=scale_shift)
    h = self.block2(h)
    return h + self.res_conv(x)

class Attention(nn.Module):
  def __init__(self, in_channels, num_heads=4, dim_head=32):
    super().__init__()
    self.num_heads = num_heads
    self.dim_head = dim_head
    self.scale_factor = 1 / (dim_head) ** 0.5
    self.hidden_dim = num_heads * dim_head
    self.input_to_qkv = nn.Conv2d(in_channels, 3 * self.hidden_dim, 1, bias=False)
    self.to_output = nn.Conv2d(self.hidden_dim, in_channels, 1)

  def forward(self, x):
    b, c, h, w = x.shape
    qkv = self.input_to_qkv(x)
    q, k, v = map(lambda t: t.view(b, self.num_heads, self.dim_head, h * w), qkv.chunk(3, dim=1))
    q = q * self.scale_factor
    # dot product between the columns of q and k
    sim = torch.einsum("b h c i, b h c j -> b h i j", q, k)
    sim = sim - sim.amax(dim=-1, keepdim=True).detach()
    attention = sim.softmax(dim=-1)

    # dot product between the rows to get the wighted values as columns
    output = torch.einsum("b h i j, b h c j -> b h i c", attention, v)
    output = output.permute(0, 1, 3, 2).reshape((b, self.hidden_dim, h, w))
    return self.to_output(output)


class LinearAttention(nn.Module):
  def __init__(self, in_channels, num_heads=4, dim_head=32):
    super().__init__()
    self.num_heads = num_heads
    self.dim_head = dim_head
    self.scale_factor = 1 / (dim_head) ** 0.5
    self.hidden_dim = num_heads * dim_head
    self.input_to_qkv = nn.Conv2d(in_channels, 3 * self.hidden_dim, 1, bias=False)
    self.to_output = nn.Sequential(
        nn.Conv2d(self.hidden_dim, in_channels, 1),
        nn.GroupNorm(1, in_channels)
    )

  def forward(self, x):
    b, c, h, w = x.shape
    qkv = self.input_to_qkv(x)
    q, k, v = map(lambda t: t.view(b, self.num_heads, self.dim_head, h * w), qkv.chunk(3, dim=1))

    q = q.softmax(dim=-2)
    k = k.softmax(dim=-1)

    q = q * self.scale_factor
    context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
    output = torch.einsum("b h d e, b h d n -> b h e n", context, q)
    output = output.view((b, self.hidden_dim, h, w))
    return self.to_output(output)
  
class PreGroupNorm(nn.Module):
  def __init__(self, dim , func, groups=1):
    super().__init__()
    self.func = func
    self.group_norm = nn.GroupNorm(groups, dim)

  def forward(self, x):
    x = self.group_norm(x)
    x = self.func(x)
    return x
     
class DiffusionUnet(nn.Module):
  def __init__(self, dim, init_dim=None, output_dim=None, dim_mults=(1, 2, 4, 8), channels=3, resnet_block_groups=4):
    super().__init__()

    self.channels = channels
    init_dim = init_dim if init_dim is not None else dim
    self.init_conv = nn.Conv2d(self.channels, init_dim, 1)
    dims = [init_dim] + [m * dim for m in dim_mults]
    input_output_dims = list(zip(dims[:-1], dims[1:]))

    time_dim = 4 * dim  # time embedding

    self.time_mlp = nn.Sequential(
        SinusodialPositionEmbedding(dim),
        nn.Linear(dim, time_dim),
        nn.GELU(),
        nn.Linear(time_dim, time_dim)
    )

    # down layers
    self.down_layers = nn.ModuleList([])
    for ii, (dim_in, dim_out) in enumerate(input_output_dims, 1):
      is_last = ii == len(input_output_dims)
      self.down_layers.append(
          nn.ModuleList(
              [
                  ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim, groups=resnet_block_groups),
                  ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim, groups=resnet_block_groups),
                  Residual(PreGroupNorm(dim_in, LinearAttention(dim_in))),
                  downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
              ]
          )
      )

      # middle layers
      mid_dim = dims[-1]
      self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, groups=resnet_block_groups)
      self.mid_attention = Residual(PreGroupNorm(mid_dim, Attention(mid_dim)))
      self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, groups=resnet_block_groups)

      # up layers
      self.up_layers = nn.ModuleList([])
      for ii, (dim_in, dim_out) in enumerate(reversed(input_output_dims), 1):
        is_last = ii == len(input_output_dims)
        self.up_layers.append(
            nn.ModuleList(
                [
                    ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_dim, groups=resnet_block_groups),
                    ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_dim, groups=resnet_block_groups),
                    Residual(PreGroupNorm(dim_out, LinearAttention(dim_out))),
                    upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
                ]
            )
        )

        self.output_dim = output_dim if output_dim is not None else channels
        self.final_res_block = ResnetBlock(2 * dim, dim, time_emb_dim=time_dim, groups=resnet_block_groups)
        self.final_conv = nn.Conv2d(dim, self.output_dim, 1)

  def forward(self, x, time):
    x = self.init_conv(x)
    init_result = x.clone()
    t = self.time_mlp(time)
    h = []

    for block1, block2, attention, downsample_block in self.down_layers:
      x = block1(x, t)
      h.append(x)

      x = block2(x, t)
      x = attention(x)

      h.append(x)

      x = downsample_block(x)

    x = self.mid_block1(x, t)
    x = self.mid_attention(x)
    x = self.mid_block2(x ,t)

    for block1, block2, attention, upsample_block in self.up_layers:
      x = torch.cat((x , h.pop()), dim=1)
      x = block1(x, t)

      x = torch.cat((x, h.pop()), dim=1)
      x = block2(x, t)

      x = attention(x)

      x = upsample_block(x)

    x = torch.cat((x, init_result), dim=1)
    x = self.final_res_block(x, t)
    x = self.final_conv(x)
    return x