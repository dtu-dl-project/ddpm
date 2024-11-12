import torch as t
import torch.nn as nn
from scorenetwork import ScoreNetwork0
import lightning as L
import torch.nn.functional as F
from betaschedule import betaschedule, compute_alphas, compute_alphas_hat

T = 1000
device = t.device('cuda' if t.cuda.is_available() else 'cpu')

betas = betaschedule(1e-4, 0.02, T).to(device)
alphas_hat = compute_alphas_hat(betas)
alphas = compute_alphas(betas)

print(betas.shape, alphas_hat.shape, alphas.shape)

def add_noise(x_0, alpha_hat_t):
    """
    Add Gaussian noise to an image tensor,
    given alpha as the noise level using PyTorch.
    """

    # Reshape alpha_hat_t to match the dimensions of x_0 for broadcasting
    alpha_hat_t = alpha_hat_t.view(-1, 1, 1, 1)

    # Generate Gaussian noise with the same shape as x_0
    noise = t.randn_like(x_0) * t.square(1 - alpha_hat_t)
    noise += t.square(alpha_hat_t)

    # Return the image tensor with added noise
    return x_0 + noise


def sample_tS(T, size):
    """
    Sample a tensor of shape size with values from [0, T] inclusive
    """

    return t.randint(0, T, size=size)


class DdpmNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.scorenet = ScoreNetwork0()

    def forward(self, x, t):
        return self.scorenet(x, t)


class DdpmLight(L.LightningModule):
    def __init__(self, ddpmnet):
        super().__init__()
        self.ddpmnet = ddpmnet

    def sample(self, count):
        x = t.rand(count, 1, 28, 28)
        for i in reversed(range(T)):
            self.forward_sample(x, i)
        return x

    def forward_sample(self, x, int_i):
        bs = x.size(0)

        i = (t.ones(bs) * int_i).int()

        x = x.to(self.device)

        i = i.to(self.device)

        alpha_t = alphas[i].view(bs, 1, 1, 1)
        alpha_hat_t = alphas_hat[i].view(bs, 1, 1, 1)
        beta_t = betas[i].view(bs, 1, 1, 1)

        pred = self.ddpmnet(x.view(bs, -1), i.view(bs, -1))

        pred = pred.view(bs, 1, 28, 28)

        x = (x - (1-alpha_t)/((1-alpha_hat_t)**0.5) * pred) * (1/(alpha_t**0.5))
        if int_i > 0:
            x += t.randn_like(x) * beta_t**0.5

    def step(self, batch, _):
        x, _ = batch

        x = x.to(self.device)

        bs = x.size(0)

        ts = sample_tS(T, size=(bs,)).to(self.device)

        alpha_hat = alphas_hat[ts]

        noised_x = add_noise(x, alpha_hat)

        # These are being flattened because
        # the score network expects a (bs, 784) tensor
        flat_noised_x = noised_x.view(bs, -1)
        flat_ts = ts.view(bs, -1)

        prediction = self.ddpmnet(flat_noised_x, flat_ts)

        return F.l1_loss((noised_x-x).view(bs, -1), prediction)


    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss


    def configure_optimizers(self):
        return t.optim.Adam(self.parameters(), lr=1e-3)
