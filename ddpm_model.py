import torch as t
import torch.nn as nn
from unet_model_2.unet import DiffusionUnet
import lightning as L
import torch.nn.functional as F
from betaschedule import betaschedule, compute_alphas, compute_alphas_hat
from utils import get_device
import logging

logger = logging.getLogger(__name__)

T = 1000

device = get_device(t)

logger.info(f"Using device: {device}")

betas = betaschedule(1e-4, 0.02, T).to(device)
alphas_hat = compute_alphas_hat(betas)
alphas = compute_alphas(betas)

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
        self.unet = DiffusionUnet(dim=28, channels=1)

    def forward(self, x, t):
        return self.unet(x, t)


class DdpmLight(L.LightningModule):
    def __init__(self, ddpmnet):
        super().__init__()
        self.ddpmnet = ddpmnet


    def sample(self, count):
        x = t.randn(count, 1, 28, 28).to(self.device)  # Use torch.randn for consistency
        for int_i in reversed(range(T)):
            x = self.forward_sample(x, int_i)
        return x

    def forward_sample(self, x, int_i):
        bs = x.size(0)
        i = t.full((bs,), int_i, device=self.device, dtype=t.long)

        # Extract parameters
        alpha_t = alphas[i].view(bs, 1, 1, 1)
        alpha_hat_t = alphas_hat[i].view(bs, 1, 1, 1)
        beta_t = betas[i].view(bs, 1, 1, 1)

        # Predict noise
        pred = self.ddpmnet(x.view(bs, -1), i.view(bs, -1)).view(bs, 1, 28, 28)

        # Compute model mean
        model_mean = (1 / alpha_t.sqrt()) * (
            x - ((1 - alpha_t) / (1 - alpha_hat_t).sqrt()) * pred
        )
        # Sample the next step
        if int_i > 0:
            model_mean += t.randn_like(x) * beta_t.sqrt()

        return model_mean

    def step(self, batch, _):
        x, _ = batch

        x = x.to(self.device)

        bs = x.size(0)

        ts = sample_tS(T, size=(bs,)).to(self.device)

        alpha_hat = alphas_hat[ts]

        noised_x = add_noise(x, alpha_hat)

        # These are being flattened because
        # the score network expects a (bs, 784) tensor
        # flat_noised_x = noised_x.view(bs, -1)
        # flat_ts = ts.view(bs, -1)

        prediction = self.ddpmnet(noised_x, ts)

        return F.l1_loss((noised_x-x), prediction)


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
