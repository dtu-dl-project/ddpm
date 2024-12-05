import torch as t
import torch.nn as nn
from unet_model_2 import unet, uncond_unet
import lightning as L
import torch.nn.functional as F
from betaschedule import linear_beta_schedule, cosine_beta_schedule, sigmoid_beta_schedule, compute_alphas, compute_alphas_hat
from utils import get_device
import logging

logger = logging.getLogger(__name__)

T = 1000

device = get_device(t)

logger.info(f"Using device: {device}")

def add_noise(x_0, alpha_hat_t):
    """
    Add Gaussian noise to an image tensor,
    given alpha as the noise level using PyTorch.
    """

    # Reshape alpha_hat_t to match the dimensions of x_0 for broadcasting
    alpha_hat_t = alpha_hat_t.view(-1, 1, 1, 1)

    gaussian_noise = t.randn_like(x_0)

    # Generate Gaussian noise with the same shape as x_0
    noise = gaussian_noise * t.sqrt(1 - alpha_hat_t)
    mean = x_0 * t.sqrt(alpha_hat_t)

    # Return the image tensor with added noise
    return (noise + mean), gaussian_noise


def sample_tS(T, size):
    """
    Sample a tensor of shape size with values from [1, T] inclusive
    """

    return t.randint(1, T+1, size=size)


class DdpmNet(nn.Module):
    def __init__(self, unet_dim, channels, img_size, beta_schedule, loss_type="smooth_l1_loss", lr=3e-4, cond=False):
        super().__init__()
        self.channels = channels
        self.img_size = img_size
        self.cond = cond
        if cond:
            self.unet = unet.DiffusionUnet(dim=unet_dim, channels=channels, cond=cond)
        else: 
            self.unet = uncond_unet.DiffusionUnet(dim=unet_dim, channels=channels)
        self.beta_schedule = beta_schedule
        if beta_schedule == "linear":
            self.betas = linear_beta_schedule(1e-4, 0.02, T).to(device)
        elif beta_schedule == "cosine":
            self.betas = cosine_beta_schedule(T, s=0.008).to(device)
        elif beta_schedule == "sigmoid":
            self.betas = sigmoid_beta_schedule(T=T).to(device)
        self.loss_type = loss_type
        self.lr = lr

        self.alphas_hat = compute_alphas_hat(self.betas)
        self.alphas = compute_alphas(self.betas)

    def forward(self, x, t, c):
        if self.cond:
            return self.unet(x, t, c)
        else:
            return self.unet(x, t)


class DdpmLight(L.LightningModule):
    def __init__(self, ddpmnet):
        super().__init__()
        self.ddpmnet = ddpmnet


    def sample(self, count, klass):
        x = t.randn(count, self.ddpmnet.channels, self.ddpmnet.img_size, self.ddpmnet.img_size).to(self.device)  # Use torch.randn for consistency
        for int_i in reversed(range(T)):
            x = self.forward_sample(x, int_i + 1, klass)
            logger.info(f"Sampled timestep {int_i + 1}")
        return x

    def forward_sample(self, x, int_i, klass):
        bs = x.size(0)
        i = t.full((bs,), int_i, device=self.device, dtype=t.long)

        if not isinstance(klass, t.Tensor):
            klass = t.full((bs,), klass).to(self.device)

        # Extract parameters
        alpha_t = self.ddpmnet.alphas[i].view(bs, 1, 1, 1)
        alpha_hat_t = self.ddpmnet.alphas_hat[i].view(bs, 1, 1, 1)
        beta_t = self.ddpmnet.betas[i].view(bs, 1, 1, 1)

        # Predict noise
        pred = self.ddpmnet(x, i, klass)


        # Compute model mean
        model_mean = (1 / alpha_t.sqrt()) * (
            x - ((1 - alpha_t) / (1 - alpha_hat_t).sqrt()) * pred
        )
        # Sample the next step
        if int_i > 1:
            model_mean += t.randn_like(x) * beta_t.sqrt()

        return model_mean

    def step(self, batch, _):
        x, y = batch

        x = x.to(self.device)

        bs = x.size(0)

        ts = sample_tS(T, size=(bs,)).to(self.device) # Timestep for each sample

        cs = y.to(self.device) # Label for each sample

        alpha_hat = self.ddpmnet.alphas_hat[ts]

        noised_x, gaussian_noise = add_noise(x, alpha_hat)

        prediction = self.ddpmnet(noised_x, ts, cs)

        if self.ddpmnet.loss_type == "smooth_l1":
            return F.smooth_l1_loss(gaussian_noise, prediction)
        elif self.ddpmnet.loss_type == "mse":
            return F.mse_loss(gaussian_noise, prediction)
        else:
            raise ValueError("Invalid loss type")

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return t.optim.Adam(self.parameters(), lr=self.ddpmnet.lr)
