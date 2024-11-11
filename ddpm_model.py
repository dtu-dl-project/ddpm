import torch as t
import torch.nn as nn
from scorenetwork import ScoreNetwork0
import lightning as L
import numpy as np
import torch.nn.functional as F

from betaschedule import betaschedule, compute_alphas, compute_alphas_hat

betas = betaschedule()
alphas_hat = compute_alphas_hat(betas)
alphas = compute_alphas(betas)

T = 1000

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
    Sample a tensor of shape size with values from [0, T]
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
            x_prev = self.forward(x, t.tensor(i))
            x = x_prev
        return x

    def forward(self, x, i):
        bs = x.size(0)

        alpha_t = alphas[i]
        alpha_hat_t = alphas_hat[i]
        beta_t = betas[i]

        pred = self.ddpmnet(x.view(bs, -1), i.view(bs, -1))

        pred = pred.view(bs, 1, 28, 28)

        xt_prev = 1/(alpha_t**0.5)
        xt_prev *= (x - (1-alpha_t)/((1-alpha_hat_t)**0.5) * pred)
        if i > 0:
            xt_prev += t.randn_like(x) * beta_t**0.5

        return xt_prev


    def training_step(self, batch, batch_idx):
        x, _ = batch

        x = x.to(self.device)

        bs = x.size(0)

        ts = sample_tS(T, size=(bs,))

        alpha_hat = alphas_hat[ts]

        noised_x = add_noise(x, t.from_numpy(alpha_hat).cuda())

        flat_noised_x = noised_x.view(bs, -1).float()
        flat_ts = ts.view(bs, -1).float().cuda()


        prediction = self.ddpmnet(flat_noised_x, flat_ts)

        # Compute l1 loss
        loss = F.l1_loss((noised_x-x).view(bs, -1), prediction)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return t.optim.Adam(self.parameters(), lr=1e-3)
