import torch as t
import torch.nn as nn
from scorenetwork import ScoreNetwork0
import lightning as L
import numpy as np
import torch.nn.functional as F

from betaschedule import betaschedule, compute_alphas

betas = betaschedule()
alphas = compute_alphas(betas)


def add_noise(x_0, alpha_hat_t):
    """
    Add gaussian noise to an image tensor,
    given alpha as the noise level
    """

    alpha_hat_t = alpha_hat_t.reshape(-1, 1, 1, 1)

    noise = t.randn_like(x_0) * np.square(1-alpha_hat_t)
    noise += np.square(alpha_hat_t)

    # x_0 shape : (batch, ch, 28, 28)
    # alpha_hat_t shape : (batch)

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

    def training_step(self, batch, batch_idx):
        x, _ = batch

        x = x.to(self.device)

        bs = x.size(0)

        ts = sample_tS(1000, size=(bs,))

        alphas_hat = alphas[ts]

        noised_x = add_noise(x, alphas_hat)

        flat_noised_x = noised_x.view(bs, -1).float()
        flat_ts = ts.view(bs, -1).float()

        prediction = self.ddpmnet(flat_noised_x, flat_ts)

        # Compute l1 loss
        loss = F.l1_loss((noised_x-x).view(bs, -1), prediction)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return t.optim.Adam(self.parameters(), lr=1e-3)
