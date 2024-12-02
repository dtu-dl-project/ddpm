import torch as t
import math

def linear_beta_schedule(min_beta: float = 1e-4, max_beta: float = 0.02, T: int = 1000) -> t.Tensor:
    """
    Function that returns the beta values at every step
    for the given range [min_beta, max_beta] and T

    Args:
        min_beta (float): Minimum beta value
        max_beta (float): Maximum beta value
        T (int): Number of steps

    Returns:
        t.Tensor: Beta values for each step
    """
    return t.cat((t.tensor([0.0]), t.linspace(min_beta, max_beta, T, dtype=t.float32)))

def cosine_beta_schedule(T: int = 1000, s: float = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = T + 1
    tt = t.linspace(0, T, steps, dtype = t.float64) / T
    alphas_cumprod = t.cos((tt + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return t.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(T: int = 1000, start: int = -3, end: int = 3, tau: int = 1, clamp_min: float = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = T + 1
    tt = t.linspace(0, T, steps, dtype = t.float64) / T
    v_start = t.tensor(start / tau).sigmoid()
    v_end = t.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((tt * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return t.clip(betas, 0, 0.999)

def compute_alphas(betas: t.Tensor) -> t.Tensor:
    """
    Compute alpha values from beta values

    Args:
        betas (t.Tensor): Beta values

    Returns:
        t.Tensor: Alpha values
    """
    return 1 - betas

def compute_alphas_hat(betas: t.Tensor) -> t.Tensor:
    """
    Compute cumulative product of alpha values from beta values

    Args:
        betas (t.Tensor): Beta values

    Returns:
        t.Tensor: Cumulative product of alpha values
    """
    alphas = 1 - betas
    alphas_hat = t.cumprod(alphas, dim=0)
    return alphas_hat

if __name__ == "__main__":
    device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    betas = t.tensor([2.0, 3.0, 4.0], device=device)

    alphas = compute_alphas(betas)
    print("Alphas:", alphas)

    betas_schedule = betaschedule()
    print("\nBeta schedule shape:", betas_schedule.shape)
    print("First few betas:", betas_schedule[:5])

    alphas_hat = compute_alphas_hat(betas_schedule)
    print("\nAlphas hat shape:", alphas_hat.shape)
    print("First few alphas hat:", alphas_hat[:5])
