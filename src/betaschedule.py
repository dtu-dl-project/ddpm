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

def cosine_beta_schedule(T: int = 1000, s: float = 0.008) -> t.Tensor:
    """
    Function that returns the beta values for a cosine schedule.
    
    Args:
        T (int): Number of steps.
        s (float): Small offset to prevent singularity issues at t=0.
    
    Returns:
        t.Tensor: Beta values for each step.
    """
    # Define the cosine function according to the schedule
    def alpha_bar_fn(timestep):
        return math.cos((timestep / T + s) / (1 + s) * math.pi / 2) ** 2
    
    alphas_bar = [alpha_bar_fn(timestep) for timestep in range(T + 1)]
    betas = []
    
    for tt in range(1, T + 1):
        beta = min(1.0 - (alphas_bar[tt] / alphas_bar[tt - 1]), 0.999)
        betas.append(beta)
    
    return t.cat((t.tensor([0.0]), t.tensor(betas, dtype=t.float32)))

import torch as t

def sigmoid_beta_schedule(min_beta: float = 1e-4, max_beta: float = 0.02, T: int = 1000, s: float = 0.008) -> t.Tensor:
    """
    Function that returns the beta values for a sigmoid schedule.
    
    Args:
        min_beta (float): Minimum beta value.
        max_beta (float): Maximum beta value.
        T (int): Number of steps.
        s (float): Scale factor to adjust the sharpness of the sigmoid curve.
    
    Returns:
        t.Tensor: Beta values for each step.
    """
    # Define the sigmoid function according to the schedule
    def sigmoid_fn(timestep):
        return min_beta + (max_beta - min_beta) / (1 + t.exp(-s * (timestep - T // 2)))

    betas = [sigmoid_fn(timestep) for timestep in range(T)]
    
    return t.cat((t.tensor([0.0]), t.tensor(betas, dtype=t.float32)))


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
