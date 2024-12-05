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

def cosine_beta_schedule_2(T: int = 1000, s: float = 0.008) -> t.Tensor:
    timesteps = t.linspace(0, T, T + 1, dtype=t.float32)
    alphas_bar = t.cos((timesteps / T + s) / (1 + s) * t.pi / 2) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]  # Normalize alphas_bar to start at 1
    betas = 1.0 - alphas_bar[1:] / alphas_bar[:-1]
    betas = t.clamp(betas, max=0.999)  # Ensure max beta is 0.999
    return t.cat((t.tensor([0.0]), t.tensor(betas, dtype=t.float32)))

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
