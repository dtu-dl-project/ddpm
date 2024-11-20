import torch as t

def betaschedule(min_beta: float = 1e-4, max_beta: float = 0.02, T: int = 1000) -> t.Tensor:
    """
    Function that returns the beta values at every step
    for the given range [min_beta, max_beta] and T

    Args:
        min_beta (float): Minimum beta value
        max_beta (float): Maximum beta value
        T (int): Number of steps

    Returns:
        torch.Tensor: Beta values for each step
    """
    return t.cat((t.tensor([0.0]), t.linspace(min_beta, max_beta, T, dtype=t.float32)))

def compute_alphas(betas: t.Tensor) -> t.Tensor:
    """
    Compute alpha values from beta values

    Args:
        betas (torch.Tensor): Beta values

    Returns:
        torch.Tensor: Alpha values
    """
    return 1 - betas

def compute_alphas_hat(betas: t.Tensor) -> t.Tensor:
    """
    Compute cumulative product of alpha values from beta values

    Args:
        betas (torch.Tensor): Beta values

    Returns:
        torch.Tensor: Cumulative product of alpha values
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
