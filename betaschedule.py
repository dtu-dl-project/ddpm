import numpy as np

def betaschedule(min_beta=1e-4, max_beta=0.02, T=1000) -> np.ndarray:
    """
    Function that returns the beta values at every step
    for the given range [min_beta, max_beta] and T
    """
    return np.linspace(min_beta, max_beta, T)

def compute_alphas(betas):
    alphas = 1 - betas

    return alphas

def compute_alphas_hat(betas):
    alphas = 1 - betas
    alphas_hat = np.cumprod(alphas)

    return alphas_hat

if __name__ == "__main__":
    betas = np.array([2,3,4])
    alphas = compute_alphas(betas)
    print(alphas)
