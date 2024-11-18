import torch as T

def get_device(torch):
    return torch.device("cuda" if T.cuda.is_available() else ("mps" if T.mps.is_available() else "cpu"))
