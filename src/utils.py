import torch as T
from torchvision import datasets, transforms
from torch.utils.data import random_split

def get_device(torch):
    return torch.device("cuda" if T.cuda.is_available() else ("mps" if T.mps.is_available() else "cpu"))

def get_dataset(dataset_name):
    image_size = 32
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),       # Resizes the image based on dataset
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
    ])
    if dataset_name == "MNIST":
        train_data = datasets.MNIST(root="data", train=True, transform=transform, download=True)
        train_size = int(0.9 * len(train_data))
        val_size = len(train_data) - train_size
        train_dataset, val_dataset = random_split(train_data, [train_size, val_size])
        test_dataset = datasets.MNIST(root="data", train=False, transform=transform, download=True)
    elif dataset_name == "CIFAR10":
        train_data = datasets.CIFAR10(root="data", train=True, transform=transform, download=True)
        train_size = int(0.9 * len(train_data))
        val_size = len(train_data) - train_size
        train_dataset, val_dataset = random_split(train_data, [train_size, val_size])
        test_dataset = datasets.CIFAR10(root="data", train=False, transform=transform, download=True)
    elif dataset_name == "Fashion-MNIST":
        train_data = datasets.FashionMNIST(root="data", train=True, transform=transform, download=True)
        train_size = int(0.9 * len(train_data))
        val_size = len(train_data) - train_size
        train_dataset, val_dataset = random_split(train_data, [train_size, val_size])
        test_dataset = datasets.FashionMNIST(root="data", train=False, transform=transform, download=True)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return train_dataset, val_dataset, test_dataset
