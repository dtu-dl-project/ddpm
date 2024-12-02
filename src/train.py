import argparse
from lightning.pytorch.callbacks import ModelCheckpoint
import torch as T
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import lightning as L
import logging
from utils import get_device

logging.basicConfig(
    level=logging.INFO,
    format=('%(filename)s: '
            '%(levelname)s: '
            '%(funcName)s(): '
            '%(lineno)d:\t'
            '%(message)s')
)

logger = logging.getLogger(__name__)

def get_dataset(dataset_name, transform):
    if dataset_name == "MNIST":
        train_data = datasets.MNIST(root="data", train=True, transform=transform, download=True)
        train_size = int(0.9 * len(train_data))
        val_size = len(train_data) - train_size
        train_dataset, val_dataset = random_split(train_data, [train_size, val_size])
        test_dataset = datasets.MNIST(root="data", train=False, transform=transform, download=True)
    elif dataset_name == "CelebA-HQ":
        train_data = datasets.ImageFolder(root="data/celeba_hq", transform=transform)
        train_size = int(0.9 * len(train_data))
        val_size = len(train_data) - train_size
        train_dataset, val_dataset = random_split(train_data, [train_size, val_size])
        test_dataset = None
    elif dataset_name == "Fashion-MNIST":
        train_data = datasets.FashionMNIST(root="data", train=True, transform=transform, download=True)
        train_size = int(0.9 * len(train_data))
        val_size = len(train_data) - train_size
        train_dataset, val_dataset = random_split(train_data, [train_size, val_size])
        test_dataset = datasets.FashionMNIST(root="data", train=False, transform=transform, download=True)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return train_dataset, val_dataset, test_dataset

def main():
    parser = argparse.ArgumentParser(description="Train DDPM with different datasets and beta schedules.")
    parser.add_argument("--dataset", type=str, choices=["MNIST", "CelebA-HQ", "Fashion-MNIST"], default="MNIST",
                        help="Dataset to use for training (MNIST, CelebA-HQ, Fashion-MNIST)")
    parser.add_argument("--unet_dim", type=int, default=32, 
                        help="Dimension of the U-Net used in DdpmNet.")
    parser.add_argument("--beta_schedule", type=str, choices=["linear", "cosine", "sigmoid"], default="linear",
                        help="Beta schedule type for DDPM (linear, cosine, sigmoid).")
    args = parser.parse_args()

    dataset_name = args.dataset
    unet_dim = args.unet_dim
    beta_schedule = args.beta_schedule

    logger.info(f"Using dataset: {dataset_name}")
    logger.info(f"Using U-Net dimension: {unet_dim}")
    logger.info(f"Using beta schedule: {beta_schedule}")

    # Set device to cuda if available, set to mps if available else cpu
    device = get_device(T)
    logger.info(f"Using device: {device}")
    
    # Import the model after setting the device
    from ddpm_model import DdpmLight, DdpmNet

    batch_size = 32

    image_size = 256 if dataset_name == "CelebA-HQ" else 32

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),       # Resizes the image based on dataset
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
    ])

    train_dataset, val_dataset, test_dataset = get_dataset(dataset_name, transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Pass the U-Net dimension and beta scheduler to the DdpmNet constructor
    num_channels = 3 if dataset_name == 'CelebA-HQ' else 1
    model = DdpmNet(unet_dim=unet_dim, channels=num_channels, img_size=image_size, beta_schedule=beta_schedule)
    ddpm_light = DdpmLight(model).to(device)

    epochs = 200

    checkpoint_callback = ModelCheckpoint(
        dirpath="ckpt", 
        save_top_k=3, 
        monitor="val_loss", 
        filename=f"{dataset_name}_unet_dim={unet_dim}_beta={beta_schedule}_{{epoch}}-{{val_loss:.4f}}"
    )

    trainer = L.Trainer(max_epochs=epochs, callbacks=checkpoint_callback)
    trainer.fit(model=ddpm_light, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

if __name__ == "__main__":
    main()
