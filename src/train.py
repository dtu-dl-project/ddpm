import argparse
from lightning.pytorch.callbacks import ModelCheckpoint
import torch as T
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import lightning as L
import logging
import os
from utils import get_device, get_dataset

logging.basicConfig(
    level=logging.INFO,
    format=('%(filename)s: '
            '%(levelname)s: '
            '%(funcName)s(): '
            '%(lineno)d:\t'
            '%(message)s')
)

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train DDPM with different datasets and beta schedules.")
    parser.add_argument("--dataset", type=str, choices=["MNIST", "CIFAR10", "Fashion-MNIST"], default="MNIST",
                        help="Dataset to use for training (MNIST, CIFAR10, Fashion-MNIST)")
    parser.add_argument("--unet_dim", type=int, default=32, 
                        help="Dimension of the U-Net used in DdpmNet.")
    parser.add_argument("--beta_schedule", type=str, choices=["linear", "cosine", "sigmoid"], default="linear",
                        help="Beta schedule type for DDPM (linear, cosine, sigmoid).")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to checkpoint file to resume training.")
    parser.add_argument("--loss", type=str, default="smooth_l1", 
                        help="Loss function to use during training (e.g., smooth_l1, mse, etc.).")
    parser.add_argument("--lr", type=float, default=3e-4, 
                        help="Learning rate for training.")
    parser.add_argument("--cond", action="store_true", help="Use conditional diffusion models.")
    parser.add_argument("--scheduler", action="store_true", help="Use cosine scheduler.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train the model.")
    parser.add_argument("--bs", type=int, default=32, help="Batch size to train the model.")
    args = parser.parse_args()

    dataset_name = args.dataset
    unet_dim = args.unet_dim
    beta_schedule = args.beta_schedule
    loss_type = args.loss
    lr = args.lr
    cond = args.cond
    epochs = args.epochs
    use_scheduler = args.scheduler
    batch_size = args.bs 

    logger.info(f"Using dataset: {dataset_name}")
    logger.info(f"Using U-Net dimension: {unet_dim}")
    logger.info(f"Using beta schedule: {beta_schedule}")
    logger.info(f"Using loss type: {loss_type}")
    logger.info(f"Using learning rate: {lr}")
    logger.info(f"Using conditional unet: {cond}")
    logger.info(f"Using batch size: {batch_size}")
    logger.info(f"Using scheduler: {use_scheduler}")

    # Set device to cuda if available, set to mps if available else cpu
    device = get_device(T)
    logger.info(f"Using device: {device}")
    
    # Import the model after setting the device
    from ddpm_model import DdpmLight, DdpmNet

    train_dataset, val_dataset, test_dataset = get_dataset(dataset_name)


    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Pass the U-Net dimension, beta scheduler, loss type, and learning rate to the DdpmNet constructor
    num_channels = 3 if dataset_name == 'CIFAR10' else 1
    image_size = 32

    if dataset_name == "CIFAR10":
        # Using the same settings of the original paper
        dim_mults = (1,2,2,2)
        resnet_block_groups = 2
        dropout = 0.1
        horizontal_flips = True
        dim_att_head = 16
    else:
        dim_mults = (1,2,4,8)
        resnet_block_groups = 4
        dropout = 0.0
        horizontal_flips = False
        dim_att_head = 32

    logger.info(f"Using dim mults: {dim_mults}")
    logger.info(f"Using resnet block groups: {resnet_block_groups}")
    logger.info(f"Using dropout: {dropout}")
    logger.info(f"Using horizontal flips: {horizontal_flips}")
    logger.info(f"Using dim att head: {dim_att_head}")

    model = DdpmNet(unet_dim=unet_dim, 
                    channels=num_channels, 
                    img_size=image_size, 
                    beta_schedule=beta_schedule, 
                    loss_type=loss_type, 
                    lr=lr, 
                    cond=cond,
                    dim_mults=dim_mults,
                    resnet_block_groups=resnet_block_groups,
                    dropout=dropout,
                    horizontal_flips=horizontal_flips,
                    dim_att_head=dim_att_head)
    
    ddpm_light = DdpmLight(model, 
                           use_scheduler=use_scheduler, 
                           len_train_set=len(train_dataloader), 
                           epochs=epochs).to(device)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="ckpt", 
        save_top_k=3, 
        monitor="val_loss", 
        filename=f"{dataset_name}_unet_dim={unet_dim}_beta={beta_schedule}_loss={loss_type}_lr={lr}_cond={cond}_bs={batch_size}_{{epoch}}-{{val_loss:.5f}}"
    )

    trainer = L.Trainer(max_epochs=epochs, callbacks=checkpoint_callback)
    if args.load_checkpoint:
        trainer.fit(model=ddpm_light, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=args.load_checkpoint)
    else:
        trainer.fit(model=ddpm_light, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

if __name__ == "__main__":
    main()
