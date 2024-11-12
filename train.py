from lightning.pytorch.callbacks import ModelCheckpoint
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
import lightning as L
import logging
from utils import get_device

logging.basicConfig(level=logging.INFO,
                        format=('%(filename)s: '
                                '%(levelname)s: '
                                '%(funcName)s(): '
                                '%(lineno)d:\t'
                                '%(message)s')
                        )

logger = logging.getLogger(__name__)

# Set device to cuda if available, set to mps if available else cpu
device = get_device(T)
logger.info(f"Using device: {device}")
# It is important that the device is set before importing the model
from ddpm_model import DdpmLight,DdpmNet


batch_size = 32

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
mnist_train = datasets.MNIST(root="mnist_data", train=True, transform=transform, download=True)
mnist_test = datasets.MNIST(root="mnist_data", train=False, transform=transform, download=True)

train_size = int(0.9 * len(mnist_train))
val_size = len(mnist_train) - train_size

train_dataset, val_dataset = random_split(mnist_train, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

model = DdpmNet()

ddpm_light = DdpmLight(model).to(device)

checkpoint_callback = ModelCheckpoint(dirpath="ckpt", save_top_k=3, monitor="val_loss", filename="{epoch}-{val_loss:.2f}")

trainer = L.Trainer(max_epochs=200, callbacks=checkpoint_callback)
trainer.fit(model=ddpm_light, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
