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
from transformers import get_cosine_schedule_with_warmup

class SchedulerCallback(L.Callback):
    def __init__(self, warmup_steps, total_steps):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
    
    def on_train_start(self, trainer, pl_module):
        optimizer = pl_module.optimizers()  # Get the optimizer
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.total_steps
        )
        trainer.lr_schedulers = [{'scheduler': scheduler, 'interval': 'step'}]

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
    transforms.Resize((32, 32)),       # Resizes the image to 32x32
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 2) - 1),
    # transforms.Normalize((0,), (1,))
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

epochs = 200

checkpoint_callback = ModelCheckpoint(dirpath="ckpt", save_top_k=3, monitor="val_loss", filename="{epoch}-{val_loss:.4f}")

num_training_steps = len(train_dataloader) * epochs
warmup_steps = int(0.1 * num_training_steps)
scheduler_callback = SchedulerCallback(warmup_steps=warmup_steps, total_steps=num_training_steps)

trainer = L.Trainer(max_epochs=epochs, callbacks=[checkpoint_callback, scheduler_callback])
trainer.fit(model=ddpm_light, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
