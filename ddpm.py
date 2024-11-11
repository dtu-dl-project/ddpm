from lightning.pytorch.callbacks import ModelCheckpoint
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
from ddpm_model import DdpmLight,DdpmNet
import lightning as L

device = T.device("cuda" if T.cuda.is_available() else "cpu")

batch_size = 64

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

model = DdpmNet().to(device)

ddpm_light = DdpmLight(model).to(device)

checkpoint_callback = ModelCheckpoint(dirpath="ckpt", save_top_k=3, monitor="val_loss", filename="{epoch}-{val_loss:.2f}")

trainer = L.Trainer(max_epochs=100, callbacks=checkpoint_callback)
trainer.fit(model=ddpm_light, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

#ddpm_light = DdpmLight.load_from_checkpoint("model.ckpt", ddpmnet=model)

#s = ddpm_light.sample(1)[0].view(28, 28, 1)

#plt.imshow(s.cpu().detach().numpy(), cmap="gray")
#plt.savefig("sample.png")

