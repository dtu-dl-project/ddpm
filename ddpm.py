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

model = DdpmNet()
ddpm_light = DdpmLight(model).to(device)

checkpoint_callback = ModelCheckpoint(dirpath="ckpt", save_top_k=3, monitor="val_loss", filename="{epoch}-{val_loss:.2f}")

trainer = L.Trainer(limit_train_batches=100, limit_val_batches=10, max_epochs=1, callbacks=checkpoint_callback)
trainer.fit(model=ddpm_light, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

#ddpm_light = DdpmLight.load_from_checkpoint("model.ckpt", ddpmnet=model)

# sample_size = 64
# ddpm_light = ddpm_light.to(device)
# s = ddpm_light.sample(sample_size).view(sample_size, 28, 28, 1)

# fig = plt.figure(figsize=(16, 16))
# columns = 8
# rows = 8
# for i in range(1, columns*rows +1):
#     img = s[i-1].cpu().detach().numpy().reshape(28, 28, 1)
#     fig.add_subplot(rows, columns, i)
#     plt.imshow(img)

# plt.savefig("sample.png")
