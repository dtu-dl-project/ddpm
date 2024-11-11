import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from ddpm_model import DdpmLight,DdpmNet
import lightning as L

device = T.device("cuda" if T.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_train = datasets.MNIST(root="mnist_data", train=True, transform=transform, download=True)
mnist_test = datasets.MNIST(root="mnist_data", train=False, transform=transform, download=True)

train_dataloader = DataLoader(mnist_train, batch_size=100, shuffle=True)
test_dataloader = DataLoader(mnist_test, batch_size=100, shuffle=False)

model = DdpmNet().to(device)

ddpm_light = DdpmLight(model).to(device)

trainer = L.Trainer(limit_train_batches=100, max_epochs=1)
trainer.fit(model=ddpm_light, train_dataloaders=train_dataloader)

#ddpm_light = DdpmLight.load_from_checkpoint("model.ckpt", ddpmnet=model)

#s = ddpm_light.sample(1)[0].view(28, 28, 1)

#plt.imshow(s.cpu().detach().numpy(), cmap="gray")
#plt.savefig("sample.png")

