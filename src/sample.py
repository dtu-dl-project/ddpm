import torch as T
import matplotlib.pyplot as plt
import logging
from utils import get_device
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("checkpoint", type=str, help="Path to the checkpoint file")
args = parser.parse_args()

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

model = DdpmNet()


model_path = "model"


ddpm_light = DdpmLight.load_from_checkpoint(args.checkpoint, ddpmnet=model)
ddpm_light.eval()

sample_size = 64
ddpm_light = ddpm_light.to(device)
with T.no_grad():
    s = ddpm_light.sample(sample_size).view(sample_size, 32, 32, 1)

fig = plt.figure(figsize=(16, 16))
columns = 8
rows = 8
for i in range(1, columns*rows +1):
    img = s[i-1].cpu().detach().numpy().reshape(32, 32, 1)
    fig.add_subplot(rows, columns, i)
    plt.imshow(img, cmap="Greys")

plt.savefig("sample.png")
