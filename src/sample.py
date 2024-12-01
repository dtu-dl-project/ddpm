import torch as T
import matplotlib.pyplot as plt
import logging
from utils import get_device
from argparse import ArgumentParser
from torchmetrics.image.fid import FrechetInceptionDistance
from train import test_dataloader
from ddpm_model import DdpmLight, DdpmNet

parser = ArgumentParser()
parser.add_argument("checkpoint", type=str, help="Path to the checkpoint file")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO,
                    format=('%(filename)s: '
                            '%(levelname)s: '
                            '%(funcName)s(): '
                            '%(lineno)d:\t'
                            '%(message)s'))

logger = logging.getLogger(__name__)

# Set device to cuda if available, set to mps if available else cpu
device = get_device(T)
logger.info(f"Using device: {device}")

# Load the model
model = DdpmNet()
ddpm_light = DdpmLight.load_from_checkpoint(args.checkpoint, ddpmnet=model)
ddpm_light.eval().to(device)

# Generate samples
sample_size = 64
logger.info(f"Generating {sample_size} samples...")
with T.no_grad():
    generated_samples = ddpm_light.sample(sample_size).view(sample_size, 1, 32, 32).to(device)

# Normalize generated samples to [0, 1]
generated_samples = ((generated_samples + 1) / 2).clamp(0, 1)
generated_samples = generated_samples.repeat(1, 3, 1, 1)  # Convert to RGB 

generated_samples = generated_samples.to(dtype=T.float64)

# Initialize FID metric
fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

# Add real images to FID
logger.info("Adding real images to FID computation...")
for real_batch in test_dataloader:
    real_images, _ = real_batch
    # Normalize real images to [0, 1]
    real_images = ((real_images + 1) / 2).clamp(0, 1).to(device)
    real_images = real_images.repeat(1, 3, 1, 1)  # Convert to RGB
    # Move real images to CPU
    real_images = real_images.to(dtype=T.float64)
    fid.update(real_images, real=True)
      # Use only the first batch for simplicity

# Add generated images to FID
logger.info("Adding generated images to FID computation...")
fid.update(generated_samples, real=False)

# Compute FID score
fid_score = fid.compute()
logger.info(f"FID Score: {fid_score}")

# Visualize generated samples
logger.info("Saving generated samples...")
fig = plt.figure(figsize=(8, 8))
columns = 8
rows = 8
for i in range(1, columns * rows + 1):
    img = generated_samples[i - 1].cpu().detach().numpy().transpose(1, 2, 0).squeeze()
    fig.add_subplot(rows, columns, i)
    plt.imshow(img, cmap="Greys")

plt.savefig("sample.png")
