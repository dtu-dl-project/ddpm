import torch as T
import matplotlib.pyplot as plt
import logging
from utils import get_device, get_dataset
from argparse import ArgumentParser
from torchmetrics.image.fid import FrechetInceptionDistance
from ddpm_model import DdpmLight, DdpmNet
from torch.utils.data import DataLoader
import math
import re

# Function to parse checkpoint filename
def parse_checkpoint_filename(filename):
    pattern = r"^(?P<dataset_name>\w+)_unet_dim=(?P<unet_dim>\d+)_beta=(?P<beta_schedule>\w+)_loss=(?P<loss>\w+)_lr=(?P<lr>[0-9.]+)_cond=(?P<cond>\w+)"
    match = re.match(pattern, filename)
    if not match:
        raise ValueError(f"Invalid checkpoint filename format: {filename}")
    
    params = match.groupdict()
    params['unet_dim'] = int(params['unet_dim'])
    params['lr'] = float(params['lr'])
    params['cond'] = params['cond'].lower() == 'true'
    return params

# Argument parser
parser = ArgumentParser()
parser.add_argument("checkpoint", type=str, help="Path to the checkpoint file")
parser.add_argument("--skip_fid", action="store_true", help="Skip the computation of the FID score")

args = parser.parse_args()

# Logging configuration
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

# Extract parameters from the checkpoint filename
try:
    checkpoint_params = parse_checkpoint_filename(args.checkpoint)
    logger.info(f"Extracted parameters from checkpoint: {checkpoint_params}")
except ValueError as e:
    logger.error(e)
    exit(1)

# Initialize the model using extracted parameters
num_channels = 3 if checkpoint_params['dataset_name'] == 'CIFAR10' else 1
model = DdpmNet(
    unet_dim=checkpoint_params['unet_dim'],
    channels=num_channels,
    img_size=32,
    beta_schedule=checkpoint_params['beta_schedule'],
    cond=checkpoint_params['cond']
)
ddpm_light = DdpmLight.load_from_checkpoint(args.checkpoint, ddpmnet=model)
ddpm_light.eval().to(device)

# Generate samples
sample_size = 100
columnrow = int(math.sqrt(sample_size))

# Generate class labels for conditional sampling
klass = T.cat([T.full((columnrow,), i) for i in range(columnrow)])
klass = klass.to(device)

logger.info(f"Generating {sample_size} samples...")
with T.no_grad():
    generated_samples = ddpm_light.sample(sample_size, klass).view(sample_size, num_channels, 32, 32).to(device)

# Normalize generated samples to [0, 1]
generated_samples = ((generated_samples + 1) / 2).clamp(0, 1)
if num_channels == 1:
    generated_samples = generated_samples.repeat(1, 3, 1, 1)  # Convert to RGB 

# Visualize generated samples
logger.info("Saving generated samples...")
fig = plt.figure(figsize=(8, 8))
columns = columnrow
rows = columnrow
for i in range(1, columns * rows + 1):
    img = generated_samples[i - 1].cpu().detach().numpy().transpose(1, 2, 0).squeeze()
    fig.add_subplot(rows, columns, i)
    plt.axis('off')
    if num_channels == 1:
        plt.imshow(img, cmap="gray")
    else:
        plt.imshow(img)

plt.savefig("sample.png")

if args.skip_fid is not True:
    generated_samples = generated_samples.to(dtype=T.float64)
    # Initialize FID metric
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    # Add real images to FID
    logger.info("Adding real images to FID computation...")
    _, _, test_dataset = get_dataset(checkpoint_params['dataset_name'])
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    for real_batch in test_dataloader:
        real_images, _ = real_batch
        # Normalize real images to [0, 1]
        real_images = ((real_images + 1) / 2).clamp(0, 1).to(device)
        if num_channels == 1:
            real_images = real_images.repeat(1, 3, 1, 1)  # Convert to RGB
        # Move real images to CPU
        real_images = real_images.to(dtype=T.float64)
        fid.update(real_images, real=True)
        break  # Use only the first batch for simplicity

    # Add generated images to FID
    logger.info("Adding generated images to FID computation...")
    fid.update(generated_samples, real=False)

    # Compute FID score
    fid_score = fid.compute()
    logger.info(f"FID Score: {fid_score}")