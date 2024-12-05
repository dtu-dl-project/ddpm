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
def parse_checkpoint_filename(filename, default_params=None):
    # Default values for missing parameters
    if default_params is None:
        default_params = {
            "dataset_name": "MNIST",
            "unet_dim": 32,
            "beta_schedule": "linear",
            "loss": "smooth_l1",
            "lr": 0.0003,
            "cond": False,
            "bs": 32,
            "epoch": 0,
            "val_loss": None
        }
    
    # Extract the base filename
    base_filename = filename.split('/')[-1]
    
    # Adjusted regex pattern to allow optional parameters
    pattern = (
        r"^(?P<dataset_name>\w+)_unet_dim=(?P<unet_dim>\d+)_beta=(?P<beta_schedule>\w+)"
        r"(?:_loss=(?P<loss>\w+))?(?:_lr=(?P<lr>[0-9.]+))?"
        r"(?:_cond=(?P<cond>\w+))?(?:_bs=(?P<bs>\d+))?_epoch=(?P<epoch>\d+)-val_loss=(?P<val_loss>[0-9.]+)\.ckpt$"
    )
    
    # Match the filename against the pattern
    match = re.match(pattern, base_filename)
    if not match:
        raise ValueError(f"Invalid checkpoint filename format: {base_filename}")
    
    # Extract parameters
    params = match.groupdict()
    
    # Convert extracted parameters and set defaults for missing ones
    params = {**default_params, **{k: v for k, v in params.items() if v is not None}}
    params['unet_dim'] = int(params['unet_dim'])
    params['lr'] = float(params['lr']) if params['lr'] is not None else default_params['lr']
    params['cond'] = params['cond'].lower() == 'true' if isinstance(params['cond'], str) else default_params['cond']
    params['bs'] = int(params['bs']) if params['bs'] is not None else default_params['bs']
    params['epoch'] = int(params['epoch'])
    params['val_loss'] = float(params['val_loss']) if params['val_loss'] is not None else default_params['val_loss']
    
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
    loss_type=checkpoint_params['loss'],
    lr=checkpoint_params['lr'],
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

if args.skip_fid is not True:
    generated_samples = generated_samples.to(dtype=T.float64)
    # Initialize FID metric
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    # Add real images to FID
    logger.info("Adding real images to FID computation...")
    _, _, test_dataset = get_dataset(checkpoint_params['dataset_name'])
    test_dataloader = DataLoader(test_dataset, batch_size=checkpoint_params['bs'], shuffle=False)
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

filename = f"samples/{checkpoint_params['dataset_name']}_{checkpoint_params['cond']}_{checkpoint_params['unet_dim']}_{checkpoint_params['beta_schedule']}_{checkpoint_params['epoch']}_{checkpoint_params['val_loss']}"
if args.skip_fid is not True:
    filename += f"_fid={fid_score:.2f}"
filename += ".png"
plt.savefig()
