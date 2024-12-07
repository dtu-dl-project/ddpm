import torch as T
import matplotlib.pyplot as plt
import logging
from utils import get_device, get_dataset
from argparse import ArgumentParser
from torchmetrics.image.fid import FrechetInceptionDistance
from ddpm_model import DdpmLight, DdpmNet
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
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
        r"(?:_loss=(?P<loss>[a-zA-Z0-9_]+))?"
        r"(?:_lr=(?P<lr>[0-9.e-]+))?"
        r"(?:_cond=(?P<cond>\w+))?"
        r"(?:_bs=(?P<bs>\d+))?"
        r"_epoch=(?P<epoch>\d+)-val_loss=(?P<val_loss>\d+\.\d+)\.ckpt$"
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
parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to generate")
parser.add_argument("--skip_fid", action="store_true", help="Skip the computation of the FID score")
parser.add_argument("--skip_plot", action="store_true", help="Skip the plotting of the generated samples")

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
sample_size = args.num_samples
batch_size = 100
columnrow = int(math.sqrt(batch_size))

# Generate class labels for conditional sampling
klass = T.cat([T.full((columnrow,), i) for i in range(columnrow)])
klass = klass.to(device)

# Initialize FID metric
fid = FrechetInceptionDistance(feature=2048, normalize=True, reset_real_features=False, input_img_size=(3,32,32)).to(device)
fid.set_dtype(T.float64)

# Add real images to FID computation
if args.skip_fid is False:
    train_dataset, _, _ = get_dataset(checkpoint_params['dataset_name'])
    subset_indices = list(range(sample_size))  # Indices for the subset
    subset_dataset = Subset(train_dataset, subset_indices)
    train_dataloader = DataLoader(subset_dataset, batch_size=checkpoint_params['bs'], shuffle=False)
    for real_batch in tqdm(train_dataloader, desc="Adding real images to FID computation"):
        real_images, _ = real_batch
        # Normalize real images to [0, 1]
        real_images = ((real_images + 1) / 2).clamp(0, 1).to(device)
        if num_channels == 1:
            real_images = real_images.repeat(1, 3, 1, 1)  # Convert to RGB
        # Move real images to CPU
        real_images = real_images.to(dtype=T.float64)
        fid.update(real_images, real=True)

with T.no_grad():
    for i in tqdm(range(sample_size // batch_size), desc="Generating Samples"):
        generated_samples = ddpm_light.sample(batch_size, klass).view(batch_size, num_channels, 32, 32).to(device)

        # Normalize generated samples to [0, 1]
        generated_samples = ((generated_samples + 1) / 2).clamp(0, 1)
        if num_channels == 1:
            generated_samples = generated_samples.repeat(1, 3, 1, 1)  # Convert to RGB 

        if args.skip_fid is False:
            # Add generated images to FID
            logger.info("Adding generated images to FID computation...")
            generated_samples = generated_samples.to(dtype=T.float64)
            fid.update(generated_samples, real=False)

# Compute FID score
fid_score = fid.compute()
logger.info(f"FID Score: {fid_score}")     

# Visualize generated samples
if args.skip_plot is False:
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

    filename = f"samples/{checkpoint_params['dataset_name']}_{checkpoint_params['cond']}_{checkpoint_params['unet_dim']}_{checkpoint_params['beta_schedule']}_{checkpoint_params['epoch']}_{checkpoint_params['val_loss']}"
    if args.skip_fid is False:
        filename += f"_fid={fid_score:.2f}"
    filename += ".png"
    plt.savefig(filename)
