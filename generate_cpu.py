"""Generate images using StyleGAN3 with a pretrained network pickle in CPU mode."""

import os
import re
from typing import List, Optional, Tuple, Union

import click
import numpy as np
import PIL.Image
import torch

import sys
import subprocess

def setup_stylegan3():
    """Ensure StyleGAN3 is set up and accessible"""
    # Check if StyleGAN3 directory exists
    if not os.path.exists('stylegan3'):
        print("StyleGAN3 directory not found. Cloning repository...")
        subprocess.run(['git', 'clone', 'https://github.com/NVlabs/stylegan3.git'], check=True)
    
    # Add StyleGAN3 to Python path
    sys.path.insert(0, 'stylegan3')

# Set up StyleGAN3
setup_stylegan3()

# Now we can import StyleGAN3 modules
import dnnlib
import legacy

# Force CPU usage
torch.cuda.is_available = lambda: False
device = torch.device('cpu')

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''
    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    outdir: str,
    class_idx: Optional[int] = None,
    noise_mode: str = 'const',
):
    """Generate images using pretrained network pickle.

    Examples:
    \b
    # Generate an image using pre-trained StyleGAN3 model
    python generate_cpu.py --outdir=out --seeds=0-5 --network=<path-to-pkl>
    """
    print('Loading networks from "%s"...' % network_pkl)
    
    # Load the model
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)

    os.makedirs(outdir, exist_ok=True)

    # Labels
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')

    # Generate images
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        
        # Use a different seed for each image
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        
        # Generate the image
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
        
        # Save the image
        output_file = f'{outdir}/seed{seed:04d}.png'
        print(f'Saving to {output_file}')
        img.save(output_file)

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def main(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    class_idx: Optional[int],
    noise_mode: str,
    outdir: str,
):
    """Generate images using pretrained network pickle."""
    # Print some debug information
    print(f"Running on device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    generate_images(
        network_pkl=network_pkl,
        seeds=seeds,
        truncation_psi=truncation_psi,
        class_idx=class_idx,
        noise_mode=noise_mode,
        outdir=outdir,
    )

if __name__ == "__main__":
    main() 