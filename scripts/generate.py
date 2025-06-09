#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.dbm import DBM
from src.sampling.gibbs_sampler import GibbsSampler
from src.utils.visualization import plot_samples

def parse_args():
    parser = argparse.ArgumentParser(description='Generate samples from trained DBM')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to trained model checkpoint')
    parser.add_argument('--num_samples', type=int, default=16,
                      help='Number of samples to generate')
    parser.add_argument('--num_steps', type=int, default=1000,
                      help='Number of Gibbs sampling steps')
    parser.add_argument('--output_dir', type=str, default='results/samples',
                      help='Directory to save generated samples')
    parser.add_argument('--temperature', type=float, default=1.0,
                      help='Sampling temperature')
    parser.add_argument('--use_annealing', action='store_true',
                      help='Use simulated annealing for sampling')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} not found!")
        return
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location='cpu')
    
    # Create model
    model_config = checkpoint.get('model_config', {})
    model = DBM(
        visible_dim=model_config['visible_dim'],
        hidden_dims=model_config['hidden_dims'],
        use_cuda=torch.cuda.is_available()
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded DBM with {len(model.hidden_dims)} hidden layers: {model.hidden_dims}")
    
    # Create sampler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sampler = GibbsSampler(model, device=device)
    
    print(f"Generating {args.num_samples} samples with {args.num_steps} Gibbs steps...")
    
    # Generate samples
    if args.use_annealing:
        # Use simulated annealing for better quality
        temperature_schedule = [5.0, 3.0, 2.0, 1.5, 1.0, 0.8]
        initial_visible = torch.bernoulli(
            torch.ones(args.num_samples, model.visible_dim, device=device) * 0.5
        )
        samples = sampler.anneal_sampling(
            initial_visible, 
            temperature_schedule, 
            steps_per_temp=args.num_steps // len(temperature_schedule)
        )
    else:
        # Standard Gibbs sampling
        samples = sampler.sample_from_model(
            batch_size=args.num_samples,
            n_steps=args.num_steps,
            temperature=args.temperature
        )
    
    # Move to CPU for visualization
    samples = samples.cpu()
    
    # Reshape samples for visualization (assuming MNIST 28x28)
    if model.visible_dim == 784:
        samples_reshaped = samples.view(args.num_samples, 1, 28, 28)
    else:
        # For other sizes, try to find square dimensions
        size = int(np.sqrt(model.visible_dim))
        if size * size == model.visible_dim:
            samples_reshaped = samples.view(args.num_samples, 1, size, size)
        else:
            print(f"Warning: Cannot reshape {model.visible_dim} dimensions to square image")
            samples_reshaped = samples.view(args.num_samples, 1, int(np.sqrt(model.visible_dim)), -1)
    
    # Plot and save samples
    output_path = os.path.join(args.output_dir, 'generated_samples.png')
    fig = plot_samples(samples_reshaped, n_samples=args.num_samples, save_path=output_path)
    
    print(f"Generated samples saved to: {output_path}")
    
    # Save raw samples as well
    raw_output_path = os.path.join(args.output_dir, 'generated_samples.pt')
    torch.save(samples, raw_output_path)
    print(f"Raw samples saved to: {raw_output_path}")
    
    # Display some statistics
    print(f"\nSample statistics:")
    print(f"  Shape: {samples.shape}")
    print(f"  Min value: {samples.min().item():.4f}")
    print(f"  Max value: {samples.max().item():.4f}")
    print(f"  Mean value: {samples.mean().item():.4f}")
    print(f"  Std value: {samples.std().item():.4f}")
    
    # Show plot if running interactively
    try:
        plt.show()
    except:
        pass

if __name__ == '__main__':
    main()