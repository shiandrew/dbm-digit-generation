#!/usr/bin/env python3
"""
Test the newly trained memory-optimized model.
This should show MUCH better generation results!
"""

import sys
import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from src.models.dbm import DBM
from src.sampling.gibbs_sampler import GibbsSampler
from src.data.loaders import get_test_dataset

def test_new_model():
    """Test the newly trained model's generation capabilities."""
    
    print("=== Testing New Memory-Optimized Model ===")
    
    # Load the new model
    model_path = "models/checkpoints_memory_opt/best_model.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Available checkpoints:")
        checkpoint_dir = "models/checkpoints_memory_opt"
        if os.path.exists(checkpoint_dir):
            for file in os.listdir(checkpoint_dir):
                print(f"  - {file}")
        return
    
    print(f"✅ Loading model from {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model_config = checkpoint.get('model_config', {})
    print(f"Model architecture: {model_config['visible_dim']} -> {model_config['hidden_dims']}")
    
    model = DBM(
        visible_dim=model_config['visible_dim'],
        hidden_dims=model_config['hidden_dims'],
        use_cuda=torch.cuda.is_available()
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Device: {device}")
    
    # Test reconstruction first
    print("\n=== Testing Reconstruction ===")
    test_dataset = get_test_dataset('mnist', normalize=False)
    
    # Get a few test samples
    test_samples = []
    test_labels = []
    for i in range(8):
        img, label = test_dataset[i]
        test_samples.append(torch.bernoulli(img.view(-1)))
        test_labels.append(label)
    
    test_batch = torch.stack(test_samples).to(device)
    
    # Reconstruct
    with torch.no_grad():
        hiddens = model.forward(test_batch)
        recon_probs, _ = model.sample_visible_given_hidden(hiddens[0])
        reconstructions = torch.bernoulli(recon_probs)
    
    recon_error = torch.nn.functional.mse_loss(test_batch, reconstructions).item()
    print(f"Reconstruction error: {recon_error:.6f}")
    
    # Test generation methods
    print("\n=== Testing Generation Methods ===")
    sampler = GibbsSampler(model, device=device)
    
    # Method 1: From random initialization
    print("Method 1: Random initialization with long sampling...")
    random_init = torch.bernoulli(torch.ones(8, model.visible_dim, device=device) * 0.3)  # Biased toward black
    samples_random = sampler.sample_from_model(
        batch_size=8,
        n_steps=5000,
        initial_visible=random_init,
        temperature=0.7
    )
    
    print(f"  Sample range: [{samples_random.min().item():.3f}, {samples_random.max().item():.3f}]")
    print(f"  Sample mean: {samples_random.mean().item():.3f}")
    print(f"  Sample std: {samples_random.std().item():.3f}")
    
    # Method 2: From real data (should work well)
    print("Method 2: Starting from real data...")
    noisy_real = test_batch.clone()
    noise = torch.bernoulli(torch.ones_like(noisy_real) * 0.1)  # 10% noise
    noisy_real = torch.where(noise == 1, 1 - noisy_real, noisy_real)
    
    samples_from_real = sampler.sample_from_model(
        batch_size=8,
        n_steps=2000,
        initial_visible=noisy_real,
        temperature=0.8
    )
    
    print(f"  Sample range: [{samples_from_real.min().item():.3f}, {samples_from_real.max().item():.3f}]")
    print(f"  Sample mean: {samples_from_real.mean().item():.3f}")
    print(f"  Sample std: {samples_from_real.std().item():.3f}")
    
    # Method 3: Annealing
    print("Method 3: Simulated annealing...")
    temp_schedule = [1.5, 1.2, 1.0, 0.8, 0.6]
    samples_annealed = sampler.anneal_sampling(
        random_init,
        temp_schedule,
        steps_per_temp=800
    )
    
    print(f"  Sample range: [{samples_annealed.min().item():.3f}, {samples_annealed.max().item():.3f}]")
    print(f"  Sample mean: {samples_annealed.mean().item():.3f}")
    print(f"  Sample std: {samples_annealed.std().item():.3f}")
    
    # Visualize results
    print("\n=== Creating Visualization ===")
    
    fig, axes = plt.subplots(5, 8, figsize=(16, 10))
    
    # Row 1: Original test images
    for i in range(8):
        axes[0, i].imshow(test_batch[i].view(28, 28).cpu().detach(), cmap='gray')
        axes[0, i].set_title(f'Real: {test_labels[i]}', fontsize=10)
        axes[0, i].axis('off')
    
    # Row 2: Reconstructions
    for i in range(8):
        axes[1, i].imshow(reconstructions[i].view(28, 28).cpu().detach(), cmap='gray')
        axes[1, i].set_title('Reconstruction', fontsize=10)
        axes[1, i].axis('off')
    
    # Row 3: Random initialization samples
    for i in range(8):
        axes[2, i].imshow(samples_random[i].view(28, 28).cpu().detach(), cmap='gray')
        axes[2, i].set_title('Random Init', fontsize=10)
        axes[2, i].axis('off')
    
    # Row 4: From real data samples
    for i in range(8):
        axes[3, i].imshow(samples_from_real[i].view(28, 28).cpu().detach(), cmap='gray')
        axes[3, i].set_title('From Real', fontsize=10)
        axes[3, i].axis('off')
    
    # Row 5: Annealed samples
    for i in range(8):
        axes[4, i].imshow(samples_annealed[i].view(28, 28).cpu().detach(), cmap='gray')
        axes[4, i].set_title('Annealed', fontsize=10)
        axes[4, i].axis('off')
    
    # Add row labels
    row_labels = ['Original', 'Reconstruction', 'Random Init', 'From Real Data', 'Annealed']
    for i, label in enumerate(row_labels):
        axes[i, 0].text(-0.1, 0.5, label, transform=axes[i, 0].transAxes, 
                       rotation=90, va='center', ha='right', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save results
    os.makedirs('results/new_model_test', exist_ok=True)
    plt.savefig('results/new_model_test/generation_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Saved results to results/new_model_test/generation_comparison.png")
    
    plt.show()
    
    # Analysis
    print("\n=== Analysis ===")
    
    # Check if we're still getting QR codes (random noise)
    for method, samples, name in [
        (1, samples_random, "Random Init"),
        (2, samples_from_real, "From Real"),
        (3, samples_annealed, "Annealed")
    ]:
        mean_val = samples.mean().item()
        std_val = samples.std().item()
        
        if abs(mean_val - 0.5) < 0.05 and abs(std_val - 0.5) < 0.05:
            print(f"⚠️  {name}: Still looks like random noise (mean={mean_val:.3f}, std={std_val:.3f})")
        elif mean_val < 0.3:
            print(f"✅ {name}: Good MNIST-like statistics (mean={mean_val:.3f}, std={std_val:.3f})")
        else:
            print(f"🔄 {name}: Improved but could be better (mean={mean_val:.3f}, std={std_val:.3f})")
    
    print(f"\n🎯 Best reconstruction error: {recon_error:.6f}")
    
    if recon_error < 0.01:
        print("✅ Excellent reconstruction capability!")
    elif recon_error < 0.05:
        print("✅ Good reconstruction capability!")
    else:
        print("⚠️  Reconstruction could be better.")

if __name__ == "__main__":
    test_new_model()