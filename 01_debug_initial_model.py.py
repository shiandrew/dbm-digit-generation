#!/usr/bin/env python3
"""
Debug script to diagnose and fix DBM generation issues.
Run this to understand why you're getting QR code patterns instead of digits.
"""

import sys
import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.dbm import DBM
from src.sampling.gibbs_sampler import GibbsSampler
from src.data.loaders import get_test_dataset

def debug_model_and_generation():
    """Debug the model and generation process."""
    
    print("=== DBM Generation Debug ===\n")
    
    # 1. Load and inspect the model
    model_path = "models/01_initial_model_500x500/best_model.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Available files:")
        checkpoint_dir = Path("models/01_initial_model_500x500")
        if checkpoint_dir.exists():
            for file in checkpoint_dir.glob("*.pt"):
                print(f"  - {file}")
        return
    
    print(f"✅ Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create model
    model_config = checkpoint.get('model_config', {})
    model = DBM(
        visible_dim=model_config['visible_dim'],
        hidden_dims=model_config['hidden_dims'],
        use_cuda=torch.cuda.is_available()
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"Model architecture: {model_config['visible_dim']} -> {model_config['hidden_dims']}")
    print(f"Device: {device}")
    
    # 2. Test model on real data first
    print("\n=== Testing Model on Real Data ===")
    test_dataset = get_test_dataset('mnist', normalize=False)
    
    # Get a few real samples
    real_samples = []
    for i in range(5):
        img, label = test_dataset[i]
        real_samples.append((img, label))
    
    # Test reconstruction
    real_batch = torch.stack([img for img, _ in real_samples])
    real_batch = real_batch.view(5, -1).to(device)
    real_batch = torch.bernoulli(real_batch)  # Binarize
    
    with torch.no_grad():
        hiddens = model.forward(real_batch)
        visible_probs, _ = model.sample_visible_given_hidden(hiddens[0])
    
    print("Real data reconstruction test:")
    print(f"  Input range: [{real_batch.min().item():.3f}, {real_batch.max().item():.3f}]")
    print(f"  Reconstruction range: [{visible_probs.min().item():.3f}, {visible_probs.max().item():.3f}]")
    print(f"  Reconstruction error: {torch.nn.functional.mse_loss(real_batch, visible_probs).item():.6f}")
    
    # 3. Test different generation methods
    print("\n=== Testing Different Generation Methods ===")
    
    sampler = GibbsSampler(model, device=device)
    
    # Method 1: Short Gibbs chain (current method)
    print("Method 1: Short Gibbs chain (1000 steps)")
    samples_short = sampler.sample_from_model(batch_size=16, n_steps=1000)
    print(f"  Sample range: [{samples_short.min().item():.3f}, {samples_short.max().item():.3f}]")
    print(f"  Sample mean: {samples_short.mean().item():.3f}")
    print(f"  Sample std: {samples_short.std().item():.3f}")
    
    # Method 2: Much longer Gibbs chain
    print("\nMethod 2: Long Gibbs chain (10000 steps)")
    samples_long = sampler.sample_from_model(batch_size=16, n_steps=10000)
    print(f"  Sample range: [{samples_long.min().item():.3f}, {samples_long.max().item():.3f}]")
    print(f"  Sample mean: {samples_long.mean().item():.3f}")
    print(f"  Sample std: {samples_long.std().item():.3f}")
    
    # Method 3: Annealing
    print("\nMethod 3: Simulated annealing")
    initial_visible = torch.bernoulli(torch.ones(16, model.visible_dim, device=device) * 0.5)
    temperature_schedule = [5.0, 3.0, 2.0, 1.5, 1.0, 0.5]
    samples_annealed = sampler.anneal_sampling(
        initial_visible, 
        temperature_schedule, 
        steps_per_temp=1000
    )
    print(f"  Sample range: [{samples_annealed.min().item():.3f}, {samples_annealed.max().item():.3f}]")
    print(f"  Sample mean: {samples_annealed.mean().item():.3f}")
    print(f"  Sample std: {samples_annealed.std().item():.3f}")
    
    # Method 4: Start from real data
    print("\nMethod 4: Start from real data and sample")
    samples_from_data = sampler.sample_from_model(
        batch_size=16, 
        n_steps=5000,
        initial_visible=real_batch[:16] if real_batch.size(0) >= 16 else real_batch.repeat(4, 1)[:16]
    )
    print(f"  Sample range: [{samples_from_data.min().item():.3f}, {samples_from_data.max().item():.3f}]")
    print(f"  Sample mean: {samples_from_data.mean().item():.3f}")
    print(f"  Sample std: {samples_from_data.std().item():.3f}")
    
    # 4. Visualize all methods
    print("\n=== Creating Comparison Visualization ===")
    
    fig, axes = plt.subplots(6, 4, figsize=(12, 18))
    
    # Real data
    for i in range(4):
        if i < len(real_samples):
            img, label = real_samples[i]
            axes[0, i].imshow(img.squeeze(), cmap='gray')
            axes[0, i].set_title(f'Real: {label}')
        axes[0, i].axis('off')
    
    # Reconstructions
    for i in range(4):
        if i < visible_probs.size(0):
            recon = visible_probs[i].view(28, 28).cpu()
            axes[1, i].imshow(recon, cmap='gray')
            axes[1, i].set_title('Reconstruction')
        axes[1, i].axis('off')
    
    # Method 1: Short
    for i in range(4):
        sample = samples_short[i].view(28, 28).cpu().detach()
        axes[2, i].imshow(sample, cmap='gray')
        axes[2, i].set_title('Short Gibbs')
        axes[2, i].axis('off')
    
    # Method 2: Long
    for i in range(4):
        sample = samples_long[i].view(28, 28).cpu().detach()
        axes[3, i].imshow(sample, cmap='gray')
        axes[3, i].set_title('Long Gibbs')
        axes[3, i].axis('off')
    
    # Method 3: Annealing
    for i in range(4):
        sample = samples_annealed[i].view(28, 28).cpu().detach()
        axes[4, i].imshow(sample, cmap='gray')
        axes[4, i].set_title('Annealing')
        axes[4, i].axis('off')
    
    # Method 4: From data
    for i in range(4):
        sample = samples_from_data[i].view(28, 28).cpu().detach()
        axes[5, i].imshow(sample, cmap='gray')
        axes[5, i].set_title('From Real Data')
        axes[5, i].axis('off')
    
    # Add row labels
    row_labels = ['Real MNIST', 'Reconstructions', 'Short Gibbs (1K)', 
                  'Long Gibbs (10K)', 'Annealing', 'From Real Data']
    for i, label in enumerate(row_labels):
        axes[i, 0].text(-0.1, 0.5, label, transform=axes[i, 0].transAxes, 
                       rotation=90, va='center', ha='right', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the comparison
    os.makedirs('results/01_initial_model_debug', exist_ok=True)
    plt.savefig('results/01_initial_model_debug/generation_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Saved comparison to results/01_initial_model_debug/generation_comparison.png")
    
    # 5. Analyze weight and bias statistics
    print("\n=== Model Weight Analysis ===")
    
    for i, weight in enumerate(model.weights):
        print(f"Weight {i}: shape={weight.shape}")
        print(f"  Mean: {weight.mean().item():.6f}")
        print(f"  Std: {weight.std().item():.6f}")
        print(f"  Min: {weight.min().item():.6f}")
        print(f"  Max: {weight.max().item():.6f}")
        
        # Check for potential issues
        if weight.std().item() < 0.001:
            print(f"  ⚠️  WARNING: Very small weight variance - model may not have learned")
        if torch.isnan(weight).any():
            print(f"  ❌ ERROR: NaN weights detected!")
        if torch.isinf(weight).any():
            print(f"  ❌ ERROR: Infinite weights detected!")
    
    print("\nBias Analysis:")
    for i, bias in enumerate(model.biases):
        layer_name = "visible" if i == 0 else f"hidden_{i-1}"
        print(f"{layer_name} bias: mean={bias.mean().item():.6f}, std={bias.std().item():.6f}")
    
    # 6. Recommendations
    print("\n=== Recommendations ===")
    
    if samples_long.std().item() < 0.1:
        print("❌ Issue: Samples have very low variance (too uniform)")
        print("💡 Fix: Try longer training or different hyperparameters")
    
    if samples_annealed.mean().item() > 0.8 or samples_annealed.mean().item() < 0.2:
        print("❌ Issue: Samples are too bright or too dark")
        print("💡 Fix: Check data preprocessing and model biases")
    
    if torch.nn.functional.mse_loss(real_batch, visible_probs).item() > 0.1:
        print("❌ Issue: Poor reconstruction quality")
        print("💡 Fix: Model needs more training or better architecture")
    
    # Check if any method produces better results
    best_method = "Short Gibbs"
    if samples_long.std().item() > samples_short.std().item():
        best_method = "Long Gibbs"
    if samples_annealed.std().item() > max(samples_short.std().item(), samples_long.std().item()):
        best_method = "Annealing"
    
    print(f"✅ Best generation method appears to be: {best_method}")
    
    print(f"\n=== Next Steps ===")
    print("1. Look at results/debug/generation_comparison.png")
    print("2. If samples still look like QR codes, try:")
    print("   - Much longer training (more epochs)")
    print("   - Different learning rate")
    print("   - Different architecture")
    print("   - Check data preprocessing")
    print("3. Use the best generation method identified above")
    
    plt.show()

if __name__ == "__main__":
    debug_model_and_generation()