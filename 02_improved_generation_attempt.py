#!/usr/bin/env python3
"""
Improved generation script that should produce better digit samples.
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

def improved_generation():
    """Generate samples using improved methods."""
    
# Load model
    model_path = "models/02_improved_attempt_300x200/best_model.pt"
    checkpoint = torch.load(model_path, map_location='cpu')
    
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
    
    # Method 1: Progressive refinement from real data
    print("Generating samples using progressive refinement...")
    
    # Get some real MNIST samples as starting points
    test_dataset = get_test_dataset('mnist', normalize=False)
    real_samples = []
    for i in range(16):
        img, label = test_dataset[i]
        real_samples.append(img.view(-1))
    
    real_batch = torch.stack(real_samples).to(device)
    real_batch = torch.bernoulli(real_batch)
    
    # Progressive sampling: start from real data, gradually make it more random
    sampler = GibbsSampler(model, device=device)
    
    # Step 1: Add noise to real data
    noisy_start = real_batch.clone()
    noise_level = 0.3  # 30% noise
    noise = torch.bernoulli(torch.ones_like(noisy_start) * noise_level)
    noisy_start = torch.where(noise == 1, 1 - noisy_start, noisy_start)
    
    # Step 2: Gentle annealing
    print("Applying gentle annealing...")
    temperature_schedule = [1.5, 1.3, 1.1, 1.0, 1.0, 1.0]  # More conservative temperatures
    refined_samples = sampler.anneal_sampling(
        noisy_start,
        temperature_schedule,
        steps_per_temp=2000  # More steps per temperature
    )
    
    # Method 2: Multiple restarts with best sample selection
    print("Generating multiple candidates...")
    
    all_candidates = []
    for restart in range(5):
        # Different initialization strategies
        if restart == 0:
            # Random initialization
            init = torch.bernoulli(torch.ones(16, model.visible_dim, device=device) * 0.5)
        elif restart == 1:
            # Biased toward MNIST-like patterns (more black pixels)
            init = torch.bernoulli(torch.ones(16, model.visible_dim, device=device) * 0.2)
        elif restart == 2:
            # Center-biased (digits often have features in center)
            init = torch.zeros(16, model.visible_dim, device=device)
            center_region = torch.zeros(28, 28)
            center_region[8:20, 8:20] = 0.7  # Higher probability in center
            init = torch.bernoulli(center_region.view(-1).repeat(16, 1).to(device))
        else:
            # Start from slightly corrupted real data
            corruption = torch.bernoulli(torch.ones_like(real_batch) * 0.4)
            init = torch.where(corruption == 1, 1 - real_batch, real_batch)
        
        # Long sampling
        candidate = sampler.sample_from_model(
            batch_size=16,
            n_steps=20000,  # Very long sampling
            initial_visible=init,
            temperature=0.8  # Slightly lower temperature
        )
        all_candidates.append(candidate)
    
    # Select best candidates based on how "digit-like" they are
    print("Selecting best samples...")
    
    best_samples = []
    for i in range(16):
        candidates_for_position = [cand[i] for cand in all_candidates]
        
        # Simple heuristic: prefer samples that aren't too random
        scores = []
        for candidate in candidates_for_position:
            # Score based on having some structure (not too random)
            img = candidate.view(28, 28)
            
            # Penalize pure randomness
            center_mass = img[10:18, 10:18].mean()  # Center should have some content
            edge_mass = torch.cat([img[0:3, :].flatten(), img[-3:, :].flatten(), 
                                  img[:, 0:3].flatten(), img[:, -3:].flatten()]).mean()
            
            # Good digits have more center content, less edge content
            structure_score = center_mass - 0.5 * edge_mass
            
            # Penalize too uniform (pure noise) or too sparse
            uniformity_penalty = -abs(candidate.mean() - 0.1)  # MNIST is mostly black
            
            total_score = structure_score + uniformity_penalty
            scores.append(total_score.item())
        
        # Pick best candidate for this position
        best_idx = np.argmax(scores)
        best_samples.append(candidates_for_position[best_idx])
    
    final_samples = torch.stack(best_samples)
    
    # Visualize results
    fig, axes = plt.subplots(3, 8, figsize=(16, 6))
    
    # Row 1: Original real data
    for i in range(8):
        axes[0, i].imshow(real_batch[i].view(28, 28).cpu().detach(), cmap='gray')
        axes[0, i].set_title('Real')
        axes[0, i].axis('off')
    
    # Row 2: Progressive refinement results
    for i in range(8):
        axes[1, i].imshow(refined_samples[i].view(28, 28).cpu().detach(), cmap='gray')
        axes[1, i].set_title('Refined')
        axes[1, i].axis('off')
    
    # Row 3: Best selected samples
    for i in range(8):
        axes[2, i].imshow(final_samples[i].view(28, 28).cpu().detach(), cmap='gray')
        axes[2, i].set_title('Best')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    
    # Save results
    os.makedirs('results/02_improved_generation_attempts', exist_ok=True)
    plt.savefig('results/02_improved_generation_attempts/improved_generation.png', dpi=150, bbox_inches='tight')
    print("✅ Saved improved samples to results/02_improved_generation_attempts/improved_generation.png")
    
    # Also save the raw tensors
    torch.save(final_samples, 'results/02_improved_generation_attempts/best_samples.pt')
    torch.save(refined_samples, 'results/02_improved_generation_attempts/refined_samples.pt')
    
    # Print statistics
    print(f"\nImproved Sample Statistics:")
    print(f"  Final samples mean: {final_samples.mean().item():.3f}")
    print(f"  Final samples std: {final_samples.std().item():.3f}")
    print(f"  Refined samples mean: {refined_samples.mean().item():.3f}")
    print(f"  Refined samples std: {refined_samples.std().item():.3f}")
    
    plt.show()

if __name__ == "__main__":
    improved_generation()