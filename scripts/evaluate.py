#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.dbm import DBM
from src.data.loaders import get_test_dataset
from src.sampling.gibbs_sampler import GibbsSampler
from src.evaluation.metrics import compute_reconstruction_error, compute_free_energy

# def parse_args():
#     parser = argparse.ArgumentParser(description='Evaluate trained DBM')
#     parser.add_argument('--model_path', type=str, required=True,
#                       help='Path to trained model checkpoint')
#     parser.add_argument('--dataset', type=str, default='mnist',
#                       choices=['mnist', 'fashion_mnist'],
#                       help='Dataset to evaluate on')
#     parser.add_argument('--batch_size', type=int, default=64,
#                       help='Batch size for evaluation')
#     parser.add_argument('--num_samples', type=int, default=100,
#                       help='Number of samples for likelihood estimation')
#     parser.add_argument('--compute_likelihood', action='store_true',
#                       help='Compute log-likelihood (expensive)')
#     return parser.parse_args()
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained DBM')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    return parser.parse_args()

def main():
#     args = parse_args()
    
#     print(f"Evaluating model: {args.model_path}")
#     print(f"Dataset: {args.dataset}")
    
    
    # Load model
    args = parse_args()

    import yaml
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_path = config["logging"]["save_dir"] + "/best_model.pt"
    dataset_name = config["data"]["dataset"]
    batch_size = config["training"].get("batch_size", 64)
    num_samples = config["evaluation"].get("sample_quality", {}).get("num_samples", 100)
    compute_likelihood = config["evaluation"].get("compute_likelihood", False)

    print(f"Evaluating model: {model_path}")
    print(f"Dataset: {dataset_name}")

    if not os.path.exists(model_path):
        print(f"Error: Model file {args.model_path} not found!")
        return
    
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
    
    print(f"Loaded DBM with {len(model.hidden_dims)} hidden layers: {model.hidden_dims}")
    
    # Load test dataset
    test_dataset = get_test_dataset(dataset_name, normalize=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Loaded test dataset with {len(test_dataset)} samples")
    
    # Initialize metrics
    total_reconstruction_error = 0.0
    total_free_energy = 0.0
    total_log_likelihood = 0.0
    num_batches = 0
    
    # Create sampler for likelihood estimation
    if compute_likelihood:
        sampler = GibbsSampler(model, device=device)
        print("Computing log-likelihood (this may take a while)...")
    
    print("Evaluating model...")
    
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):
            data = data.to(device)
            data = data.view(data.size(0), -1)  # Flatten
            
            # Binarize data for evaluation
            data = torch.bernoulli(data)
            
            # Compute reconstruction error
            hiddens = model.forward(data)
            visible_probs, _ = model.sample_visible_given_hidden(hiddens[0])
            recon_error = compute_reconstruction_error(data, visible_probs)
            
            # Compute free energy
            free_energy = model.free_energy(data)
            avg_free_energy = compute_free_energy(data, free_energy)
            
            total_reconstruction_error += recon_error
            total_free_energy += avg_free_energy
            
            # Compute log-likelihood if requested
            if compute_likelihood:
                log_likelihood = sampler.compute_likelihood(
                    data, 
                    n_samples=args.num_samples,
                    n_steps=100  # Reduced for speed
                )
                total_log_likelihood += log_likelihood.mean().item()
            
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    # Compute average metrics
    avg_reconstruction_error = total_reconstruction_error / num_batches
    avg_free_energy = total_free_energy / num_batches
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Average Reconstruction Error: {avg_reconstruction_error:.6f}")
    print(f"Average Free Energy: {avg_free_energy:.6f}")
    
    if compute_likelihood:
        avg_log_likelihood = total_log_likelihood / num_batches
        print(f"Average Log-Likelihood: {avg_log_likelihood:.6f}")
    
    # Additional model information
    print("\nModel Information:")
    print(f"  Visible dimensions: {model.visible_dim}")
    print(f"  Hidden dimensions: {model.hidden_dims}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Weight statistics
    print("\nWeight Statistics:")
    for i, weight in enumerate(model.weights):
        print(f"  Weight {i}: shape={weight.shape}, mean={weight.mean().item():.6f}, std={weight.std().item():.6f}")
    
    print("\nBias Statistics:")
    for i, bias in enumerate(model.biases):
        layer_name = "visible" if i == 0 else f"hidden_{i-1}"
        print(f"  {layer_name} bias: shape={bias.shape}, mean={bias.mean().item():.6f}, std={bias.std().item():.6f}")
    
    print("\nEvaluation completed!")

if __name__ == '__main__':
    main()