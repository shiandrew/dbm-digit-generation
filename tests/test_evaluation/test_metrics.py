import pytest
import torch
import numpy as np

from src.evaluation.metrics import (
    compute_reconstruction_error,
    compute_free_energy,
    compute_log_likelihood,
    compute_kl_divergence
)

def test_reconstruction_error():
    """Test reconstruction error computation."""
    # Create dummy data
    x = torch.randn(32, 784)
    x_recon = torch.randn(32, 784)
    
    # Compute error
    error = compute_reconstruction_error(x, x_recon)
    
    assert isinstance(error, float)
    assert error >= 0
    assert not np.isnan(error)
    assert not np.isinf(error)

def test_free_energy():
    """Test free energy computation."""
    # Create dummy data
    x = torch.randn(32, 784)
    energy = torch.randn(32)
    
    # Compute free energy
    fe = compute_free_energy(x, energy)
    
    assert isinstance(fe, float)
    assert not np.isnan(fe)
    assert not np.isinf(fe)

def test_log_likelihood():
    """Test log likelihood computation."""
    # Create dummy data
    x = torch.randn(32, 784)
    log_probs = torch.randn(32)
    
    # Compute log likelihood
    ll = compute_log_likelihood(x, log_probs)
    
    assert isinstance(ll, float)
    assert not np.isnan(ll)
    assert not np.isinf(ll)

def test_kl_divergence():
    """Test KL divergence computation."""
    # Create dummy distributions
    p = torch.rand(32, 784)
    q = torch.rand(32, 784)
    
    # Normalize
    p = p / p.sum(dim=1, keepdim=True)
    q = q / q.sum(dim=1, keepdim=True)
    
    # Compute KL divergence
    kl = compute_kl_divergence(p, q)
    
    assert isinstance(kl, float)
    assert kl >= 0
    assert not np.isnan(kl)
    assert not np.isinf(kl)

def test_reconstruction_error_zero():
    """Test reconstruction error with identical inputs."""
    x = torch.randn(32, 784)
    error = compute_reconstruction_error(x, x)
    assert error == 0

def test_kl_divergence_identical():
    """Test KL divergence with identical distributions."""
    p = torch.rand(32, 784)
    p = p / p.sum(dim=1, keepdim=True)
    kl = compute_kl_divergence(p, p)
    assert kl == 0

def test_log_likelihood_consistency():
    """Test log likelihood consistency."""
    x = torch.randn(32, 784)
    log_probs = torch.randn(32)
    
    # Compute log likelihood
    ll1 = compute_log_likelihood(x, log_probs)
    ll2 = compute_log_likelihood(x, log_probs + 1)  # Shift by constant
    
    # Log likelihood should differ by constant
    assert abs(ll1 - ll2 + 1) < 1e-6

def test_free_energy_scale():
    """Test free energy scale invariance."""
    x = torch.randn(32, 784)
    energy = torch.randn(32)
    
    # Compute free energy
    fe1 = compute_free_energy(x, energy)
    fe2 = compute_free_energy(x, energy * 2)  # Scale energy
    
    # Free energy should scale with energy
    assert abs(fe1 - fe2/2) < 1e-6 