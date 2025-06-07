import pytest
import torch
import matplotlib.pyplot as plt
import numpy as np

from src.utils.visualization import (
    plot_samples,
    plot_training_curves,
    plot_interpolation
)

def test_plot_samples():
    """Test sample plotting."""
    # Create dummy samples
    samples = torch.randn(16, 1, 28, 28)
    
    # Test plotting
    fig = plot_samples(samples, n_samples=16, figsize=(8, 8))
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_training_curves():
    """Test training curves plotting."""
    # Create dummy training history
    history = {
        'train_loss': [0.5, 0.4, 0.3],
        'val_loss': [0.6, 0.5, 0.4]
    }
    
    # Test plotting
    fig = plot_training_curves(history)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_interpolation():
    """Test latent space interpolation plotting."""
    # Create dummy samples
    samples = torch.randn(10, 1, 28, 28)
    
    # Test plotting
    fig = plot_interpolation(samples, n_steps=5)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_samples_invalid_input():
    """Test sample plotting with invalid input."""
    with pytest.raises(ValueError):
        plot_samples(torch.randn(5, 1, 28, 28), n_samples=10)  # More samples than available

def test_plot_training_curves_empty_history():
    """Test training curves plotting with empty history."""
    with pytest.raises(ValueError):
        plot_training_curves({})

def test_plot_interpolation_invalid_steps():
    """Test interpolation plotting with invalid steps."""
    samples = torch.randn(10, 1, 28, 28)
    with pytest.raises(ValueError):
        plot_interpolation(samples, n_steps=0)  # Invalid number of steps

def test_plot_samples_save(tmp_path):
    """Test sample plotting with save option."""
    samples = torch.randn(16, 1, 28, 28)
    save_path = tmp_path / "samples.png"
    
    fig = plot_samples(samples, n_samples=16, save_path=save_path)
    assert save_path.exists()
    plt.close(fig)

def test_plot_training_curves_save(tmp_path):
    """Test training curves plotting with save option."""
    history = {
        'train_loss': [0.5, 0.4, 0.3],
        'val_loss': [0.6, 0.5, 0.4]
    }
    save_path = tmp_path / "curves.png"
    
    fig = plot_training_curves(history, save_path=save_path)
    assert save_path.exists()
    plt.close(fig) 