import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_samples(
    samples: torch.Tensor,
    n_samples: int = 16,
    figsize: tuple = (8, 8),
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot generated samples in a grid.
    
    Args:
        samples (torch.Tensor): Generated samples of shape (n_samples, channels, height, width)
        n_samples (int): Number of samples to plot
        figsize (tuple): Figure size
        save_path (str or Path, optional): Path to save the figure
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    if n_samples > samples.size(0):
        raise ValueError(f"Requested {n_samples} samples but only {samples.size(0)} available")
    
    # Convert to numpy and move to CPU
    samples = samples[:n_samples].cpu().numpy()
    
    # Calculate grid dimensions
    n_rows = int(np.sqrt(n_samples))
    n_cols = int(np.ceil(n_samples / n_rows))
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    # Plot samples
    for i, ax in enumerate(axes):
        if i < n_samples:
            ax.imshow(samples[i, 0], cmap='gray')
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    
    return fig

def plot_training_curves(
    history: Dict[str, List[float]],
    figsize: tuple = (10, 6),
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot training curves.
    
    Args:
        history (dict): Training history with metrics
        figsize (tuple): Figure size
        save_path (str or Path, optional): Path to save the figure
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    if not history:
        raise ValueError("Empty training history")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for metric, values in history.items():
        ax.plot(values, label=metric)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)
    
    # Save figure if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    
    return fig

def plot_interpolation(
    samples: torch.Tensor,
    n_steps: int = 5,
    figsize: tuple = (10, 2),
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot interpolation between samples.
    
    Args:
        samples (torch.Tensor): Samples to interpolate between
        n_steps (int): Number of interpolation steps
        figsize (tuple): Figure size
        save_path (str or Path, optional): Path to save the figure
        
    Returns:
        plt.Figure: Matplotlib figure
    """
    if n_steps < 1:
        raise ValueError("Number of steps must be at least 1")
    
    # Convert to numpy and move to CPU
    samples = samples.cpu().numpy()
    
    # Create interpolation
    n_samples = samples.shape[0]
    interpolated = []
    
    for i in range(n_samples - 1):
        for t in np.linspace(0, 1, n_steps):
            interp = (1 - t) * samples[i] + t * samples[i + 1]
            interpolated.append(interp)
    
    interpolated = np.array(interpolated)
    
    # Create figure
    fig, axes = plt.subplots(1, len(interpolated), figsize=figsize)
    if len(interpolated) == 1:
        axes = [axes]
    
    # Plot interpolated samples
    for i, ax in enumerate(axes):
        ax.imshow(interpolated[i, 0], cmap='gray')
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    
    return fig 