import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class RBM(nn.Module):
    """
    Restricted Boltzmann Machine implementation.
    
    Args:
        visible_dim (int): Number of visible units
        hidden_dim (int): Number of hidden units
        k (int): Number of Gibbs sampling steps for CD-k
        learning_rate (float): Learning rate for weight updates
        use_cuda (bool): Whether to use CUDA if available
    """
    
    def __init__(
        self, 
        visible_dim: int, 
        hidden_dim: int, 
        k: int = 1,
        learning_rate: float = 0.01,
        use_cuda: bool = True
    ):
        super(RBM, self).__init__()
        
        self.visible_dim = visible_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self.learning_rate = learning_rate
        
        # Initialize weights and biases
        self.W = nn.Parameter(torch.randn(visible_dim, hidden_dim) * 0.01)
        self.visible_bias = nn.Parameter(torch.zeros(visible_dim))
        self.hidden_bias = nn.Parameter(torch.zeros(hidden_dim))
        
        # Device setup
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def sample_hidden(self, visible: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample hidden units given visible units.
        
        Args:
            visible (torch.Tensor): Visible units of shape (batch_size, visible_dim)
            
        Returns:
            tuple: (hidden_probabilities, hidden_samples)
        """
        # Compute hidden probabilities
        hidden_activations = F.linear(visible, self.W.t(), self.hidden_bias)
        hidden_probs = torch.sigmoid(hidden_activations)
        
        # Sample from Bernoulli distribution
        hidden_samples = torch.bernoulli(hidden_probs)
        
        return hidden_probs, hidden_samples
    
    def sample_visible(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample visible units given hidden units.
        
        Args:
            hidden (torch.Tensor): Hidden units of shape (batch_size, hidden_dim)
            
        Returns:
            tuple: (visible_probabilities, visible_samples)
        """
        # Compute visible probabilities
        visible_activations = F.linear(hidden, self.W, self.visible_bias)
        visible_probs = torch.sigmoid(visible_activations)
        
        # Sample from Bernoulli distribution
        visible_samples = torch.bernoulli(visible_probs)
        
        return visible_probs, visible_samples
    
    def contrastive_divergence(self, visible: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform contrastive divergence to compute gradients.
        
        Args:
            visible (torch.Tensor): Input visible units
            
        Returns:
            tuple: (positive_grad, negative_grad, reconstruction_error)
        """
        batch_size = visible.size(0)
        
        # Positive phase
        pos_hidden_probs, pos_hidden_samples = self.sample_hidden(visible)
        pos_associations = torch.mm(visible.t(), pos_hidden_probs)
        
        # Negative phase - k steps of Gibbs sampling
        neg_visible = visible.clone()
        for _ in range(self.k):
            _, neg_hidden_samples = self.sample_hidden(neg_visible)
            neg_visible_probs, neg_visible = self.sample_visible(neg_hidden_samples)
        
        neg_hidden_probs, _ = self.sample_hidden(neg_visible)
        neg_associations = torch.mm(neg_visible.t(), neg_hidden_probs)
        
        # Compute gradients
        pos_grad = pos_associations / batch_size
        neg_grad = neg_associations / batch_size
        
        # Reconstruction error
        recon_error = F.mse_loss(visible, neg_visible_probs)
        
        return pos_grad, neg_grad, recon_error
    
    def free_energy(self, visible: torch.Tensor) -> torch.Tensor:
        """
        Compute the free energy of visible units.
        
        Args:
            visible (torch.Tensor): Visible units
            
        Returns:
            torch.Tensor: Free energy values
        """
        # Visible bias term
        visible_bias_term = torch.mv(visible, self.visible_bias)
        
        # Hidden unit term
        hidden_activations = F.linear(visible, self.W.t(), self.hidden_bias)
        hidden_term = torch.sum(F.softplus(hidden_activations), dim=1)
        
        return -visible_bias_term - hidden_term
    
    def forward(self, visible: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through RBM.
        
        Args:
            visible (torch.Tensor): Input visible units
            
        Returns:
            torch.Tensor: Hidden unit probabilities
        """
        hidden_probs, _ = self.sample_hidden(visible)
        return hidden_probs
    
    def train_step(self, visible: torch.Tensor) -> float:
        """
        Perform one training step using contrastive divergence.
        
        Args:
            visible (torch.Tensor): Training batch
            
        Returns:
            float: Reconstruction error
        """
        visible = visible.to(self.device)
        
        # Compute gradients
        pos_grad, neg_grad, recon_error = self.contrastive_divergence(visible)
        
        # Update weights and biases
        self.W.data += self.learning_rate * (pos_grad - neg_grad)
        
        # Update biases
        pos_visible_bias_grad = torch.mean(visible, dim=0)
        neg_visible_bias_grad = torch.mean(neg_grad.t(), dim=0)  # From negative phase
        self.visible_bias.data += self.learning_rate * (pos_visible_bias_grad - neg_visible_bias_grad)
        
        pos_hidden_probs, _ = self.sample_hidden(visible)
        pos_hidden_bias_grad = torch.mean(pos_hidden_probs, dim=0)
        neg_hidden_probs, _ = self.sample_hidden(neg_grad.t())  # From negative phase
        neg_hidden_bias_grad = torch.mean(neg_hidden_probs, dim=0)
        self.hidden_bias.data += self.learning_rate * (pos_hidden_bias_grad - neg_hidden_bias_grad)
        
        return recon_error.item()
    
    def reconstruct(self, visible: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct visible units through one forward-backward pass.
        
        Args:
            visible (torch.Tensor): Input visible units
            
        Returns:
            torch.Tensor: Reconstructed visible units
        """
        _, hidden_samples = self.sample_hidden(visible)
        visible_probs, _ = self.sample_visible(hidden_samples)
        return visible_probs
    
    def generate_samples(self, num_samples: int, num_gibbs_steps: int = 1000) -> torch.Tensor:
        """
        Generate samples using Gibbs sampling.
        
        Args:
            num_samples (int): Number of samples to generate
            num_gibbs_steps (int): Number of Gibbs sampling steps
            
        Returns:
            torch.Tensor: Generated samples
        """
        # Initialize with random visible units
        visible = torch.rand(num_samples, self.visible_dim, device=self.device)
        
        # Run Gibbs chain
        for _ in range(num_gibbs_steps):
            _, hidden_samples = self.sample_hidden(visible)
            visible_probs, visible = self.sample_visible(hidden_samples)
        
        return visible_probs