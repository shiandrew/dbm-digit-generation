import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


class DBM(nn.Module):
    """
    Deep Boltzmann Machine implementation.
    
    A DBM consists of multiple layers where each layer is connected to its neighboring layers.
    The energy function captures interactions between all connected layers.
    
    Args:
        visible_dim (int): Dimension of the visible layer
        hidden_dims (list): List of hidden layer dimensions
        learning_rate (float): Learning rate for training
        use_cuda (bool): Whether to use CUDA if available
    """
    
    def __init__(
        self, 
        visible_dim: int, 
        hidden_dims: List[int],
        learning_rate: float = 0.001,
        use_cuda: bool = True
    ):
        super(DBM, self).__init__()
        
        self.visible_dim = visible_dim
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)
        self.learning_rate = learning_rate
        
        # Create weight matrices between adjacent layers
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        
        # Visible to first hidden layer
        self.weights.append(nn.Parameter(torch.randn(visible_dim, hidden_dims[0]) * 0.01))
        self.biases.append(nn.Parameter(torch.zeros(visible_dim)))  # Visible bias
        
        # Hidden to hidden layers
        for i in range(len(hidden_dims) - 1):
            self.weights.append(nn.Parameter(torch.randn(hidden_dims[i], hidden_dims[i+1]) * 0.01))
        
        # Hidden layer biases
        for hidden_dim in hidden_dims:
            self.biases.append(nn.Parameter(torch.zeros(hidden_dim)))
        
        # Device setup
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def energy(self, visible: torch.Tensor, hiddens: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute the energy of the configuration.
        
        Energy = -sum(bias_terms) - sum(weight_interactions)
        
        Args:
            visible (torch.Tensor): Visible units of shape (batch_size, visible_dim)
            hiddens (List[torch.Tensor]): List of hidden layer activations
            
        Returns:
            torch.Tensor: Energy values of shape (batch_size,)
        """
        batch_size = visible.size(0)
        energy = torch.zeros(batch_size, device=self.device)
        
        # Visible bias term
        energy -= torch.mv(visible, self.biases[0])
        
        # Hidden bias terms
        for i, hidden in enumerate(hiddens):
            energy -= torch.mv(hidden, self.biases[i + 1])
        
        # Interaction terms between adjacent layers
        # Visible to first hidden
        energy -= torch.sum(visible.unsqueeze(2) * self.weights[0].unsqueeze(0) * hiddens[0].unsqueeze(1), dim=(1, 2))
        
        # Hidden to hidden interactions
        for i in range(len(hiddens) - 1):
            energy -= torch.sum(hiddens[i].unsqueeze(2) * self.weights[i + 1].unsqueeze(0) * hiddens[i + 1].unsqueeze(1), dim=(1, 2))
        
        return energy
    
    def free_energy(self, visible: torch.Tensor) -> torch.Tensor:
        """
        Compute the free energy by marginalizing over hidden units.
        
        Free Energy = -log(sum_h exp(-Energy(v,h)))
                    = -bias_v^T * v - sum_i log(1 + exp(bias_h_i + sum_j W_ij * v_j))
        
        Args:
            visible (torch.Tensor): Visible units
            
        Returns:
            torch.Tensor: Free energy values
        """
        batch_size = visible.size(0)
        
        # Visible bias term
        free_energy = -torch.mv(visible, self.biases[0])
        
        # Compute contributions from each hidden layer
        current_input = visible
        
        for layer_idx in range(self.num_layers):
            # Compute pre-activation for this hidden layer
            if layer_idx == 0:
                # First hidden layer gets input from visible
                pre_activation = F.linear(current_input, self.weights[layer_idx].t(), self.biases[layer_idx + 1])
            else:
                # Subsequent layers - we need to account for top-down connections too
                # For simplicity in free energy computation, we'll use mean-field approximation
                pre_activation = F.linear(current_input, self.weights[layer_idx].t(), self.biases[layer_idx + 1])
            
            # Add softplus term: log(1 + exp(x))
            free_energy -= torch.sum(F.softplus(pre_activation), dim=1)
            
            # For next layer, use sigmoid probabilities as input
            current_input = torch.sigmoid(pre_activation)
        
        return free_energy
    
    def sample_hidden_given_visible(self, visible: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample hidden units at a specific layer given visible units.
        
        Args:
            visible (torch.Tensor): Visible units
            layer_idx (int): Index of hidden layer to sample
            
        Returns:
            tuple: (probabilities, samples)
        """
        if layer_idx >= self.num_layers:
            raise ValueError(f"Layer index {layer_idx} exceeds number of hidden layers {self.num_layers}")
        
        # For first hidden layer
        if layer_idx == 0:
            pre_activation = F.linear(visible, self.weights[0].t(), self.biases[1])
        else:
            # For deeper layers, we need to propagate through previous layers
            current = visible
            for i in range(layer_idx + 1):
                if i == 0:
                    # First hidden layer from visible
                    pre_activation = F.linear(current, self.weights[i].t(), self.biases[i + 1])
                else:
                    # Subsequent hidden layers
                    pre_activation = F.linear(current, self.weights[i].t(), self.biases[i + 1])
                
                # If this is not the target layer, compute probabilities for next layer
                if i < layer_idx:
                    current = torch.sigmoid(pre_activation)
        
        probs = torch.sigmoid(pre_activation)
        samples = torch.bernoulli(probs)
        
        return probs, samples
    
    def sample_visible_given_hidden(self, hidden: torch.Tensor, layer_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample visible units given hidden units at a specific layer.
        
        Args:
            hidden (torch.Tensor): Hidden units
            layer_idx (int): Index of hidden layer (0 = first hidden layer)
            
        Returns:
            tuple: (probabilities, samples)
        """
        if layer_idx >= self.num_layers:
            raise ValueError(f"Layer index {layer_idx} exceeds number of hidden layers {self.num_layers}")
        
        # Reconstruct visible from the specified hidden layer
        pre_activation = F.linear(hidden, self.weights[layer_idx], self.biases[0])
        probs = torch.sigmoid(pre_activation)
        samples = torch.bernoulli(probs)
        
        return probs, samples
    
    def gibbs_step(self, visible: torch.Tensor, hiddens: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Perform one step of Gibbs sampling.
        
        Args:
            visible (torch.Tensor): Current visible state
            hiddens (List[torch.Tensor]): Current hidden states
            
        Returns:
            tuple: (new_visible, new_hiddens)
        """
        new_hiddens = []
        
        # Sample each hidden layer sequentially
        for i in range(self.num_layers):
            if i == 0:
                # First hidden layer depends on visible
                input_from_below = visible
                pre_activation = F.linear(input_from_below, self.weights[i].t(), self.biases[i + 1])
            else:
                # Higher layers depend on layer below
                input_from_below = new_hiddens[i - 1]
                pre_activation = F.linear(input_from_below, self.weights[i].t(), self.biases[i + 1])
            
            # Add top-down connections if not the top layer
            if i < self.num_layers - 1:
                input_from_above = hiddens[i + 1] if i + 1 < len(hiddens) else torch.zeros_like(pre_activation)
                pre_activation += F.linear(input_from_above, self.weights[i + 1])
            
            probs = torch.sigmoid(pre_activation)
            samples = torch.bernoulli(probs)
            new_hiddens.append(samples)
        
        # Sample visible units
        visible_pre_activation = F.linear(new_hiddens[0], self.weights[0], self.biases[0])
        visible_probs = torch.sigmoid(visible_pre_activation)
        new_visible = torch.bernoulli(visible_probs)
        
        return new_visible, new_hiddens
    
    def contrastive_divergence(self, visible: torch.Tensor, k: int = 1) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Perform k steps of contrastive divergence.
        
        Args:
            visible (torch.Tensor): Input visible units
            k (int): Number of Gibbs sampling steps
            
        Returns:
            tuple: (negative_visible, negative_hiddens)
        """
        # Initialize hidden units properly for multi-layer networks
        hiddens = []
        current_input = visible
        
        for i in range(self.num_layers):
            if i == 0:
                pre_activation = F.linear(current_input, self.weights[i].t(), self.biases[i + 1])
            else:
                pre_activation = F.linear(current_input, self.weights[i].t(), self.biases[i + 1])
            
            probs = torch.sigmoid(pre_activation)
            samples = torch.bernoulli(probs)
            hiddens.append(samples)
            current_input = probs  # Use probabilities for next layer
        
        # Run k steps of Gibbs sampling
        neg_visible = visible.clone()
        neg_hiddens = [h.clone() for h in hiddens]
        
        for _ in range(k):
            neg_visible, neg_hiddens = self.gibbs_step(neg_visible, neg_hiddens)
        
        return neg_visible, neg_hiddens
    
    def forward(self, visible: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through the DBM to get hidden representations.
        
        Args:
            visible (torch.Tensor): Input visible units
            
        Returns:
            List[torch.Tensor]: Hidden layer representations
        """
        hiddens = []
        current_input = visible
        
        for i in range(self.num_layers):
            # Each layer gets input from the layer below
            if i == 0:
                # First hidden layer from visible
                pre_activation = F.linear(current_input, self.weights[i].t(), self.biases[i + 1])
            else:
                # Subsequent hidden layers from previous hidden layer
                pre_activation = F.linear(current_input, self.weights[i].t(), self.biases[i + 1])
            
            probs = torch.sigmoid(pre_activation)
            hiddens.append(probs)
            current_input = probs  # Use probabilities for next layer
        
        return hiddens
    
    def generate_samples(self, num_samples: int, num_gibbs_steps: int = 1000) -> torch.Tensor:
        """
        Generate samples using Gibbs sampling.
        
        Args:
            num_samples (int): Number of samples to generate
            num_gibbs_steps (int): Number of Gibbs sampling steps
            
        Returns:
            torch.Tensor: Generated visible samples
        """
        # Initialize random configuration
        visible = torch.rand(num_samples, self.visible_dim, device=self.device)
        hiddens = []
        for hidden_dim in self.hidden_dims:
            hiddens.append(torch.rand(num_samples, hidden_dim, device=self.device))
        
        # Run Gibbs chain
        for _ in range(num_gibbs_steps):
            visible, hiddens = self.gibbs_step(visible, hiddens)
        
        return visible
    
    def save(self, path: str):
        """Save model state."""
        torch.save({
            'state_dict': self.state_dict(),
            'visible_dim': self.visible_dim,
            'hidden_dims': self.hidden_dims,
            'learning_rate': self.learning_rate
        }, path)
    
    @classmethod
    def load(cls, path: str):
        """Load model from saved state."""
        checkpoint = torch.load(path)
        model = cls(
            visible_dim=checkpoint['visible_dim'],
            hidden_dims=checkpoint['hidden_dims'],
            learning_rate=checkpoint['learning_rate']
        )
        model.load_state_dict(checkpoint['state_dict'])
        return model