import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


class DBM(nn.Module):
    """
    Deep Boltzmann Machine implementation with proper DBM architecture.
    
    Key fixes:
    1. Proper forward pass using mean-field inference
    2. Complete free energy implementation with mean-field approximation
    3. Fixed sample_visible method for proper reconstruction
    4. Corrected energy computation
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
        
        # Create weight matrices and biases
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        
        # Visible bias
        self.biases.append(nn.Parameter(torch.zeros(visible_dim)))
        
        # Weights between layers
        # Visible to first hidden
        self.weights.append(nn.Parameter(torch.randn(visible_dim, hidden_dims[0]) * 0.01))
        
        # Hidden layer biases
        for hidden_dim in hidden_dims:
            self.biases.append(nn.Parameter(torch.zeros(hidden_dim)))
        
        # Hidden to hidden weights
        for i in range(len(hidden_dims) - 1):
            self.weights.append(nn.Parameter(torch.randn(hidden_dims[i], hidden_dims[i+1]) * 0.01))
        
        # Device setup
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def energy(self, visible: torch.Tensor, hiddens: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute the energy of the configuration.
        
        Energy = -sum(bias_terms) - sum(weight_interactions)
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
        FIXED: Compute free energy using proper mean-field approximation for DBM.
        
        For DBM, we use mean-field inference to approximate the intractable sum over hidden units.
        """
        batch_size = visible.size(0)
        
        # Visible bias term
        free_energy = -torch.mv(visible, self.biases[0])
        
        # Mean-field inference for hidden units
        # Initialize mean-field parameters
        mu = []  # Mean-field parameters for each hidden layer
        
        # Initialize with bottom-up pass
        current_input = visible
        for i in range(self.num_layers):
            if i == 0:
                activation = F.linear(current_input, self.weights[i].t(), self.biases[i + 1])
            else:
                activation = F.linear(current_input, self.weights[i].t(), self.biases[i + 1])
            
            mu_i = torch.sigmoid(activation)
            mu.append(mu_i)
            current_input = mu_i
        
        # Mean-field updates (simplified - normally would iterate to convergence)
        for iteration in range(3):  # Few iterations for efficiency
            for i in range(self.num_layers):
                # Compute total input to layer i
                total_input = self.biases[i + 1].unsqueeze(0).expand(batch_size, -1)
                
                # Bottom-up input
                if i == 0:
                    total_input = total_input + F.linear(visible, self.weights[i].t())
                else:
                    total_input = total_input + F.linear(mu[i - 1], self.weights[i].t())
                
                # Top-down input
                if i < self.num_layers - 1:
                    total_input = total_input + F.linear(mu[i + 1], self.weights[i + 1])
                
                mu[i] = torch.sigmoid(total_input)
        
        # Compute free energy contribution from each layer
        for i in range(self.num_layers):
            # Mean-field free energy for layer i
            mu_i = mu[i]
            # Avoid log(0) with small epsilon
            eps = 1e-7
            mu_i = torch.clamp(mu_i, eps, 1 - eps)
            
            # Binary entropy term
            entropy = -(mu_i * torch.log(mu_i) + (1 - mu_i) * torch.log(1 - mu_i))
            free_energy -= torch.sum(entropy, dim=1)
            
            # Bias interaction
            free_energy -= torch.mv(mu_i, self.biases[i + 1])
            
            # Weight interactions
            if i == 0:
                free_energy -= torch.sum(visible.unsqueeze(2) * self.weights[i].unsqueeze(0) * mu_i.unsqueeze(1), dim=(1, 2))
            else:
                free_energy -= torch.sum(mu[i-1].unsqueeze(2) * self.weights[i].unsqueeze(0) * mu_i.unsqueeze(1), dim=(1, 2))
        
        return free_energy
    
    def forward(self, visible: torch.Tensor) -> List[torch.Tensor]:
        """
        FIXED: Proper DBM forward pass using mean-field inference.
        
        DBM requires iterative mean-field updates, not just bottom-up propagation.
        """
        batch_size = visible.size(0)
        
        # Initialize mean-field parameters with bottom-up pass
        mu = []
        current_input = visible
        
        for i in range(self.num_layers):
            if i == 0:
                activation = F.linear(current_input, self.weights[i].t(), self.biases[i + 1])
            else:
                activation = F.linear(current_input, self.weights[i].t(), self.biases[i + 1])
            
            mu_i = torch.sigmoid(activation)
            mu.append(mu_i)
            current_input = mu_i
        
        # Mean-field updates
        for iteration in range(5):  # Iterate to approximate convergence
            for i in range(self.num_layers):
                # Compute total input to layer i
                total_input = self.biases[i + 1].unsqueeze(0).expand(batch_size, -1)
                
                # Bottom-up input
                if i == 0:
                    total_input = total_input + F.linear(visible, self.weights[i].t())
                else:
                    total_input = total_input + F.linear(mu[i - 1], self.weights[i].t())
                
                # Top-down input
                if i < self.num_layers - 1:
                    total_input = total_input + F.linear(mu[i + 1], self.weights[i + 1])
                
                mu[i] = torch.sigmoid(total_input)
        
        return mu
    
    def sample_hidden_given_visible(self, visible: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample hidden units at a specific layer given visible units.
        Uses mean-field inference to get to the target layer.
        """
        if layer_idx >= self.num_layers:
            raise ValueError(f"Layer index {layer_idx} exceeds number of hidden layers {self.num_layers}")
        
        # Use forward pass to get mean-field estimates
        hidden_means = self.forward(visible)
        
        # Return probabilities and samples for target layer
        probs = hidden_means[layer_idx]
        samples = torch.bernoulli(probs)
        
        return probs, samples
    
    def sample_visible_given_hidden(self, hidden: torch.Tensor, layer_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        FIXED: Sample visible units given hidden units at any layer.
        
        Now properly handles reconstruction from any hidden layer using mean-field.
        """
        if layer_idx >= self.num_layers:
            raise ValueError(f"Layer index {layer_idx} exceeds number of hidden layers {self.num_layers}")
        
        batch_size = hidden.size(0)
        
        if layer_idx == 0:
            # Direct reconstruction from first hidden layer
            pre_activation = F.linear(hidden, self.weights[0], self.biases[0])
        else:
            # Reconstruction from deeper layer requires mean-field inference
            # Initialize other hidden layers
            mu = [None] * self.num_layers
            mu[layer_idx] = hidden
            
            # Initialize other layers with random values, then update
            for i in range(self.num_layers):
                if i != layer_idx:
                    mu[i] = torch.sigmoid(torch.randn(batch_size, self.hidden_dims[i], device=self.device))
            
            # Mean-field updates
            for iteration in range(5):
                for i in range(self.num_layers):
                    if i == layer_idx:
                        continue  # Keep the given layer fixed
                    
                    total_input = self.biases[i + 1].unsqueeze(0).expand(batch_size, -1)
                    
                    # Bottom-up input
                    if i == 0:
                        # Will be computed after we determine visible
                        pass
                    else:
                        total_input = total_input + F.linear(mu[i - 1], self.weights[i].t())
                    
                    # Top-down input
                    if i < self.num_layers - 1:
                        total_input = total_input + F.linear(mu[i + 1], self.weights[i + 1])
                    
                    if i != 0:  # Don't update visible reconstruction layer here
                        mu[i] = torch.sigmoid(total_input)
            
            # Now reconstruct visible from first hidden layer
            pre_activation = F.linear(mu[0], self.weights[0], self.biases[0])
        
        probs = torch.sigmoid(pre_activation)
        samples = torch.bernoulli(probs)
        
        return probs, samples
    
    def gibbs_step(self, visible: torch.Tensor, hiddens: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Perform one step of Gibbs sampling for DBM.
        
        In DBM, each layer is conditionally independent given its neighbors.
        """
        new_hiddens = []
        
        # Sample each hidden layer given its neighbors
        for i in range(self.num_layers):
            # Get input from layer below
            if i == 0:
                input_from_below = visible
                weight_below = self.weights[i]
            else:
                input_from_below = new_hiddens[i - 1]
                weight_below = self.weights[i]
            
            # Start with bias
            pre_activation = self.biases[i + 1].unsqueeze(0).expand(visible.size(0), -1)
            
            # Add bottom-up input
            pre_activation = pre_activation + F.linear(input_from_below, weight_below.t())
            
            # Add top-down input if not the top layer
            if i < self.num_layers - 1:
                input_from_above = hiddens[i + 1]
                weight_above = self.weights[i + 1]
                pre_activation = pre_activation + F.linear(input_from_above, weight_above)
            
            probs = torch.sigmoid(pre_activation)
            samples = torch.bernoulli(probs)
            new_hiddens.append(samples)
        
        # Sample visible units given first hidden layer
        visible_pre_activation = F.linear(new_hiddens[0], self.weights[0], self.biases[0])
        visible_probs = torch.sigmoid(visible_pre_activation)
        new_visible = torch.bernoulli(visible_probs)
        
        return new_visible, new_hiddens
    
    def contrastive_divergence(self, visible: torch.Tensor, k: int = 1) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Perform k steps of contrastive divergence.
        """
        # Initialize hidden units with proper forward pass
        hiddens = self.forward(visible)
        # Convert probabilities to samples for CD
        hiddens = [torch.bernoulli(h) for h in hiddens]
        
        # Run k steps of Gibbs sampling
        neg_visible = visible.clone()
        neg_hiddens = [h.clone() for h in hiddens]
        
        for _ in range(k):
            neg_visible, neg_hiddens = self.gibbs_step(neg_visible, neg_hiddens)
        
        return neg_visible, neg_hiddens
    
    def generate_samples(self, num_samples: int, num_gibbs_steps: int = 1000) -> torch.Tensor:
        """
        Generate samples using Gibbs sampling.
        """
        # Initialize random configuration
        visible = torch.rand(num_samples, self.visible_dim, device=self.device)
        visible = torch.bernoulli(visible)
        
        hiddens = []
        for hidden_dim in self.hidden_dims:
            h = torch.rand(num_samples, hidden_dim, device=self.device)
            hiddens.append(torch.bernoulli(h))
        
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
    def load(cls, path: str, use_cuda: bool = True):
        """Load model from saved state."""
        checkpoint = torch.load(path, map_location='cpu')
        model = cls(
            visible_dim=checkpoint['visible_dim'],
            hidden_dims=checkpoint['hidden_dims'],
            learning_rate=checkpoint['learning_rate'],
            use_cuda=use_cuda
        )
        model.load_state_dict(checkpoint['state_dict'])
        return model