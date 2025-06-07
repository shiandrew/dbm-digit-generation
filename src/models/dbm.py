import torch
import torch.nn as nn
import torch.nn.functional as F

class DBM(nn.Module):
    """
    Deep Boltzmann Machine implementation.
    
    Args:
        visible_dim (int): Dimension of the visible layer
        hidden_dims (list): List of hidden layer dimensions
        dropout_rate (float): Dropout rate for regularization
        batch_norm (bool): Whether to use batch normalization
    """
    
    def __init__(self, visible_dim, hidden_dims, dropout_rate=0.2, batch_norm=True):
        super(DBM, self).__init__()
        
        self.visible_dim = visible_dim
        self.hidden_dims = hidden_dims
        self.num_layers = len(hidden_dims)
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        
        # Create layers
        self.layers = nn.ModuleList()
        
        # First layer (visible to first hidden)
        self.layers.append(nn.Linear(visible_dim, hidden_dims[0]))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        
        # Batch normalization layers
        if batch_norm:
            self.bn_layers = nn.ModuleList()
            for dim in hidden_dims:
                self.bn_layers.append(nn.BatchNorm1d(dim))
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, visible_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_dims[-1])
        """
        h = x
        
        for i in range(self.num_layers):
            h = self.layers[i](h)
            if self.batch_norm:
                h = self.bn_layers[i](h)
            h = torch.sigmoid(h)
            h = self.dropout(h)
        
        return h
    
    def sample_visible(self, h):
        """
        Sample visible units given hidden units.
        
        Args:
            h (torch.Tensor): Hidden units of shape (batch_size, hidden_dims[-1])
            
        Returns:
            torch.Tensor: Sampled visible units
        """
        v = self.layers[0].weight.t() @ h.t()
        v = torch.sigmoid(v).t()
        return v
    
    def sample_hidden(self, v):
        """
        Sample hidden units given visible units.
        
        Args:
            v (torch.Tensor): Visible units of shape (batch_size, visible_dim)
            
        Returns:
            torch.Tensor: Sampled hidden units
        """
        h = self.forward(v)
        return h
    
    def free_energy(self, v):
        """
        Compute the free energy of the model.
        
        Args:
            v (torch.Tensor): Visible units of shape (batch_size, visible_dim)
            
        Returns:
            torch.Tensor: Free energy
        """
        h = self.forward(v)
        energy = -torch.sum(v * self.layers[0].bias)
        
        for i in range(self.num_layers):
            energy = energy - torch.sum(h * self.layers[i].bias)
            if i < self.num_layers - 1:
                h = self.layers[i + 1](h)
                if self.batch_norm:
                    h = self.bn_layers[i + 1](h)
                h = torch.sigmoid(h)
        
        return energy
    
    def save(self, path):
        """
        Save model state.
        
        Args:
            path (str): Path to save the model
        """
        torch.save({
            'state_dict': self.state_dict(),
            'visible_dim': self.visible_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm
        }, path)
    
    @classmethod
    def load(cls, path):
        """
        Load model from saved state.
        
        Args:
            path (str): Path to saved model
            
        Returns:
            DBM: Loaded model
        """
        checkpoint = torch.load(path)
        model = cls(
            visible_dim=checkpoint['visible_dim'],
            hidden_dims=checkpoint['hidden_dims'],
            dropout_rate=checkpoint['dropout_rate'],
            batch_norm=checkpoint['batch_norm']
        )
        model.load_state_dict(checkpoint['state_dict'])
        return model 