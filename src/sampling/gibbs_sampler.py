from typing import Callable, List, Optional, Tuple, Union

import torch

from src.models.dbm import DBM

class GibbsSampler:
    """
    Gibbs sampler for Deep Boltzmann Machine.
    
    Args:
        model (DBM): DBM model to sample from
        device (torch.device, optional): Device to use for sampling
    """
    
    def __init__(
        self,
        model: DBM,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.device = device or torch.device('cpu')
        self.model.to(self.device)
    
    def sample_visible(self, h: torch.Tensor) -> torch.Tensor:
        """
        Sample visible units given hidden units.
        
        Args:
            h (torch.Tensor): Hidden units of shape (batch_size, hidden_dim)
            
        Returns:
            torch.Tensor: Sampled visible units
        """
        return self.model.sample_visible(h)
    
    def sample_hidden(self, v: torch.Tensor) -> torch.Tensor:
        """
        Sample hidden units given visible units.
        
        Args:
            v (torch.Tensor): Visible units of shape (batch_size, visible_dim)
            
        Returns:
            torch.Tensor: Sampled hidden units
        """
        return self.model.sample_hidden(v)
    
    def gibbs_step(
        self,
        v: torch.Tensor,
        h: torch.Tensor,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one Gibbs sampling step.
        
        Args:
            v (torch.Tensor): Current visible units
            h (torch.Tensor): Current hidden units
            temperature (float): Sampling temperature
            
        Returns:
            tuple: (new visible units, new hidden units)
        """
        # Sample hidden units
        h_new = self.sample_hidden(v)
        if temperature != 1.0:
            h_new = h_new / temperature
        
        # Sample visible units
        v_new = self.sample_visible(h_new)
        if temperature != 1.0:
            v_new = v_new / temperature
        
        return v_new, h_new
    
    def gibbs_chain(
        self,
        v: torch.Tensor,
        n_steps: int = 1,
        temperature: float = 1.0,
        callback: Optional[Callable[[int, torch.Tensor, torch.Tensor], None]] = None
    ) -> List[torch.Tensor]:
        """
        Run a chain of Gibbs sampling steps.
        
        Args:
            v (torch.Tensor): Initial visible units
            n_steps (int): Number of Gibbs steps
            temperature (float): Sampling temperature
            callback (callable, optional): Callback function called after each step
            
        Returns:
            list: List of visible unit samples
        """
        samples = [v]
        h = self.sample_hidden(v)
        
        for step in range(n_steps):
            v, h = self.gibbs_step(v, h, temperature)
            samples.append(v)
            
            if callback is not None:
                callback(step, v, h)
        
        return samples
    
    def sample_from_model(
        self,
        batch_size: int,
        n_steps: int = 1000,
        temperature: float = 1.0,
        initial_state: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[int, torch.Tensor, torch.Tensor], None]] = None
    ) -> torch.Tensor:
        """
        Sample from the model using Gibbs sampling.
        
        Args:
            batch_size (int): Number of samples to generate
            n_steps (int): Number of Gibbs steps
            temperature (float): Sampling temperature
            initial_state (torch.Tensor, optional): Initial visible units
            callback (callable, optional): Callback function called after each step
            
        Returns:
            torch.Tensor: Generated samples
        """
        if initial_state is None:
            v = torch.randn(batch_size, self.model.visible_dim, device=self.device)
        else:
            v = initial_state.to(self.device)
        
        samples = self.gibbs_chain(
            v,
            n_steps=n_steps,
            temperature=temperature,
            callback=callback
        )
        
        return samples[-1]  # Return final sample 