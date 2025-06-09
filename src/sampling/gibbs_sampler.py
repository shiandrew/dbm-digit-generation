from typing import Callable, List, Optional, Tuple, Union
import torch
import numpy as np


class GibbsSampler:
    """
    Gibbs sampler for Deep Boltzmann Machine.
    
    Performs block Gibbs sampling where we alternately sample all units
    in each layer conditioned on the neighboring layers.
    
    Args:
        model: DBM model to sample from
        device (torch.device, optional): Device to use for sampling
    """
    
    def __init__(
        self,
        model,  # DBM model
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.device = device or torch.device('cpu')
        self.model.to(self.device)
    
    def sample_layer(
        self, 
        layer_idx: int,
        bottom_layer: Optional[torch.Tensor] = None,
        top_layer: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a specific layer given its neighboring layers.
        
        Args:
            layer_idx (int): Index of layer to sample (0 = visible, 1+ = hidden)
            bottom_layer (torch.Tensor, optional): Layer below (if any)
            top_layer (torch.Tensor, optional): Layer above (if any)
            temperature (float): Sampling temperature
            
        Returns:
            tuple: (probabilities, samples)
        """
        if layer_idx == 0:
            # Sample visible layer
            if bottom_layer is not None:
                raise ValueError("Visible layer has no bottom layer")
            
            if top_layer is None:
                raise ValueError("Must provide hidden layer to sample visible layer")
            
            # Visible units receive input from first hidden layer
            # Use weights[0] which connects visible to first hidden
            pre_activation = torch.mm(top_layer, self.model.weights[0].t()) + self.model.biases[0]
            
        else:
            # Sample hidden layer (layer_idx >= 1)
            # Determine which hidden layer this is (0-indexed for hidden layers)
            hidden_layer_idx = layer_idx - 1  # 0 for first hidden, 1 for second hidden, etc.
            
            batch_size = bottom_layer.size(0) if bottom_layer is not None else top_layer.size(0)
            
            # Start with bias term for this hidden layer
            pre_activation = self.model.biases[layer_idx].unsqueeze(0).expand(batch_size, -1)
            
            # Input from bottom layer
            if bottom_layer is not None:
                if layer_idx == 1:
                    # First hidden layer receives from visible via weights[0]
                    pre_activation = pre_activation + torch.mm(bottom_layer, self.model.weights[0])
                else:
                    # Higher hidden layers receive from lower hidden layers
                    # For layer_idx=2 (hidden2), bottom is hidden1, use weights[1]
                    # For layer_idx=3 (hidden3), bottom is hidden2, use weights[2]
                    # Pattern: use weights[layer_idx-1]
                    weight_idx = layer_idx - 1
                    pre_activation = pre_activation + torch.mm(bottom_layer, self.model.weights[weight_idx])
            
            # Input from top layer  
            if top_layer is not None:
                # For top-down connections:
                # When sampling layer_idx=1 (hidden1), top is hidden2, use weights[1].t()
                # When sampling layer_idx=2 (hidden2), top is hidden3, use weights[2].t()
                # Pattern: use weights[layer_idx].t() 
                weight_idx = layer_idx
                if weight_idx < len(self.model.weights):
                    pre_activation = pre_activation + torch.mm(top_layer, self.model.weights[weight_idx].t())
        
        # Apply temperature scaling
        if temperature != 1.0:
            pre_activation = pre_activation / temperature
        
        # Compute probabilities and sample
        probs = torch.sigmoid(pre_activation)
        samples = torch.bernoulli(probs)
        
        return probs, samples
    
    def gibbs_step(
        self,
        visible: torch.Tensor,
        hiddens: List[torch.Tensor],
        temperature: float = 1.0,
        sample_visible: bool = True
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Perform one complete Gibbs sampling step.
        
        Args:
            visible (torch.Tensor): Current visible units
            hiddens (List[torch.Tensor]): Current hidden units
            temperature (float): Sampling temperature
            sample_visible (bool): Whether to sample visible units
            
        Returns:
            tuple: (new_visible, new_hiddens)
        """
        new_hiddens = []
        
        # Sample hidden layers from bottom to top
        for i in range(len(hiddens)):
            bottom_layer = visible if i == 0 else new_hiddens[i - 1]
            top_layer = hiddens[i + 1] if i + 1 < len(hiddens) else None
            
            _, hidden_samples = self.sample_layer(
                layer_idx=i + 1,  # +1 because layer 0 is visible
                bottom_layer=bottom_layer,
                top_layer=top_layer,
                temperature=temperature
            )
            new_hiddens.append(hidden_samples)
        
        # Sample visible layer if requested
        if sample_visible:
            _, visible_samples = self.sample_layer(
                layer_idx=0,
                top_layer=new_hiddens[0],
                temperature=temperature
            )
            new_visible = visible_samples
        else:
            new_visible = visible
        
        return new_visible, new_hiddens
    
    def gibbs_chain(
        self,
        initial_visible: torch.Tensor,
        n_steps: int = 1000,
        temperature: float = 1.0,
        burn_in: int = 100,
        sample_interval: int = 10,
        callback: Optional[Callable[[int, torch.Tensor, List[torch.Tensor]], None]] = None
    ) -> List[Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Run a Gibbs sampling chain.
        
        Args:
            initial_visible (torch.Tensor): Initial visible state
            n_steps (int): Total number of Gibbs steps
            temperature (float): Sampling temperature
            burn_in (int): Number of burn-in steps to discard
            sample_interval (int): Interval between saved samples
            callback (callable, optional): Callback called after each step
            
        Returns:
            list: List of (visible, hiddens) tuples
        """
        batch_size = initial_visible.size(0)
        
        # Initialize hidden layers
        hiddens = []
        for hidden_dim in self.model.hidden_dims:
            hiddens.append(torch.rand(batch_size, hidden_dim, device=self.device))
        
        visible = initial_visible.clone()
        samples = []
        
        for step in range(n_steps):
            visible, hiddens = self.gibbs_step(visible, hiddens, temperature)
            
            # Save sample if past burn-in and at sample interval
            # Use > instead of >= to exclude the burn-in step itself
            if step > burn_in and (step - burn_in - 1) % sample_interval == 0:
                samples.append((visible.clone(), [h.clone() for h in hiddens]))
            
            if callback is not None:
                callback(step, visible, hiddens)
        
        return samples
    
    def sample_from_model(
        self,
        batch_size: int,
        n_steps: int = 1000,
        temperature: float = 1.0,
        initial_visible: Optional[torch.Tensor] = None,
        return_chain: bool = False,
        callback: Optional[Callable[[int, torch.Tensor, List[torch.Tensor]], None]] = None
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Generate samples from the model using Gibbs sampling.
        
        Args:
            batch_size (int): Number of samples to generate
            n_steps (int): Number of Gibbs steps
            temperature (float): Sampling temperature
            initial_visible (torch.Tensor, optional): Initial visible state
            return_chain (bool): Whether to return the entire chain
            callback (callable, optional): Callback called after each step
            
        Returns:
            torch.Tensor or list: Generated samples or sampling chain
        """
        if initial_visible is None:
            # Initialize with random visible units (0.5 probability for binary units)
            initial_visible = torch.bernoulli(
                torch.ones(batch_size, self.model.visible_dim, device=self.device) * 0.5
            )
        else:
            initial_visible = initial_visible.to(self.device)
        
        if return_chain:
            samples = self.gibbs_chain(
                initial_visible,
                n_steps=n_steps,
                temperature=temperature,
                callback=callback
            )
            return [sample[0] for sample in samples]  # Return only visible states
        else:
            samples = self.gibbs_chain(
                initial_visible,
                n_steps=n_steps,
                temperature=temperature,
                burn_in=n_steps - 1,  # Only keep final sample
                sample_interval=1,
                callback=callback
            )
            return samples[-1][0] if samples else initial_visible
    
    def compute_likelihood(
        self,
        visible: torch.Tensor,
        n_samples: int = 100,
        n_steps: int = 1000
    ) -> torch.Tensor:
        """
        Estimate log-likelihood using importance sampling.
        
        Args:
            visible (torch.Tensor): Visible data
            n_samples (int): Number of importance samples
            n_steps (int): Number of Gibbs steps per sample
            
        Returns:
            torch.Tensor: Estimated log-likelihood
        """
        batch_size = visible.size(0)
        
        # Generate samples for partition function estimation
        importance_samples = []
        for _ in range(n_samples):
            sample = self.sample_from_model(
                batch_size=1,
                n_steps=n_steps
            )
            importance_samples.append(sample)
        
        importance_samples = torch.cat(importance_samples, dim=0)
        
        # Compute free energies
        data_free_energy = self.model.free_energy(visible)
        sample_free_energies = self.model.free_energy(importance_samples)
        
        # Estimate log partition function
        log_z_estimate = -torch.logsumexp(-sample_free_energies, dim=0) + torch.log(torch.tensor(n_samples, dtype=torch.float))
        
        # Compute log-likelihood
        log_likelihood = -data_free_energy - log_z_estimate
        
        return log_likelihood
    
    def anneal_sampling(
        self,
        initial_visible: torch.Tensor,
        temperature_schedule: List[float],
        steps_per_temp: int = 100
    ) -> torch.Tensor:
        """
        Perform simulated annealing for better sampling.
        
        Args:
            initial_visible (torch.Tensor): Initial visible state
            temperature_schedule (list): List of temperatures (high to low)
            steps_per_temp (int): Number of steps at each temperature
            
        Returns:
            torch.Tensor: Final sample after annealing
        """
        visible = initial_visible.clone()
        batch_size = visible.size(0)
        
        # Initialize hidden layers
        hiddens = []
        for hidden_dim in self.model.hidden_dims:
            hiddens.append(torch.rand(batch_size, hidden_dim, device=self.device))
        
        for temperature in temperature_schedule:
            for _ in range(steps_per_temp):
                visible, hiddens = self.gibbs_step(visible, hiddens, temperature)
        
        return visible
    
    def persistent_chain_sampling(
        self,
        batch_size: int,
        n_samples: int,
        steps_between_samples: int = 100,
        temperature: float = 1.0
    ) -> List[torch.Tensor]:
        """
        Use persistent contrastive divergence approach for sampling.
        
        Args:
            batch_size (int): Batch size for persistent chains
            n_samples (int): Number of samples to collect
            steps_between_samples (int): Gibbs steps between collected samples
            temperature (float): Sampling temperature
            
        Returns:
            list: List of samples
        """
        # Initialize persistent chains
        visible = torch.bernoulli(
            torch.ones(batch_size, self.model.visible_dim, device=self.device) * 0.5
        )
        
        hiddens = []
        for hidden_dim in self.model.hidden_dims:
            hiddens.append(torch.rand(batch_size, hidden_dim, device=self.device))
        
        samples = []
        
        for sample_idx in range(n_samples):
            # Run chain for specified steps
            for _ in range(steps_between_samples):
                visible, hiddens = self.gibbs_step(visible, hiddens, temperature)
            
            # Collect sample
            samples.append(visible.clone())
        
        return samples