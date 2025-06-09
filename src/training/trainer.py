import logging
import os
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class DBMTrainer:
    """
    Trainer for Deep Boltzmann Machine using proper DBM training procedure.
    
    The training consists of two phases:
    1. Layer-wise pre-training of RBMs 
    2. Fine-tuning the entire DBM
    """
    
    def __init__(
        self,
        model,  # DBM model
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: dict,
        writer: Optional[SummaryWriter] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.writer = writer
        
        self.logger = logging.getLogger(__name__)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Create save directory
        self.save_dir = Path(config['logging']['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        self.learning_rate = config['training']['learning_rate']
        self.cd_steps = config['training'].get('cd_steps', 1)
        self.persistent_cd = config['training'].get('persistent_cd', False)
        
        # Initialize persistent chains for PCD if enabled
        if self.persistent_cd:
            self.persistent_visible = None
            self.persistent_hiddens = None
    
    def contrastive_divergence_step(
        self, 
        visible_data: torch.Tensor, 
        k: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Perform contrastive divergence for DBM training.
        
        Args:
            visible_data (torch.Tensor): Training batch
            k (int): Number of Gibbs sampling steps
            
        Returns:
            tuple: (reconstruction_error, positive_phase_data, negative_phase_data)
        """
        batch_size = visible_data.size(0)
        
        # Positive phase: compute hidden probabilities given data
        pos_hiddens = self.model.forward(visible_data)
        
        # Negative phase: start from data or persistent chains
        if self.persistent_cd and self.persistent_visible is not None:
            # Use persistent contrastive divergence
            neg_visible = self.persistent_visible[:batch_size].clone()
            neg_hiddens = [h[:batch_size].clone() for h in self.persistent_hiddens]
        else:
            # Start negative chain from current data
            neg_visible = visible_data.clone()
            neg_hiddens = [h.clone() for h in pos_hiddens]
        
        # Run k steps of Gibbs sampling
        for step in range(k):
            # Sample hiddens given visible
            new_hiddens = []
            current_input = neg_visible
            
            for layer_idx in range(self.model.num_layers):
                # Get input from layer below
                if layer_idx == 0:
                    bottom_input = neg_visible
                else:
                    bottom_input = new_hiddens[layer_idx - 1]
                
                # Get input from layer above (if exists)
                if layer_idx < self.model.num_layers - 1:
                    top_input = neg_hiddens[layer_idx + 1]
                    # Combine bottom-up and top-down
                    pre_activation = (
                        F.linear(bottom_input, self.model.weights[layer_idx].t(), self.model.biases[layer_idx + 1]) +
                        F.linear(top_input, self.model.weights[layer_idx + 1])
                    )
                else:
                    # Top layer only gets bottom-up input
                    pre_activation = F.linear(bottom_input, self.model.weights[layer_idx].t(), self.model.biases[layer_idx + 1])
                
                hidden_probs = torch.sigmoid(pre_activation)
                hidden_samples = torch.bernoulli(hidden_probs)
                new_hiddens.append(hidden_samples)
            
            neg_hiddens = new_hiddens
            
            # Sample visible given hiddens
            visible_pre_activation = F.linear(neg_hiddens[0], self.model.weights[0], self.model.biases[0])
            visible_probs = torch.sigmoid(visible_pre_activation)
            neg_visible = torch.bernoulli(visible_probs)
        
        # Update persistent chains
        if self.persistent_cd:
            if self.persistent_visible is None:
                # Initialize persistent chains
                self.persistent_visible = neg_visible.clone()
                self.persistent_hiddens = [h.clone() for h in neg_hiddens]
            else:
                # Update persistent chains
                self.persistent_visible[:batch_size] = neg_visible
                for i, h in enumerate(neg_hiddens):
                    self.persistent_hiddens[i][:batch_size] = h
        
        # Compute reconstruction error
        recon_error = F.mse_loss(visible_data, visible_probs)
        
        return recon_error, pos_hiddens, (neg_visible, neg_hiddens)
    
    def compute_gradients(
        self, 
        visible_data: torch.Tensor, 
        pos_hiddens: List[torch.Tensor],
        neg_data: Tuple[torch.Tensor, List[torch.Tensor]]
    ) -> dict:
        """
        Compute gradients for DBM parameters.
        
        Args:
            visible_data (torch.Tensor): Positive phase visible data
            pos_hiddens (List[torch.Tensor]): Positive phase hidden activations
            neg_data (tuple): Negative phase (visible, hiddens)
            
        Returns:
            dict: Gradients for each parameter
        """
        neg_visible, neg_hiddens = neg_data
        batch_size = visible_data.size(0)
        
        gradients = {}
        
        # Gradient for visible bias
        pos_visible_bias = torch.mean(visible_data, dim=0)
        neg_visible_bias = torch.mean(neg_visible, dim=0)
        gradients['visible_bias'] = pos_visible_bias - neg_visible_bias
        
        # Gradients for hidden biases
        for i in range(self.model.num_layers):
            pos_hidden_bias = torch.mean(pos_hiddens[i], dim=0)
            neg_hidden_bias = torch.mean(neg_hiddens[i], dim=0)
            gradients[f'hidden_bias_{i}'] = pos_hidden_bias - neg_hidden_bias
        
        # Gradients for weights
        # Visible to first hidden
        pos_weight_grad_0 = torch.mm(visible_data.t(), pos_hiddens[0]) / batch_size
        neg_weight_grad_0 = torch.mm(neg_visible.t(), neg_hiddens[0]) / batch_size
        gradients['weight_0'] = pos_weight_grad_0 - neg_weight_grad_0
        
        # Hidden to hidden weights
        for i in range(self.model.num_layers - 1):
            pos_weight_grad = torch.mm(pos_hiddens[i].t(), pos_hiddens[i + 1]) / batch_size
            neg_weight_grad = torch.mm(neg_hiddens[i].t(), neg_hiddens[i + 1]) / batch_size
            gradients[f'weight_{i + 1}'] = pos_weight_grad - neg_weight_grad
        
        return gradients
    
    def update_parameters(self, gradients: dict):
        """
        Update DBM parameters using computed gradients.
        
        Args:
            gradients (dict): Computed gradients
        """
        # Update visible bias
        self.model.biases[0].data += self.learning_rate * gradients['visible_bias']
        
        # Update hidden biases
        for i in range(self.model.num_layers):
            self.model.biases[i + 1].data += self.learning_rate * gradients[f'hidden_bias_{i}']
        
        # Update weights
        for i in range(len(self.model.weights)):
            self.model.weights[i].data += self.learning_rate * gradients[f'weight_{i}']
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            epoch (int): Current epoch number
            
        Returns:
            float: Average training loss
        """
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(self.device)
            data = data.view(data.size(0), -1)  # Flatten
            
            # Binarize data for Bernoulli units
            data = torch.bernoulli(data)
            
            # Perform contrastive divergence
            recon_error, pos_hiddens, neg_data = self.contrastive_divergence_step(
                data, k=self.cd_steps
            )
            
            # Compute gradients
            gradients = self.compute_gradients(data, pos_hiddens, neg_data)
            
            # Update parameters
            self.update_parameters(gradients)
            
            total_loss += recon_error.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': recon_error.item()})
            
            # Log to tensorboard
            if self.writer and batch_idx % self.config['logging']['log_interval'] == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('train/reconstruction_error', recon_error.item(), global_step)
                
                # Log parameter statistics
                for i, weight in enumerate(self.model.weights):
                    self.writer.add_scalar(f'params/weight_{i}_mean', weight.data.mean().item(), global_step)
                    self.writer.add_scalar(f'params/weight_{i}_std', weight.data.std().item(), global_step)
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self, epoch: int) -> float:
        """
        Validate the model.
        
        Args:
            epoch (int): Current epoch number
            
        Returns:
            float: Average validation loss
        """
        self.model.eval()
        total_loss = 0
        total_free_energy = 0
        
        with torch.no_grad():
            for data, _ in self.val_loader:
                data = data.to(self.device)
                data = data.view(data.size(0), -1)  # Flatten
                data = torch.bernoulli(data)
                
                # Compute reconstruction error
                hiddens = self.model.forward(data)
                visible_probs, _ = self.model.sample_visible_given_hidden(hiddens[0])
                recon_error = F.mse_loss(data, visible_probs)
                
                # Compute free energy
                free_energy = self.model.free_energy(data)
                
                total_loss += recon_error.item()
                total_free_energy += free_energy.mean().item()
        
        avg_loss = total_loss / len(self.val_loader)
        avg_free_energy = total_free_energy / len(self.val_loader)
        
        # Log to tensorboard
        if self.writer:
            self.writer.add_scalar('val/reconstruction_error', avg_loss, epoch)
            self.writer.add_scalar('val/free_energy', avg_free_energy, epoch)
        
        return avg_loss
    
    def train(self):
        """
        Train the DBM model.
        """
        self.logger.info("Starting DBM training...")
        
        for epoch in range(self.config['training']['epochs']):
            # Train
            train_loss = self.train_epoch(epoch)
            self.logger.info(f'Epoch {epoch}: Train Loss = {train_loss:.4f}')
            
            # Validate
            val_loss = self.validate(epoch)
            self.logger.info(f'Epoch {epoch}: Val Loss = {val_loss:.4f}')
            
            # Save checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
            
            # Save regular checkpoint
            if epoch % self.config['logging']['save_interval'] == 0:
                self.save_checkpoint(epoch)
            
            # Early stopping
            if self.patience_counter >= self.config['training']['early_stopping_patience']:
                self.logger.info('Early stopping triggered')
                break
        
        self.logger.info("Training completed!")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch (int): Current epoch number
            is_best (bool): Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'model_config': {
                'visible_dim': self.model.visible_dim,
                'hidden_dims': self.model.hidden_dims
            }
        }
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.save_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: str):
        """
        Load model checkpoint.
        
        Args:
            path (str): Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.logger.info(f'Loaded checkpoint from epoch {checkpoint["epoch"]}')
    
    def generate_samples(self, num_samples: int = 16, num_steps: int = 1000) -> torch.Tensor:
        """
        Generate samples from the trained model.
        
        Args:
            num_samples (int): Number of samples to generate
            num_steps (int): Number of Gibbs sampling steps
            
        Returns:
            torch.Tensor: Generated samples
        """
        self.model.eval()
        with torch.no_grad():
            samples = self.model.generate_samples(num_samples, num_steps)
        return samples