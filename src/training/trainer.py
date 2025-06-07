import logging
import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class DBMTrainer:
    """
    Trainer for Deep Boltzmann Machine.
    """
    
    def __init__(
        self,
        model,
        optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: dict,
        writer: Optional[SummaryWriter] = None
    ):
        self.model = model
        self.optimizer = optimizer
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
            
            # Forward pass
            self.optimizer.zero_grad()
            hidden = self.model(data)
            visible = self.model.sample_visible(hidden)
            
            # Compute loss (free energy difference)
            loss = self.model.free_energy(data) - self.model.free_energy(visible)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to tensorboard
            if self.writer and batch_idx % self.config['logging']['log_interval'] == 0:
                self.writer.add_scalar('train/loss', loss.item(),
                                     epoch * len(self.train_loader) + batch_idx)
        
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
        
        with torch.no_grad():
            for data, _ in self.val_loader:
                data = data.to(self.device)
                data = data.view(data.size(0), -1)  # Flatten
                
                # Forward pass
                hidden = self.model(data)
                visible = self.model.sample_visible(hidden)
                
                # Compute loss
                loss = self.model.free_energy(data) - self.model.free_energy(visible)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Log to tensorboard
        if self.writer:
            self.writer.add_scalar('val/loss', avg_loss, epoch)
        
        return avg_loss
    
    def train(self):
        """
        Train the model.
        """
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
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
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
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.logger.info(f'Loaded checkpoint from epoch {checkpoint["epoch"]}') 