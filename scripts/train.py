#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.dbm import DBM
from src.data.loaders import get_dataset
from src.training.trainer import DBMTrainer
from src.utils.config import load_config
from src.utils.logging_utils import setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description='Train DBM model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                      help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume from')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(config['logging']['log_dir'])
    logger = logging.getLogger(__name__)
    
    # Setup device
    device = torch.device(config['hardware']['device'])
    
    # Create model
    model = DBM(
        visible_dim=config['model']['visible_dim'],
        hidden_dims=config['model']['hidden_dims'],
        dropout_rate=config['model']['dropout_rate'],
        batch_norm=config['model']['batch_norm']
    ).to(device)
    
    # Load dataset
    train_dataset, val_dataset = get_dataset(
        name=config['data']['dataset'],
        train_val_split=config['data']['train_val_split'],
        normalize=config['data']['normalize']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Setup tensorboard
    writer = SummaryWriter(config['logging']['log_dir']) if config['logging']['tensorboard'] else None
    
    # Create trainer
    trainer = DBMTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        writer=writer
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train model
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    finally:
        if writer:
            writer.close()

if __name__ == '__main__':
    main() 