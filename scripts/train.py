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
    parser.add_argument('--config', type=str, default='configs/01_baseline_config.yaml',
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
    device_name = config['hardware']['device']
    if device_name == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_name)
    
    logger.info(f"Using device: {device}")
    
    # Create model with correct parameters
    model = DBM(
        visible_dim=config['model']['visible_dim'],
        hidden_dims=config['model']['hidden_dims'],
        learning_rate=config['training']['learning_rate'],
        use_cuda=(device.type == 'cuda')
    ).to(device)
    
    logger.info(f"Created DBM with {len(model.hidden_dims)} hidden layers: {model.hidden_dims}")
    
    # Load dataset
    train_dataset, val_dataset = get_dataset(
        name=config['data']['dataset'],
        train_val_split=config['data']['train_val_split'],
        normalize=config['data'].get('normalize', False)
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
    
    # Setup tensorboard
    writer = SummaryWriter(config['logging']['log_dir']) if config['logging']['tensorboard'] else None
    
    # Create trainer (no optimizer needed as DBM handles its own parameter updates)
    trainer = DBMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        writer=writer
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed from checkpoint: {args.resume}")
    
    # Train model
    try:
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training completed successfully!")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    finally:
        if writer:
            writer.close()

if __name__ == '__main__':
    main()