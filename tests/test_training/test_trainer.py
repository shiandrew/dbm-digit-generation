import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from src.models.dbm import DBM
from src.training.trainer import DBMTrainer

@pytest.fixture
def model():
    return DBM(
        visible_dim=784,
        hidden_dims=[500, 500],
        dropout_rate=0.2,
        batch_norm=True
    )

@pytest.fixture
def optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=0.001)

@pytest.fixture
def train_loader():
    # Create dummy dataset
    x = torch.randn(100, 1, 28, 28)
    y = torch.randint(0, 10, (100,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=32, shuffle=True)

@pytest.fixture
def val_loader():
    # Create dummy dataset
    x = torch.randn(20, 1, 28, 28)
    y = torch.randint(0, 10, (20,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=32, shuffle=False)

@pytest.fixture
def config():
    return {
        'training': {
            'epochs': 2,
            'early_stopping_patience': 3
        },
        'logging': {
            'log_dir': 'results/logs',
            'save_dir': 'models/checkpoints',
            'log_interval': 1,
            'save_interval': 1
        }
    }

def test_trainer_initialization(model, optimizer, train_loader, val_loader, config):
    """Test trainer initialization."""
    trainer = DBMTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device('cpu'),
        config=config
    )
    
    assert trainer.model == model
    assert trainer.optimizer == optimizer
    assert trainer.train_loader == train_loader
    assert trainer.val_loader == val_loader
    assert trainer.best_val_loss == float('inf')
    assert trainer.patience_counter == 0

def test_train_epoch(model, optimizer, train_loader, val_loader, config):
    """Test training for one epoch."""
    trainer = DBMTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device('cpu'),
        config=config
    )
    
    loss = trainer.train_epoch(0)
    assert isinstance(loss, float)
    assert not torch.isnan(torch.tensor(loss))
    assert not torch.isinf(torch.tensor(loss))

def test_validate(model, optimizer, train_loader, val_loader, config):
    """Test validation."""
    trainer = DBMTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device('cpu'),
        config=config
    )
    
    loss = trainer.validate(0)
    assert isinstance(loss, float)
    assert not torch.isnan(torch.tensor(loss))
    assert not torch.isinf(torch.tensor(loss))

def test_save_load_checkpoint(model, optimizer, train_loader, val_loader, config, tmp_path):
    """Test saving and loading checkpoints."""
    trainer = DBMTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device('cpu'),
        config=config
    )
    
    # Save checkpoint
    trainer.save_checkpoint(0)
    
    # Load checkpoint
    trainer.load_checkpoint(trainer.save_dir / 'checkpoint_epoch_0.pt')
    
    # Check if model parameters are preserved
    for p1, p2 in zip(model.parameters(), trainer.model.parameters()):
        assert torch.allclose(p1, p2)

def test_early_stopping(model, optimizer, train_loader, val_loader, config):
    """Test early stopping."""
    trainer = DBMTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device('cpu'),
        config=config
    )
    
    # Train for a few epochs
    trainer.train()
    
    # Check if training stopped early if validation loss didn't improve
    assert trainer.patience_counter <= config['training']['early_stopping_patience']

def test_tensorboard_logging(model, optimizer, train_loader, val_loader, config, tmp_path):
    """Test tensorboard logging."""
    writer = SummaryWriter(tmp_path)
    trainer = DBMTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=torch.device('cpu'),
        config=config,
        writer=writer
    )
    
    # Train for one epoch
    trainer.train_epoch(0)
    
    # Check if tensorboard files are created
    assert len(list(tmp_path.glob('*'))) > 0 