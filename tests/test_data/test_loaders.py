import pytest
import torch
from torch.utils.data import DataLoader

from src.data.loaders import get_dataset, get_test_dataset

def test_get_dataset():
    """Test dataset loading and splitting."""
    train_dataset, val_dataset = get_dataset(
        name='mnist',
        train_val_split=0.9,
        normalize=True
    )
    
    # Check dataset sizes
    assert len(train_dataset) > 0
    assert len(val_dataset) > 0
    assert len(train_dataset) + len(val_dataset) == 60000  # MNIST train size
    
    # Check data format
    x, y = train_dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, int)
    assert x.shape == (1, 28, 28)  # MNIST image shape
    assert 0 <= y <= 9  # MNIST labels

def test_get_test_dataset():
    """Test test dataset loading."""
    test_dataset = get_test_dataset(
        name='mnist',
        normalize=True
    )
    
    # Check dataset size
    assert len(test_dataset) == 10000  # MNIST test size
    
    # Check data format
    x, y = test_dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, int)
    assert x.shape == (1, 28, 28)  # MNIST image shape
    assert 0 <= y <= 9  # MNIST labels

def test_data_normalization():
    """Test data normalization."""
    train_dataset, _ = get_dataset(
        name='mnist',
        train_val_split=0.9,
        normalize=True
    )
    
    # Check normalization
    x, _ = train_dataset[0]
    assert x.min() >= 0
    assert x.max() <= 1

def test_data_loader():
    """Test data loader creation."""
    train_dataset, _ = get_dataset(
        name='mnist',
        train_val_split=0.9,
        normalize=True
    )
    
    batch_size = 32
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    # Check batch format
    x, y = next(iter(loader))
    assert x.shape == (batch_size, 1, 28, 28)
    assert y.shape == (batch_size,)
    assert torch.all((x >= 0) & (x <= 1))  # Check normalization

def test_invalid_dataset():
    """Test invalid dataset name."""
    with pytest.raises(ValueError):
        get_dataset(name='invalid_dataset')

def test_fashion_mnist():
    """Test Fashion-MNIST dataset loading."""
    train_dataset, val_dataset = get_dataset(
        name='fashion_mnist',
        train_val_split=0.9,
        normalize=True
    )
    
    # Check dataset sizes
    assert len(train_dataset) > 0
    assert len(val_dataset) > 0
    assert len(train_dataset) + len(val_dataset) == 60000  # Fashion-MNIST train size
    
    # Check data format
    x, y = train_dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, int)
    assert x.shape == (1, 28, 28)  # Fashion-MNIST image shape
    assert 0 <= y <= 9  # Fashion-MNIST labels 