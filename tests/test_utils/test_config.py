import pytest
import yaml
from pathlib import Path

from src.utils.config import load_config, save_config, update_config

@pytest.fixture
def config():
    return {
        'model': {
            'type': 'dbm',
            'hidden_dims': [500, 500],
            'visible_dim': 784
        },
        'training': {
            'batch_size': 128,
            'learning_rate': 0.001
        },
        'logging': {
            'log_dir': 'results/logs',
            'save_dir': 'models/checkpoints'
        }
    }

def test_load_config(tmp_path, config):
    """Test loading configuration from file."""
    # Create config file
    config_path = tmp_path / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Load config
    loaded_config = load_config(config_path)
    
    # Check values
    assert loaded_config['model']['type'] == 'dbm'
    assert loaded_config['model']['hidden_dims'] == [500, 500]
    assert loaded_config['training']['batch_size'] == 128
    assert loaded_config['training']['learning_rate'] == 0.001

def test_save_config(tmp_path, config):
    """Test saving configuration to file."""
    # Save config
    config_path = tmp_path / "config.yaml"
    save_config(config, config_path)
    
    # Load and check
    with open(config_path, 'r') as f:
        saved_config = yaml.safe_load(f)
    
    assert saved_config == config

def test_update_config(config):
    """Test updating configuration."""
    updates = {
        'model': {
            'hidden_dims': [1000, 1000]
        },
        'training': {
            'batch_size': 256
        }
    }
    
    updated_config = update_config(config, updates)
    
    # Check updated values
    assert updated_config['model']['hidden_dims'] == [1000, 1000]
    assert updated_config['training']['batch_size'] == 256
    
    # Check unchanged values
    assert updated_config['model']['type'] == 'dbm'
    assert updated_config['training']['learning_rate'] == 0.001

def test_load_config_invalid_path():
    """Test loading configuration from invalid path."""
    with pytest.raises(FileNotFoundError):
        load_config("invalid/path/config.yaml")

def test_save_config_invalid_path():
    """Test saving configuration to invalid path."""
    config = {'key': 'value'}
    with pytest.raises(FileNotFoundError):
        save_config(config, "invalid/path/config.yaml")

def test_update_config_nested():
    """Test updating nested configuration."""
    config = {
        'model': {
            'type': 'dbm',
            'params': {
                'hidden_dims': [500, 500],
                'visible_dim': 784
            }
        }
    }
    
    updates = {
        'model': {
            'params': {
                'hidden_dims': [1000, 1000]
            }
        }
    }
    
    updated_config = update_config(config, updates)
    
    # Check updated values
    assert updated_config['model']['params']['hidden_dims'] == [1000, 1000]
    
    # Check unchanged values
    assert updated_config['model']['type'] == 'dbm'
    assert updated_config['model']['params']['visible_dim'] == 784 