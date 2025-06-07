import logging
import pytest
from pathlib import Path

from src.utils.logging_utils import setup_logging, get_logger, log_config

def test_setup_logging(tmp_path):
    """Test logging setup."""
    log_dir = tmp_path / "logs"
    setup_logging(log_dir)
    
    # Check log directory creation
    assert log_dir.exists()
    assert (log_dir / "training.log").exists()
    
    # Check logger configuration
    logger = logging.getLogger()
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 2  # File and stream handlers

def test_get_logger():
    """Test getting logger instance."""
    logger = get_logger("test_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"

def test_log_config(tmp_path):
    """Test logging configuration."""
    # Setup logging
    log_dir = tmp_path / "logs"
    setup_logging(log_dir)
    
    # Get logger
    logger = get_logger("test_logger")
    
    # Test logging configuration
    config = {
        'model': {
            'type': 'dbm',
            'hidden_dims': [500, 500]
        },
        'training': {
            'batch_size': 128
        }
    }
    
    # Capture log output
    with open(log_dir / "training.log", 'r') as f:
        log_config(logger, config)
        log_content = f.read()
    
    # Check log content
    assert "Configuration:" in log_content
    assert "model:" in log_content
    assert "training:" in log_content
    assert "type: dbm" in log_content
    assert "batch_size: 128" in log_content

def test_setup_logging_custom_file(tmp_path):
    """Test logging setup with custom log file."""
    log_dir = tmp_path / "logs"
    log_file = "custom.log"
    setup_logging(log_dir, log_file=log_file)
    
    # Check log file creation
    assert (log_dir / log_file).exists()

def test_setup_logging_custom_level(tmp_path):
    """Test logging setup with custom level."""
    log_dir = tmp_path / "logs"
    setup_logging(log_dir, level=logging.DEBUG)
    
    # Check logger level
    logger = logging.getLogger()
    assert logger.level == logging.DEBUG

def test_log_config_empty():
    """Test logging empty configuration."""
    logger = get_logger("test_logger")
    log_config(logger, {})  # Should not raise any errors

def test_log_config_nested():
    """Test logging nested configuration."""
    logger = get_logger("test_logger")
    config = {
        'model': {
            'type': 'dbm',
            'params': {
                'hidden_dims': [500, 500],
                'visible_dim': 784
            }
        }
    }
    log_config(logger, config)  # Should not raise any errors 