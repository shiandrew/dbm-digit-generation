import logging
import os
from pathlib import Path
from typing import Optional

def setup_logging(
    log_dir: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_dir (str): Directory to store log files
        level (int): Logging level
        log_file (str, optional): Name of log file. If None, uses 'training.log'
    """
    # Create log directory
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup log file
    if log_file is None:
        log_file = 'training.log'
    log_path = log_dir / log_file
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

def get_logger(name: str) -> logging.Logger:
    """
    Get logger instance.
    
    Args:
        name (str): Logger name
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)

def log_config(logger: logging.Logger, config: dict) -> None:
    """
    Log configuration dictionary.
    
    Args:
        logger (logging.Logger): Logger instance
        config (dict): Configuration dictionary
    """
    logger.info("Configuration:")
    for section, values in config.items():
        logger.info(f"\n{section}:")
        for key, value in values.items():
            logger.info(f"  {key}: {value}") 