import os
from pathlib import Path
from typing import Dict, Any

import yaml

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to config file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert paths to absolute paths
    for key in ['log_dir', 'save_dir']:
        if key in config.get('logging', {}):
            config['logging'][key] = str(Path(config['logging'][key]).resolve())
    
    return config

def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config (dict): Configuration dictionary
        save_path (str): Path to save config file
    """
    # Convert Path objects to strings
    config_copy = config.copy()
    for key in ['log_dir', 'save_dir']:
        if key in config_copy.get('logging', {}):
            config_copy['logging'][key] = str(config_copy['logging'][key])
    
    with open(save_path, 'w') as f:
        yaml.dump(config_copy, f, default_flow_style=False)

def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with new values.
    
    Args:
        config (dict): Original configuration
        updates (dict): Updates to apply
        
    Returns:
        dict: Updated configuration
    """
    def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = deep_update(d[k], v)
            else:
                d[k] = v
        return d
    
    return deep_update(config.copy(), updates) 