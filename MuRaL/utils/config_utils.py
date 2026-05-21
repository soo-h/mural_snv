import json

from pathlib import Path
from typing import Dict, Any, Union

def read_feature_config(config_path: Union[str, Path]) -> Dict[str, Any]:

    config_path = Path(config_path)
    
    # 1. check if the file exists
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    # 2. read JSON
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {config_path}: {e}")
    return config

def read_bnn_config(config_path, mode='train'):

    if mode not in ['train', 'pred']:
        raise ValueError(f"mode must be 'train' or 'pred', but got {mode}")

    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # set default values
    defaults = {
        "num_monte_carlo": 10,
        "prior_mu": 0.0,
        "prior_sigma": 1.0,
        "posterior_mu_init": 0.0,
        "posterior_rho_init": -3.0,
        "type": "Flipout",
        "moped_enable": False,
        "moped_delta": 0.5,
        "train_monte_carlo" : 10,
    }

    # set default values if not config
    for key, value in defaults.items():
        config.setdefault(key, value)
    
    # check
    if mode == 'pred':
        if config['moped_enable']:
            raise ValueError("moped_enable should be False in prediction mode")

    return config