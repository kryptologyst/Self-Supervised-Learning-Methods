"""Utility functions for self-supervised learning framework."""

import logging
import os
import random
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from omegaconf import DictConfig, OmegaConf


def setup_logging(level: str = "INFO", log_dir: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to save log files
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("ssl_framework")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_dir provided)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, "ssl_framework.log"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CUDA operations deterministic
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    # Set environment variables for additional reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device: Device specification ("auto", "cuda", "mps", "cpu")
        
    Returns:
        PyTorch device object
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    device_obj = torch.device(device)
    
    # Log device information
    logger = logging.getLogger("ssl_framework")
    logger.info(f"Using device: {device_obj}")
    
    if device_obj.type == "cuda":
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif device_obj.type == "mps":
        logger.info("Using Apple Silicon GPU (MPS)")
    
    return device_obj


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        OmegaConf configuration object
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = OmegaConf.load(config_path)
    return config


def save_config(config: DictConfig, save_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object to save
        save_path: Path to save configuration
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    OmegaConf.save(config, save_path)


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_number(num: Union[int, float]) -> str:
    """Format large numbers with appropriate suffixes.
    
    Args:
        num: Number to format
        
    Returns:
        Formatted string
    """
    if num >= 1e9:
        return f"{num/1e9:.1f}B"
    elif num >= 1e6:
        return f"{num/1e6:.1f}M"
    elif num >= 1e3:
        return f"{num/1e3:.1f}K"
    else:
        return str(num)


def sanitize_log_data(data: Any) -> Any:
    """Sanitize data for logging to remove potential PII.
    
    Args:
        data: Data to sanitize
        
    Returns:
        Sanitized data
    """
    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            # Skip potentially sensitive keys
            if any(sensitive in key.lower() for sensitive in ["password", "token", "key", "secret", "id"]):
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = sanitize_log_data(value)
        return sanitized
    elif isinstance(data, (list, tuple)):
        return [sanitize_log_data(item) for item in data]
    elif isinstance(data, str) and len(data) > 100:
        # Truncate long strings
        return data[:100] + "..."
    else:
        return data


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best_weights: bool = True,
    ):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: "min" for loss, "max" for accuracy
            restore_best_weights: Whether to restore best weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        """Check if training should stop.
        
        Args:
            score: Current validation score
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        return self.counter >= self.patience
    
    def _is_improvement(self, score: float) -> bool:
        """Check if score is an improvement."""
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta
    
    def restore_weights(self, model: torch.nn.Module) -> None:
        """Restore best weights to model."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
