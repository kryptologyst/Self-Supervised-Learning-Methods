#!/usr/bin/env python3
"""Evaluation script for self-supervised learning models."""

import argparse
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import set_seed, get_device, setup_logging, load_config
from src.data import CIFAR10Dataset, ContrastiveTransform, create_dataloader
from src.models import SimCLR, MoCo
from src.metrics import LinearProbeEvaluator, KNNEvaluator, EmbeddingEvaluator


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate self-supervised learning models")
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10"],
        help="Dataset to use"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, mps, cpu)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        device: Device to use
        
    Returns:
        Loaded model
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Determine model type from checkpoint
    if "base_model" in checkpoint:
        base_model = checkpoint["base_model"]
    else:
        # Try to infer from checkpoint path
        if "simclr" in checkpoint_path.lower():
            base_model = "simclr"
        elif "moco" in checkpoint_path.lower():
            base_model = "moco"
        else:
            raise ValueError("Cannot determine model type from checkpoint")
    
    # Load model
    if base_model == "simclr":
        model = SimCLR.load_checkpoint(checkpoint_path)
    elif base_model == "moco":
        model = MoCo.load_checkpoint(checkpoint_path)
    else:
        raise ValueError(f"Unsupported model type: {base_model}")
    
    return model


def load_data(config: OmegaConf) -> tuple:
    """Load dataset and create data loaders.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_config = config.data
    
    # Create transform (no augmentation for evaluation)
    transform = ContrastiveTransform(
        image_size=data_config.transform.get("image_size", 224),
        normalize=data_config.transform.get("normalize", True),
        color_jitter_strength=0.0,  # No augmentation for evaluation
        gaussian_blur_prob=0.0,
        horizontal_flip_prob=0.0,
    )
    
    # Load dataset
    if data_config._target_ == "src.data.datasets.CIFAR10Dataset":
        train_dataset = CIFAR10Dataset(
            root=data_config.get("root", "data/raw"),
            train=True,
            download=data_config.get("download", True),
            transform=transform,
        )
        
        val_dataset = CIFAR10Dataset(
            root=data_config.get("root", "data/raw"),
            train=False,
            download=data_config.get("download", True),
            transform=transform,
        )
    else:
        raise ValueError(f"Unsupported dataset: {data_config._target_}")
    
    # Create data loaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=data_config.get("batch_size", 256),
        shuffle=False,  # No shuffling for evaluation
        num_workers=data_config.get("num_workers", 4),
        pin_memory=data_config.get("pin_memory", True),
        drop_last=False,
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=data_config.get("batch_size", 256),
        shuffle=False,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=data_config.get("pin_memory", True),
        drop_last=False,
    )
    
    return train_loader, val_loader


def main() -> None:
    """Main evaluation function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup logging
    logger = setup_logging(
        level=config.logging.get("level", "INFO"),
        log_dir=config.logging.get("log_dir", "logs")
    )
    
    # Get device
    device = get_device(args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from checkpoint: {args.checkpoint}")
    model = load_model_from_checkpoint(args.checkpoint, device)
    model = model.to(device)
    model.eval()
    
    # Load data
    logger.info("Loading data...")
    train_loader, val_loader = load_data(config)
    
    # Linear probe evaluation
    logger.info("Running linear probe evaluation...")
    linear_probe_evaluator = LinearProbeEvaluator(
        epochs=config.evaluation.get("probe_epochs", 50),
        learning_rate=config.evaluation.get("probe_learning_rate", 0.001),
        batch_size=config.evaluation.get("probe_batch_size", 256),
    )
    
    linear_probe_metrics = linear_probe_evaluator.evaluate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )
    
    logger.info("Linear Probe Results:")
    for metric, value in linear_probe_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # K-NN evaluation
    if config.evaluation.knn_eval.get("enabled", True):
        logger.info("Running K-NN evaluation...")
        knn_evaluator = KNNEvaluator(
            k_values=config.evaluation.knn_eval.get("k_values", [1, 5, 10, 20]),
            distance_metric=config.evaluation.knn_eval.get("distance_metric", "cosine"),
        )
        
        knn_metrics = knn_evaluator.evaluate(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
        )
        
        logger.info("K-NN Results:")
        for metric, value in knn_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
    
    # Save results
    results = {
        "linear_probe": linear_probe_metrics,
        "knn": knn_metrics if config.evaluation.knn_eval.get("enabled", True) else {},
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "device": str(device),
    }
    
    results_path = os.path.join(args.output_dir, "evaluation_results.yaml")
    OmegaConf.save(results, results_path)
    logger.info(f"Results saved to: {results_path}")
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
