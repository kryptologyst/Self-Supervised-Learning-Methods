#!/usr/bin/env python3
"""Main training script for self-supervised learning."""

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
from src.train import ContrastiveTrainer
from src.metrics import LinearProbeEvaluator, KNNEvaluator


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train self-supervised learning models")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="simclr",
        choices=["simclr", "moco"],
        help="Model to train"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10"],
        help="Dataset to use"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate"
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
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only evaluate, don't train"
    )
    
    return parser.parse_args()


def load_model(config: OmegaConf, device: torch.device) -> torch.nn.Module:
    """Load model from configuration.
    
    Args:
        config: Configuration object
        device: Device to use
        
    Returns:
        Loaded model
    """
    model_config = config.model
    
    if model_config.base_model == "simclr":
        model = SimCLR(
            base_model=model_config.get("base_model", "resnet50"),
            projection_dim=model_config.get("projection_dim", 128),
            hidden_dim=model_config.get("hidden_dim", 512),
            temperature=model_config.get("temperature", 0.5),
            pretrained=model_config.get("pretrained", True),
            freeze_backbone=model_config.get("freeze_backbone", False),
        )
    elif model_config.base_model == "moco":
        model = MoCo(
            base_model=model_config.get("base_model", "resnet50"),
            projection_dim=model_config.get("projection_dim", 128),
            hidden_dim=model_config.get("hidden_dim", 512),
            temperature=model_config.get("temperature", 0.07),
            momentum=model_config.get("momentum", 0.999),
            queue_size=model_config.get("queue_size", 65536),
            pretrained=model_config.get("pretrained", True),
            freeze_backbone=model_config.get("freeze_backbone", False),
        )
    else:
        raise ValueError(f"Unsupported model: {model_config.base_model}")
    
    return model


def load_data(config: OmegaConf) -> tuple:
    """Load dataset and create data loaders.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_config = config.data
    
    # Create transform
    transform = ContrastiveTransform(
        image_size=data_config.transform.get("image_size", 224),
        normalize=data_config.transform.get("normalize", True),
        color_jitter_strength=data_config.transform.get("color_jitter_strength", 0.8),
        gaussian_blur_prob=data_config.transform.get("gaussian_blur_prob", 0.5),
        horizontal_flip_prob=data_config.transform.get("horizontal_flip_prob", 0.5),
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
        shuffle=data_config.get("shuffle", True),
        num_workers=data_config.get("num_workers", 4),
        pin_memory=data_config.get("pin_memory", True),
        drop_last=data_config.get("drop_last", True),
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
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.device != "auto":
        config.experiment.device = args.device
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup logging
    logger = setup_logging(
        level=config.logging.get("level", "INFO"),
        log_dir=config.logging.get("log_dir", "logs")
    )
    
    # Get device
    device = get_device(config.experiment.device)
    
    # Log configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(config))
    
    # Load model
    logger.info("Loading model...")
    model = load_model(config, device)
    
    # Load data
    logger.info("Loading data...")
    train_loader, val_loader = load_data(config)
    
    # Create trainer
    trainer = ContrastiveTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.training,
        device=device,
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train model
    if not args.eval_only:
        logger.info("Starting training...")
        training_history = trainer.train()
        
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {training_history['best_val_loss']:.4f}")
        logger.info(f"Training time: {training_history['training_time']:.2f} seconds")
    
    # Evaluate model
    logger.info("Evaluating model...")
    
    # Linear probe evaluation
    linear_probe_evaluator = LinearProbeEvaluator(
        epochs=config.evaluation.get("probe_epochs", 50),
        learning_rate=config.evaluation.get("probe_learning_rate", 0.001),
        batch_size=config.evaluation.get("probe_batch_size", 256),
    )
    
    linear_probe_metrics = linear_probe_evaluator.evaluate(
        model=trainer.model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )
    
    logger.info("Linear Probe Results:")
    for metric, value in linear_probe_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # K-NN evaluation
    if config.evaluation.knn_eval.get("enabled", True):
        knn_evaluator = KNNEvaluator(
            k_values=config.evaluation.knn_eval.get("k_values", [1, 5, 10, 20]),
            distance_metric=config.evaluation.knn_eval.get("distance_metric", "cosine"),
        )
        
        knn_metrics = knn_evaluator.evaluate(
            model=trainer.model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
        )
        
        logger.info("K-NN Results:")
        for metric, value in knn_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()
