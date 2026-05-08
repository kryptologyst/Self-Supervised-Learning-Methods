#!/usr/bin/env python3
"""Benchmark script for comparing different self-supervised learning methods."""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils import set_seed, get_device, setup_logging
from src.data import CIFAR10Dataset, ContrastiveTransform, create_dataloader
from src.models import SimCLR, MoCo
from src.train import ContrastiveTrainer
from src.metrics import LinearProbeEvaluator, KNNEvaluator


def benchmark_model(model_name: str, config: OmegaConf, device: torch.device) -> dict:
    """Benchmark a single model.
    
    Args:
        model_name: Name of the model to benchmark
        config: Configuration object
        device: Device to use
        
    Returns:
        Benchmark results
    """
    logger = setup_logging()
    logger.info(f"Benchmarking {model_name}...")
    
    # Load model
    if model_name == "simclr":
        model = SimCLR(
            base_model="resnet50",
            projection_dim=128,
            temperature=0.5,
        )
    elif model_name == "moco":
        model = MoCo(
            base_model="resnet50",
            projection_dim=128,
            temperature=0.07,
            momentum=0.999,
            queue_size=65536,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    model = model.to(device)
    
    # Load data
    transform = ContrastiveTransform(image_size=224, normalize=True)
    
    train_dataset = CIFAR10Dataset(
        root="data/raw",
        train=True,
        download=True,
        transform=transform,
    )
    
    val_dataset = CIFAR10Dataset(
        root="data/raw",
        train=False,
        download=True,
        transform=transform,
    )
    
    train_loader = create_dataloader(train_dataset, batch_size=256, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=256, shuffle=False)
    
    # Training benchmark
    trainer = ContrastiveTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.training,
        device=device,
    )
    
    start_time = time.time()
    training_history = trainer.train()
    training_time = time.time() - start_time
    
    # Evaluation benchmark
    start_time = time.time()
    
    # Linear probe evaluation
    linear_probe_evaluator = LinearProbeEvaluator(epochs=20)  # Quick evaluation
    linear_probe_metrics = linear_probe_evaluator.evaluate(
        model=trainer.model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )
    
    # K-NN evaluation
    knn_evaluator = KNNEvaluator(k_values=[1, 5, 10])
    knn_metrics = knn_evaluator.evaluate(
        model=trainer.model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )
    
    evaluation_time = time.time() - start_time
    
    # Compile results
    results = {
        "model": model_name,
        "training_time": training_time,
        "evaluation_time": evaluation_time,
        "total_time": training_time + evaluation_time,
        "linear_probe": linear_probe_metrics,
        "knn": knn_metrics,
        "training_history": training_history,
    }
    
    logger.info(f"Benchmark completed for {model_name}")
    return results


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="Benchmark self-supervised learning methods")
    parser.add_argument("--models", nargs="+", default=["simclr", "moco"], help="Models to benchmark")
    parser.add_argument("--config", default="configs/config.yaml", help="Configuration file")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--output", default="benchmark_results.yaml", help="Output file")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    config.training.epochs = args.epochs
    
    # Setup
    set_seed(42)
    device = get_device(args.device)
    logger = setup_logging()
    
    logger.info("Starting benchmark...")
    logger.info(f"Models: {args.models}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Device: {device}")
    
    # Run benchmarks
    all_results = []
    
    for model_name in args.models:
        try:
            results = benchmark_model(model_name, config, device)
            all_results.append(results)
        except Exception as e:
            logger.error(f"Benchmark failed for {model_name}: {e}")
    
    # Save results
    benchmark_data = {
        "config": OmegaConf.to_yaml(config),
        "device": str(device),
        "epochs": args.epochs,
        "results": all_results,
    }
    
    OmegaConf.save(benchmark_data, args.output)
    logger.info(f"Benchmark results saved to {args.output}")
    
    # Print summary
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    
    for result in all_results:
        print(f"\n{result['model'].upper()}:")
        print(f"  Training time: {result['training_time']:.2f}s")
        print(f"  Evaluation time: {result['evaluation_time']:.2f}s")
        print(f"  Total time: {result['total_time']:.2f}s")
        print(f"  Linear probe accuracy: {result['linear_probe']['accuracy']:.3f}")
        print(f"  K-NN accuracy (k=1): {result['knn']['knn_1_accuracy']:.3f}")
    
    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()
