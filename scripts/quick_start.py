#!/usr/bin/env python3
"""Quick start script for self-supervised learning."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False


def main():
    """Main quick start function."""
    print("🚀 Self-Supervised Learning Methods - Quick Start")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required!")
        sys.exit(1)
    
    print("✅ Python version check passed")
    
    # Install dependencies
    if not run_command("pip install -e .", "Installing dependencies"):
        sys.exit(1)
    
    # Create necessary directories
    directories = ["data/raw", "checkpoints", "logs", "assets"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Download CIFAR-10 dataset
    print("🔄 Downloading CIFAR-10 dataset...")
    try:
        import torch
        from torchvision import datasets
        datasets.CIFAR10(root="data/raw", train=True, download=True)
        datasets.CIFAR10(root="data/raw", train=False, download=True)
        print("✅ CIFAR-10 dataset downloaded successfully!")
    except Exception as e:
        print(f"❌ Failed to download CIFAR-10 dataset: {e}")
        sys.exit(1)
    
    # Train a quick model
    print("🔄 Training a quick SimCLR model (5 epochs)...")
    if not run_command(
        "python src/train.py --config configs/config.yaml --model simclr --epochs 5",
        "Training SimCLR model"
    ):
        print("⚠️ Training failed, but setup is complete!")
    
    print("\n🎉 Quick start completed!")
    print("\nNext steps:")
    print("1. Train a full model: python src/train.py --config configs/config.yaml --model simclr")
    print("2. Launch demo: streamlit run src/demo.py")
    print("3. Evaluate model: python src/eval.py --checkpoint checkpoints/best.pth")


if __name__ == "__main__":
    main()
