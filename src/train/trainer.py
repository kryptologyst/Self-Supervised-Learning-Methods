"""Training utilities for self-supervised learning."""

import os
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from ..utils import setup_logging, EarlyStopping, sanitize_log_data
from ..losses import NTXentLoss


class ContrastiveTrainer:
    """Trainer for contrastive learning models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[DictConfig] = None,
        device: Optional[torch.device] = None,
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Device to use
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or DictConfig({})
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.logger = setup_logging()
        
        # Training parameters
        self.epochs = self.config.get("epochs", 100)
        self.learning_rate = self.config.get("learning_rate", 0.0001)
        self.weight_decay = self.config.get("weight_decay", 0.0001)
        self.optimizer_name = self.config.get("optimizer", "adam")
        self.scheduler_name = self.config.get("scheduler", "cosine")
        self.warmup_epochs = self.config.get("warmup_epochs", 10)
        self.gradient_clip_norm = self.config.get("gradient_clip_norm", 1.0)
        self.accumulate_grad_batches = self.config.get("accumulate_grad_batches", 1)
        self.precision = self.config.get("precision", "16-mixed")
        self.compile_model = self.config.get("compile_model", False)
        
        # Validation parameters
        self.validate_every_n_epochs = self.config.get("validate_every_n_epochs", 5)
        self.save_top_k = self.config.get("save_top_k", 3)
        
        # Loss function
        loss_config = self.config.get("loss", {})
        self.loss_fn = NTXentLoss(
            temperature=loss_config.get("temperature", 0.5),
            normalize=loss_config.get("normalize", True),
        )
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        self._setup_scheduler()
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Compile model if requested
        if self.compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
            self.logger.info("Model compiled with torch.compile")
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.get("patience", 10),
            min_delta=self.config.get("min_delta", 0.001),
            mode="min",
        )
        
        self.logger.info(f"Initialized trainer with {self.epochs} epochs")
        self.logger.info(f"Learning rate: {self.learning_rate}")
        self.logger.info(f"Optimizer: {self.optimizer_name}")
        self.logger.info(f"Scheduler: {self.scheduler_name}")
        self.logger.info(f"Device: {self.device}")
    
    def _setup_optimizer(self) -> None:
        """Setup optimizer."""
        if self.optimizer_name == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
    
    def _setup_scheduler(self) -> None:
        """Setup learning rate scheduler."""
        if self.scheduler_name == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs
            )
        elif self.scheduler_name == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        elif self.scheduler_name == "exponential":
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.95
            )
        else:
            self.scheduler = None
    
    def train(self) -> Dict[str, Any]:
        """Train the model.
        
        Returns:
            Training history
        """
        self.logger.info("Starting training...")
        start_time = time.time()
        
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            
            # Training
            train_loss = self._train_epoch()
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = None
            if self.val_loader is not None and epoch % self.validate_every_n_epochs == 0:
                val_loss = self._validate_epoch()
                self.val_losses.append(val_loss)
                
                # Check for improvement
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint("best")
                
                # Early stopping
                if self.early_stopping(val_loss, self.model):
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Log progress
            self._log_epoch(epoch, train_loss, val_loss)
            
            # Save checkpoint
            if epoch % 10 == 0:
                self._save_checkpoint(f"epoch_{epoch}")
        
        # Restore best weights
        self.early_stopping.restore_weights(self.model)
        
        # Final checkpoint
        self._save_checkpoint("final")
        
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "training_time": training_time,
            "epochs_trained": self.current_epoch + 1,
        }
    
    def _train_epoch(self) -> float:
        """Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Get batch data
            if len(batch) == 3:  # (view1, view2, label)
                view1, view2, _ = batch
            else:  # (data, label)
                view1, view2 = batch[0], batch[0]  # Use same data for both views
            
            view1 = view1.to(self.device)
            view2 = view2.to(self.device)
            
            # Forward pass
            if hasattr(self.model, 'contrastive_loss'):
                # Model has built-in contrastive loss
                loss = self.model.contrastive_loss(view1, view2)
            else:
                # Use external loss function
                z1 = self.model(view1)
                z2 = self.model(view2)
                loss = self.loss_fn(z1, z2)
            
            # Backward pass
            loss = loss / self.accumulate_grad_batches
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.accumulate_grad_batches == 0:
                # Gradient clipping
                if self.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_norm
                    )
                
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.accumulate_grad_batches
            num_batches += 1
        
        return total_loss / num_batches
    
    def _validate_epoch(self) -> float:
        """Validate for one epoch.
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Get batch data
                if len(batch) == 3:  # (view1, view2, label)
                    view1, view2, _ = batch
                else:  # (data, label)
                    view1, view2 = batch[0], batch[0]
                
                view1 = view1.to(self.device)
                view2 = view2.to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'contrastive_loss'):
                    loss = self.model.contrastive_loss(view1, view2)
                else:
                    z1 = self.model(view1)
                    z2 = self.model(view2)
                    loss = self.loss_fn(z1, z2)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _log_epoch(self, epoch: int, train_loss: float, val_loss: Optional[float]) -> None:
        """Log epoch progress."""
        log_data = {
            "epoch": epoch,
            "train_loss": train_loss,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }
        
        if val_loss is not None:
            log_data["val_loss"] = val_loss
        
        # Sanitize log data
        log_data = sanitize_log_data(log_data)
        
        self.logger.info(f"Epoch {epoch}: {log_data}")
    
    def _save_checkpoint(self, name: str) -> None:
        """Save model checkpoint."""
        checkpoint_dir = self.config.get("checkpointing", {}).get("save_dir", "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f"{name}.pth")
        
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.current_epoch = checkpoint.get("epoch", 0)
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {self.current_epoch}")
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model on given data loader.
        
        Args:
            data_loader: Data loader for evaluation
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                if len(batch) == 3:  # (view1, view2, label)
                    view1, view2, _ = batch
                else:  # (data, label)
                    view1, view2 = batch[0], batch[0]
                
                view1 = view1.to(self.device)
                view2 = view2.to(self.device)
                
                if hasattr(self.model, 'contrastive_loss'):
                    loss = self.model.contrastive_loss(view1, view2)
                else:
                    z1 = self.model(view1)
                    z2 = self.model(view2)
                    loss = self.loss_fn(z1, z2)
                
                total_loss += loss.item()
                num_batches += 1
        
        return {"loss": total_loss / num_batches}
