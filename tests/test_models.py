"""Tests for self-supervised learning models."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models import SimCLR, MoCo
from src.losses import NTXentLoss, InfoNCE
from src.metrics import LinearProbeEvaluator, KNNEvaluator
from src.data import ContrastiveTransform


class TestSimCLR:
    """Test SimCLR model."""
    
    def test_simclr_initialization(self):
        """Test SimCLR model initialization."""
        model = SimCLR(
            base_model="resnet18",  # Use smaller model for testing
            projection_dim=64,
            hidden_dim=256,
            temperature=0.5,
        )
        
        assert isinstance(model, SimCLR)
        assert model.projection_dim == 64
        assert model.temperature == 0.5
    
    def test_simclr_forward(self):
        """Test SimCLR forward pass."""
        model = SimCLR(
            base_model="resnet18",
            projection_dim=64,
            temperature=0.5,
        )
        
        # Create dummy input
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)
        
        # Forward pass
        output = model(x)
        
        assert output.shape == (batch_size, 64)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_simclr_contrastive_loss(self):
        """Test SimCLR contrastive loss."""
        model = SimCLR(
            base_model="resnet18",
            projection_dim=64,
            temperature=0.5,
        )
        
        batch_size = 4
        x1 = torch.randn(batch_size, 3, 224, 224)
        x2 = torch.randn(batch_size, 3, 224, 224)
        
        loss = model.contrastive_loss(x1, x2)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestMoCo:
    """Test MoCo model."""
    
    def test_moco_initialization(self):
        """Test MoCo model initialization."""
        model = MoCo(
            base_model="resnet18",
            projection_dim=64,
            temperature=0.07,
            momentum=0.999,
            queue_size=1024,
        )
        
        assert isinstance(model, MoCo)
        assert model.projection_dim == 64
        assert model.temperature == 0.07
        assert model.momentum == 0.999
        assert model.queue_size == 1024
    
    def test_moco_forward(self):
        """Test MoCo forward pass."""
        model = MoCo(
            base_model="resnet18",
            projection_dim=64,
            temperature=0.07,
            momentum=0.999,
            queue_size=1024,
        )
        
        batch_size = 4
        x_q = torch.randn(batch_size, 3, 224, 224)
        x_k = torch.randn(batch_size, 3, 224, 224)
        
        q, k = model(x_q, x_k)
        
        assert q.shape == (batch_size, 64)
        assert k.shape == (batch_size, 64)
        assert not torch.isnan(q).any()
        assert not torch.isnan(k).any()
    
    def test_moco_contrastive_loss(self):
        """Test MoCo contrastive loss."""
        model = MoCo(
            base_model="resnet18",
            projection_dim=64,
            temperature=0.07,
            momentum=0.999,
            queue_size=1024,
        )
        
        batch_size = 4
        x_q = torch.randn(batch_size, 3, 224, 224)
        x_k = torch.randn(batch_size, 3, 224, 224)
        
        q, k = model(x_q, x_k)
        loss = model.contrastive_loss(q, k)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestLosses:
    """Test loss functions."""
    
    def test_ntxent_loss(self):
        """Test NT-Xent loss."""
        loss_fn = NTXentLoss(temperature=0.5)
        
        batch_size = 4
        z_i = torch.randn(batch_size, 64)
        z_j = torch.randn(batch_size, 64)
        
        loss = loss_fn(z_i, z_j)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_infonce_loss(self):
        """Test InfoNCE loss."""
        loss_fn = InfoNCE(temperature=0.07)
        
        batch_size = 4
        query = torch.randn(batch_size, 64)
        positive = torch.randn(batch_size, 64)
        negatives = torch.randn(batch_size, 32, 64)
        
        loss = loss_fn(query, positive, negatives)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_linear_probe_evaluator(self):
        """Test linear probe evaluator."""
        evaluator = LinearProbeEvaluator(epochs=2)  # Quick test
        
        # Create dummy model
        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )
        
        # Create dummy data
        train_data = torch.randn(100, 64)
        train_labels = torch.randint(0, 10, (100,))
        val_data = torch.randn(50, 64)
        val_labels = torch.randint(0, 10, (50,))
        
        train_dataset = TensorDataset(train_data, train_labels)
        val_dataset = TensorDataset(val_data, val_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Test evaluation
        metrics = evaluator.evaluate(model, train_loader, val_loader, torch.device("cpu"))
        
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1
    
    def test_knn_evaluator(self):
        """Test K-NN evaluator."""
        evaluator = KNNEvaluator(k_values=[1, 3])
        
        # Create dummy model
        model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )
        
        # Create dummy data
        train_data = torch.randn(100, 64)
        train_labels = torch.randint(0, 10, (100,))
        val_data = torch.randn(50, 64)
        val_labels = torch.randint(0, 10, (50,))
        
        train_dataset = TensorDataset(train_data, train_labels)
        val_dataset = TensorDataset(val_data, val_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Test evaluation
        metrics = evaluator.evaluate(model, train_loader, val_loader, torch.device("cpu"))
        
        assert isinstance(metrics, dict)
        assert "knn_1_accuracy" in metrics
        assert "knn_3_accuracy" in metrics
        assert 0 <= metrics["knn_1_accuracy"] <= 1
        assert 0 <= metrics["knn_3_accuracy"] <= 1


class TestDataTransforms:
    """Test data transforms."""
    
    def test_contrastive_transform(self):
        """Test contrastive transform."""
        transform = ContrastiveTransform(
            image_size=224,
            normalize=True,
            color_jitter_strength=0.8,
            gaussian_blur_prob=0.5,
        )
        
        # Create dummy image
        x = torch.randn(3, 224, 224)
        
        # Apply transform
        view1, view2 = transform(x)
        
        assert view1.shape == (3, 224, 224)
        assert view2.shape == (3, 224, 224)
        assert not torch.isnan(view1).any()
        assert not torch.isnan(view2).any()


if __name__ == "__main__":
    pytest.main([__file__])
