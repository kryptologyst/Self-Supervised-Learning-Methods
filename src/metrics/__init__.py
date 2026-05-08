"""Evaluation metrics for self-supervised learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.neighbors import KNeighborsClassifier
from typing import Dict, List, Optional, Tuple, Union


class LinearProbeEvaluator:
    """Linear probe evaluation for self-supervised learning."""
    
    def __init__(
        self,
        epochs: int = 50,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        weight_decay: float = 0.0,
        optimizer: str = "sgd",
        scheduler: str = "cosine",
    ):
        """Initialize linear probe evaluator.
        
        Args:
            epochs: Number of training epochs
            learning_rate: Learning rate for linear probe
            batch_size: Batch size for training
            weight_decay: Weight decay for regularization
            optimizer: Optimizer type ("sgd" or "adam")
            scheduler: Learning rate scheduler
        """
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def evaluate(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        """Evaluate model using linear probe.
        
        Args:
            model: Trained model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to use
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Extract features
        train_features, train_labels = self._extract_features(model, train_loader, device)
        val_features, val_labels = self._extract_features(model, val_loader, device)
        
        # Train linear probe
        probe = self._train_linear_probe(train_features, train_labels, device)
        
        # Evaluate on validation set
        metrics = self._evaluate_probe(probe, val_features, val_labels, device)
        
        return metrics
    
    def _extract_features(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from model.
        
        Args:
            model: Model to extract features from
            data_loader: Data loader
            device: Device to use
            
        Returns:
            Tuple of (features, labels)
        """
        model.eval()
        features_list = []
        labels_list = []
        
        with torch.no_grad():
            for batch in data_loader:
                if len(batch) == 3:  # (view1, view2, label)
                    _, _, labels = batch
                else:  # (data, label)
                    _, labels = batch
                
                # Use first view for feature extraction
                if len(batch) == 3:
                    data = batch[0]
                else:
                    data = batch[0]
                
                data = data.to(device)
                labels = labels.to(device)
                
                # Extract features
                if hasattr(model, 'encode'):
                    features = model.encode(data)
                else:
                    features = model(data)
                
                features_list.append(features.cpu())
                labels_list.append(labels.cpu())
        
        features = torch.cat(features_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        
        return features, labels
    
    def _train_linear_probe(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        device: torch.device,
    ) -> nn.Module:
        """Train linear probe classifier.
        
        Args:
            features: Feature embeddings
            labels: Class labels
            device: Device to use
            
        Returns:
            Trained linear probe
        """
        num_classes = len(torch.unique(labels))
        feature_dim = features.size(1)
        
        # Create linear probe
        probe = nn.Linear(feature_dim, num_classes).to(device)
        
        # Setup optimizer
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                probe.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=0.9,
            )
        else:
            optimizer = torch.optim.Adam(
                probe.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        
        # Setup scheduler
        if self.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs
            )
        else:
            scheduler = None
        
        # Training loop
        probe.train()
        for epoch in range(self.epochs):
            # Create mini-batches
            num_batches = len(features) // self.batch_size
            indices = torch.randperm(len(features))
            
            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(features))
                batch_indices = indices[start_idx:end_idx]
                
                batch_features = features[batch_indices].to(device)
                batch_labels = labels[batch_indices].to(device)
                
                optimizer.zero_grad()
                
                logits = probe(batch_features)
                loss = F.cross_entropy(logits, batch_labels)
                
                loss.backward()
                optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
        
        return probe
    
    def _evaluate_probe(
        self,
        probe: nn.Module,
        features: torch.Tensor,
        labels: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, float]:
        """Evaluate linear probe.
        
        Args:
            probe: Trained linear probe
            features: Feature embeddings
            labels: Class labels
            device: Device to use
            
        Returns:
            Dictionary of evaluation metrics
        """
        probe.eval()
        
        with torch.no_grad():
            features = features.to(device)
            labels = labels.to(device)
            
            logits = probe(features)
            predictions = torch.argmax(logits, dim=1)
            probabilities = F.softmax(logits, dim=1)
        
        # Convert to numpy
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()
        probabilities = probabilities.cpu().numpy()
        
        # Compute metrics
        metrics = {
            "accuracy": accuracy_score(labels, predictions),
            "f1_macro": f1_score(labels, predictions, average="macro"),
            "f1_weighted": f1_score(labels, predictions, average="weighted"),
            "precision_macro": precision_score(labels, predictions, average="macro"),
            "recall_macro": recall_score(labels, predictions, average="macro"),
        }
        
        # Compute AUC if binary classification
        if len(np.unique(labels)) == 2:
            metrics["auroc"] = roc_auc_score(labels, probabilities[:, 1])
            metrics["auprc"] = average_precision_score(labels, probabilities[:, 1])
        else:
            # Multi-class AUC
            try:
                metrics["auroc"] = roc_auc_score(labels, probabilities, multi_class="ovr")
            except ValueError:
                metrics["auroc"] = 0.0
        
        return metrics


class KNNEvaluator:
    """K-NN evaluation for self-supervised learning."""
    
    def __init__(
        self,
        k_values: List[int] = [1, 5, 10, 20],
        distance_metric: str = "cosine",
    ):
        """Initialize K-NN evaluator.
        
        Args:
            k_values: List of k values to evaluate
            distance_metric: Distance metric ("cosine" or "euclidean")
        """
        self.k_values = k_values
        self.distance_metric = distance_metric
    
    def evaluate(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        """Evaluate model using K-NN.
        
        Args:
            model: Trained model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to use
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Extract features
        train_features, train_labels = self._extract_features(model, train_loader, device)
        val_features, val_labels = self._extract_features(model, val_loader, device)
        
        # Convert to numpy
        train_features = train_features.cpu().numpy()
        train_labels = train_labels.cpu().numpy()
        val_features = val_features.cpu().numpy()
        val_labels = val_labels.cpu().numpy()
        
        # Normalize features
        train_features = train_features / np.linalg.norm(train_features, axis=1, keepdims=True)
        val_features = val_features / np.linalg.norm(val_features, axis=1, keepdims=True)
        
        metrics = {}
        
        for k in self.k_values:
            # Train K-NN classifier
            knn = KNeighborsClassifier(
                n_neighbors=k,
                metric=self.distance_metric,
            )
            knn.fit(train_features, train_labels)
            
            # Evaluate
            predictions = knn.predict(val_features)
            accuracy = accuracy_score(val_labels, predictions)
            
            metrics[f"knn_{k}_accuracy"] = accuracy
        
        return metrics
    
    def _extract_features(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from model."""
        model.eval()
        features_list = []
        labels_list = []
        
        with torch.no_grad():
            for batch in data_loader:
                if len(batch) == 3:  # (view1, view2, label)
                    _, _, labels = batch
                else:  # (data, label)
                    _, labels = batch
                
                # Use first view for feature extraction
                if len(batch) == 3:
                    data = batch[0]
                else:
                    data = batch[0]
                
                data = data.to(device)
                labels = labels.to(device)
                
                # Extract features
                if hasattr(model, 'encode'):
                    features = model.encode(data)
                else:
                    features = model(data)
                
                features_list.append(features.cpu())
                labels_list.append(labels.cpu())
        
        features = torch.cat(features_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        
        return features, labels


class EmbeddingEvaluator:
    """General embedding evaluation utilities."""
    
    @staticmethod
    def compute_similarity_matrix(embeddings: torch.Tensor) -> torch.Tensor:
        """Compute similarity matrix between embeddings.
        
        Args:
            embeddings: Feature embeddings
            
        Returns:
            Similarity matrix
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=-1, p=2)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T)
        
        return similarity_matrix
    
    @staticmethod
    def compute_retrieval_metrics(
        similarity_matrix: torch.Tensor,
        labels: torch.Tensor,
        k_values: List[int] = [1, 5, 10, 20],
    ) -> Dict[str, float]:
        """Compute retrieval metrics.
        
        Args:
            similarity_matrix: Similarity matrix
            labels: Class labels
            k_values: List of k values for recall@k
            
        Returns:
            Dictionary of retrieval metrics
        """
        batch_size = similarity_matrix.size(0)
        
        # Remove diagonal (self-similarity)
        mask = torch.eye(batch_size).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        # Get top-k indices
        _, top_k_indices = torch.topk(similarity_matrix, max(k_values), dim=1)
        
        metrics = {}
        
        for k in k_values:
            # Get top-k predictions
            top_k_pred = top_k_indices[:, :k]
            
            # Compute recall@k
            recall_k = 0
            for i in range(batch_size):
                true_label = labels[i]
                pred_labels = labels[top_k_pred[i]]
                recall_k += (pred_labels == true_label).any().float()
            
            recall_k /= batch_size
            metrics[f"recall@{k}"] = recall_k.item()
        
        return metrics
