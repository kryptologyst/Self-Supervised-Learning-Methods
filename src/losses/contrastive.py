"""Loss functions for self-supervised learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy (NT-Xent) loss.
    
    This is the contrastive loss used in SimCLR and other contrastive learning methods.
    """
    
    def __init__(self, temperature: float = 0.5, normalize: bool = True):
        """Initialize NT-Xent loss.
        
        Args:
            temperature: Temperature parameter for scaling
            normalize: Whether to normalize embeddings
        """
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
    
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """Compute NT-Xent loss.
        
        Args:
            z_i: First set of embeddings
            z_j: Second set of embeddings
            
        Returns:
            NT-Xent loss
        """
        batch_size = z_i.size(0)
        
        # Normalize embeddings
        if self.normalize:
            z_i = F.normalize(z_i, dim=-1, p=2)
            z_j = F.normalize(z_j, dim=-1, p=2)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(z_i, z_j.T) / self.temperature
        
        # Create labels for positive pairs (diagonal)
        labels = torch.arange(batch_size).long().to(z_i.device)
        
        # Compute loss for both directions
        loss_ij = F.cross_entropy(similarity_matrix, labels)
        loss_ji = F.cross_entropy(similarity_matrix.T, labels)
        
        return (loss_ij + loss_ji) / 2


class InfoNCE(nn.Module):
    """InfoNCE loss for contrastive learning."""
    
    def __init__(self, temperature: float = 0.07, normalize: bool = True):
        """Initialize InfoNCE loss.
        
        Args:
            temperature: Temperature parameter for scaling
            normalize: Whether to normalize embeddings
        """
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
    
    def forward(self, query: torch.Tensor, positive: torch.Tensor, negatives: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute InfoNCE loss.
        
        Args:
            query: Query embeddings
            positive: Positive embeddings
            negatives: Negative embeddings (optional)
            
        Returns:
            InfoNCE loss
        """
        batch_size = query.size(0)
        
        # Normalize embeddings
        if self.normalize:
            query = F.normalize(query, dim=-1, p=2)
            positive = F.normalize(positive, dim=-1, p=2)
            if negatives is not None:
                negatives = F.normalize(negatives, dim=-1, p=2)
        
        # Compute positive similarity
        pos_sim = torch.sum(query * positive, dim=-1) / self.temperature
        
        if negatives is not None:
            # Compute negative similarities
            neg_sim = torch.matmul(query, negatives.T) / self.temperature
            
            # Combine positive and negative similarities
            logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
        else:
            # Use other samples in batch as negatives
            all_sim = torch.matmul(query, torch.cat([positive.unsqueeze(0), query], dim=0).T) / self.temperature
            logits = all_sim
        
        # Create labels (positive is at index 0)
        labels = torch.zeros(batch_size, dtype=torch.long).to(query.device)
        
        return F.cross_entropy(logits, labels)


class TripletLoss(nn.Module):
    """Triplet loss for metric learning."""
    
    def __init__(self, margin: float = 1.0, distance_metric: str = "euclidean"):
        """Initialize triplet loss.
        
        Args:
            margin: Margin for triplet loss
            distance_metric: Distance metric ("euclidean" or "cosine")
        """
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """Compute triplet loss.
        
        Args:
            anchor: Anchor embeddings
            positive: Positive embeddings
            negative: Negative embeddings
            
        Returns:
            Triplet loss
        """
        if self.distance_metric == "euclidean":
            pos_dist = F.pairwise_distance(anchor, positive, p=2)
            neg_dist = F.pairwise_distance(anchor, negative, p=2)
        elif self.distance_metric == "cosine":
            pos_dist = 1 - F.cosine_similarity(anchor, positive)
            neg_dist = 1 - F.cosine_similarity(anchor, negative)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss."""
    
    def __init__(self, temperature: float = 0.07, normalize: bool = True):
        """Initialize supervised contrastive loss.
        
        Args:
            temperature: Temperature parameter for scaling
            normalize: Whether to normalize embeddings
        """
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute supervised contrastive loss.
        
        Args:
            features: Feature embeddings
            labels: Class labels
            
        Returns:
            Supervised contrastive loss
        """
        batch_size = features.size(0)
        
        # Normalize embeddings
        if self.normalize:
            features = F.normalize(features, dim=-1, p=2)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs (same class)
        mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()
        
        # Remove diagonal (self-similarity)
        mask = mask - torch.eye(batch_size).to(features.device)
        
        # Compute log probabilities
        log_prob = F.log_softmax(similarity_matrix, dim=-1)
        
        # Compute loss for each sample
        loss = -torch.sum(mask * log_prob, dim=-1) / torch.sum(mask, dim=-1)
        
        return loss.mean()


class BarlowTwinsLoss(nn.Module):
    """Barlow Twins loss for self-supervised learning."""
    
    def __init__(self, lambda_param: float = 0.005):
        """Initialize Barlow Twins loss.
        
        Args:
            lambda_param: Lambda parameter for off-diagonal terms
        """
        super().__init__()
        self.lambda_param = lambda_param
    
    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """Compute Barlow Twins loss.
        
        Args:
            z_a: First set of embeddings
            z_b: Second set of embeddings
            
        Returns:
            Barlow Twins loss
        """
        # Normalize embeddings
        z_a = F.normalize(z_a, dim=-1, p=2)
        z_b = F.normalize(z_b, dim=-1, p=2)
        
        # Compute cross-correlation matrix
        batch_size = z_a.size(0)
        cross_corr = torch.matmul(z_a.T, z_b) / batch_size
        
        # Compute loss
        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = cross_corr.flatten()[1:].view(cross_corr.size(0) - 1, cross_corr.size(0) + 1)[:, :-1].pow_(2).sum()
        
        loss = on_diag + self.lambda_param * off_diag
        
        return loss


class SimSiamLoss(nn.Module):
    """SimSiam loss for self-supervised learning."""
    
    def __init__(self, normalize: bool = True):
        """Initialize SimSiam loss.
        
        Args:
            normalize: Whether to normalize embeddings
        """
        super().__init__()
        self.normalize = normalize
    
    def forward(self, p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Compute SimSiam loss.
        
        Args:
            p: Predictions from one view
            z: Target embeddings from another view
            
        Returns:
            SimSiam loss
        """
        if self.normalize:
            p = F.normalize(p, dim=-1, p=2)
            z = F.normalize(z, dim=-1, p=2)
        
        # Compute negative cosine similarity
        loss = -(p * z).sum(dim=-1).mean()
        
        return loss
