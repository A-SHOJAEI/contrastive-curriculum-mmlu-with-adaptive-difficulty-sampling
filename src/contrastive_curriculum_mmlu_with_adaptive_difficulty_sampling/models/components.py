"""Custom loss functions and model components."""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ContrastiveLoss(nn.Module):
    """Contrastive loss for learning discriminative representations."""

    def __init__(
        self,
        temperature: float = 0.07,
        normalize: bool = True,
    ):
        """Initialize contrastive loss.

        Args:
            temperature: Temperature parameter for scaling similarities.
            normalize: Whether to L2-normalize embeddings before computing similarity.
        """
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contrastive loss.

        Args:
            embeddings: Embedding vectors of shape (batch_size, embedding_dim).
            labels: Class labels of shape (batch_size,).

        Returns:
            Scalar contrastive loss.
        """
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create positive mask (same class)
        labels = labels.unsqueeze(1)
        positive_mask = (labels == labels.T).float()

        # Remove self-similarity
        batch_size = embeddings.shape[0]
        diagonal_mask = torch.eye(batch_size, device=embeddings.device)
        positive_mask = positive_mask * (1 - diagonal_mask)

        # Create negative mask
        negative_mask = 1 - positive_mask - diagonal_mask

        # Compute loss using InfoNCE formulation
        exp_sim = torch.exp(similarity_matrix)

        # For each anchor, sum over positives and negatives
        pos_sum = (exp_sim * positive_mask).sum(dim=1)
        neg_sum = (exp_sim * negative_mask).sum(dim=1)

        # Avoid division by zero
        loss = -torch.log((pos_sum + 1e-8) / (pos_sum + neg_sum + 1e-8))

        # Average over batch
        return loss.mean()


class SubjectAwareContrastiveLoss(nn.Module):
    """Contrastive loss with subject-domain awareness.

    This is a novel component that enforces clustering within subject domains
    while maintaining separation across different subjects.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        subject_weight: float = 0.5,
        normalize: bool = True,
    ):
        """Initialize subject-aware contrastive loss.

        Args:
            temperature: Temperature parameter for scaling.
            subject_weight: Weight for subject-level clustering (vs class-level).
            normalize: Whether to L2-normalize embeddings.
        """
        super().__init__()
        self.temperature = temperature
        self.subject_weight = subject_weight
        self.normalize = normalize

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        subject_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute subject-aware contrastive loss.

        Args:
            embeddings: Embedding vectors of shape (batch_size, embedding_dim).
            labels: Class labels of shape (batch_size,).
            subject_ids: Subject domain IDs of shape (batch_size,).

        Returns:
            Scalar subject-aware contrastive loss.
        """
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        batch_size = embeddings.shape[0]
        diagonal_mask = torch.eye(batch_size, device=embeddings.device)

        # Create class-level positive mask
        labels = labels.unsqueeze(1)
        class_positive_mask = (labels == labels.T).float() * (1 - diagonal_mask)

        # Create subject-level positive mask
        subject_ids = subject_ids.unsqueeze(1)
        subject_positive_mask = (subject_ids == subject_ids.T).float() * (1 - diagonal_mask)

        # Combine masks with weighting
        positive_mask = (
            (1 - self.subject_weight) * class_positive_mask +
            self.subject_weight * subject_positive_mask
        )

        # Negative mask
        negative_mask = 1 - positive_mask - diagonal_mask

        # InfoNCE loss
        exp_sim = torch.exp(similarity_matrix)
        pos_sum = (exp_sim * positive_mask).sum(dim=1)
        neg_sum = (exp_sim * negative_mask).sum(dim=1)

        loss = -torch.log((pos_sum + 1e-8) / (pos_sum + neg_sum + 1e-8))

        return loss.mean()


class CurriculumWeightedLoss(nn.Module):
    """Weighted loss that adapts based on curriculum difficulty.

    This novel component dynamically adjusts sample weights based on
    model uncertainty gradients for adaptive difficulty sampling.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        curriculum_strategy: str = "adaptive",
        initial_threshold: float = 0.3,
    ):
        """Initialize curriculum-weighted loss.

        Args:
            base_loss: Base loss function to wrap.
            curriculum_strategy: Strategy for weighting ('adaptive', 'easy_first', 'hard_first').
            initial_threshold: Initial difficulty threshold.
        """
        super().__init__()
        self.base_loss = base_loss
        self.curriculum_strategy = curriculum_strategy
        self.difficulty_threshold = initial_threshold
        self.step_count = 0

    def update_threshold(self, new_threshold: float) -> None:
        """Update difficulty threshold.

        Args:
            new_threshold: New threshold value.
        """
        self.difficulty_threshold = new_threshold
        logger.debug(f"Updated difficulty threshold to {new_threshold:.4f}")

    def compute_sample_weights(
        self,
        uncertainties: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sample weights based on uncertainties.

        Args:
            uncertainties: Uncertainty scores of shape (batch_size,).

        Returns:
            Sample weights of shape (batch_size,).
        """
        if self.curriculum_strategy == "adaptive":
            # Weight samples near the difficulty threshold more
            weights = 1.0 - torch.abs(uncertainties - self.difficulty_threshold)
            weights = torch.clamp(weights, min=0.1)
        elif self.curriculum_strategy == "easy_first":
            # Higher weight for lower uncertainty (easier samples)
            weights = 1.0 - uncertainties
        elif self.curriculum_strategy == "hard_first":
            # Higher weight for higher uncertainty (harder samples)
            weights = uncertainties
        else:
            weights = torch.ones_like(uncertainties)

        # Normalize
        weights = weights / (weights.sum() + 1e-8) * len(weights)

        return weights

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainties: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute curriculum-weighted loss.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            uncertainties: Optional uncertainty scores. If None, uniform weighting is used.

        Returns:
            Weighted loss value.
        """
        # Compute base loss (unreduced)
        if isinstance(self.base_loss, nn.CrossEntropyLoss):
            base_loss_unreduced = F.cross_entropy(predictions, targets, reduction='none')
        else:
            base_loss_unreduced = self.base_loss(predictions, targets)

        # Apply curriculum weighting
        if uncertainties is not None and self.curriculum_strategy != "none":
            weights = self.compute_sample_weights(uncertainties)
            weighted_loss = (base_loss_unreduced * weights).mean()
        else:
            weighted_loss = base_loss_unreduced.mean()

        self.step_count += 1

        return weighted_loss


class UncertaintyEstimator(nn.Module):
    """Estimate model uncertainty from predictions."""

    def __init__(self, method: str = "entropy"):
        """Initialize uncertainty estimator.

        Args:
            method: Method for uncertainty estimation ('entropy', 'margin', 'variance').
        """
        super().__init__()
        self.method = method

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute uncertainty from logits.

        Args:
            logits: Model logits of shape (batch_size, num_classes).

        Returns:
            Uncertainty scores of shape (batch_size,).
        """
        if self.method == "entropy":
            # Entropy of predicted distribution
            probs = F.softmax(logits, dim=1)
            log_probs = F.log_softmax(logits, dim=1)
            entropy = -(probs * log_probs).sum(dim=1)
            # Normalize by max entropy
            max_entropy = torch.log(torch.tensor(logits.shape[1], dtype=torch.float32))
            uncertainty = entropy / max_entropy
        elif self.method == "margin":
            # Margin between top two predictions
            probs = F.softmax(logits, dim=1)
            top2 = torch.topk(probs, k=2, dim=1).values
            uncertainty = 1.0 - (top2[:, 0] - top2[:, 1])
        else:  # variance
            probs = F.softmax(logits, dim=1)
            uncertainty = probs.var(dim=1)

        return uncertainty
