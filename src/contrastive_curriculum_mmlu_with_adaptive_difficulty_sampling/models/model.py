"""Core model implementation for contrastive curriculum learning."""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class ContrastiveCurriculumModel(nn.Module):
    """Model combining contrastive learning with curriculum strategies for MMLU.

    This model uses a pre-trained sentence transformer as the encoder and adds:
    1. A projection head for contrastive learning
    2. A classification head for MMLU tasks
    3. Support for subject-aware embeddings
    """

    def __init__(
        self,
        base_model_name: str,
        num_classes: int,
        hidden_dim: int = 384,
        projection_dim: int = 256,
        dropout: float = 0.1,
        pooling_mode: str = "mean",
    ):
        """Initialize the model.

        Args:
            base_model_name: Name of the pre-trained sentence transformer.
            num_classes: Number of output classes (MMLU subjects).
            hidden_dim: Hidden dimension of the base model.
            projection_dim: Dimension of projection space for contrastive learning.
            dropout: Dropout probability.
            pooling_mode: Pooling strategy ('mean', 'max', 'cls').
        """
        super().__init__()

        self.base_model_name = base_model_name
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.pooling_mode = pooling_mode

        # Load pre-trained encoder
        logger.info(f"Loading base model: {base_model_name}")
        self.encoder = SentenceTransformer(base_model_name)

        # Freeze encoder initially (can be unfrozen during training)
        self._freeze_encoder()

        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
        )

        # Classification head for MMLU
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 4),  # 4 choices (A, B, C, D)
        )

        # Subject embedding for subject-aware learning
        self.subject_embeddings = nn.Embedding(num_classes, hidden_dim)

        logger.info(f"Model initialized with {self.count_parameters():,} parameters")

    def _freeze_encoder(self) -> None:
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder frozen")

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder for fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        logger.info("Encoder unfrozen")

    def count_parameters(self) -> int:
        """Count trainable parameters.

        Returns:
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode input text to embeddings.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).

        Returns:
            Encoded embeddings of shape (batch_size, hidden_dim).
        """
        # Get the device from the model's parameters
        device = next(self.encoder.parameters()).device

        # Ensure input tensors are on the correct device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Use the encoder's forward method
        features = self.encoder._first_module().forward({
            'input_ids': input_ids,
            'attention_mask': attention_mask
        })

        # Apply pooling
        if self.pooling_mode == "mean":
            # Mean pooling with attention mask
            token_embeddings = features['token_embeddings']
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
        elif self.pooling_mode == "max":
            # Max pooling
            embeddings = torch.max(features['token_embeddings'], dim=1)[0]
        else:  # cls
            # Use [CLS] token
            embeddings = features['token_embeddings'][:, 0]

        return embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        subject_idx: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).
            subject_idx: Subject indices of shape (batch_size,).
            return_embeddings: Whether to return intermediate embeddings.

        Returns:
            Dictionary containing:
                - logits: Classification logits (batch_size, 4)
                - embeddings: Base embeddings (batch_size, hidden_dim) if return_embeddings=True
                - projections: Projected embeddings (batch_size, projection_dim) if return_embeddings=True
        """
        # Encode input
        embeddings = self.encode(input_ids, attention_mask)

        # Add subject-aware information if provided
        if subject_idx is not None:
            # Ensure subject_idx is on the same device as embeddings
            subject_idx = subject_idx.to(embeddings.device)
            subject_emb = self.subject_embeddings(subject_idx)
            # Combine with residual connection
            embeddings = embeddings + 0.1 * subject_emb

        # Get projections for contrastive learning
        projections = self.projection_head(embeddings)

        # Get classification logits
        logits = self.classification_head(embeddings)

        output = {'logits': logits}

        if return_embeddings:
            output['embeddings'] = embeddings
            output['projections'] = projections

        return output

    def get_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get embeddings for input text (for inference).

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.

        Returns:
            Embeddings tensor.
        """
        with torch.no_grad():
            embeddings = self.encode(input_ids, attention_mask)
        return embeddings
