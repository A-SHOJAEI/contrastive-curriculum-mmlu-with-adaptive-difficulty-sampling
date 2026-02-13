"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

# Add src/ to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import pytest
import torch
import numpy as np
from typing import Dict, Any


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Sample configuration for testing.

    Returns:
        Configuration dictionary.
    """
    return {
        'model': {
            'base_model': 'sentence-transformers/all-MiniLM-L6-v2',
            'hidden_dim': 384,
            'projection_dim': 128,
            'num_classes': 10,
            'dropout': 0.1,
            'pooling_mode': 'mean',
        },
        'training': {
            'batch_size': 4,
            'num_epochs': 2,
            'learning_rate': 0.0001,
            'weight_decay': 0.01,
            'gradient_clip_norm': 1.0,
            'early_stopping_patience': 3,
            'mixed_precision': False,
        },
        'optimizer': {
            'type': 'adamw',
            'betas': [0.9, 0.999],
            'eps': 0.00000001,
        },
        'scheduler': {
            'type': 'cosine',
            'num_warmup_steps': 10,
            'num_training_steps': 100,
        },
        'contrastive': {
            'temperature': 0.07,
            'use_hard_negatives': False,
            'subject_aware_sampling': True,
            'embedding_normalize': True,
        },
        'curriculum': {
            'enabled': True,
            'strategy': 'adaptive',
            'initial_difficulty_threshold': 0.3,
            'warmup_epochs': 1,
        },
        'data': {
            'dataset_name': 'cais/mmlu',
            'max_seq_length': 128,
            'few_shot_k': 3,
            'validation_split': 0.2,
            'test_split': 0.2,
            'seed': 42,
            'num_workers': 0,
        },
        'loss': {
            'contrastive_weight': 0.5,
            'classification_weight': 0.5,
            'curriculum_weight': 0.0,
            'label_smoothing': 0.0,
        },
        'paths': {
            'checkpoint_dir': 'test_checkpoints',
            'results_dir': 'test_results',
            'cache_dir': '.test_cache',
        },
        'seed': 42,
    }


@pytest.fixture
def sample_batch() -> Dict[str, torch.Tensor]:
    """Sample batch for testing.

    Returns:
        Dictionary with sample batch tensors.
    """
    batch_size = 4
    seq_length = 32

    return {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_length)),
        'attention_mask': torch.ones(batch_size, seq_length),
        'label': torch.randint(0, 4, (batch_size,)),
        'subject_idx': torch.randint(0, 10, (batch_size,)),
    }


@pytest.fixture
def sample_embeddings() -> torch.Tensor:
    """Sample embeddings for testing.

    Returns:
        Tensor of shape (batch_size, embedding_dim).
    """
    return torch.randn(8, 128)


@pytest.fixture
def sample_labels() -> torch.Tensor:
    """Sample labels for testing.

    Returns:
        Tensor of shape (batch_size,).
    """
    return torch.randint(0, 4, (8,))


@pytest.fixture
def device() -> torch.device:
    """Get device for testing.

    Returns:
        CPU device for testing.
    """
    return torch.device('cpu')
