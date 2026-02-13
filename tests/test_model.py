"""Tests for model architecture and components."""

import pytest
import torch

from contrastive_curriculum_mmlu_with_adaptive_difficulty_sampling.models import (
    ContrastiveCurriculumModel,
    ContrastiveLoss,
    SubjectAwareContrastiveLoss,
    CurriculumWeightedLoss,
)
from contrastive_curriculum_mmlu_with_adaptive_difficulty_sampling.models.components import UncertaintyEstimator


def test_model_initialization(sample_config):
    """Test model initialization."""
    model = ContrastiveCurriculumModel(
        base_model_name=sample_config['model']['base_model'],
        num_classes=sample_config['model']['num_classes'],
        hidden_dim=sample_config['model']['hidden_dim'],
        projection_dim=sample_config['model']['projection_dim'],
        dropout=sample_config['model']['dropout'],
    )

    assert model.num_classes == sample_config['model']['num_classes']
    assert model.hidden_dim == sample_config['model']['hidden_dim']
    assert model.projection_dim == sample_config['model']['projection_dim']


def test_model_forward(sample_config, sample_batch):
    """Test model forward pass."""
    import torch
    model = ContrastiveCurriculumModel(
        base_model_name=sample_config['model']['base_model'],
        num_classes=sample_config['model']['num_classes'],
        hidden_dim=sample_config['model']['hidden_dim'],
        projection_dim=sample_config['model']['projection_dim'],
    )
    # Explicitly move model to CPU for consistent testing
    model = model.to('cpu')

    outputs = model(
        input_ids=sample_batch['input_ids'],
        attention_mask=sample_batch['attention_mask'],
        subject_idx=sample_batch['subject_idx'],
        return_embeddings=True,
    )

    assert 'logits' in outputs
    assert 'embeddings' in outputs
    assert 'projections' in outputs

    batch_size = sample_batch['input_ids'].shape[0]
    assert outputs['logits'].shape == (batch_size, 4)
    assert outputs['embeddings'].shape[1] == sample_config['model']['hidden_dim']
    assert outputs['projections'].shape[1] == sample_config['model']['projection_dim']


def test_contrastive_loss(sample_embeddings, sample_labels):
    """Test contrastive loss computation."""
    loss_fn = ContrastiveLoss(temperature=0.07, normalize=True)

    loss = loss_fn(sample_embeddings, sample_labels)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar
    assert loss.item() >= 0


def test_subject_aware_contrastive_loss(sample_embeddings, sample_labels):
    """Test subject-aware contrastive loss."""
    loss_fn = SubjectAwareContrastiveLoss(temperature=0.07, subject_weight=0.5)

    subject_ids = torch.randint(0, 5, (sample_embeddings.shape[0],))
    loss = loss_fn(sample_embeddings, sample_labels, subject_ids)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() >= 0


def test_curriculum_weighted_loss():
    """Test curriculum-weighted loss."""
    base_loss = torch.nn.CrossEntropyLoss()
    curriculum_loss = CurriculumWeightedLoss(
        base_loss=base_loss,
        curriculum_strategy='adaptive',
        initial_threshold=0.3,
    )

    predictions = torch.randn(8, 4)
    targets = torch.randint(0, 4, (8,))
    uncertainties = torch.rand(8)

    loss = curriculum_loss(predictions, targets, uncertainties)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0
    assert loss.item() >= 0


def test_uncertainty_estimator():
    """Test uncertainty estimation."""
    estimator = UncertaintyEstimator(method='entropy')

    logits = torch.randn(8, 4)
    uncertainties = estimator(logits)

    assert uncertainties.shape == (8,)
    assert torch.all(uncertainties >= 0)
    assert torch.all(uncertainties <= 1)

    # Test with peaked distribution (low uncertainty)
    peaked_logits = torch.zeros(8, 4)
    peaked_logits[:, 0] = 10.0
    peaked_uncertainties = estimator(peaked_logits)
    assert torch.all(peaked_uncertainties < 0.2)


def test_model_parameter_count(sample_config):
    """Test parameter counting."""
    model = ContrastiveCurriculumModel(
        base_model_name=sample_config['model']['base_model'],
        num_classes=sample_config['model']['num_classes'],
        hidden_dim=sample_config['model']['hidden_dim'],
        projection_dim=sample_config['model']['projection_dim'],
    )

    param_count = model.count_parameters()
    assert param_count > 0
    assert isinstance(param_count, int)


def test_model_freeze_unfreeze(sample_config):
    """Test encoder freezing/unfreezing."""
    model = ContrastiveCurriculumModel(
        base_model_name=sample_config['model']['base_model'],
        num_classes=sample_config['model']['num_classes'],
        hidden_dim=sample_config['model']['hidden_dim'],
        projection_dim=sample_config['model']['projection_dim'],
    )

    # Initially encoder should be frozen
    encoder_params = list(model.encoder.parameters())
    assert all(not p.requires_grad for p in encoder_params)

    # Unfreeze
    model.unfreeze_encoder()
    encoder_params = list(model.encoder.parameters())
    assert all(p.requires_grad for p in encoder_params)
