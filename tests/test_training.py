"""Tests for training utilities."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from contrastive_curriculum_mmlu_with_adaptive_difficulty_sampling.models import ContrastiveCurriculumModel
from contrastive_curriculum_mmlu_with_adaptive_difficulty_sampling.training import (
    ContrastiveCurriculumTrainer,
    train_epoch,
    evaluate,
)


@pytest.fixture
def simple_dataloader():
    """Create a simple dataloader for testing.

    Returns:
        DataLoader with dummy data.
    """
    batch_size = 4
    num_samples = 16
    seq_length = 32

    input_ids = torch.randint(0, 1000, (num_samples, seq_length))
    attention_mask = torch.ones(num_samples, seq_length)
    labels = torch.randint(0, 4, (num_samples,))
    subject_idx = torch.randint(0, 10, (num_samples,))

    # Create custom dataset
    class SimpleDataset(torch.utils.data.Dataset):
        def __len__(self):
            return num_samples

        def __getitem__(self, idx):
            return {
                'input_ids': input_ids[idx],
                'attention_mask': attention_mask[idx],
                'label': labels[idx],
                'subject_idx': subject_idx[idx],
            }

    dataset = SimpleDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def test_trainer_initialization(sample_config, device):
    """Test trainer initialization."""
    model = ContrastiveCurriculumModel(
        base_model_name=sample_config['model']['base_model'],
        num_classes=sample_config['model']['num_classes'],
        hidden_dim=sample_config['model']['hidden_dim'],
        projection_dim=sample_config['model']['projection_dim'],
    )

    trainer = ContrastiveCurriculumTrainer(
        model=model,
        config=sample_config,
        device=device,
    )

    assert trainer.model is not None
    assert trainer.optimizer is not None
    assert trainer.current_epoch == 0
    assert trainer.global_step == 0


def test_trainer_train_epoch(sample_config, simple_dataloader, device):
    """Test training for one epoch."""
    model = ContrastiveCurriculumModel(
        base_model_name=sample_config['model']['base_model'],
        num_classes=sample_config['model']['num_classes'],
        hidden_dim=sample_config['model']['hidden_dim'],
        projection_dim=sample_config['model']['projection_dim'],
    )

    trainer = ContrastiveCurriculumTrainer(
        model=model,
        config=sample_config,
        device=device,
    )

    metrics = trainer.train_epoch(simple_dataloader, epoch=1)

    assert 'loss' in metrics
    assert 'accuracy' in metrics
    assert isinstance(metrics['loss'], float)
    assert 0.0 <= metrics['accuracy'] <= 1.0


def test_trainer_evaluate(sample_config, simple_dataloader, device):
    """Test evaluation."""
    model = ContrastiveCurriculumModel(
        base_model_name=sample_config['model']['base_model'],
        num_classes=sample_config['model']['num_classes'],
        hidden_dim=sample_config['model']['hidden_dim'],
        projection_dim=sample_config['model']['projection_dim'],
    )

    trainer = ContrastiveCurriculumTrainer(
        model=model,
        config=sample_config,
        device=device,
    )

    metrics = trainer.evaluate(simple_dataloader)

    assert 'loss' in metrics
    assert 'accuracy' in metrics
    assert isinstance(metrics['loss'], float)
    assert 0.0 <= metrics['accuracy'] <= 1.0


def test_train_epoch_function(sample_config, simple_dataloader, device):
    """Test standalone train_epoch function."""
    model = ContrastiveCurriculumModel(
        base_model_name=sample_config['model']['base_model'],
        num_classes=sample_config['model']['num_classes'],
        hidden_dim=sample_config['model']['hidden_dim'],
        projection_dim=sample_config['model']['projection_dim'],
    )
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    avg_loss = train_epoch(model, simple_dataloader, optimizer, device, epoch=1)

    assert isinstance(avg_loss, float)
    assert avg_loss >= 0


def test_evaluate_function(sample_config, simple_dataloader, device):
    """Test standalone evaluate function."""
    model = ContrastiveCurriculumModel(
        base_model_name=sample_config['model']['base_model'],
        num_classes=sample_config['model']['num_classes'],
        hidden_dim=sample_config['model']['hidden_dim'],
        projection_dim=sample_config['model']['projection_dim'],
    )
    model = model.to(device)

    loss, accuracy = evaluate(model, simple_dataloader, device)

    assert isinstance(loss, float)
    assert isinstance(accuracy, float)
    assert loss >= 0
    assert 0.0 <= accuracy <= 1.0


def test_checkpoint_saving(sample_config, device, tmp_path):
    """Test checkpoint saving."""
    model = ContrastiveCurriculumModel(
        base_model_name=sample_config['model']['base_model'],
        num_classes=sample_config['model']['num_classes'],
        hidden_dim=sample_config['model']['hidden_dim'],
        projection_dim=sample_config['model']['projection_dim'],
    )

    trainer = ContrastiveCurriculumTrainer(
        model=model,
        config=sample_config,
        device=device,
    )

    checkpoint_path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(str(checkpoint_path), is_best=True)

    assert checkpoint_path.exists()

    # Check that best model was also saved
    best_path = tmp_path / "best_model.pt"
    assert best_path.exists()

    # Load and verify
    checkpoint = torch.load(checkpoint_path, map_location=device)
    assert 'model_state_dict' in checkpoint
    assert 'optimizer_state_dict' in checkpoint
    assert 'config' in checkpoint
