"""Training loop implementation with curriculum learning and early stopping."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.components import (
    CurriculumWeightedLoss,
    SubjectAwareContrastiveLoss,
    UncertaintyEstimator,
)
from ..models.model import ContrastiveCurriculumModel

logger = logging.getLogger(__name__)


class ContrastiveCurriculumTrainer:
    """Trainer for contrastive curriculum learning on MMLU."""

    def __init__(
        self,
        model: ContrastiveCurriculumModel,
        config: Dict[str, Any],
        device: torch.device,
    ):
        """Initialize trainer.

        Args:
            model: The model to train.
            config: Configuration dictionary.
            device: Device to train on.
        """
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Training config
        self.train_config = config['training']
        self.loss_config = config['loss']
        self.curriculum_config = config.get('curriculum', {})
        self.contrastive_config = config.get('contrastive', {})

        # Initialize optimizer
        self.optimizer = self._create_optimizer()

        # Initialize scheduler
        self.scheduler = None

        # Initialize loss functions
        self._setup_loss_functions()

        # Mixed precision training
        self.use_amp = self.train_config.get('mixed_precision', False)
        self.scaler = GradScaler('cuda') if self.use_amp and torch.cuda.is_available() else None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_accuracy = 0.0
        self.patience_counter = 0

        # Uncertainty estimator for curriculum
        self.uncertainty_estimator = UncertaintyEstimator(
            method=self.curriculum_config.get('uncertainty_metric', 'entropy')
        )

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer.

        Returns:
            Optimizer instance.
        """
        opt_config = self.config.get('optimizer', {})
        opt_type = opt_config.get('type', 'adamw').lower()

        if opt_type == 'adamw':
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.train_config['learning_rate'],
                betas=opt_config.get('betas', [0.9, 0.999]),
                eps=opt_config.get('eps', 1e-8),
                weight_decay=self.train_config.get('weight_decay', 0.01),
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")

        logger.info(f"Created optimizer: {opt_type}")
        return optimizer

    def _create_scheduler(self, num_training_steps: int) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler.

        Args:
            num_training_steps: Total number of training steps.

        Returns:
            Learning rate scheduler.
        """
        sched_config = self.config.get('scheduler', {})
        sched_type = sched_config.get('type', 'cosine').lower()
        warmup_steps = sched_config.get('num_warmup_steps', 500)

        if sched_type == 'cosine':
            # Linear warmup followed by cosine annealing
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps - warmup_steps,
                eta_min=1e-6,
            )
            scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )
        else:
            raise ValueError(f"Unknown scheduler type: {sched_type}")

        logger.info(f"Created scheduler: {sched_type} with {warmup_steps} warmup steps")
        return scheduler

    def _setup_loss_functions(self) -> None:
        """Setup loss functions."""
        # Classification loss
        label_smoothing = self.loss_config.get('label_smoothing', 0.0)
        self.classification_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Contrastive loss
        if self.contrastive_config.get('subject_aware_sampling', False):
            self.contrastive_loss = SubjectAwareContrastiveLoss(
                temperature=self.contrastive_config.get('temperature', 0.07),
                subject_weight=0.5,
                normalize=self.contrastive_config.get('embedding_normalize', True),
            )
        else:
            from ..models.components import ContrastiveLoss
            self.contrastive_loss = ContrastiveLoss(
                temperature=self.contrastive_config.get('temperature', 0.07),
                normalize=self.contrastive_config.get('embedding_normalize', True),
            )

        # Curriculum-weighted loss
        if self.curriculum_config.get('enabled', False):
            self.curriculum_loss = CurriculumWeightedLoss(
                base_loss=self.classification_loss,
                curriculum_strategy=self.curriculum_config.get('strategy', 'adaptive'),
                initial_threshold=self.curriculum_config.get('initial_difficulty_threshold', 0.3),
            )
        else:
            self.curriculum_loss = None

        logger.info("Loss functions initialized")

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader.
            epoch: Current epoch number.

        Returns:
            Dictionary with training metrics.
        """
        self.model.train()

        total_loss = 0.0
        total_cls_loss = 0.0
        total_cont_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            subject_idx = batch['subject_idx'].to(self.device)

            # Forward pass with mixed precision
            with autocast(device_type='cuda', enabled=self.use_amp and torch.cuda.is_available()):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    subject_idx=subject_idx,
                    return_embeddings=True,
                )

                logits = outputs['logits']
                embeddings = outputs['embeddings']
                projections = outputs['projections']

                # Compute losses
                # 1. Classification loss with curriculum weighting
                if self.curriculum_loss is not None and self.current_epoch >= self.curriculum_config.get('warmup_epochs', 0):
                    uncertainties = self.uncertainty_estimator(logits)
                    cls_loss = self.curriculum_loss(logits, labels, uncertainties)
                else:
                    cls_loss = self.classification_loss(logits, labels)

                # 2. Contrastive loss
                if isinstance(self.contrastive_loss, SubjectAwareContrastiveLoss):
                    cont_loss = self.contrastive_loss(projections, labels, subject_idx)
                else:
                    cont_loss = self.contrastive_loss(projections, labels)

                # Combined loss
                loss = (
                    self.loss_config['classification_weight'] * cls_loss +
                    self.loss_config['contrastive_weight'] * cont_loss
                )

            # Backward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_config['gradient_clip_norm']
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_config['gradient_clip_norm']
                )
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            # Update metrics
            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_cont_loss += cont_loss.item()

            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            self.global_step += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': correct / total,
            })

        metrics = {
            'loss': total_loss / len(train_loader),
            'cls_loss': total_cls_loss / len(train_loader),
            'cont_loss': total_cont_loss / len(train_loader),
            'accuracy': correct / total,
        }

        return metrics

    def evaluate(
        self,
        eval_loader: DataLoader,
    ) -> Dict[str, float]:
        """Evaluate the model.

        Args:
            eval_loader: Evaluation data loader.

        Returns:
            Dictionary with evaluation metrics.
        """
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                subject_idx = batch['subject_idx'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    subject_idx=subject_idx,
                )

                logits = outputs['logits']
                loss = self.classification_loss(logits, labels)

                total_loss += loss.item()

                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        metrics = {
            'loss': total_loss / len(eval_loader),
            'accuracy': correct / total,
        }

        return metrics

    def save_checkpoint(
        self,
        save_path: str,
        is_best: bool = False,
    ) -> None:
        """Save model checkpoint.

        Args:
            save_path: Path to save checkpoint.
            is_best: Whether this is the best model so far.
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'config': self.config,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, save_path)

        if is_best:
            best_path = str(Path(save_path).parent / 'best_model.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")

        logger.info(f"Saved checkpoint to {save_path}")


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """Train for one epoch (simplified version).

    Args:
        model: Model to train.
        train_loader: Training data loader.
        optimizer: Optimizer.
        device: Device to train on.
        epoch: Current epoch number.

    Returns:
        Average training loss.
    """
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs['logits']

        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(
    model: nn.Module,
    eval_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model.

    Args:
        model: Model to evaluate.
        eval_loader: Evaluation data loader.
        device: Device to evaluate on.

    Returns:
        Tuple of (loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']

            loss = nn.functional.cross_entropy(logits, labels)
            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(eval_loader), correct / total
