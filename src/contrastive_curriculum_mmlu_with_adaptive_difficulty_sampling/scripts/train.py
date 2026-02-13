#!/usr/bin/env python
"""Training script for contrastive curriculum MMLU model."""

import argparse
import logging
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

from contrastive_curriculum_mmlu_with_adaptive_difficulty_sampling.data import MMLUDataLoader
from contrastive_curriculum_mmlu_with_adaptive_difficulty_sampling.models import ContrastiveCurriculumModel
from contrastive_curriculum_mmlu_with_adaptive_difficulty_sampling.training import ContrastiveCurriculumTrainer
from contrastive_curriculum_mmlu_with_adaptive_difficulty_sampling.utils import load_config, set_seed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train contrastive curriculum MMLU model")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        help='Disable CUDA even if available'
    )

    return parser.parse_args()


def setup_mlflow(config: dict) -> None:
    """Setup MLflow tracking.

    Args:
        config: Configuration dictionary.
    """
    try:
        import mlflow

        mlflow_config = config.get('mlflow', {})
        if mlflow_config.get('enabled', False):
            mlflow.set_tracking_uri(mlflow_config.get('tracking_uri', 'mlruns'))
            mlflow.set_experiment(mlflow_config.get('experiment_name', 'contrastive-curriculum-mmlu'))
            mlflow.start_run(run_name=mlflow_config.get('run_name', 'default'))

            # Log parameters
            mlflow.log_params({
                'learning_rate': config['training']['learning_rate'],
                'batch_size': config['training']['batch_size'],
                'num_epochs': config['training']['num_epochs'],
                'model': config['model']['base_model'],
            })

            logger.info("MLflow tracking enabled")
    except Exception as e:
        logger.warning(f"MLflow initialization failed: {e}. Continuing without MLflow.")


def log_metrics_to_mlflow(metrics: dict, step: int) -> None:
    """Log metrics to MLflow.

    Args:
        metrics: Metrics dictionary.
        step: Current step.
    """
    try:
        import mlflow
        mlflow.log_metrics(metrics, step=step)
    except Exception:
        pass


def main() -> None:
    """Main training function."""
    args = parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Set random seed
    seed = config.get('seed', 42)
    set_seed(seed)
    logger.info(f"Set random seed to {seed}")

    # Setup device
    if args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Setup MLflow
    setup_mlflow(config)

    try:
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config['model']['base_model'])

        # Load data
        logger.info("Loading MMLU dataset...")
        data_loader = MMLUDataLoader(config, tokenizer)

        # Prepare splits
        train_data, val_data, test_data = data_loader.prepare_splits(
            val_split=config['data']['validation_split'],
            test_split=config['data']['test_split'],
            seed=seed,
        )

        logger.info(f"Dataset splits: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

        # Create dataloaders
        train_loader = data_loader.create_dataloader(
            train_data,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['data'].get('num_workers', 0),
        )

        val_loader = data_loader.create_dataloader(
            val_data,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['data'].get('num_workers', 0),
        )

        # Initialize model
        logger.info("Initializing model...")
        model = ContrastiveCurriculumModel(
            base_model_name=config['model']['base_model'],
            num_classes=len(data_loader.subjects),
            hidden_dim=config['model']['hidden_dim'],
            projection_dim=config['model']['projection_dim'],
            dropout=config['model']['dropout'],
            pooling_mode=config['model']['pooling_mode'],
        )

        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = ContrastiveCurriculumTrainer(
            model=model,
            config=config,
            device=device,
        )

        # Create scheduler
        num_training_steps = len(train_loader) * config['training']['num_epochs']
        trainer.scheduler = trainer._create_scheduler(num_training_steps)

        # Training loop
        logger.info("Starting training...")
        num_epochs = config['training']['num_epochs']
        patience = config['training']['early_stopping_patience']
        best_val_acc = 0.0
        epochs_without_improvement = 0

        checkpoint_dir = Path(config['paths']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, num_epochs + 1):
            logger.info(f"\nEpoch {epoch}/{num_epochs}")

            # Train
            train_metrics = trainer.train_epoch(train_loader, epoch)
            logger.info(f"Train metrics: {train_metrics}")

            # Log to MLflow
            log_metrics_to_mlflow(
                {f'train_{k}': v for k, v in train_metrics.items()},
                step=epoch
            )

            # Evaluate
            val_metrics = trainer.evaluate(val_loader)
            logger.info(f"Validation metrics: {val_metrics}")

            # Log to MLflow
            log_metrics_to_mlflow(
                {f'val_{k}': v for k, v in val_metrics.items()},
                step=epoch
            )

            # Check for improvement
            val_acc = val_metrics['accuracy']
            is_best = val_acc > best_val_acc

            if is_best:
                best_val_acc = val_acc
                epochs_without_improvement = 0
                logger.info(f"New best validation accuracy: {best_val_acc:.4f}")
            else:
                epochs_without_improvement += 1
                logger.info(f"No improvement for {epochs_without_improvement} epoch(s)")

            # Save checkpoint
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            trainer.save_checkpoint(str(checkpoint_path), is_best=is_best)

            # Early stopping
            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break

            trainer.current_epoch = epoch

        # Save final results
        results_dir = Path(config['paths']['results_dir'])
        results_dir.mkdir(parents=True, exist_ok=True)

        import json
        final_results = {
            'best_val_accuracy': best_val_acc,
            'total_epochs': epoch,
            'final_train_metrics': train_metrics,
            'final_val_metrics': val_metrics,
        }

        with open(results_dir / 'training_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)

        logger.info(f"\nTraining completed!")
        logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
        logger.info(f"Model saved to: {checkpoint_dir / 'best_model.pt'}")

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise

    finally:
        # End MLflow run
        try:
            import mlflow
            mlflow.end_run()
        except Exception:
            pass


if __name__ == "__main__":
    main()
