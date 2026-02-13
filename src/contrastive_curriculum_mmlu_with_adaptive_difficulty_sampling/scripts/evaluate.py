#!/usr/bin/env python
"""Evaluation script for contrastive curriculum MMLU model."""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from contrastive_curriculum_mmlu_with_adaptive_difficulty_sampling.data import MMLUDataLoader
from contrastive_curriculum_mmlu_with_adaptive_difficulty_sampling.evaluation import (
    analyze_results,
    print_summary,
    save_results,
)
from contrastive_curriculum_mmlu_with_adaptive_difficulty_sampling.models import ContrastiveCurriculumModel
from contrastive_curriculum_mmlu_with_adaptive_difficulty_sampling.utils import load_config, set_seed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate contrastive curriculum MMLU model")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['val', 'test'],
        help='Which split to evaluate on'
    )

    return parser.parse_args()


def load_model(
    checkpoint_path: str,
    config: dict,
    device: torch.device,
) -> ContrastiveCurriculumModel:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        config: Configuration dictionary.
        device: Device to load model on.

    Returns:
        Loaded model.
    """
    logger.info(f"Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get number of classes from checkpoint state dict (most reliable)
    if 'subject_embeddings.weight' in checkpoint['model_state_dict']:
        num_classes = checkpoint['model_state_dict']['subject_embeddings.weight'].shape[0]
    elif 'config' in checkpoint:
        num_classes = checkpoint['config']['model'].get('num_classes', 57)
    else:
        num_classes = config['model'].get('num_classes', 57)

    logger.info(f"Loading model with {num_classes} classes")

    model = ContrastiveCurriculumModel(
        base_model_name=config['model']['base_model'],
        num_classes=num_classes,
        hidden_dim=config['model']['hidden_dim'],
        projection_dim=config['model']['projection_dim'],
        dropout=config['model']['dropout'],
        pooling_mode=config['model']['pooling_mode'],
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    logger.info("Model loaded successfully")
    return model


def evaluate_model(
    model: ContrastiveCurriculumModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate model on a dataset.

    Args:
        model: Model to evaluate.
        dataloader: Data loader.
        device: Device to run evaluation on.

    Returns:
        Dictionary with predictions, labels, and other outputs.
    """
    all_predictions = []
    all_labels = []
    all_confidences = []
    all_subject_indices = []
    all_logits = []

    logger.info("Running evaluation...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            subject_idx = batch['subject_idx'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                subject_idx=subject_idx,
            )

            logits = outputs['logits']
            probs = torch.softmax(logits, dim=1)
            confidences, predictions = torch.max(probs, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            all_subject_indices.extend(subject_idx.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())

    return {
        'predictions': np.array(all_predictions),
        'labels': np.array(all_labels),
        'confidences': np.array(all_confidences),
        'subject_indices': np.array(all_subject_indices),
        'logits': np.array(all_logits),
    }


def main() -> None:
    """Main evaluation function."""
    args = parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Set random seed
    seed = config.get('seed', 42)
    set_seed(seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

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

    # Select evaluation split
    if args.split == 'val':
        eval_data = val_data
    else:
        eval_data = test_data

    logger.info(f"Evaluating on {args.split} split with {len(eval_data)} examples")

    # Create dataloader
    eval_loader = data_loader.create_dataloader(
        eval_data,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
    )

    # Load model
    model = load_model(args.checkpoint, config, device)

    # Evaluate
    outputs = evaluate_model(model, eval_loader, device)

    # Analyze results
    logger.info("Analyzing results...")
    results = analyze_results(
        predictions=outputs['predictions'],
        labels=outputs['labels'],
        subjects=data_loader.subjects,
        subject_indices=outputs['subject_indices'],
        confidences=outputs['confidences'],
        output_dir=args.output_dir,
    )

    # Print summary
    print_summary(results)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    save_results(results, str(output_dir / 'evaluation_results.json'), format='json')

    # Save as CSV
    save_results(results, str(output_dir / 'evaluation_results.csv'), format='csv')

    # Save detailed outputs
    detailed_output = {
        'predictions': outputs['predictions'].tolist(),
        'labels': outputs['labels'].tolist(),
        'confidences': outputs['confidences'].tolist(),
        'subject_indices': outputs['subject_indices'].tolist(),
    }

    with open(output_dir / 'detailed_predictions.json', 'w') as f:
        json.dump(detailed_output, f, indent=2)

    logger.info(f"Evaluation complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
