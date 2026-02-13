#!/usr/bin/env python
"""Prediction script for contrastive curriculum MMLU model."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoTokenizer

from contrastive_curriculum_mmlu_with_adaptive_difficulty_sampling.models import ContrastiveCurriculumModel
from contrastive_curriculum_mmlu_with_adaptive_difficulty_sampling.utils import load_config

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
    parser = argparse.ArgumentParser(description="Make predictions with trained MMLU model")
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
        '--question',
        type=str,
        help='Question text to predict on'
    )
    parser.add_argument(
        '--choices',
        type=str,
        nargs=4,
        help='Four answer choices (A, B, C, D)'
    )
    parser.add_argument(
        '--input-file',
        type=str,
        help='JSON file with questions to predict on'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='predictions.json',
        help='File to save predictions'
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

    # Get number of classes from checkpoint config if available
    if 'config' in checkpoint:
        num_classes = checkpoint['config']['model'].get('num_classes', 57)
    else:
        num_classes = config['model'].get('num_classes', 57)

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


def format_question(question: str, choices: List[str]) -> str:
    """Format question and choices for model input.

    Args:
        question: Question text.
        choices: List of 4 answer choices.

    Returns:
        Formatted text.
    """
    formatted = f"Question: {question}\n"
    for i, choice in enumerate(choices):
        formatted += f"{chr(65+i)}. {choice}\n"
    formatted += "Answer:"
    return formatted


def predict(
    model: ContrastiveCurriculumModel,
    tokenizer: AutoTokenizer,
    question: str,
    choices: List[str],
    device: torch.device,
    max_length: int = 512,
) -> Dict[str, any]:
    """Make prediction for a single question.

    Args:
        model: Trained model.
        tokenizer: Tokenizer.
        question: Question text.
        choices: List of answer choices.
        device: Device to run on.
        max_length: Maximum sequence length.

    Returns:
        Dictionary with prediction results.
    """
    # Format input
    formatted_text = format_question(question, choices)

    # Tokenize
    encoding = tokenizer(
        formatted_text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Predict
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits = outputs['logits']
        probs = torch.softmax(logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)

    # Format output
    answer_idx = prediction.item()
    answer_letter = chr(65 + answer_idx)  # Convert to A, B, C, D

    return {
        'question': question,
        'choices': choices,
        'predicted_answer': answer_letter,
        'predicted_index': answer_idx,
        'confidence': confidence.item(),
        'probabilities': {
            chr(65+i): prob.item()
            for i, prob in enumerate(probs[0])
        }
    }


def main() -> None:
    """Main prediction function."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['base_model'])

    # Load model
    model = load_model(args.checkpoint, config, device)

    predictions = []

    # Process input
    if args.question and args.choices:
        # Single question from command line
        logger.info("Making prediction for single question...")
        result = predict(
            model, tokenizer, args.question, args.choices, device,
            max_length=config['data']['max_seq_length']
        )
        predictions.append(result)

        # Print result
        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        print(f"Question: {result['question']}")
        print("\nChoices:")
        for i, choice in enumerate(result['choices']):
            print(f"  {chr(65+i)}. {choice}")
        print(f"\nPredicted Answer: {result['predicted_answer']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("\nProbabilities:")
        for letter, prob in result['probabilities'].items():
            print(f"  {letter}: {prob:.4f}")
        print("="*60 + "\n")

    elif args.input_file:
        # Batch prediction from file
        logger.info(f"Loading questions from {args.input_file}...")
        with open(args.input_file, 'r') as f:
            questions = json.load(f)

        logger.info(f"Making predictions for {len(questions)} questions...")
        for item in questions:
            result = predict(
                model, tokenizer,
                item['question'],
                item['choices'],
                device,
                max_length=config['data']['max_seq_length']
            )
            predictions.append(result)

        logger.info(f"Completed {len(predictions)} predictions")

    else:
        logger.error("Must provide either --question and --choices, or --input-file")
        sys.exit(1)

    # Save predictions
    if predictions:
        output_file = Path(args.output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(predictions, f, indent=2)

        logger.info(f"Saved predictions to {output_file}")


if __name__ == "__main__":
    main()
