"""Results analysis and visualization utilities."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
) -> None:
    """Plot confusion matrix.

    Args:
        predictions: Predicted labels.
        labels: Ground truth labels.
        class_names: Names of classes for axis labels.
        save_path: Path to save the figure.
        figsize: Figure size.
    """
    cm = confusion_matrix(labels, predictions)

    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names or range(cm.shape[1]),
        yticklabels=class_names or range(cm.shape[0]),
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_subject_performance(
    subject_metrics: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
) -> None:
    """Plot per-subject performance.

    Args:
        subject_metrics: Dictionary mapping subjects to their metrics.
        save_path: Path to save the figure.
        figsize: Figure size.
    """
    subjects = list(subject_metrics.keys())
    accuracies = [metrics['accuracy'] for metrics in subject_metrics.values()]

    plt.figure(figsize=figsize)
    plt.bar(range(len(subjects)), accuracies)
    plt.xlabel('Subject')
    plt.ylabel('Accuracy')
    plt.title('Per-Subject Accuracy')
    plt.xticks(range(len(subjects)), subjects, rotation=90)
    plt.ylim(0, 1)
    plt.axhline(y=np.mean(accuracies), color='r', linestyle='--', label='Average')
    plt.legend()
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved subject performance plot to {save_path}")
    else:
        plt.show()

    plt.close()


def analyze_results(
    predictions: np.ndarray,
    labels: np.ndarray,
    subjects: List[str],
    subject_indices: np.ndarray,
    confidences: Optional[np.ndarray] = None,
    output_dir: str = "results",
) -> Dict[str, Any]:
    """Comprehensive analysis of results.

    Args:
        predictions: Predicted labels.
        labels: Ground truth labels.
        subjects: List of subject names.
        subject_indices: Subject index for each example.
        confidences: Optional confidence scores.
        output_dir: Directory to save analysis outputs.

    Returns:
        Dictionary containing all analysis results.
    """
    from .metrics import (
        compute_metrics,
        compute_per_subject_metrics,
        compute_confidence_metrics,
    )

    # Overall metrics
    overall_metrics = compute_metrics(predictions, labels)

    # Per-subject metrics
    subject_metrics = compute_per_subject_metrics(
        predictions, labels, subjects, subject_indices
    )

    # Confidence metrics if available
    confidence_metrics = {}
    if confidences is not None:
        confidence_metrics = compute_confidence_metrics(
            predictions, labels, confidences
        )

    # Create visualizations
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Confusion matrix for answer choices (A, B, C, D)
    plot_confusion_matrix(
        predictions,
        labels,
        class_names=['A', 'B', 'C', 'D'],
        save_path=str(output_path / 'confusion_matrix.png'),
    )

    # Subject performance
    plot_subject_performance(
        subject_metrics,
        save_path=str(output_path / 'subject_performance.png'),
    )

    # Combine all results
    analysis = {
        'overall_metrics': overall_metrics,
        'subject_metrics': subject_metrics,
        'confidence_metrics': confidence_metrics,
    }

    return analysis


def save_results(
    results: Dict[str, Any],
    save_path: str,
    format: str = 'json',
) -> None:
    """Save results to file.

    Args:
        results: Results dictionary.
        save_path: Path to save results.
        format: Save format ('json' or 'csv').
    """
    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)

    if format == 'json':
        with open(save_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {save_path}")

    elif format == 'csv':
        # Flatten results for CSV
        flat_results = {}

        if 'overall_metrics' in results:
            for k, v in results['overall_metrics'].items():
                flat_results[f'overall_{k}'] = v

        if 'subject_metrics' in results:
            for subject, metrics in results['subject_metrics'].items():
                for k, v in metrics.items():
                    flat_results[f'{subject}_{k}'] = v

        df = pd.DataFrame([flat_results])
        df.to_csv(save_file, index=False)
        logger.info(f"Saved results to {save_path}")

    else:
        raise ValueError(f"Unknown format: {format}")


def print_summary(results: Dict[str, Any]) -> None:
    """Print summary of results.

    Args:
        results: Results dictionary.
    """
    print("\n" + "="*60)
    print("EVALUATION RESULTS SUMMARY")
    print("="*60)

    if 'overall_metrics' in results:
        print("\nOverall Metrics:")
        for metric, value in results['overall_metrics'].items():
            print(f"  {metric:20s}: {value:.4f}")

    if 'subject_metrics' in results:
        print(f"\nPer-Subject Metrics ({len(results['subject_metrics'])} subjects):")
        subject_accs = [m['accuracy'] for m in results['subject_metrics'].values()]
        print(f"  Mean Accuracy: {np.mean(subject_accs):.4f}")
        print(f"  Std Accuracy:  {np.std(subject_accs):.4f}")
        print(f"  Min Accuracy:  {np.min(subject_accs):.4f}")
        print(f"  Max Accuracy:  {np.max(subject_accs):.4f}")

    if 'confidence_metrics' in results and results['confidence_metrics']:
        print("\nConfidence Metrics:")
        for metric, value in results['confidence_metrics'].items():
            print(f"  {metric:30s}: {value:.4f}")

    print("="*60 + "\n")
