"""Evaluation metrics for MMLU tasks."""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    average: str = "macro",
) -> Dict[str, float]:
    """Compute classification metrics.

    Args:
        predictions: Predicted labels of shape (n_samples,).
        labels: Ground truth labels of shape (n_samples,).
        average: Averaging strategy for multi-class metrics.

    Returns:
        Dictionary containing accuracy, precision, recall, and F1 scores.
    """
    metrics = {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, average=average, zero_division=0),
        'recall': recall_score(labels, predictions, average=average, zero_division=0),
        'f1_macro': f1_score(labels, predictions, average='macro', zero_division=0),
        'f1_weighted': f1_score(labels, predictions, average='weighted', zero_division=0),
    }

    return metrics


def compute_per_subject_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    subjects: List[str],
    subject_indices: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Compute metrics per subject.

    Args:
        predictions: Predicted labels.
        labels: Ground truth labels.
        subjects: List of subject names.
        subject_indices: Subject index for each example.

    Returns:
        Dictionary mapping subject names to their metrics.
    """
    subject_metrics = {}

    unique_subjects = np.unique(subject_indices)

    for subj_idx in unique_subjects:
        mask = subject_indices == subj_idx

        if mask.sum() == 0:
            continue

        subj_preds = predictions[mask]
        subj_labels = labels[mask]

        subject_name = subjects[subj_idx] if subj_idx < len(subjects) else f"subject_{subj_idx}"

        subject_metrics[subject_name] = {
            'accuracy': accuracy_score(subj_labels, subj_preds),
            'f1': f1_score(subj_labels, subj_preds, average='macro', zero_division=0),
            'n_samples': int(mask.sum()),
        }

    return subject_metrics


def compute_confidence_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    confidences: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute metrics based on prediction confidence.

    Args:
        predictions: Predicted labels.
        labels: Ground truth labels.
        confidences: Confidence scores for predictions.
        threshold: Confidence threshold.

    Returns:
        Dictionary with confidence-based metrics.
    """
    high_conf_mask = confidences >= threshold

    metrics = {
        'high_confidence_ratio': high_conf_mask.mean(),
    }

    if high_conf_mask.sum() > 0:
        metrics['high_confidence_accuracy'] = accuracy_score(
            labels[high_conf_mask],
            predictions[high_conf_mask]
        )
    else:
        metrics['high_confidence_accuracy'] = 0.0

    low_conf_mask = ~high_conf_mask
    if low_conf_mask.sum() > 0:
        metrics['low_confidence_accuracy'] = accuracy_score(
            labels[low_conf_mask],
            predictions[low_conf_mask]
        )
    else:
        metrics['low_confidence_accuracy'] = 0.0

    return metrics


def compute_calibration_metrics(
    confidences: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compute calibration metrics (ECE, MCE).

    Args:
        confidences: Confidence scores.
        predictions: Predicted labels.
        labels: Ground truth labels.
        n_bins: Number of bins for calibration.

    Returns:
        Dictionary with calibration metrics and bin statistics.
    """
    # Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()

            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))

            bin_accuracies.append(float(accuracy_in_bin))
            bin_confidences.append(float(avg_confidence_in_bin))
            bin_counts.append(int(in_bin.sum()))
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)
            bin_counts.append(0)

    return {
        'ece': float(ece),
        'mce': float(mce),
        'bin_accuracies': bin_accuracies,
        'bin_confidences': bin_confidences,
        'bin_counts': bin_counts,
    }
