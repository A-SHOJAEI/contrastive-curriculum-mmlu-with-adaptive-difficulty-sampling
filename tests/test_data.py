"""Tests for data loading and preprocessing."""

import pytest
import numpy as np

from contrastive_curriculum_mmlu_with_adaptive_difficulty_sampling.data.preprocessing import (
    MMLUPreprocessor,
    create_few_shot_examples,
    compute_uncertainty,
)


def test_preprocessor_initialization():
    """Test MMLUPreprocessor initialization."""
    subjects = ['math', 'physics', 'chemistry']
    preprocessor = MMLUPreprocessor(subjects, few_shot_k=5, seed=42)

    assert preprocessor.subjects == subjects
    assert preprocessor.few_shot_k == 5
    assert len(preprocessor.difficulty_scores) == 3
    assert all(score == 0.5 for score in preprocessor.difficulty_scores.values())


def test_update_difficulty_scores():
    """Test difficulty score updates."""
    subjects = ['math', 'physics']
    preprocessor = MMLUPreprocessor(subjects, seed=42)

    # Update with high uncertainty
    preprocessor.update_difficulty_scores('math', 0.9)
    assert preprocessor.difficulty_scores['math'] > 0.5

    # Update with low uncertainty
    preprocessor.update_difficulty_scores('physics', 0.1)
    assert preprocessor.difficulty_scores['physics'] < 0.5


def test_get_difficulty_weights():
    """Test difficulty weight computation."""
    subjects = ['math', 'physics', 'chemistry']
    preprocessor = MMLUPreprocessor(subjects, seed=42)

    # Test random strategy
    weights = preprocessor.get_difficulty_weights(strategy='random')
    assert len(weights) == 3
    assert all(w == 1.0 for w in weights.values())

    # Test adaptive strategy
    preprocessor.difficulty_scores = {'math': 0.3, 'physics': 0.5, 'chemistry': 0.7}
    weights = preprocessor.get_difficulty_weights(strategy='adaptive')
    assert len(weights) == 3
    assert abs(sum(weights.values()) - 1.0) < 0.01  # Should be normalized


def test_sample_by_curriculum():
    """Test curriculum-based sampling."""
    subjects = ['math', 'physics']
    preprocessor = MMLUPreprocessor(subjects, seed=42)

    data = [
        {'subject': 'math', 'question': 'q1', 'answer': 0},
        {'subject': 'math', 'question': 'q2', 'answer': 1},
        {'subject': 'physics', 'question': 'q3', 'answer': 2},
        {'subject': 'physics', 'question': 'q4', 'answer': 3},
    ]

    sampled = preprocessor.sample_by_curriculum(data, n_samples=2, strategy='random')
    assert len(sampled) == 2
    assert all(item in data for item in sampled)


def test_create_few_shot_examples():
    """Test few-shot example creation."""
    data = [
        {'subject': 'math', 'question': f'q{i}', 'answer': i % 4}
        for i in range(10)
    ]

    # Test with subject filter
    examples = create_few_shot_examples(data, k=3, subject='math', seed=42)
    assert len(examples) == 3
    assert all(ex['subject'] == 'math' for ex in examples)

    # Test without subject filter
    examples = create_few_shot_examples(data, k=5, subject=None, seed=42)
    assert len(examples) == 5


def test_compute_uncertainty():
    """Test uncertainty computation."""
    # Uniform distribution (high uncertainty)
    logits_uniform = np.array([1.0, 1.0, 1.0, 1.0])
    uncertainty_uniform = compute_uncertainty(logits_uniform)
    assert 0.9 < uncertainty_uniform <= 1.0

    # Peaked distribution (low uncertainty)
    logits_peaked = np.array([10.0, 0.0, 0.0, 0.0])
    uncertainty_peaked = compute_uncertainty(logits_peaked)
    assert 0.0 <= uncertainty_peaked < 0.1

    # Test edge case with zeros
    logits_zeros = np.array([0.0, 0.0, 0.0, 0.0])
    uncertainty_zeros = compute_uncertainty(logits_zeros)
    assert 0.0 <= uncertainty_zeros <= 1.0
