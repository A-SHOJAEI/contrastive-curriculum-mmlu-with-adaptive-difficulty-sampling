# Project Summary: Contrastive Curriculum MMLU with Adaptive Difficulty Sampling

## Overview

This is a **research-tier** machine learning project that implements a novel approach to few-shot learning on the MMLU benchmark. The project combines contrastive representation learning with adaptive curriculum strategies to improve sample efficiency and cross-domain knowledge transfer.

## Novel Contributions

### 1. Subject-Aware Contrastive Loss
**Location**: `src/.../models/components.py:67-113`

A custom contrastive loss function that enforces clustering within subject domains while maintaining separation across different subjects. Unlike standard contrastive learning that only considers class-level similarity, this approach incorporates subject-domain information:

```python
positive_mask = (
    (1 - subject_weight) * class_positive_mask +
    subject_weight * subject_positive_mask
)
```

**Impact**: Enhances cross-domain knowledge transfer by learning domain-aware representations.

### 2. Curriculum-Weighted Loss
**Location**: `src/.../models/components.py:124-207`

An adaptive loss weighting mechanism that dynamically adjusts sample importance based on model uncertainty gradients. Samples near the difficulty threshold receive higher weights:

```python
weights = 1.0 - torch.abs(uncertainties - self.difficulty_threshold)
```

**Impact**: Enables efficient curriculum progression by focusing on samples at the right difficulty level.

### 3. Adaptive Difficulty Sampling
**Location**: `src/.../data/preprocessing.py:51-105`

A sampling strategy that tracks per-subject difficulty scores using exponential moving averages of model uncertainty:

```python
alpha = 0.3
current_score = self.difficulty_scores.get(subject, 0.5)
self.difficulty_scores[subject] = alpha * uncertainty + (1 - alpha) * current_score
```

**Impact**: Dynamically adjusts task exposure based on model learning progress.

## Technical Architecture

### Model Components
- **Encoder**: Pre-trained sentence transformer (sentence-transformers/all-MiniLM-L6-v2)
- **Projection Head**: For contrastive learning (hidden_dim → projection_dim)
- **Classification Head**: For MMLU task prediction (hidden_dim → 4 choices)
- **Subject Embeddings**: Learned embeddings for 57 MMLU subjects

### Training Pipeline
- **Optimizer**: AdamW with weight decay
- **LR Scheduler**: Cosine annealing with linear warmup
- **Mixed Precision**: Automatic mixed precision (AMP) support
- **Early Stopping**: Patience-based with validation monitoring
- **Gradient Clipping**: For training stability
- **MLflow Tracking**: Experiment tracking (optional)

### Multi-Objective Loss
```python
total_loss = (
    classification_weight * classification_loss +
    contrastive_weight * contrastive_loss +
    curriculum_weight * curriculum_loss
)
```

## Project Statistics

- **Total Python Files**: 22
- **Total Lines of Code**: 3,314
- **Test Coverage**: 3 test modules with comprehensive fixtures
- **README Length**: 169 lines (under 200 limit)
- **Configuration Files**: 2 (default + ablation baseline)

## File Structure

```
contrastive-curriculum-mmlu-with-adaptive-difficulty-sampling/
├── src/contrastive_curriculum_mmlu_with_adaptive_difficulty_sampling/
│   ├── data/               # MMLU data loading and preprocessing
│   ├── models/             # Model architecture and custom components
│   ├── training/           # Training loop with curriculum support
│   ├── evaluation/         # Comprehensive metrics and analysis
│   └── utils/              # Configuration and utilities
├── configs/                # YAML configurations (default + ablation)
├── scripts/                # train.py, evaluate.py, predict.py
├── tests/                  # Unit tests with pytest
├── checkpoints/            # Model checkpoints (created during training)
├── results/                # Evaluation results (created during evaluation)
└── README.md               # Concise project documentation
```

## Usage Examples

### Training
```bash
# Full model with novel components
python scripts/train.py --config configs/default.yaml

# Baseline for ablation study
python scripts/train.py --config configs/ablation.yaml
```

### Evaluation
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --split test \
    --output-dir results/
```

### Prediction
```bash
python scripts/predict.py \
    --checkpoint checkpoints/best_model.pt \
    --question "What is the derivative of x^2?" \
    --choices "2x" "x^2" "x" "2"
```

## Key Design Decisions

1. **Sentence Transformers Base**: Chosen for strong pre-trained representations and efficient inference
2. **Curriculum Strategy**: Adaptive strategy that focuses on medium-difficulty samples (Goldilocks zone)
3. **Subject Embeddings**: Learned rather than fixed to adapt to task-specific patterns
4. **Multi-Task Learning**: Simultaneous training across all 57 MMLU subjects
5. **Uncertainty Estimation**: Entropy-based for interpretability and computational efficiency

## Reproducibility

- All random seeds are set in `utils/config.py:set_seed()`
- Configuration files contain all hyperparameters
- Deterministic PyTorch operations enabled
- Requirements pinned with minimum versions

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=term-missing
```

## Evaluation Metrics

The project tracks:
- **Overall Accuracy**: Across all MMLU subjects
- **Per-Subject Accuracy**: With statistical analysis (mean, std, min, max)
- **F1 Scores**: Macro and weighted
- **Confidence Metrics**: High/low confidence accuracy
- **Calibration Metrics**: ECE (Expected Calibration Error)

## Ablation Study

The project includes a baseline configuration (`configs/ablation.yaml`) that disables:
- Contrastive learning (contrastive_weight = 0.0)
- Curriculum learning (enabled = false)
- Subject-aware sampling (subject_aware_sampling = false)

This allows for fair comparison of the novel components' contribution.

## Expected Performance

Target metrics (from specification):
- MMLU Average Accuracy: 0.68
- Low Resource Task Improvement: 0.15
- Training Efficiency Gain: 0.25

Note: Actual performance will depend on hardware, training time, and hyperparameter tuning.

## License

MIT License - Copyright (c) 2026 Alireza Shojaei

---

**Project Type**: Research-tier ML implementation
**Domain**: Natural Language Processing (NLP)
**Frameworks**: PyTorch, Transformers, sentence-transformers
**Techniques**: Contrastive Learning, Curriculum Learning, Multi-Task Learning
