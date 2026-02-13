# Contrastive Curriculum MMLU with Adaptive Difficulty Sampling

A contrastive learning approach for MMLU classification that combines subject-aware representation learning with adaptive curriculum strategies.

## Overview

This project implements three integrated techniques for MMLU task learning:

1. **Subject-Aware Contrastive Learning** - Uses both class labels and subject indices to construct positive/negative pairs, enabling domain-aware representation learning across 58 MMLU subjects.

2. **Adaptive Curriculum Sampling** - Tracks per-subject difficulty using exponential moving averages of entropy-based uncertainty, implementing a zone of proximal development approach.

3. **Uncertainty-Weighted Loss** - Dynamically adjusts sample importance based on uncertainty gradients for smooth curriculum progression.

## Installation

```bash
pip install -e .
```

Or install dependencies only:
```bash
pip install -r requirements.txt
```

## Quick Start

### Training

```bash
python scripts/train.py --config configs/default.yaml
```

### Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --split test
```

### Prediction

```bash
python scripts/predict.py \
    --checkpoint checkpoints/best_model.pt \
    --question "What is the capital of France?" \
    --choices "London" "Paris" "Berlin" "Madrid"
```

## Results

### Full Model Performance

| Metric | Score |
|--------|-------|
| Overall Accuracy | 26.94% |
| Macro F1 | 21.31% |
| Weighted F1 | 22.14% |
| Best Subject (Professional Medicine) | 47.14% |

The model exceeds the random baseline (25% for 4-choice questions) and shows subject-specific specialization. Top performing subjects include Electrical Engineering (44.44%), High School Statistics (44.89%), and Formal Logic (43.33%).

### Baseline Comparison

| Model | Accuracy | F1 Macro | F1 Weighted |
|-------|----------|----------|-------------|
| Random Baseline | 25.00% | - | - |
| Standard Classifier (Baseline) | ~24.5-26.0% | ~20-22% | ~21-23% |
| Contrastive Only | ~25.5-27.5% | ~20.5-22.5% | ~21.5-23.5% |
| Full Model (Ours) | 26.94% | 21.31% | 22.14% |

See `ABLATION_RESULTS.md` for complete ablation study details and analysis.

## Configuration

Key parameters in `configs/default.yaml`:

**Model:**
- `base_model`: sentence-transformers/all-MiniLM-L6-v2
- `projection_dim`: 256
- `hidden_dim`: 384

**Training:**
- `batch_size`: 32
- `num_epochs`: 20
- `learning_rate`: 0.0001

**Contrastive Learning:**
- `temperature`: 0.07
- `subject_aware_sampling`: true

**Curriculum:**
- `strategy`: adaptive_difficulty
- `initial_difficulty_threshold`: 0.3
- `uncertainty_metric`: entropy

**Loss Weights:**
- `contrastive_weight`: 0.5
- `classification_weight`: 0.3
- `curriculum_weight`: 0.2

## Architecture

The model consists of:
- Pre-trained sentence transformer encoder (frozen by default)
- Projection head for contrastive learning (384 -> 256)
- Classification head for 4-way MMLU choice prediction
- Subject embeddings for domain-aware representations

## Key Components

### SubjectAwareContrastiveLoss
Contrastive loss that constructs positive pairs using both class labels and subject indices, weighted by a subject_weight parameter to balance fine-grained and coarse-grained clustering.

### CurriculumWeightedLoss
Adaptive loss weighting based on model uncertainty and difficulty thresholds. Implements smooth curriculum progression by adjusting sample importance dynamically.

### AdaptiveDifficultyTracker
Tracks per-subject difficulty using exponential moving average of entropy-based uncertainty. Updates difficulty thresholds periodically to maintain focus on appropriately challenging samples.

## Project Structure

```
├── src/contrastive_curriculum_mmlu_with_adaptive_difficulty_sampling/
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model architecture and loss components
│   ├── training/          # Trainer with curriculum sampling
│   ├── evaluation/        # Metrics and analysis
│   ├── scripts/           # Training, evaluation, prediction
│   └── utils/             # Configuration utilities
├── configs/               # YAML configuration files
├── scripts/               # Convenience CLI wrappers
├── tests/                 # Unit tests
└── results/               # Evaluation outputs
```

## Testing

```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=term-missing
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.35+
- sentence-transformers 2.2+

See `requirements.txt` for complete dependency list.

## Known Limitations

The current model achieves 26.94% accuracy, barely exceeding the 25% random baseline. This indicates architectural limitations with using a frozen sentence-transformer for MMLU reasoning tasks. Potential improvements include:

1. Unfreezing the encoder for task-specific fine-tuning
2. Using larger pre-trained models with reasoning capabilities
3. Implementing multi-hop reasoning architectures
4. Adding explicit knowledge retrieval mechanisms

## License

MIT License - Copyright (c) 2026 Alireza Shojaei

See LICENSE file for full terms.
