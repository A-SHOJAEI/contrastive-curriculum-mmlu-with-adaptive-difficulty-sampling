#!/usr/bin/env python
"""Training script for contrastive curriculum MMLU model.

This is a convenience wrapper that calls the main training entry point.
For production use, install the package and use: train-mmlu
"""

import sys
from pathlib import Path

# Add src/ to path for development mode
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contrastive_curriculum_mmlu_with_adaptive_difficulty_sampling.scripts.train import main


if __name__ == "__main__":
    main()
