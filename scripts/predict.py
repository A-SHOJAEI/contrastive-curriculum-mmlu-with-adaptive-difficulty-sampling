#!/usr/bin/env python
"""Prediction script for contrastive curriculum MMLU model.

This is a convenience wrapper that calls the main prediction entry point.
For production use, install the package and use: predict-mmlu
"""

import sys
from pathlib import Path

# Add src/ to path for development mode
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from contrastive_curriculum_mmlu_with_adaptive_difficulty_sampling.scripts.predict import main


if __name__ == "__main__":
    main()
