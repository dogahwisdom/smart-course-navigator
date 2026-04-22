"""CLI entry-point for training reproducibility.

Usage:
    python scripts/train.py

This script is intentionally thin: it validates dataset presence, delegates
model training to ``MLPipeline``, and prints artifact locations for operators.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.ml_pipeline import MLPipeline
from utils.paths import DATASET_CSV, METRICS_PATH, MODEL_PATH


def main() -> None:
    """Run training from the command line and emit artifact summary."""
    if not DATASET_CSV.exists():
        raise SystemExit(f"Dataset missing: {DATASET_CSV}. Run python data/generate_dataset.py first.")

    # Delegate full training/evaluation/persistence to the pipeline class.
    result = MLPipeline().train_and_save(DATASET_CSV, MODEL_PATH, METRICS_PATH)

    # Minimal operational logging for quick CI/manual verification.
    print("Saved:", result.model_path)
    print("Selected:", result.metrics["selected_model"])


if __name__ == "__main__":
    main()
