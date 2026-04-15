"""CLI entry to regenerate metrics and persist the best estimator."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.ml_pipeline import MLPipeline
from utils.paths import DATASET_CSV, METRICS_PATH, MODEL_PATH


def main() -> None:
    if not DATASET_CSV.exists():
        raise SystemExit(f"Dataset missing: {DATASET_CSV}. Run python data/generate_dataset.py first.")
    result = MLPipeline().train_and_save(DATASET_CSV, MODEL_PATH, METRICS_PATH)
    print("Saved:", result.model_path)
    print("Selected:", result.metrics["selected_model"])


if __name__ == "__main__":
    main()
