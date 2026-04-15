"""Project root paths for data, models, and reports."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORT_DIR = ROOT / "report"
DATASET_CSV = DATA_DIR / "dataset.csv"
MODEL_PATH = MODELS_DIR / "trained_model.joblib"
METRICS_PATH = MODELS_DIR / "evaluation_metrics.json"
