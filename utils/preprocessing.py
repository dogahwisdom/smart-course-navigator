"""Feature matrix construction and preprocessing helpers for ML."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


NUMERIC_FEATURES = [
    "gpa",
    "course_difficulty",
    "course_level",
    "attempts",
    "attendance",
    "credit_load",
]
CATEGORICAL_FEATURES = ["program", "course_id"]
TARGET = "pass_fail"


@dataclass(frozen=True)
class DatasetSchema:
    """Documents the canonical column contract for training and inference."""

    numeric: tuple[str, ...] = tuple(NUMERIC_FEATURES)
    categorical: tuple[str, ...] = tuple(CATEGORICAL_FEATURES)
    target: str = TARGET


def build_preprocessor() -> ColumnTransformer:
    """Scaling for numeric features; OHE for high-cardinality categoricals."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), list(NUMERIC_FEATURES)),
            ("cat", _one_hot_encoder(), list(CATEGORICAL_FEATURES)),
        ]
    )


def load_training_frame(csv_path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    missing = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")
    X = df[list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)]
    y = df[TARGET].astype(int)
    return X, y


def row_dict_to_frame(row: dict) -> pd.DataFrame:
    """Single-row inference helper aligned with training columns."""
    payload = {k: row[k] for k in NUMERIC_FEATURES + CATEGORICAL_FEATURES}
    return pd.DataFrame([payload])
