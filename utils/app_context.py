"""Shared cached loaders and view-model helpers for Streamlit pages."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st

from data.generate_dataset import SyntheticAcademicDatasetBuilder
from utils.ml_pipeline import MLPipeline, load_model_bundle
from utils.paths import DATASET_CSV, METRICS_PATH, MODEL_PATH
from utils.risk_analysis import build_program_course_index


@dataclass(frozen=True)
class AppContext:
    """Typed container for immutable runtime resources."""

    dataset: pd.DataFrame
    model_bundle: dict[str, Any]
    metrics: dict[str, Any]
    catalog_index: pd.DataFrame


def _validate_required_columns(df: pd.DataFrame) -> None:
    required = {
        "program",
        "course_id",
        "course_name",
        "pass_fail",
        "course_difficulty",
        "gpa",
        "course_level",
        "attempts",
        "attendance",
        "credit_load",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")


@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Load and validate the analytics/training snapshot."""
    df = pd.read_csv(DATASET_CSV)
    _validate_required_columns(df)
    return df


@st.cache_resource(show_spinner=False)
def load_bundle() -> dict[str, Any]:
    """Load serialized model bundle once per process.

    Streamlit Cloud/runtime Python versions can differ from the training
    environment, which may make old joblib artifacts unreadable. In that case,
    regenerate dataset/model artifacts and load a fresh compatible bundle.
    """
    if not DATASET_CSV.exists():
        DATASET_CSV.parent.mkdir(parents=True, exist_ok=True)
        SyntheticAcademicDatasetBuilder(seed=42).build(1500).to_csv(DATASET_CSV, index=False)

    if not MODEL_PATH.exists():
        MLPipeline(random_state=42).train_and_save(DATASET_CSV, MODEL_PATH, METRICS_PATH)
        return load_model_bundle(MODEL_PATH)

    try:
        return load_model_bundle(MODEL_PATH)
    except Exception:
        MLPipeline(random_state=42).train_and_save(DATASET_CSV, MODEL_PATH, METRICS_PATH)
        return load_model_bundle(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict[str, Any]:
    """Load persisted training metrics for dashboards."""
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def build_context() -> AppContext:
    """Build page context with all shared resources."""
    df = load_dataset()
    return AppContext(
        dataset=df,
        model_bundle=load_bundle(),
        metrics=load_metrics(),
        catalog_index=build_program_course_index(df),
    )
