"""Model training and inference utilities for pass/fail prediction.

This module encapsulates the full supervised-learning lifecycle used by the
project:
1. Load and validate feature matrices.
2. Train three candidate classifiers behind a shared preprocessing pipeline.
3. Evaluate with hold-out metrics (accuracy, precision, recall, F1).
4. Select the best model by F1-score.
5. Persist the production bundle and JSON metrics for the app/report.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from utils.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
    load_training_frame,
)


@dataclass
class TrainingResult:
    """Container for training outputs returned to CLI/notebook callers."""

    metrics: dict[str, Any]
    model_path: Path


class MLPipeline:
    """Train/evaluate candidate models and persist the selected estimator.

    Notes for teammates:
    - The same preprocessing transformer is reused for each candidate model to
      keep comparisons fair.
    - F1-score is used as the selection criterion because it balances precision
      and recall for classification tasks where class balance may shift.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def _estimators(self) -> dict[str, Any]:
        """Return configured candidate estimators for model benchmarking."""
        return {
            "logistic_regression": LogisticRegression(
                max_iter=2500,
                class_weight="balanced",
                solver="lbfgs",
                random_state=self.random_state,
            ),
            "decision_tree": DecisionTreeClassifier(
                max_depth=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=self.random_state,
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=240,
                max_depth=16,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
            ),
        }

    def train_and_save(self, csv_path: Path, model_path: Path, metrics_path: Path) -> TrainingResult:
        """Run the full training pipeline and persist artifacts.

        Args:
            csv_path: Input dataset path.
            model_path: Output path for serialized model bundle (joblib).
            metrics_path: Output path for evaluation metrics JSON.

        Returns:
            TrainingResult containing metrics payload and model file path.
        """
        # Phase 1: Data load and hold-out split.
        X, y = load_training_frame(str(csv_path))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=self.random_state
        )

        # Phase 2: Candidate model training and evaluation.
        rows: list[dict[str, Any]] = []
        best_name = ""
        best_score = -1.0
        best_pipe: Pipeline | None = None

        for name, est in self._estimators().items():
            pipe = Pipeline(steps=[("prep", build_preprocessor()), ("clf", est)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            rows.append(
                {
                    "model": name,
                    "accuracy": round(float(acc), 4),
                    "precision": round(float(prec), 4),
                    "recall": round(float(rec), 4),
                    "f1_score": round(float(f1), 4),
                }
            )
            if f1 > best_score:
                best_score = f1
                best_name = name
                best_pipe = pipe

        assert best_pipe is not None
        # Phase 3: Explainability summary for tree-based selected models.
        importances = self._feature_importance(best_pipe)

        # Phase 4: Metrics packaging for reporting and UI dashboard display.
        metrics = {
            "selected_model": best_name,
            "selection_criterion": "Highest F1-score on stratified hold-out test set (imbalance handled via class_weight).",
            "results": rows,
            "feature_columns": list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES),
            "feature_importance": importances,
        }

        # Phase 5: Persist model bundle and companion files.
        model_path.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "pipeline": best_pipe,
            "name": best_name,
            "numeric": list(NUMERIC_FEATURES),
            "categorical": list(CATEGORICAL_FEATURES),
        }
        joblib.dump(bundle, model_path)
        pkl_path = model_path.with_name("trained_model.pkl")
        shutil.copyfile(model_path, pkl_path)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return TrainingResult(metrics=metrics, model_path=model_path)

    def _feature_importance(self, pipe: Pipeline) -> list[dict[str, float]]:
        """Extract top encoded feature importances if classifier supports it."""
        clf = pipe.named_steps["clf"]
        prep = pipe.named_steps["prep"]
        if not hasattr(clf, "feature_importances_"):
            return []
        names = prep.get_feature_names_out()
        scores = clf.feature_importances_
        order = np.argsort(scores)[::-1][:12]
        return [{"feature": str(names[i]), "importance": float(scores[i])} for i in order]


def load_model_bundle(model_path: Path) -> dict[str, Any]:
    """Load persisted model bundle from disk."""
    return joblib.load(model_path)


def predict_pass_probability(bundle: dict[str, Any], row: dict[str, Any]) -> float:
    """Predict pass probability for a single candidate course row."""
    frame = pd.DataFrame([{k: row[k] for k in bundle["numeric"] + bundle["categorical"]}])
    proba = bundle["pipeline"].predict_proba(frame)[:, 1]
    return float(proba[0])
