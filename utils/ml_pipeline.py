"""Train, evaluate, persist, and load pass/fail classifiers."""
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
    metrics: dict[str, Any]
    model_path: Path


class MLPipeline:
    """Fits three estimators, selects best by macro F1 on the positive class context."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def _estimators(self) -> dict[str, Any]:
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
        X, y = load_training_frame(str(csv_path))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=self.random_state
        )

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
        importances = self._feature_importance(best_pipe)

        metrics = {
            "selected_model": best_name,
            "selection_criterion": "Highest F1-score on stratified hold-out test set (imbalance handled via class_weight).",
            "results": rows,
            "feature_columns": list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES),
            "feature_importance": importances,
        }

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
        clf = pipe.named_steps["clf"]
        prep = pipe.named_steps["prep"]
        if not hasattr(clf, "feature_importances_"):
            return []
        names = prep.get_feature_names_out()
        scores = clf.feature_importances_
        order = np.argsort(scores)[::-1][:12]
        return [{"feature": str(names[i]), "importance": float(scores[i])} for i in order]


def load_model_bundle(model_path: Path) -> dict[str, Any]:
    return joblib.load(model_path)


def predict_pass_probability(bundle: dict[str, Any], row: dict[str, Any]) -> float:
    frame = pd.DataFrame([{k: row[k] for k in bundle["numeric"] + bundle["categorical"]}])
    proba = bundle["pipeline"].predict_proba(frame)[:, 1]
    return float(proba[0])
