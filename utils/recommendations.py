"""Semester course recommendations with cohort-based reasoning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from utils.ml_pipeline import predict_pass_probability


@dataclass
class RecommendationEngine:
    """Ranks catalog courses using historical performance and individualized ML scores."""

    dataset: pd.DataFrame
    model_bundle: dict[str, Any]
    probability_floor: float = 0.55
    max_credits: int = 18

    def _catalog(self, program: str) -> pd.DataFrame:
        subset = self.dataset[self.dataset["program"] == program]
        if subset.empty:
            subset = self.dataset
        g = (
            subset.groupby(["course_id", "course_name"], as_index=False)
            .agg(
                mean_pass=("pass_fail", "mean"),
                mean_difficulty=("course_difficulty", "mean"),
                credits=("credit_load", "first"),
                cohort_gpa_mean=("gpa", "mean"),
            )
            .sort_values(["mean_pass", "mean_difficulty"], ascending=[False, True])
        )
        return g

    def suggest(
        self,
        program: str,
        gpa: float,
        course_level: int,
        attendance: float,
    ) -> dict[str, Any]:
        catalog = self._catalog(program)
        picks: list[dict[str, Any]] = []
        credits_used = 0

        for _, row in catalog.iterrows():
            cred = int(row["credits"])
            if credits_used + cred > self.max_credits:
                continue
            feats = {
                "gpa": gpa,
                "course_difficulty": float(row["mean_difficulty"]),
                "course_level": int(course_level),
                "attempts": 1,
                "attendance": attendance,
                "credit_load": cred,
                "program": program,
                "course_id": str(row["course_id"]),
            }
            p = predict_pass_probability(self.model_bundle, feats)
            if p < self.probability_floor:
                continue
            cohort = float(row["mean_pass"])
            reason = (
                f"Recommended because historical pass rate for this unit is {cohort:.0%} and "
                f"students near your GPA (~{row['cohort_gpa_mean']:.2f} in this slice) combined with "
                f"your attendance profile yield a modeled pass probability of {p:.0%}."
            )
            picks.append(
                {
                    "course_id": row["course_id"],
                    "course_name": row["course_name"],
                    "credits": cred,
                    "historical_pass_rate": round(cohort, 4),
                    "pass_probability": round(p, 4),
                    "reasoning": reason,
                }
            )
            credits_used += cred
            if credits_used >= self.max_credits - 1:
                break

        return {
            "program": program,
            "planned_credits": credits_used,
            "max_credits": self.max_credits,
            "courses": picks,
        }
