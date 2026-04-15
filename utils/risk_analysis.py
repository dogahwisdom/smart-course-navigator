"""Academic load risk classification for stacked course selections."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from utils.ml_pipeline import predict_pass_probability


@dataclass
class RiskAssessmentService:
    """Combines difficulty mass, credit load, and ML pass probabilities."""

    catalog: pd.DataFrame
    model_bundle: dict[str, Any]
    difficulty_soft_cap: float = 26.0
    min_safe_probability: float = 0.48

    def _course_row(self, program: str, course_id: str) -> pd.Series | None:
        rows = self.catalog[
            (self.catalog["course_id"] == course_id) & (self.catalog["program"] == program)
        ]
        if rows.empty:
            rows = self.catalog[self.catalog["course_id"] == course_id]
        if rows.empty:
            return None
        return rows.iloc[0]

    def assess(
        self,
        program: str,
        gpa: float,
        course_level: int,
        attendance: float,
        course_ids: list[str],
    ) -> dict[str, Any]:
        difficulties: list[float] = []
        credits: list[int] = []
        details: list[dict[str, Any]] = []

        for cid in course_ids:
            row = self._course_row(program, cid)
            if row is None:
                details.append({"course_id": cid, "error": "Unknown course for program"})
                continue
            diff = float(row.get("mean_difficulty", row.get("course_difficulty", 6.5)))
            cred = int(row.get("credits", row.get("credit_load", 3)))
            difficulties.append(diff)
            credits.append(cred)
            feats = {
                "gpa": gpa,
                "course_difficulty": diff,
                "course_level": int(course_level),
                "attempts": 1,
                "attendance": attendance,
                "credit_load": cred,
                "program": program,
                "course_id": cid,
            }
            p = predict_pass_probability(self.model_bundle, feats)
            details.append(
                {
                    "course_id": cid,
                    "pass_probability": round(p, 4),
                    "course_difficulty": diff,
                    "credits": cred,
                }
            )

        diff_sum = float(sum(difficulties)) if difficulties else 0.0
        credit_sum = int(sum(credits)) if credits else 0
        min_p = min((d["pass_probability"] for d in details if "pass_probability" in d), default=1.0)

        risk_score = 0
        if diff_sum > self.difficulty_soft_cap:
            risk_score += 2
        if credit_sum > 19:
            risk_score += 1
        if min_p < self.min_safe_probability:
            risk_score += 2

        if risk_score >= 3:
            band = "High risk"
        elif risk_score == 2:
            band = "Medium risk"
        else:
            band = "Low risk"

        warnings: list[str] = []
        if diff_sum > self.difficulty_soft_cap:
            warnings.append(
                f"Combined difficulty index ({diff_sum:.1f}) exceeds the advisory threshold "
                f"({self.difficulty_soft_cap:.1f}) for a single semester."
            )
        if credit_sum > 19:
            warnings.append("Credit load is aggressive relative to typical engineering pacing.")
        if min_p < self.min_safe_probability:
            warnings.append("At least one course shows a modeled pass probability below the comfort zone.")

        return {
            "risk_band": band,
            "difficulty_sum": diff_sum,
            "credit_sum": credit_sum,
            "min_modeled_pass_probability": min_p,
            "warnings": warnings,
            "details": details,
        }


def build_program_course_index(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregated lookup used by risk and recommendations."""
    return (
        df.groupby(["program", "course_id", "course_name"], as_index=False)
        .agg(
            mean_difficulty=("course_difficulty", "mean"),
            mean_pass=("pass_fail", "mean"),
            credit_load=("credit_load", "first"),
        )
        .rename(columns={"credit_load": "credits"})
    )
