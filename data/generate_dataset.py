"""
Generate a realistic synthetic UMaT-style dataset (>=1000 rows).
Programs: Mining, Geological, Electrical, Mechanical Engineering.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROGRAMS = [
    "Mining Engineering",
    "Geological Engineering",
    "Electrical Engineering",
    "Mechanical Engineering",
]

# (program, course_id, course_name, credit_load, base_difficulty)
CATALOG: list[tuple[str, str, str, int, float]] = [
    ("Mining Engineering", "MN201", "Introduction to Mining", 2, 6.2),
    ("Mining Engineering", "MN301", "Rock Mechanics", 3, 7.4),
    ("Mining Engineering", "MN401", "Mine Planning", 3, 7.8),
    ("Geological Engineering", "GL201", "Structural Geology", 3, 6.9),
    ("Geological Engineering", "GL301", "Hydrogeology", 3, 7.1),
    ("Geological Engineering", "GL401", "Engineering Geology", 3, 7.3),
    ("Electrical Engineering", "EE201", "Circuit Theory II", 3, 7.0),
    ("Electrical Engineering", "EE301", "Power Systems I", 3, 7.6),
    ("Electrical Engineering", "EE401", "Renewable Energy", 3, 6.9),
    ("Mechanical Engineering", "ME201", "Thermodynamics I", 3, 7.0),
    ("Mechanical Engineering", "ME301", "Machine Design", 3, 7.5),
    ("Mechanical Engineering", "ME401", "Finite Element Methods", 3, 7.9),
]


class SyntheticAcademicDatasetBuilder:
    """Builds rows using a latent readiness model with attendance coupling."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def _pass_logit(
        self,
        gpa: float,
        difficulty: float,
        level: int,
        attempts: int,
        attendance: float,
        program_idx: int,
    ) -> float:
        z = (
            2.35 * (gpa - 2.15)
            - 0.38 * (difficulty - 6.0)
            - 0.85 * max(0, attempts - 1)
            + 0.018 * (attendance - 80)
            - 0.0008 * abs(level - 300)
            + 0.04 * program_idx
            + self._rng.normal(0, 0.32)
        )
        return float(1 / (1 + np.exp(-z)))

    def _letter_grade(self, mastery: float) -> str:
        if mastery >= 0.78:
            return "A"
        if mastery >= 0.62:
            return "B"
        if mastery >= 0.45:
            return "C"
        if mastery >= 0.28:
            return "D"
        return "F"

    def build(self, n_rows: int) -> pd.DataFrame:
        rows: list[dict] = []
        prog_idx = {p: i for i, p in enumerate(PROGRAMS)}

        for _ in range(n_rows):
            sid = f"STU-{self._rng.integers(1, 600):04d}"
            program = str(self._rng.choice(PROGRAMS))
            in_catalog = [c for c in CATALOG if c[0] == program]
            if in_catalog and self._rng.random() < 0.88:
                _, cid, cname, credits, base_d = in_catalog[int(self._rng.integers(0, len(in_catalog)))]
            else:
                _, cid, cname, credits, base_d = CATALOG[int(self._rng.integers(0, len(CATALOG)))]

            difficulty = float(np.clip(base_d + self._rng.normal(0, 0.35), 1.0, 10.0))
            level = int(self._rng.choice([200, 200, 300, 300, 400]))
            gpa = float(np.clip(self._rng.normal(2.75, 0.45), 1.5, 4.0))
            attempts = int(self._rng.choice([1, 1, 1, 2, 3], p=[0.74, 0.12, 0.06, 0.05, 0.03]))
            attendance = float(np.clip(self._rng.normal(82, 12), 40, 100))

            p_pass = self._pass_logit(
                gpa, difficulty, level, attempts, attendance, prog_idx[program]
            )
            passed = self._rng.random() < p_pass
            mastery = float(np.clip(p_pass + self._rng.normal(0, 0.07), 0.02, 0.98))
            if passed:
                grade = self._letter_grade(mastery)
            else:
                grade = str(self._rng.choice(["D", "F"], p=[0.32, 0.68]))
            pass_fail = 1 if grade != "F" else 0

            rows.append(
                {
                    "student_id": sid,
                    "program": program,
                    "gpa": round(gpa, 3),
                    "course_id": cid,
                    "course_name": cname,
                    "course_level": level,
                    "course_difficulty": round(difficulty, 2),
                    "attempts": attempts,
                    "attendance": round(attendance, 1),
                    "credit_load": credits,
                    "grade": grade,
                    "pass_fail": pass_fail,
                }
            )

        return pd.DataFrame(rows).sample(frac=1.0, random_state=42).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1500)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "dataset.csv",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = SyntheticAcademicDatasetBuilder().build(args.rows)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
