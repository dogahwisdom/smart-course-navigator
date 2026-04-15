"""Generate figures for the academic DOCX report."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).resolve().parent / "assets"


class ReportFigureFactory:
    """Creates PNG diagrams referenced by the Word report."""

    def __init__(self, out_dir: Path | None = None) -> None:
        self.out = out_dir or ASSETS
        self.out.mkdir(parents=True, exist_ok=True)

    def streamlit_architecture(self) -> Path:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.axis("off")
        boxes = [
            ("Browser\n(Streamlit UI)", 0.06, 0.55),
            ("Python logic\n(utils/)", 0.36, 0.55),
            ("joblib model\n(Random Forest)", 0.66, 0.72),
            ("CSV dataset\n(data/)", 0.66, 0.32),
        ]
        for txt, x, y in boxes:
            self._box(ax, txt, x, y)
        self._arrow(ax, 0.26, 0.6, 0.35, 0.6)
        self._arrow(ax, 0.55, 0.62, 0.65, 0.72)
        self._arrow(ax, 0.55, 0.55, 0.65, 0.4)
        ax.set_title("Figure 1. Streamlit-centric architecture")
        path = self.out / "fig_architecture.png"
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
        return path

    def ml_workflow(self) -> Path:
        fig, ax = plt.subplots(figsize=(9, 2.8))
        ax.axis("off")
        steps = ["Data ingest", "Preprocess", "Train / test split", "Train 3 models", "Select best F1", "Save joblib"]
        x = 0.04
        for i, label in enumerate(steps):
            self._box(ax, label, x + i * 0.15, 0.35, w=0.13, h=0.32)
            if i < len(steps) - 1:
                self._arrow(ax, x + i * 0.15 + 0.11, 0.5, x + (i + 1) * 0.15, 0.5)
        ax.set_title("Figure 2. Machine learning workflow")
        path = self.out / "fig_ml_workflow.png"
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
        return path

    def model_comparison(self) -> Path:
        metrics_path = ROOT / "models" / "evaluation_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows = metrics["results"]
        names = [r["model"].replace("_", " ").title() for r in rows]
        x = np.arange(len(names))
        width = 0.2
        fig, ax = plt.subplots(figsize=(8, 4.2))
        for i, key in enumerate(["accuracy", "precision", "recall", "f1_score"]):
            vals = [r[key] for r in rows]
            ax.bar(x + (i - 1.5) * width, vals, width=width, label=key.replace("_", " ").title())
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=12)
        ax.set_ylim(0, 1.05)
        ax.legend(ncol=2)
        ax.set_title("Figure 3. Model comparison (hold-out)")
        path = self.out / "fig_model_compare.png"
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
        return path

    def feature_importance(self) -> Path:
        metrics_path = ROOT / "models" / "evaluation_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        items = metrics.get("feature_importance") or []
        if not items:
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(0.1, 0.5, "Feature importance unavailable for selected estimator.")
            path = self.out / "fig_feature_importance.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            return path
        df = pd.DataFrame(items).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.barh(df["feature"], df["importance"], color="#0b3d5c")
        ax.set_title("Figure 4. Random Forest feature importance (top encoded features)")
        path = self.out / "fig_feature_importance.png"
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
        return path

    def streamlit_ui_mock(self) -> Path:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.05, 0.12), 0.9, 0.7, fill=False, linestyle="--", linewidth=1.2))
        ax.text(0.5, 0.82, "Streamlit UI - multi-page navigator (mock)", ha="center", weight="bold")
        ax.text(
            0.08,
            0.68,
            "• Sidebar GPA / program / attendance\n• Prediction + risk bundle\n• Analytics + Plotly charts\n• Recommendations with reasoning",
            fontsize=11,
            va="top",
        )
        ax.text(
            0.5,
            0.2,
            "Replace with real screenshots: streamlit run app.py → capture each page.",
            ha="center",
            fontsize=10,
            style="italic",
        )
        path = self.out / "fig_streamlit_ui.png"
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)
        return path

    def _box(self, ax, text: str, x: float, y: float, w: float = 0.2, h: float = 0.26) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.01", linewidth=1.1, facecolor="#f8fafc"
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    def _arrow(self, ax, x1: float, y1: float, x2: float, y2: float) -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, linewidth=1.1, color="#0b3d5c"
            )
        )


def main() -> None:
    factory = ReportFigureFactory()
    factory.streamlit_architecture()
    factory.ml_workflow()
    factory.model_comparison()
    factory.feature_importance()
    factory.streamlit_ui_mock()
    print("Figures saved to", ASSETS)


if __name__ == "__main__":
    main()
