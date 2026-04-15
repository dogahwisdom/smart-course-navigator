"""Classifier benchmarking and feature importance."""
from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.paths import METRICS_PATH


st.title("Model performance dashboard")

if not METRICS_PATH.exists():
    st.error("Run `python scripts/train.py` after generating the dataset.")
    st.stop()

metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
st.subheader("Model selection")
st.write(metrics["selection_criterion"])
st.success(f"Selected model: **{metrics['selected_model']}**")

res = pd.DataFrame(metrics["results"])
melted = res.melt(id_vars="model", var_name="metric", value_name="score")
fig = px.bar(
    melted,
    x="model",
    y="score",
    color="metric",
    barmode="group",
    title="Hold-out metrics by model",
    labels={"score": "Score", "model": "Model"},
)
st.plotly_chart(fig, use_container_width=True)

st.dataframe(res, use_container_width=True, hide_index=True)

imps = metrics.get("feature_importance") or []
if imps:
    imp_df = pd.DataFrame(imps)
    fig2 = px.bar(imp_df, x="importance", y="feature", orientation="h", title="Top feature importances (Random Forest)")
    st.plotly_chart(fig2, use_container_width=True)

st.markdown(
    """
    **Interpretation guidance**
    - **Accuracy** summarizes overall agreement but can mask minority-class behaviour.
    - **Precision** captures how trustworthy positive (pass) predictions are.
    - **Recall** tracks sensitivity to actual passes (important when supporting at-risk students).
    - **F1** balances precision and recall; it was used as the primary selector for deployment.
    """
)
