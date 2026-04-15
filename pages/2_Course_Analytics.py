"""Course-level analytics and visualizations."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.app_context import build_context


ctx = build_context()
df = ctx.dataset
st.title("Course analytics")

agg = (
    df.groupby(["course_id", "course_name", "program"], as_index=False)
    .agg(
        pass_rate=("pass_fail", "mean"),
        difficulty=("course_difficulty", "mean"),
        enrollments=("pass_fail", "count"),
    )
    .assign(trail_rate=lambda x: 1 - x["pass_rate"])
    .sort_values("pass_rate")
)

st.dataframe(
    agg.rename(columns={"difficulty": "difficulty_score"}),
    use_container_width=True,
    hide_index=True,
)

c1, c2 = st.columns(2)
with c1:
    fig1 = px.bar(
        agg.head(12),
        x="course_id",
        y="pass_rate",
        color="program",
        title="Pass rate by course (top 12 lowest)",
        labels={"pass_rate": "Pass rate", "course_id": "Course"},
    )
    st.plotly_chart(fig1, use_container_width=True)

with c2:
    fig2 = px.histogram(df, x="course_difficulty", color="program", nbins=24, title="Difficulty distribution")
    st.plotly_chart(fig2, use_container_width=True)

sample = df.sample(min(800, len(df)), random_state=42)
scatter = px.scatter(
    sample,
    x="gpa",
    y="pass_fail",
    color="program",
    size="course_difficulty",
    title="GPA vs pass outcome (sample; marker size ~ difficulty)",
    labels={"pass_fail": "Pass (1) / Fail (0)"},
    height=480,
)
st.plotly_chart(scatter, use_container_width=True)

st.info("Trail rate is computed as 1 − pass rate for each aggregated course-program cohort.")
