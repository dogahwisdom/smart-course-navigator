"""Semester planning with reasoning strings."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.ml_pipeline import load_model_bundle
from utils.paths import DATASET_CSV, MODEL_PATH
from utils.recommendations import RecommendationEngine


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATASET_CSV)


@st.cache_resource
def bundle():
    return load_model_bundle(MODEL_PATH)


df = load_data()
model = bundle()
st.title("Recommendation system")

col_a, col_b = st.columns(2)
with col_a:
    program = st.selectbox("Program", sorted(df["program"].unique()))
with col_b:
    cap = st.slider("Credit ceiling", 12, 22, 18)

gpa = st.slider("Assumed GPA", 1.5, 4.0, 2.8, 0.05)
level = st.selectbox("Target course level", [200, 300, 400], index=1)
attendance = st.slider("Attendance (%)", 50.0, 100.0, 86.0, 1.0)

if st.button("Generate recommendations", type="primary"):
    engine = RecommendationEngine(df, model, max_credits=cap)
    plan = engine.suggest(program=program, gpa=gpa, course_level=level, attendance=attendance)
    st.metric("Planned credits", plan["planned_credits"])
    for item in plan["courses"]:
        with st.expander(f"{item['course_id']} — {item['course_name']}"):
            st.write(item["reasoning"])
            st.write(
                {
                    "historical_pass_rate": item["historical_pass_rate"],
                    "modeled_pass_probability": item["pass_probability"],
                    "credits": item["credits"],
                }
            )
