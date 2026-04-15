"""Student dashboard, ML prediction, and bundle risk analysis."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.ml_pipeline import load_model_bundle, predict_pass_probability
from utils.paths import DATASET_CSV, METRICS_PATH, MODEL_PATH
from utils.risk_analysis import RiskAssessmentService, build_program_course_index

st.title("Student workspace & prediction")

if "gpa" not in st.session_state:
    st.session_state.update(
        {
            "gpa": 2.7,
            "program": "Mining Engineering",
            "course_level": 300,
            "attendance": 85.0,
        }
    )


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATASET_CSV)


@st.cache_resource
def bundle():
    return load_model_bundle(MODEL_PATH)


df = load_data()
model = bundle()
catalog = build_program_course_index(df)
programs = sorted(df["program"].unique())
levels = [100, 200, 300, 400]

with st.sidebar:
    st.header("Academic profile")
    st.session_state["gpa"] = st.slider(
        "Current GPA", 1.5, 4.0, float(st.session_state["gpa"]), 0.05
    )
    prog_idx = (
        programs.index(st.session_state["program"])
        if st.session_state["program"] in programs
        else 0
    )
    st.session_state["program"] = st.selectbox("Program", programs, index=prog_idx)
    cl = int(st.session_state["course_level"])
    lvl_idx = levels.index(cl) if cl in levels else 2
    st.session_state["course_level"] = st.selectbox("Course level focus", levels, index=lvl_idx)
    st.session_state["attendance"] = st.slider(
        "Typical attendance (%)", 50.0, 100.0, float(st.session_state["attendance"]), 1.0
    )

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("GPA", f"{st.session_state['gpa']:.2f}")
with c2:
    st.metric("Program", st.session_state["program"].split()[0])
with c3:
    st.metric("Level", int(st.session_state["course_level"]))

st.subheader("Academic summary")
hist = df[df["program"] == st.session_state["program"]]
st.write(
    {
        "records_in_program_sample": int(len(hist)),
        "mean_pass_rate": float(hist["pass_fail"].mean()) if len(hist) else None,
        "mean_difficulty": float(hist["course_difficulty"].mean()) if len(hist) else None,
    }
)

st.divider()
st.subheader("Pass probability for a single course")
course_options = sorted(df["course_id"].unique())
choice = st.selectbox("Course", course_options)
row_meta = df[df["course_id"] == choice].iloc[0]
difficulty = st.number_input(
    "Course difficulty (1-10)",
    min_value=1.0,
    max_value=10.0,
    value=float(row_meta["course_difficulty"]),
    step=0.1,
)
attempts = st.number_input("Attempts", min_value=1, max_value=4, value=1)
credits = int(st.number_input("Credit load", min_value=1, max_value=6, value=int(row_meta["credit_load"])))

features = {
    "gpa": float(st.session_state["gpa"]),
    "course_difficulty": float(difficulty),
    "course_level": int(st.session_state["course_level"]),
    "attempts": int(attempts),
    "attendance": float(st.session_state["attendance"]),
    "credit_load": credits,
    "program": st.session_state["program"],
    "course_id": choice,
}

if st.button("Run prediction", type="primary"):
    prob = predict_pass_probability(model, features)
    st.success(f"Modeled pass probability: **{prob:.1%}**")
    bullets = [
        f"GPA signal: {'supportive' if st.session_state['gpa'] >= 3.0 else 'elevates modeled risk'}.",
        f"Difficulty {difficulty:.1f} relative to catalog for {choice}.",
        f"Attendance at {st.session_state['attendance']:.0f}% shifts readiness.",
        "Additional attempts increase modeled hazard when >1.",
    ]
    for b in bullets:
        st.write(f"- {b}")

st.divider()
st.subheader("Risk analysis for a course bundle")
defaults = st.multiselect(
    "Select multiple courses",
    options=course_options,
    default=course_options[: min(3, len(course_options))],
)
if st.button("Assess bundle risk"):
    risk_svc = RiskAssessmentService(catalog, model)
    report = risk_svc.assess(
        program=st.session_state["program"],
        gpa=float(st.session_state["gpa"]),
        course_level=int(st.session_state["course_level"]),
        attendance=float(st.session_state["attendance"]),
        course_ids=defaults,
    )
    band = report["risk_band"]
    if band == "Low risk":
        st.success(band)
    elif band == "Medium risk":
        st.warning(band)
    else:
        st.error(band)
    for w in report["warnings"]:
        st.warning(w)
    st.dataframe(pd.DataFrame(report["details"]), use_container_width=True)

if METRICS_PATH.exists():
    meta = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    st.caption(f"Production estimator in bundle: **{meta['selected_model']}**")
