"""
Smart Undergraduate Course Navigator — UMaT (Streamlit home).
Run from project root: streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Smart Course Navigator | UMaT",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Smart Undergraduate Course Navigator")
st.caption("Intelligent decision support for course selection and academic performance optimization at UMaT")

col1, col2 = st.columns(2)
with col1:
    st.markdown(
        """
        ### What this system does
        - **Predict** modeled pass probabilities using GPA, difficulty, attendance, attempts, and load.
        - **Analyze** cohort-level pass/trail rates and difficulty distributions.
        - **Recommend** semester plans that respect credit ceilings and risk thresholds.
        - **Benchmark** Logistic Regression, Decision Tree, and Random Forest with full metrics.
        """
    )
with col2:
    st.info(
        "Use the sidebar pages to move between **Student & Prediction**, **Course Analytics**, "
        "**Recommendations**, and **Model Performance**. "
        "Synthetic data are provided for academic demonstration—replace with governed registrar extracts "
        "before any production deployment."
    )

st.divider()
st.markdown(
    """
    **Repository (placeholder):** https://github.com/dogahwisdom/smart-course-navigator  
    **Tech stack:** Python · Streamlit · scikit-learn · pandas · plotly/matplotlib · joblib
    """
)
