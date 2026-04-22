# Smart Undergraduate Course Navigator (UMaT) - Streamlit

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/stable/)
[![License](https://img.shields.io/badge/License-Academic-lightgrey.svg)](#ethics)

Python-only intelligent decision support system for **course selection** and **academic performance optimization** at the **University of Mines and Technology (UMaT)**.  
The interface is built with **Streamlit**, machine learning is powered by **scikit-learn**, and visual analytics use **Plotly** and **Matplotlib**.

**GitHub repository:** https://github.com/dogahwisdom/smart-course-navigator

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Regenerate Data, Train Models, and Rebuild the Report](#regenerate-data-train-models-and-rebuild-the-report)
- [Run the Streamlit App](#run-the-streamlit-app)
- [Jupyter Notebook](#jupyter-notebook)
- [Contributors](#contributors)
- [Screenshots for Your Binder](#screenshots-for-your-binder)
- [Ethics](#ethics)

## Features

- **Student workspace & prediction:** GPA, program, level, attendance; single-course pass probability; bundle risk (low / medium / high).
- **Course analytics:** pass rate, trail rate, difficulty, Plotly charts.
- **Recommendations:** credit-aware semester suggestions with cohort + ML-based reasoning.
- **Model performance:** accuracy, precision, recall, F1; model comparison; Random Forest feature importance.
- **Full ML workflow:** preprocessing, class imbalance handling via `class_weight`, stratified split, joblib persistence.
- **Academic report:** generated in both Word and PDF under `report/`, with embedded diagrams and screenshots.

## Quick Start

```bash
cd smart-course-navigator
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`).

## Repository Layout

```
smart-course-navigator/
├── app.py                      # Home page
├── pages/                      # Multi-page Streamlit app
├── utils/                      # preprocessing, ML, recommendations, risk
├── data/
│   ├── generate_dataset.py
│   └── dataset.csv             # generated (≥1000 rows)
├── models/
│   ├── trained_model.joblib    # best pipeline + metadata
│   ├── trained_model.pkl       # compatibility copy
│   └── evaluation_metrics.json # metrics + feature importance
├── notebooks/
│   └── training.ipynb
├── scripts/
│   └── train.py
├── report/
│   ├── diagram_generator.py
│   ├── build_docx.py
│   ├── Smart_Course_Navigator_Streamlit_Report.docx
│   └── Smart_Course_Navigator_Streamlit_Report.pdf
├── presentation/
│   ├── Demo_Script_and_QA_Guide.docx
│   └── Demo_Script_and_QA_Guide.pdf
├── requirements.txt
└── README.md
```

## Installation

```bash
cd smart-course-navigator
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If `pip` times out on slow networks, retry with:

```bash
pip install --default-timeout=300 -r requirements.txt
```

## Regenerate Data, Train Models, and Rebuild the Report

```bash
python data/generate_dataset.py --rows 1500
python scripts/train.py
python report/diagram_generator.py
python report/build_docx.py
```

Metrics and the **evaluation summary** are written to `models/evaluation_metrics.json`.

## Run the Streamlit App

From `smart-course-navigator/` with the virtual environment activated:

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (typically `http://localhost:8501`).  
Use the sidebar to switch between **Student Prediction**, **Course Analytics**, **Recommendation System**, and **Model Performance**.

## Jupyter Notebook

`notebooks/training.ipynb` is team-friendly and can be executed with **Run All** on a fresh machine (after dependency install).

The notebook setup phase automatically:
- resolves project path imports
- checks/creates required folders
- generates `data/dataset.csv` if it is missing

### Run with JupyterLab (recommended for BI coursework)

```bash
cd smart-course-navigator
source .venv/bin/activate
jupyter lab
```

Then open `notebooks/training.ipynb` and run all cells.

## Contributors

| Name | Index Number |
| --- | --- |
| Wisdom Dogah | FCM.41.020.099.23 |
| Faustian Nyamekye | FCM.41.020.146.23 |
| Owusu Appiah Barimah Kofi Duodu | FCM.41.020.159.23 |
| De-Graft Prince Kweku | F.C.M. 41.020.098.23 |
| Cudjoe Jennifer Abena | FCM. 41.020.096.23 |
| Danquah Joseph | FCM.41.020.097.23 |
| Kelvin Kandibiga | FCM.41.020.118.23 |
| Emmanuel Cudjoe | F.CM.41.020.095.23 |

## Screenshots for Your Binder

App screenshots are stored under `screenshots/app/` and embedded automatically when generating the report:

- `01_home.png`
- `02_student_prediction.png`
- `03_course_analytics_overview.png`
- `04_course_analytics_charts.png`
- `05_recommendation_inputs.png`
- `06_model_performance_top.png`
- `07_model_performance_detail.png`

Notebook evidence screenshots are stored under `screenshots/notebook/` and are embedded in the appendices section of the generated report.

## Ethics

Synthetic data are for **academic demonstration** only. Replace with governed, anonymized registrar extracts before any institutional deployment.
