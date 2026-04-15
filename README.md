# Smart Undergraduate Course Navigator (UMaT) - Streamlit

Python-only intelligent decision support system for **course selection** and **academic performance** optimization at the **University of Mines and Technology (UMaT)**. The UI is built with **Streamlit**; models use **scikit-learn**; visuals use **Plotly** and **Matplotlib**.

**GitHub repository:** https://github.com/dogahwisdom/smart-course-navigator

## Features

- **Student workspace & prediction:** GPA, program, level, attendance; single-course pass probability; bundle risk (low / medium / high).
- **Course analytics:** pass rate, trail rate, difficulty, Plotly charts.
- **Recommendations:** credit-aware semester suggestions with cohort + ML-based reasoning.
- **Model performance:** accuracy, precision, recall, F1; model comparison; Random Forest feature importance.
- **Full ML workflow:** preprocessing, class imbalance handling via `class_weight`, stratified split, joblib persistence.
- **Academic report:** Word document under `report/` with embedded diagrams.

## Repository layout

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
│   └── evaluation_metrics.json # metrics + feature importance
├── notebooks/
│   └── training.ipynb
├── scripts/
│   └── train.py
├── report/
│   ├── diagram_generator.py
│   ├── build_docx.py
│   └── Smart_Course_Navigator_Streamlit_Report.docx
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

## Regenerate data, train models, and rebuild the report

```bash
python data/generate_dataset.py --rows 1500
python scripts/train.py
python report/diagram_generator.py
python report/build_docx.py
```

Metrics and the **evaluation summary** are written to `models/evaluation_metrics.json`.

## Run the Streamlit app

From this directory (with the virtual environment activated):

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (typically `http://localhost:8501`). Use the sidebar to switch between **Student Prediction**, **Course Analytics**, **Recommendation System**, and **Model Performance**.

## Jupyter notebook

Open `notebooks/training.ipynb` and execute cells **with the working directory set to `smart-course-navigator/`** so `utils` imports resolve (the first cell adjusts `sys.path` if needed).

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

## Screenshots for your binder

Captured app screenshots are stored under `screenshots/app/` and are embedded automatically when generating the Word report. The current set includes:

- `01_home.png`
- `02_student_prediction.png`
- `03_course_analytics_overview.png`
- `04_course_analytics_charts.png`
- `05_recommendation_inputs.png`
- `06_model_performance_top.png`
- `07_model_performance_detail.png`

You can add notebook screenshots next (for appendices) and they will be integrated in a follow-up update.

## Ethics

Synthetic data are for **academic demonstration** only. Replace with governed, anonymized registrar extracts before any institutional deployment.
