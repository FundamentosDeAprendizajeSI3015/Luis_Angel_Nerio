from pathlib import Path

# Rutas base
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
REPORTS_DIR = BASE_DIR / "reports"
RESULTS_DIR = REPORTS_DIR / "results"
FIGURES_DIR = REPORTS_DIR / "figures"

# Reproducibilidad
RANDOM_STATE = 42

# Columnas exactas del dataset
COL_ID = "Student_ID"
COL_GPA = "GPA"
COL_STRESS = "Stress_Level"

COL_HOURS = {
    "Study_Hours_Per_Day",
    "Extracurricular_Hours_Per_Day",
    "Sleep_Hours_Per_Day",
    "Social_Hours_Per_Day",
    "Physical_Activity_Hours_Per_Day",
}

# Rangos esperados (heurísticos)
RANGE_HOURS_MIN = 0
RANGE_HOURS_MAX = 24
RANGE_GPA_MIN = 0.0
RANGE_GPA_MAX = 4.0

# Analisis de correlacion
CORR_THRESHOLD = 0.7
SPECIFIC_CORR_PAIRS = [
    ("Pclass", "Fare"),
    ("Age", "Fare"),
    ("SibSp", "Parch"),
]

# Transformacion logaritmica
SKEWNESS_THRESHOLD = 1.0
