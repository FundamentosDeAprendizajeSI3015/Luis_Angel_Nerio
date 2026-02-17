import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import pandas as pd

from .config import (
    RAW_DIR, RESULTS_DIR,
    COL_GPA, COL_STRESS, COL_ID, COL_HOURS,
    RANGE_HOURS_MIN, RANGE_HOURS_MAX,
    RANGE_GPA_MIN, RANGE_GPA_MAX
)

@dataclass
class DataOverview:
    file_path: str
    n_rows: int
    n_cols: int
    columns: list
    dtypes: Dict[str, str]
    missing_by_col: Dict[str, int]
    missing_total: int
    duplicate_rows: int
    basic_stats: Dict[str, Dict[str, float]]
    range_issues: Dict[str, Any]


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def find_first_csv(raw_dir: Path = RAW_DIR) -> Path:
    if not raw_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta: {raw_dir}")
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No encontré .csv en: {raw_dir}")
    return csv_files[0]


def load_dataset(csv_path: Path) -> pd.DataFrame:
    # Intenta leer con separador estándar; si falla, prueba con ';'
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.read_csv(csv_path, sep=";")
    return df


def _numeric_basic_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    num = df.select_dtypes(include=["number"])
    if num.shape[1] == 0:
        return {}
    desc = num.describe().T  # count, mean, std, min, 25%, 50%, 75%, max
    # Convertir a dict simple
    out: Dict[str, Dict[str, float]] = {}
    for col in desc.index:
        out[col] = {k: float(desc.loc[col, k]) for k in desc.columns}
    return out


def _check_ranges(df: pd.DataFrame) -> Dict[str, Any]:
    issues: Dict[str, Any] = {}

    # Chequeo GPA
    if COL_GPA in df.columns:
        gpa = pd.to_numeric(df[COL_GPA], errors="coerce")
        bad = df[(gpa < RANGE_GPA_MIN) | (gpa > RANGE_GPA_MAX)]
        issues["gpa_out_of_range_count"] = int(bad.shape[0])
    else:
        issues["gpa_out_of_range_count"] = None
        issues["gpa_note"] = f"No encontré columna '{COL_GPA}'"

    # Chequeo horas
    hours_issues = {}
    for c in COL_HOURS:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce")
            bad_count = int(((x < RANGE_HOURS_MIN) | (x > RANGE_HOURS_MAX)).sum(skipna=True))
            hours_issues[c] = bad_count
        else:
            hours_issues[c] = None
    issues["hours_out_of_range_by_col"] = hours_issues

    # Chequeo Stress
    if COL_STRESS in df.columns:
        issues["stress_unique_values"] = sorted(df[COL_STRESS].dropna().astype(str).unique().tolist())
    else:
        issues["stress_unique_values"] = None

    # Chequeo ID
    if COL_ID in df.columns:
        issues["id_unique_count"] = int(df[COL_ID].nunique(dropna=True))
    else:
        issues["id_unique_count"] = None

    return issues


def build_overview(df: pd.DataFrame, csv_path: Path) -> DataOverview:
    missing_by_col = df.isna().sum().to_dict()
    overview = DataOverview(
        file_path=str(csv_path),
        n_rows=int(df.shape[0]),
        n_cols=int(df.shape[1]),
        columns=df.columns.tolist(),
        dtypes={c: str(t) for c, t in df.dtypes.items()},
        missing_by_col={k: int(v) for k, v in missing_by_col.items()},
        missing_total=int(df.isna().sum().sum()),
        duplicate_rows=int(df.duplicated().sum()),
        basic_stats=_numeric_basic_stats(df),
        range_issues=_check_ranges(df),
    )
    return overview


def save_overview(overview: DataOverview) -> Tuple[Path, Path]:
    ensure_dirs()
    json_path = RESULTS_DIR / "data_overview.json"
    csv_path = RESULTS_DIR / "data_overview.csv"

    # JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(overview.__dict__, f, ensure_ascii=False, indent=2)

    # CSV plano (resumen rápido)
    rows = [
        ("file_path", overview.file_path),
        ("n_rows", overview.n_rows),
        ("n_cols", overview.n_cols),
        ("missing_total", overview.missing_total),
        ("duplicate_rows", overview.duplicate_rows),
    ]
    pd.DataFrame(rows, columns=["metric", "value"]).to_csv(csv_path, index=False, encoding="utf-8")

    return json_path, csv_path


def ingest_and_profile(csv_path: Optional[Path] = None) -> Dict[str, Any]:
    if csv_path is None:
        csv_path = find_first_csv()

    df = load_dataset(csv_path)
    overview = build_overview(df, csv_path)
    json_path, csv_summary_path = save_overview(overview)

    return {
        "df": df,
        "overview": overview,
        "json_path": json_path,
        "csv_summary_path": csv_summary_path,
        "csv_path": csv_path,
    }
