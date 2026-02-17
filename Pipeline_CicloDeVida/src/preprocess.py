from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import (
    DATA_DIR, RANDOM_STATE,
    COL_GPA, COL_STRESS, COL_ID, COL_HOURS
)

PROCESSED_DIR = DATA_DIR / "processed"


@dataclass
class PreprocessOutput:
    X: pd.DataFrame
    y_gpa: pd.Series
    y_stress: pd.Series
    stress_mapping: Dict[str, int]
    scaler: StandardScaler


def ensure_processed_dir() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _select_features(df: pd.DataFrame) -> pd.DataFrame:
    # Usamos solo hábitos como features
    feature_cols = [c for c in sorted(COL_HOURS) if c in df.columns]
    if not feature_cols:
        raise ValueError("No encontré columnas de hábitos para features (COL_HOURS).")
    X = df[feature_cols].copy()
    return X


def _clean_basic(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()

    # Eliminar ID si existe
    if COL_ID in df2.columns:
        df2 = df2.drop(columns=[COL_ID])

    # Eliminar duplicados
    df2 = df2.drop_duplicates()

    return df2


def _encode_stress_ordinal(s: pd.Series) -> Tuple[pd.Series, Dict[str, int]]:
    """
    Convierte Stress Level a ordinal: Low=0, Moderate=1, High=2.
    Si los valores no están exactamente así, mapea por orden alfabético como fallback.
    """
    s_str = s.astype(str).str.strip()

    preferred = {"Low": 0, "Moderate": 1, "High": 2}
    unique_vals = set(s_str.unique())

    if unique_vals.issubset(set(preferred.keys())):
        mapping = preferred
    else:
        # Fallback: orden alfabético (menos ideal, pero evita romper el pipeline)
        classes = sorted(s_str.unique())
        mapping = {c: i for i, c in enumerate(classes)}

    encoded = s_str.map(mapping)
    return encoded, mapping


def preprocess(df: pd.DataFrame) -> PreprocessOutput:
    ensure_processed_dir()

    df2 = _clean_basic(df)

    # Verificar columnas objetivo
    if COL_GPA not in df2.columns:
        raise ValueError(f"No encontré la columna objetivo de GPA: '{COL_GPA}'")
    if COL_STRESS not in df2.columns:
        raise ValueError(f"No encontré la columna objetivo de estrés: '{COL_STRESS}'")

    # Features
    X = _select_features(df2)

    # Targets
    y_gpa = pd.to_numeric(df2[COL_GPA], errors="coerce")
    y_stress_raw = df2[COL_STRESS]

    # Manejo simple de nulos: eliminar filas incompletas
    # (Si tu dataset no tiene nulos, no cambia nada)
    mask = X.notna().all(axis=1) & y_gpa.notna() & y_stress_raw.notna()
    X = X.loc[mask].copy()
    y_gpa = y_gpa.loc[mask].copy()
    y_stress_raw = y_stress_raw.loc[mask].copy()

    # Codificar estrés
    y_stress, mapping = _encode_stress_ordinal(y_stress_raw)

    # Escalamiento (para features numéricas)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    # Guardar dataset procesado (opcional)
    processed_df = X_scaled.copy()
    processed_df["GPA"] = y_gpa.values
    processed_df["Stress_Level_Ordinal"] = y_stress.values
    processed_df.to_csv(PROCESSED_DIR / "dataset_processed.csv", index=False, encoding="utf-8")

    return PreprocessOutput(
        X=X_scaled,
        y_gpa=y_gpa,
        y_stress=y_stress,
        stress_mapping=mapping,
        scaler=scaler,
    )


def split_for_regression(out: PreprocessOutput, test_size: float = 0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        out.X, out.y_gpa, test_size=test_size, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


def split_for_classification(out: PreprocessOutput, test_size: float = 0.2):
    # Stratify para mantener proporciones de clases
    X_train, X_test, y_train, y_test = train_test_split(
        out.X, out.y_stress,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=out.y_stress
    )
    return X_train, X_test, y_train, y_test
