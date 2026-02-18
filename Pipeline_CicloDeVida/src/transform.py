from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import category_encoders as ce

from .config import DATA_DIR, SKEWNESS_THRESHOLD

PROCESSED_DIR = DATA_DIR / "processed"


@dataclass
class TransformOutput:
    """Contenedor de outputs de transformaciones de features.
    
    Almacena múltiples versiones del dataset (base, onehot, label, binary)
    y metadatos sobre transformaciones aplicadas.
    """
    base_df: pd.DataFrame
    onehot_df: pd.DataFrame
    label_df: pd.DataFrame
    binary_df: pd.DataFrame
    output_paths: Dict[str, Path]
    label_mappings: Dict[str, Dict[str, int]]
    log_transform_cols: List[str]
    log_skew_report: Dict[str, Dict[str, float]]
    categorical_columns: List[str]
    engineered_features: List[str]


def ensure_processed_dir() -> None:
    """Crea directorio processed si no existe."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    """Retorna lista de nombres de columnas con tipo numérico.
    
    Args:
        df: DataFrame a inspeccionar.
    
    Returns:
        Lista de nombres de columnas numéricas.
    """
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _categorical_columns(df: pd.DataFrame) -> List[str]:
    """Retorna lista de nombres de columnas categóricas (objeto, bool, categoría).
    
    Args:
        df: DataFrame a inspeccionar.
    
    Returns:
        Lista de nombres de columnas categóricas.
    """
    out = []
    for c in df.columns:
        if pd.api.types.is_bool_dtype(df[c]):
            out.append(c)
        elif pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c]):
            out.append(c)
    return out


def _apply_feature_engineering(df: pd.DataFrame) -> List[str]:
    """Crea nuevas features a partir de variables existentes.
    
    Para Titanic: crea FamilySize (SibSp + Parch + 1) e IsAlone (FamilySize == 1).
    
    Args:
        df: DataFrame a enriquecer.
    
    Returns:
        Lista de nombres de features nuevas creadas.
    """
    engineered = []
    if "SibSp" in df.columns and "Parch" in df.columns:
        df["FamilySize"] = pd.to_numeric(df["SibSp"], errors="coerce") + pd.to_numeric(df["Parch"], errors="coerce") + 1
        df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
        engineered.extend(["FamilySize", "IsAlone"])
    return engineered


def _apply_log_transform(df: pd.DataFrame) -> tuple[List[str], Dict[str, Dict[str, float]]]:
    """Aplica transformación log10 a columnas numéricas con alta asimetría.
    
    Sólo aplica log a columnas con:
    - Asimetría (|skew|) > SKEWNESS_THRESHOLD
    - Sin valores negativos
    
    Args:
        df: DataFrame a transformar.
    
    Returns:
        Tupla (lista_columnas_transformadas, reporte_asimetría_antes_despues).
    """
    log_cols = []
    skew_report: Dict[str, Dict[str, float]] = {}

    for col in _numeric_columns(df):
        series = pd.to_numeric(df[col], errors="coerce")
        series_no_na = series.dropna()
        if len(series_no_na) == 0:
            continue

        skew_val = float(series_no_na.skew())
        # Solo si la asimetría es significativa
        if abs(skew_val) < SKEWNESS_THRESHOLD:
            continue

        # No aplicar log a valores negativos
        if (series_no_na < 0).any():
            continue

        # Aplicar transformación log10
        log_col = f"{col}_log10"
        df[log_col] = np.log10(series + 1)
        log_cols.append(col)

        # Calcular asimetría después de transformación
        log_skew = float(df[log_col].dropna().skew()) if df[log_col].dropna().size > 0 else 0.0
        skew_report[col] = {
            "skew_before": skew_val,
            "skew_after": log_skew,
        }

    return log_cols, skew_report


def _label_encode(df: pd.DataFrame, cat_cols: List[str]) -> tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """Codifica variables categóricas con etiquetas numéricas (0, 1, 2, ...).
    
    Ordena alfabeticamente y asigna enteros secuenciales.
    
    Args:
        df: DataFrame a codificar.
        cat_cols: Lista de columnas categóricas a codificar.
    
    Returns:
        Tupla (df_codificado, diccionario_mapeos_categorias).
    """
    out = df.copy()
    mappings: Dict[str, Dict[str, int]] = {}

    for col in cat_cols:
        series = out[col].astype(str).str.strip()
        # Orden alfabético de clases
        classes = sorted([c for c in series.dropna().unique().tolist()])
        mapping = {cls: i for i, cls in enumerate(classes)}
        mappings[col] = mapping
        out[col] = series.map(mapping).astype("Int64")

    return out, mappings


def _one_hot_encode(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    """Aplica One-Hot Encoding a variables categóricas.
    
    Crea una columna binaria por cada categoría única.
    
    Args:
        df: DataFrame a codificar.
        cat_cols: Lista de columnas categóricas.
    
    Returns:
        DataFrame con one-hot encoding aplicado.
    """
    if not cat_cols:
        return df.copy()
    return pd.get_dummies(df, columns=cat_cols, drop_first=False)


def _binary_encode(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    """Aplica Binary Encoding (category_encoders) a variables categóricas.
    
    Más compacto que one-hot para categorías con muchas clases.
    
    Args:
        df: DataFrame a codificar.
        cat_cols: Lista de columnas categóricas.
    
    Returns:
        DataFrame con binary encoding aplicado.
    """
    if not cat_cols:
        return df.copy()
    encoder = ce.BinaryEncoder(cols=cat_cols, return_df=True)
    return encoder.fit_transform(df)


def _build_output_path(csv_path: Optional[Path], suffix: str) -> Path:
    """Construye path de salida con sufijo descriptivo.
    
    Args:
        csv_path: Path del CSV original (puede ser None).
        suffix: Sufijo a añadir antes de .csv (ej: '_onehot', '_label').
    
    Returns:
        Path completo para guardar el archivo transformado.
    """
    stem = csv_path.stem if csv_path is not None else "dataset"
    name = f"{stem}_transformado{suffix}.csv"
    return PROCESSED_DIR / name


def apply_transformations(df: pd.DataFrame, csv_path: Optional[Path] = None) -> TransformOutput:
    """Función principal: aplica todas las transformaciones de features.
    
    Pasos realizados:
    1. Feature engineering (si aplica, ej: FamilySize en Titanic)
    2. Transformación log de variables asimétricas
    3. Identificación de columnas categóricas
    4. Aplicación de tres métodos de encoding:
       - One-Hot: columnas binarias por categoría
       - Label: enteros secuenciales
       - Binary: encode binario compacto
    5. Guardado de cuatro versiones del dataset
    
    Args:
        df: DataFrame a transformar.
        csv_path: Path al CSV original (para nombrar salidas).
    
    Returns:
        TransformOutput con todos los datasets y metadatos de transformaciones.
    """
    ensure_processed_dir()

    base_df = df.copy()
    engineered_features = _apply_feature_engineering(base_df)
    log_cols, skew_report = _apply_log_transform(base_df)

    cat_cols = _categorical_columns(base_df)
    onehot_df = _one_hot_encode(base_df, cat_cols)
    label_df, mappings = _label_encode(base_df, cat_cols)
    binary_df = _binary_encode(base_df, cat_cols)

    # Construir rutas de salida
    output_paths = {
        "base": _build_output_path(csv_path, ""),
        "onehot": _build_output_path(csv_path, "_onehot"),
        "label": _build_output_path(csv_path, "_label"),
        "binary": _build_output_path(csv_path, "_binary"),
    }

    # Guardar cada versión
    base_df.to_csv(output_paths["base"], index=False, encoding="utf-8")
    onehot_df.to_csv(output_paths["onehot"], index=False, encoding="utf-8")
    label_df.to_csv(output_paths["label"], index=False, encoding="utf-8")
    binary_df.to_csv(output_paths["binary"], index=False, encoding="utf-8")

    return TransformOutput(
        base_df=base_df,
        onehot_df=onehot_df,
        label_df=label_df,
        binary_df=binary_df,
        output_paths=output_paths,
        label_mappings=mappings,
        log_transform_cols=log_cols,
        log_skew_report=skew_report,
        categorical_columns=cat_cols,
        engineered_features=engineered_features,
    )
