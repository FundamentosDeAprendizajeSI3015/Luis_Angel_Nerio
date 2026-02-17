from dataclasses import dataclass, field
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from .config import COL_GPA, COL_STRESS, COL_HOURS, CORR_THRESHOLD, SPECIFIC_CORR_PAIRS


@dataclass
class EDAReport:
    numeric_summary: Dict[str, Any]
    stress_distribution: Dict[str, int]
    gpa_by_stress: Dict[str, float]
    statistical_measures: Dict[str, Any] = field(default_factory=dict)
    outlier_analysis: Dict[str, Any] = field(default_factory=dict)
    percentiles_deciles: Dict[str, Any] = field(default_factory=dict)
    correlation_analysis: Dict[str, Any] = field(default_factory=dict)
    skewness_analysis: Dict[str, Any] = field(default_factory=dict)


def _get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Obtiene columnas numéricas del dataframe."""
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _calculate_statistical_measures(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
    """Calcula media, mediana, moda, desv. estándar, min, max."""
    measures = {}
    
    for col in numeric_cols:
        data = df[col].dropna()
        measures[col] = {
            "media": float(data.mean()),
            "mediana": float(data.median()),
            "moda": float(data.mode()[0]) if len(data.mode()) > 0 else None,
            "desv_estandar": float(data.std()),
            "minimo": float(data.min()),
            "maximo": float(data.max()),
            "varianza": float(data.var()),
        }
    
    return measures


def _calculate_quartiles_iqr(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
    """Calcula cuartiles (Q1, Q2, Q3), IQR y límites para outliers."""
    quartiles = {}
    
    for col in numeric_cols:
        data = df[col].dropna()
        Q1 = data.quantile(0.25)
        Q2 = data.quantile(0.50)  # mediana
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        quartiles[col] = {
            "Q1": float(Q1),
            "Q2_mediana": float(Q2),
            "Q3": float(Q3),
            "IQR": float(IQR),
            "limite_inferior_outliers": float(lower_bound),
            "limite_superior_outliers": float(upper_bound),
        }
    
    return quartiles


def _calculate_percentiles_deciles(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
    """Calcula percentiles (P10, P50, P70, P90) y deciles para cada variable numérica."""
    percentiles_deciles = {}
    
    for col in numeric_cols:
        data = df[col].dropna()
        
        # Percentiles
        percentiles = {
            "P10": float(data.quantile(0.10)),
            "P25": float(data.quantile(0.25)),
            "P50": float(data.quantile(0.50)),
            "P75": float(data.quantile(0.75)),
            "P90": float(data.quantile(0.90)),
        }
        
        # Deciles (10%, 20%, ..., 90%)
        deciles = {
            f"D{i}": float(data.quantile(i / 10))
            for i in range(1, 10)
        }
        
        # Quintiles (20%, 40%, 60%, 80%)
        quintiles = {
            f"Q{i}": float(data.quantile(i / 5))
            for i in range(1, 5)
        }
        
        percentiles_deciles[col] = {
            "percentiles": percentiles,
            "deciles": deciles,
            "quintiles": quintiles,
        }
    
    return percentiles_deciles


def _detect_and_analyze_outliers(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
    """Detecta outliers usando regla 1.5 × IQR y analiza impacto."""
    outlier_analysis = {}
    
    for col in numeric_cols:
        data = df[col].dropna()
        
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identificar outliers
        outliers_mask = (data < lower_bound) | (data > upper_bound)
        num_outliers = outliers_mask.sum()
        outliers_values = data[outliers_mask].tolist()
        
        # Estadísticas ANTES de eliminar outliers
        stats_before = {
            "media": float(data.mean()),
            "mediana": float(data.median()),
            "desv_estandar": float(data.std()),
            "varianza": float(data.var()),
            "minimo": float(data.min()),
            "maximo": float(data.max()),
        }
        
        # Estadísticas DESPUÉS de eliminar outliers
        data_without_outliers = data[~outliers_mask]
        stats_after = {
            "media": float(data_without_outliers.mean()) if len(data_without_outliers) > 0 else None,
            "mediana": float(data_without_outliers.median()) if len(data_without_outliers) > 0 else None,
            "desv_estandar": float(data_without_outliers.std()) if len(data_without_outliers) > 0 else None,
            "varianza": float(data_without_outliers.var()) if len(data_without_outliers) > 0 else None,
            "minimo": float(data_without_outliers.min()) if len(data_without_outliers) > 0 else None,
            "maximo": float(data_without_outliers.max()) if len(data_without_outliers) > 0 else None,
        }
        
        # Impacto (cambio en %)
        impacto = {}
        for metric in stats_before.keys():
            if stats_before[metric] is not None and stats_after[metric] is not None:
                cambio_abs = abs(stats_after[metric] - stats_before[metric])
                cambio_pct = (cambio_abs / abs(stats_before[metric]) * 100) if stats_before[metric] != 0 else 0
                impacto[metric] = {
                    "cambio_absoluto": float(cambio_abs),
                    "cambio_porcentaje": float(cambio_pct),
                }
        
        outlier_analysis[col] = {
            "cantidad_outliers": int(num_outliers),
            "porcentaje_outliers": float((num_outliers / len(data) * 100) if len(data) > 0 else 0),
            "valores_outliers": [float(x) for x in outliers_values],
            "limite_inferior": float(lower_bound),
            "limite_superior": float(upper_bound),
            "estadisticas_antes": stats_before,
            "estadisticas_despues": stats_after,
            "impacto_remocion": impacto,
        }
    
    return outlier_analysis


def _calculate_correlations(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
    if len(numeric_cols) < 2:
        return {
            "pearson": {},
            "spearman": {},
            "high_corr_pairs": [],
            "specific_pairs": {},
        }

    corr_pearson = df[numeric_cols].corr(method="pearson", numeric_only=True)
    corr_spearman = df[numeric_cols].corr(method="spearman", numeric_only=True)

    high_pairs = []
    for i, col_i in enumerate(numeric_cols):
        for j in range(i + 1, len(numeric_cols)):
            col_j = numeric_cols[j]
            val = corr_pearson.loc[col_i, col_j]
            if pd.notna(val) and abs(val) >= CORR_THRESHOLD:
                high_pairs.append({
                    "col_1": col_i,
                    "col_2": col_j,
                    "corr": float(val),
                    "method": "pearson",
                })

    specific_pairs = {}
    for col_1, col_2 in SPECIFIC_CORR_PAIRS:
        if col_1 in df.columns and col_2 in df.columns:
            if pd.api.types.is_numeric_dtype(df[col_1]) and pd.api.types.is_numeric_dtype(df[col_2]):
                pearson_val = df[[col_1, col_2]].corr(method="pearson").iloc[0, 1]
                spearman_val = df[[col_1, col_2]].corr(method="spearman").iloc[0, 1]
                specific_pairs[f"{col_1}__{col_2}"] = {
                    "pearson": float(pearson_val),
                    "spearman": float(spearman_val),
                }

    return {
        "pearson": corr_pearson.round(4).to_dict(),
        "spearman": corr_spearman.round(4).to_dict(),
        "high_corr_pairs": high_pairs,
        "specific_pairs": specific_pairs,
    }


def _calculate_skewness(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, Any]:
    skewness = {}
    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) == 0:
            continue
        skewness[col] = float(data.skew())
    return skewness


def run_basic_eda(df: pd.DataFrame) -> EDAReport:
    # Obtener columnas numéricas
    numeric_cols = _get_numeric_columns(df)
    
    # Resumen numérico básico
    numeric_summary = {}
    if numeric_cols:
        desc = df[numeric_cols].describe().T
        numeric_summary = desc.to_dict(orient="index")

    # Distribución de estrés
    if COL_STRESS in df.columns:
        stress_distribution = df[COL_STRESS].astype(str).value_counts(dropna=False).to_dict()
    else:
        stress_distribution = {}

    # GPA por estrés (promedio)
    gpa_by_stress = {}
    if COL_GPA in df.columns and COL_STRESS in df.columns:
        gpa_by_stress = (
            df.groupby(COL_STRESS)[COL_GPA]
            .mean()
            .sort_index()
            .to_dict()
        )

    # Calcular medidas estadísticas avanzadas
    statistical_measures = _calculate_statistical_measures(df, numeric_cols)
    quartiles_iqr = _calculate_quartiles_iqr(df, numeric_cols)
    percentiles_deciles = _calculate_percentiles_deciles(df, numeric_cols)
    outlier_analysis = _detect_and_analyze_outliers(df, numeric_cols)
    correlation_analysis = _calculate_correlations(df, numeric_cols)
    skewness_analysis = _calculate_skewness(df, numeric_cols)
    
    return EDAReport(
        numeric_summary=numeric_summary,
        stress_distribution={str(k): int(v) for k, v in stress_distribution.items()},
        gpa_by_stress={str(k): float(v) for k, v in gpa_by_stress.items()},
        statistical_measures={col: statistical_measures.get(col, {}) for col in numeric_cols},
        outlier_analysis=outlier_analysis,
        percentiles_deciles={col: percentiles_deciles.get(col, {}) for col in numeric_cols},
        correlation_analysis=correlation_analysis,
        skewness_analysis=skewness_analysis,
    )
