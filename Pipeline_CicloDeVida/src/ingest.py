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
    """Contenedor de información de perfil del dataset cargado.
    
    Almacena resumen estadístico, tipos de datos, nulos, duplicados y validaciones de rango.
    """
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
    """Crea directorio de resultados si no existe."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def find_first_csv(raw_dir: Path = RAW_DIR) -> Path:
    """Busca y retorna el primer archivo CSV en la carpeta raw.
    
    Args:
        raw_dir: Ruta al directorio con datos crudos.
    
    Returns:
        Path al archivo CSV encontrado.
    
    Raises:
        FileNotFoundError: Si no existe la carpeta o no hay archivos CSV.
    """
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
    """Calcula estadísticas básicas descriptivas de columnas numéricas.
    
    Retorna count, mean, std, min, percentiles (25%, 50%, 75%) y max.
    
    Args:
        df: DataFrame a analizar.
    
    Returns:
        Diccionario con estadísticas por columna numérica.
    """
    num = df.select_dtypes(include=["number"])
    if num.shape[1] == 0:
        return {}
    # Calcular describe (count, mean, std, min, 25%, 50%, 75%, max)
    desc = num.describe().T
    # Convertir a diccionario simple para serialización JSON
    out: Dict[str, Dict[str, float]] = {}
    for col in desc.index:
        out[col] = {k: float(desc.loc[col, k]) for k in desc.columns}
    return out


def _check_ranges(df: pd.DataFrame) -> Dict[str, Any]:
    """Valida que los valores estén dentro de rangos esperados.
    
    Verifica:
    - GPA dentro de [0.0, 4.0]
    - Horas dentro de [0, 24]
    - Valores únicos de Stress Level
    - Cantidad de IDs únicos
    
    Args:
        df: DataFrame a validar.
    
    Returns:
        Diccionario con resultados de validaciones y anomalías encontradas.
    """
    issues: Dict[str, Any] = {}

    # Chequeo GPA: validar rango [0.0, 4.0]
    if COL_GPA in df.columns:
        gpa = pd.to_numeric(df[COL_GPA], errors="coerce")
        bad = df[(gpa < RANGE_GPA_MIN) | (gpa > RANGE_GPA_MAX)]
        issues["gpa_out_of_range_count"] = int(bad.shape[0])
    else:
        issues["gpa_out_of_range_count"] = None
        issues["gpa_note"] = f"No encontré columna '{COL_GPA}'"

    # Chequeo horas: validar rango [0, 24] para cada tipo de hora
    hours_issues = {}
    for c in COL_HOURS:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce")
            bad_count = int(((x < RANGE_HOURS_MIN) | (x > RANGE_HOURS_MAX)).sum(skipna=True))
            hours_issues[c] = bad_count
        else:
            hours_issues[c] = None
    issues["hours_out_of_range_by_col"] = hours_issues

    # Chequeo Stress: listar valores únicos
    if COL_STRESS in df.columns:
        issues["stress_unique_values"] = sorted(df[COL_STRESS].dropna().astype(str).unique().tolist())
    else:
        issues["stress_unique_values"] = None

    # Chequeo ID: contar IDs únicos
    if COL_ID in df.columns:
        issues["id_unique_count"] = int(df[COL_ID].nunique(dropna=True))
    else:
        issues["id_unique_count"] = None

    return issues


def build_overview(df: pd.DataFrame, csv_path: Path) -> DataOverview:
    """Construye un perfil completo del dataset: dimensiones, tipos, nulos y validaciones.
    
    Agrupa información de:
    - Dimensiones y nombre de columnas
    - Tipos de datos
    - Conteo de nulos por columna
    - Duplicados
    - Estadísticas descriptivas numéricas
    - Validaciones de rango
    
    Args:
        df: DataFrame a perfilar.
        csv_path: Ruta del archivo CSV cargado.
    
    Returns:
        DataOverview con información completa del dataset.
    """
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
    """Guarda el perfil del dataset en JSON y CSV.
    
    Genera dos archivos:
    - JSON: Perfil completo con todos los detalles
    - CSV: Resumen ejecutivo rápido
    
    Args:
        overview: DataOverview a guardar.
    
    Returns:
        Tupla con rutas a (json_path, csv_path).
    """
    ensure_dirs()
    json_path = RESULTS_DIR / "data_overview.json"
    csv_path = RESULTS_DIR / "data_overview.csv"

    # Guardar perfil completo en JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(overview.__dict__, f, ensure_ascii=False, indent=2)

    # Guardar resumen ejecutivo en CSV (métricas clave solamente)
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
    """Función principal: carga dataset, lo perfila y guarda resumen.
    
    Orquesta la ingesta de datos:
    1. Encuentra o recibe path a CSV
    2. Carga el dataset (intenta , en separadores si es necesario)
    3. Construye perfil completo (estadísticas, validaciones)
    4. Guarda resumen en JSON y CSV
    
    Args:
        csv_path: Ruta opcional al CSV. Si es None, busca primer CSV en raw_dir.
    
    Returns:
        Diccionario con dataframe, overview y rutas de archivos generados.
    """
    # Si no proporcionan path, buscar primer CSV disponible
    if csv_path is None:
        csv_path = find_first_csv()

    # Cargar dataset desde CSV
    df = load_dataset(csv_path)
    
    # Construir perfil completo
    overview = build_overview(df, csv_path)
    
    # Guardar resumen en archivos
    json_path, csv_summary_path = save_overview(overview)

    return {
        "df": df,
        "overview": overview,
        "json_path": json_path,
        "csv_summary_path": csv_summary_path,
        "csv_path": csv_path,
    }
