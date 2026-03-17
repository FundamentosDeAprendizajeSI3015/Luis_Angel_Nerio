"""
Preprocesamiento para los dos datasets de semana 9.

Salida esperada:
- Un CSV limpio y listo para clustering por cada dataset original.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
	"""Limpia espacios y nombres de columnas para evitar inconsistencias."""
	df = df.copy()
	df.columns = [str(col).strip().lower() for col in df.columns]
	return df


def standardize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
	"""Normaliza representaciones de faltantes en columnas de texto."""
	df = df.copy()
	text_cols = df.select_dtypes(include=["object"]).columns
	missing_tokens = {
		"": np.nan,
		" ": np.nan,
		"na": np.nan,
		"n/a": np.nan,
		"null": np.nan,
		"none": np.nan,
		"nan": np.nan,
	}
	for col in text_cols:
		cleaned = df[col].astype(str).str.strip().str.lower().replace(missing_tokens)
		df[col] = cleaned
	return df


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Intenta convertir columnas object a numéricas cuando la gran mayoría de valores lo permita.
	"""
	df = df.copy()
	for col in df.columns:
		if df[col].dtype == "object":
			converted = pd.to_numeric(df[col], errors="coerce")
			ratio_numeric = converted.notna().mean()
			if ratio_numeric >= 0.8:
				df[col] = converted
	return df


def split_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
	"""Separa columnas numéricas y categóricas para tratarlas distinto."""
	numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
	categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
	return numeric_cols, categorical_cols


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
	"""Imputa faltantes con mediana (numéricas) y moda (categóricas)."""
	df = df.copy()
	numeric_cols, categorical_cols = split_feature_types(df)

	for col in numeric_cols:
		if df[col].isna().any():
			df[col] = df[col].fillna(df[col].median())

	for col in categorical_cols:
		if df[col].isna().any():
			mode_series = df[col].mode(dropna=True)
			fill_value = mode_series.iloc[0] if not mode_series.empty else "desconocido"
			df[col] = df[col].fillna(fill_value)

	return df


def cap_outliers_iqr(df: pd.DataFrame, k: float = 1.5) -> pd.DataFrame:
	"""Recorta outliers con regla IQR para estabilizar clustering."""
	df = df.copy()
	numeric_cols = df.select_dtypes(include=[np.number]).columns

	for col in numeric_cols:
		q1 = df[col].quantile(0.25)
		q3 = df[col].quantile(0.75)
		iqr = q3 - q1
		if pd.isna(iqr) or iqr == 0:
			continue
		lower = q1 - k * iqr
		upper = q3 + k * iqr
		df[col] = df[col].clip(lower=lower, upper=upper)

	return df


def encode_categorical_variables(df: pd.DataFrame) -> pd.DataFrame:
	"""Codifica variables categóricas en formato one-hot."""
	categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
	if not categorical_cols:
		return df
	return pd.get_dummies(df, columns=categorical_cols, drop_first=False, dtype=int)


def preprocess_dataset(file_path: Path, output_dir: Path) -> Path:
	"""Ejecuta limpieza completa para un dataset y guarda su CSV final."""
	print(f"\nProcesando: {file_path.name}")
	df = pd.read_csv(file_path)

	df = normalize_column_names(df)
	df = standardize_missing_values(df)
	df = coerce_numeric_columns(df)
	df = df.drop_duplicates().reset_index(drop=True)
	df = impute_missing_values(df)

	# Para clustering no supervisado no usamos la etiqueta objetivo.
	if "label" in df.columns:
		df = df.drop(columns=["label"])

	# Si existe anio, se excluye para no forzar agrupamientos por tiempo.
	if "anio" in df.columns:
		df = df.drop(columns=["anio"])

	df = cap_outliers_iqr(df)
	df = encode_categorical_variables(df)

	# Garantiza salida solo numérica y sin nulos.
	for col in df.columns:
		df[col] = pd.to_numeric(df[col], errors="coerce")
	df = df.fillna(df.median(numeric_only=True))

	output_path = output_dir / f"{file_path.stem}_preprocesado.csv"
	df.to_csv(output_path, index=False)

	print(f"Filas: {df.shape[0]} | Columnas finales: {df.shape[1]}")
	print(f"Salida: {output_path.name}")
	return output_path


def run_preprocessing() -> Dict[str, Path]:
	"""Procesa los dos datasets de semana 9 y retorna sus rutas de salida."""
	base_dir = Path(__file__).resolve().parent
	output_dir = base_dir / "datos_preprocesados"
	output_dir.mkdir(exist_ok=True)

	datasets = [
		base_dir / "dataset_sintetico_FIRE_UdeA.csv",
		base_dir / "dataset_sintetico_FIRE_UdeA_realista.csv",
	]

	outputs: Dict[str, Path] = {}
	for dataset_path in datasets:
		if not dataset_path.exists():
			raise FileNotFoundError(f"No existe el dataset: {dataset_path}")
		output_path = preprocess_dataset(dataset_path, output_dir)
		outputs[dataset_path.name] = output_path

	print("\nPreprocesamiento finalizado correctamente.")
	return outputs


if __name__ == "__main__":
	run_preprocessing()
