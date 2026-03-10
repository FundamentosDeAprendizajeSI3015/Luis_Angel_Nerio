"""
Preprocesamiento completo para dataset_sintetico_FIRE_UdeA_realista.csv.

Objetivo:
- Dejar los datos listos para entrenar un modelo mejor.
- Evitar fuga de datos con split temporal.
- Guardar salidas reproducibles para modelado.
"""

import json
import os
import sys
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

warnings.filterwarnings("ignore")


class Tee:
	"""Duplica salida en pantalla y archivo de log."""

	def __init__(self, *files):
		self.files = files

	def write(self, obj: str) -> None:
		for file_obj in self.files:
			file_obj.write(obj)
			file_obj.flush()

	def flush(self) -> None:
		for file_obj in self.files:
			file_obj.flush()


def print_section(title: str) -> None:
	print("\n" + "=" * 90)
	print(title)
	print("=" * 90)


def ensure_dirs(base_dir: str) -> Dict[str, str]:
	paths = {
		"processed": os.path.join(base_dir, "preprocess_outputs", "processed"),
		"reports": os.path.join(base_dir, "preprocess_outputs", "reports"),
	}
	for path in paths.values():
		os.makedirs(path, exist_ok=True)
	return paths


def safe_divide(a: pd.Series, b: pd.Series, eps: float = 1e-8) -> pd.Series:
	return a / (b + eps)


def clean_base_data(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()

	# Limpieza de espacios en nombres de columnas y texto.
	df.columns = [str(c).strip() for c in df.columns]
	for col in df.select_dtypes(include="object").columns:
		df[col] = df[col].astype(str).str.strip()
		df[col] = df[col].replace({"nan": np.nan, "": np.nan})

	# Conversión de tipos esperados.
	if "anio" in df.columns:
		df["anio"] = pd.to_numeric(df["anio"], errors="coerce").astype("Int64")
	if "label" in df.columns:
		df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")

	for col in df.columns:
		if col not in ["unidad"]:
			if col not in ["anio", "label"]:
				# Convierte a numerico cuando aplica; deja texto si la columna no es numerica.
				converted = pd.to_numeric(df[col], errors="coerce")
				if converted.notna().sum() > 0:
					df[col] = converted

	# Duplicados exactos.
	df = df.drop_duplicates().reset_index(drop=True)
	return df


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()

	if {"gastos_personal", "ingresos_totales"}.issubset(df.columns):
		df["ratio_gastos_personal_ingresos"] = safe_divide(df["gastos_personal"], df["ingresos_totales"])

	if {"cfo", "ingresos_totales"}.issubset(df.columns):
		df["ratio_cfo_ingresos"] = safe_divide(df["cfo"], df["ingresos_totales"])

	if {"liquidez", "endeudamiento"}.issubset(df.columns):
		df["ratio_liquidez_endeudamiento"] = safe_divide(df["liquidez"], df["endeudamiento"])

	if "hhi_fuentes" in df.columns:
		df["diversificacion_fuentes"] = 1.0 - df["hhi_fuentes"]

	if "dias_efectivo" in df.columns:
		df["anios_cobertura_efectivo"] = df["dias_efectivo"] / 365.0

	if {"unidad", "anio", "gp_ratio"}.issubset(df.columns):
		df = df.sort_values(["unidad", "anio"]).reset_index(drop=True)
		df["delta_gp_ratio_anual"] = df.groupby("unidad")["gp_ratio"].diff()

	if {"unidad", "anio", "liquidez"}.issubset(df.columns):
		df = df.sort_values(["unidad", "anio"]).reset_index(drop=True)
		df["delta_liquidez_anual"] = df.groupby("unidad")["liquidez"].diff()

	if "anio" in df.columns:
		df = df.sort_values(["anio", "unidad" if "unidad" in df.columns else df.columns[0]]).reset_index(drop=True)

	return df


def build_temporal_splits(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
	if "anio" not in df.columns:
		raise ValueError("El dataset no tiene la columna 'anio', necesaria para split temporal.")

	years = sorted(df["anio"].dropna().astype(int).unique().tolist())
	if len(years) < 4:
		raise ValueError(f"Se requieren al menos 4 anios para split temporal robusto. Anios encontrados: {years}")

	train_years = years[:-3]
	valid_year = years[-3]
	test_year = years[-2]
	holdout_year = years[-1]

	splits = {
		"train": df[df["anio"].isin(train_years)].copy(),
		"valid": df[df["anio"] == valid_year].copy(),
		"test": df[df["anio"] == test_year].copy(),
		"holdout": df[df["anio"] == holdout_year].copy(),
	}

	print(f"[INFO] Split temporal -> train:{train_years} | valid:{valid_year} | test:{test_year} | holdout:{holdout_year}")
	for name, split_df in splits.items():
		print(f"[INFO] {name}: {split_df.shape}")

	return splits


def compute_iqr_bounds(train_df: pd.DataFrame, numeric_cols: List[str], k: float = 1.5) -> Dict[str, Tuple[float, float]]:
	bounds: Dict[str, Tuple[float, float]] = {}
	for col in numeric_cols:
		series = train_df[col].dropna()
		if series.empty:
			continue
		q1 = series.quantile(0.25)
		q3 = series.quantile(0.75)
		iqr = q3 - q1
		if pd.isna(iqr):
			continue
		bounds[col] = (float(q1 - k * iqr), float(q3 + k * iqr))
	return bounds


def apply_iqr_capping(df: pd.DataFrame, bounds: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
	df = df.copy()
	for col, (lower, upper) in bounds.items():
		if col in df.columns:
			df[col] = df[col].clip(lower=lower, upper=upper)
	return df


def build_preprocessor(X_train: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
	numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
	cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

	num_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="median")),
			("scaler", RobustScaler()),
		]
	)

	cat_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="most_frequent")),
			("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
		]
	)

	preprocessor = ColumnTransformer(
		transformers=[
			("num", num_pipeline, numeric_cols),
			("cat", cat_pipeline, cat_cols),
		],
		remainder="drop",
	)

	return preprocessor, numeric_cols, cat_cols


def transform_splits(
	preprocessor: ColumnTransformer,
	X_train: pd.DataFrame,
	X_valid: pd.DataFrame,
	X_test: pd.DataFrame,
	X_holdout: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
	Z_train = preprocessor.fit_transform(X_train)
	Z_valid = preprocessor.transform(X_valid)
	Z_test = preprocessor.transform(X_test)
	Z_holdout = preprocessor.transform(X_holdout)

	feature_names = preprocessor.get_feature_names_out().tolist()
	feature_names = [name.replace("num__", "").replace("cat__", "") for name in feature_names]

	return {
		"train": pd.DataFrame(Z_train, columns=feature_names, index=X_train.index),
		"valid": pd.DataFrame(Z_valid, columns=feature_names, index=X_valid.index),
		"test": pd.DataFrame(Z_test, columns=feature_names, index=X_test.index),
		"holdout": pd.DataFrame(Z_holdout, columns=feature_names, index=X_holdout.index),
	}


def save_outputs(
	outputs_dir: Dict[str, str],
	raw_df: pd.DataFrame,
	splits_raw: Dict[str, pd.DataFrame],
	splits_model: Dict[str, pd.DataFrame],
	y_splits: Dict[str, pd.Series],
	metadata: Dict[str, object],
) -> None:
	processed_dir = outputs_dir["processed"]
	reports_dir = outputs_dir["reports"]

	raw_df.to_csv(os.path.join(processed_dir, "00_dataset_clean_featured.csv"), index=False)

	for split_name, split_df in splits_raw.items():
		split_df.to_csv(os.path.join(processed_dir, f"01_raw_{split_name}.csv"), index=False)

	for split_name, matrix_df in splits_model.items():
		model_df = matrix_df.copy()
		model_df["label"] = y_splits[split_name].values
		model_df.to_csv(os.path.join(processed_dir, f"02_model_{split_name}.csv"), index=False)

	missing_report = raw_df.isnull().sum().rename("missing_count").reset_index().rename(columns={"index": "column"})
	missing_report["missing_pct"] = (missing_report["missing_count"] / len(raw_df) * 100).round(2)
	missing_report.to_csv(os.path.join(reports_dir, "missing_report.csv"), index=False)

	with open(os.path.join(reports_dir, "preprocessing_metadata.json"), "w", encoding="utf-8") as f:
		json.dump(metadata, f, indent=2, ensure_ascii=False)


def run_preprocessing() -> None:
	script_dir = os.path.dirname(os.path.abspath(__file__))
	dataset_path = os.path.join(script_dir, "dataset_sintetico_FIRE_UdeA_realista.csv")
	outputs_dir = ensure_dirs(script_dir)

	log_path = os.path.join(outputs_dir["reports"], "preprocessing_log.txt")
	log_file = open(log_path, "w", encoding="utf-8")
	original_stdout = sys.stdout
	sys.stdout = Tee(original_stdout, log_file)

	try:
		print_section("INICIO PREPROCESAMIENTO FIRE-UDEA")
		print(f"[INFO] Dataset: {dataset_path}")
		if not os.path.exists(dataset_path):
			raise FileNotFoundError(f"No existe el dataset en: {dataset_path}")

		df = pd.read_csv(dataset_path)
		print(f"[OK] Dataset cargado: {df.shape}")

		print_section("1) LIMPIEZA BASE")
		df = clean_base_data(df)
		print(f"[OK] Dataset limpio base: {df.shape}")
		print(f"[INFO] Duplicados despues de limpiar: {int(df.duplicated().sum())}")

		if "label" not in df.columns:
			raise ValueError("El dataset no contiene la columna objetivo 'label'.")

		before_drop = len(df)
		df = df[df["label"].notna()].copy()
		df["label"] = df["label"].astype(int)
		print(f"[INFO] Filas removidas por label nulo: {before_drop - len(df)}")

		print_section("2) FEATURE ENGINEERING")
		df = add_feature_engineering(df)
		print(f"[OK] Columnas despues de FE: {len(df.columns)}")

		print_section("3) SPLIT TEMPORAL")
		splits_raw = build_temporal_splits(df)

		X_cols = [c for c in df.columns if c != "label"]
		y_splits = {name: split_df["label"].copy() for name, split_df in splits_raw.items()}
		X_splits = {name: split_df[X_cols].copy() for name, split_df in splits_raw.items()}

		print_section("4) CONTROL DE OUTLIERS (IQR CAPPING)")
		numeric_cols = X_splits["train"].select_dtypes(include=[np.number]).columns.tolist()
		iqr_bounds = compute_iqr_bounds(X_splits["train"], numeric_cols)
		for name in X_splits:
			X_splits[name] = apply_iqr_capping(X_splits[name], iqr_bounds)
		print(f"[OK] IQR capping aplicado en {len(iqr_bounds)} columnas numericas")

		print_section("5) IMPUTACION + ESCALADO + ONE-HOT (SIN FUGA)")
		preprocessor, numeric_used, cat_used = build_preprocessor(X_splits["train"])
		splits_model = transform_splits(
			preprocessor=preprocessor,
			X_train=X_splits["train"],
			X_valid=X_splits["valid"],
			X_test=X_splits["test"],
			X_holdout=X_splits["holdout"],
		)
		print(f"[OK] Matriz train procesada: {splits_model['train'].shape}")
		print(f"[OK] Matriz valid procesada: {splits_model['valid'].shape}")
		print(f"[OK] Matriz test procesada: {splits_model['test'].shape}")
		print(f"[OK] Matriz holdout procesada: {splits_model['holdout'].shape}")

		train_label_dist = y_splits["train"].value_counts().to_dict()
		n_total = len(y_splits["train"])
		n_classes = len(train_label_dist)
		class_weights = {
			str(cls): float(n_total / (n_classes * cnt))
			for cls, cnt in train_label_dist.items()
			if cnt > 0
		}

		metadata: Dict[str, object] = {
			"input_dataset": dataset_path,
			"total_rows": int(len(df)),
			"total_columns": int(df.shape[1]),
			"split_sizes": {k: int(v.shape[0]) for k, v in splits_raw.items()},
			"feature_count_processed": int(splits_model["train"].shape[1]),
			"numeric_features_used": numeric_used,
			"categorical_features_used": cat_used,
			"iqr_capped_columns": list(iqr_bounds.keys()),
			"train_class_distribution": {str(k): int(v) for k, v in train_label_dist.items()},
			"recommended_class_weights": class_weights,
			"recommended_modeling_note": "Usar valid para tuning de hiperparametros y umbral; dejar test y holdout solo para evaluacion final.",
		}

		print_section("6) GUARDADO DE SALIDAS")
		save_outputs(
			outputs_dir=outputs_dir,
			raw_df=df,
			splits_raw=splits_raw,
			splits_model=splits_model,
			y_splits=y_splits,
			metadata=metadata,
		)
		print("[OK] Archivos guardados en preprocess_outputs")
		print(f"[INFO] Procesados: {outputs_dir['processed']}")
		print(f"[INFO] Reportes: {outputs_dir['reports']}")

		print_section("PREPROCESAMIENTO FINALIZADO")

	finally:
		sys.stdout = original_stdout
		log_file.close()


if __name__ == "__main__":
	run_preprocessing()
