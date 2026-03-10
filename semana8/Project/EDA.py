"""
EDA completo para dataset_sintetico_FIRE_UdeA_realista.csv.

Este script sigue el estilo de semanas anteriores:
- Mensajes por pasos en consola
- Guardado de tablas de analisis en CSV
- Guardado de figuras en carpeta dedicada
"""

import io
import json
import os
import sys
import warnings
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class Tee:
	"""Duplica salida de consola a archivo de log y pantalla."""

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
	print("\n" + "=" * 80)
	print(title)
	print("=" * 80)


def ensure_dirs(base_dir: str) -> Dict[str, str]:
	paths = {
		"figures": os.path.join(base_dir, "eda_figures"),
		"results": os.path.join(base_dir, "eda_results"),
	}
	for path in paths.values():
		os.makedirs(path, exist_ok=True)
	return paths


def get_numeric_columns(df: pd.DataFrame, exclude: List[str]) -> List[str]:
	cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
	return [c for c in cols if c not in exclude]


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
	missing_count = df.isnull().sum()
	missing_pct = (missing_count / len(df) * 100).round(2)
	report = pd.DataFrame(
		{
			"column": df.columns,
			"missing_count": missing_count.values,
			"missing_pct": missing_pct.values,
			"dtype": [str(df[c].dtype) for c in df.columns],
		}
	)
	return report.sort_values(["missing_count", "column"], ascending=[False, True])


def iqr_outlier_report(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
	rows = []
	for col in numeric_cols:
		series = df[col].dropna()
		if series.empty:
			continue
		q1 = series.quantile(0.25)
		q3 = series.quantile(0.75)
		iqr = q3 - q1
		lower = q1 - 1.5 * iqr
		upper = q3 + 1.5 * iqr
		mask = (series < lower) | (series > upper)
		count = int(mask.sum())
		pct = (count / len(series) * 100) if len(series) > 0 else 0.0
		rows.append(
			{
				"feature": col,
				"q1": float(q1),
				"q3": float(q3),
				"iqr": float(iqr),
				"lower_bound": float(lower),
				"upper_bound": float(upper),
				"outlier_count": count,
				"outlier_pct": round(float(pct), 2),
			}
		)
	return pd.DataFrame(rows).sort_values("outlier_count", ascending=False)


def save_dataset_overview(df: pd.DataFrame, results_dir: str) -> None:
	info_buffer = io.StringIO()
	df.info(buf=info_buffer)
	overview_path = os.path.join(results_dir, "01_dataset_info.txt")
	with open(overview_path, "w", encoding="utf-8") as f:
		f.write(info_buffer.getvalue())

	head_path = os.path.join(results_dir, "02_head.csv")
	df.head(20).to_csv(head_path, index=False)

	describe_num_path = os.path.join(results_dir, "03_describe_numeric.csv")
	df.describe(include=[np.number]).T.to_csv(describe_num_path)

	describe_all_path = os.path.join(results_dir, "04_describe_all.csv")
	df.describe(include="all").T.to_csv(describe_all_path)


def plot_target_distribution(df: pd.DataFrame, target_col: str, figures_dir: str) -> None:
	counts = df[target_col].value_counts(dropna=False).sort_index()
	fig, axes = plt.subplots(1, 2, figsize=(12, 5))

	counts.plot(kind="bar", ax=axes[0], color=["#4E79A7", "#E15759"])
	axes[0].set_title(f"Distribucion de {target_col}")
	axes[0].set_xlabel(target_col)
	axes[0].set_ylabel("Frecuencia")
	axes[0].tick_params(axis="x", rotation=0)

	counts.plot(kind="pie", ax=axes[1], autopct="%1.1f%%", startangle=90)
	axes[1].set_title(f"Proporcion de {target_col}")
	axes[1].set_ylabel("")

	plt.tight_layout()
	plt.savefig(os.path.join(figures_dir, "01_target_distribution.png"), dpi=300, bbox_inches="tight")
	plt.close()


def plot_temporal_patterns(df: pd.DataFrame, year_col: str, target_col: str, figures_dir: str) -> None:
	if year_col not in df.columns:
		return

	prevalence = df.groupby(year_col)[target_col].mean().reset_index()
	counts = df.groupby(year_col).size().reset_index(name="n")

	fig, ax1 = plt.subplots(figsize=(10, 5))
	sns.lineplot(data=prevalence, x=year_col, y=target_col, marker="o", ax=ax1, color="#E15759")
	ax1.set_title("Prevalencia de label por anio")
	ax1.set_ylabel("Prevalencia label=1")
	ax1.set_xlabel("Anio")
	ax1.set_ylim(0, 1)

	ax2 = ax1.twinx()
	sns.barplot(data=counts, x=year_col, y="n", alpha=0.25, ax=ax2, color="#4E79A7")
	ax2.set_ylabel("Numero de registros")

	plt.tight_layout()
	plt.savefig(os.path.join(figures_dir, "02_prevalencia_por_anio.png"), dpi=300, bbox_inches="tight")
	plt.close()


def plot_numeric_distributions(df: pd.DataFrame, numeric_cols: List[str], figures_dir: str) -> None:
	if not numeric_cols:
		return

	n_cols = 3
	n_rows = int(np.ceil(len(numeric_cols) / n_cols))
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.6 * n_rows))
	axes = np.array(axes).reshape(-1)

	for i, col in enumerate(numeric_cols):
		sns.histplot(df[col], kde=True, bins=20, ax=axes[i], color="#76B7B2")
		axes[i].set_title(f"Histograma: {col}")
		axes[i].set_xlabel(col)

	for j in range(i + 1, len(axes)):
		axes[j].axis("off")

	plt.tight_layout()
	plt.savefig(os.path.join(figures_dir, "03_histogramas_numericas.png"), dpi=300, bbox_inches="tight")
	plt.close()


def plot_boxplots(df: pd.DataFrame, numeric_cols: List[str], figures_dir: str) -> None:
	if not numeric_cols:
		return

	long_df = df[numeric_cols].melt(var_name="feature", value_name="value")
	plt.figure(figsize=(max(11, len(numeric_cols) * 0.75), 5.5))
	sns.boxplot(data=long_df, x="feature", y="value", color="#F28E2B")
	plt.xticks(rotation=45, ha="right")
	plt.title("Boxplots de variables numericas")
	plt.xlabel("Variable")
	plt.ylabel("Valor")
	plt.tight_layout()
	plt.savefig(os.path.join(figures_dir, "04_boxplots_numericas.png"), dpi=300, bbox_inches="tight")
	plt.close()


def compute_correlations(df: pd.DataFrame, numeric_cols: List[str], target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
	corr_matrix = df[numeric_cols].corr(method="pearson")
	corr_with_target = pd.DataFrame(columns=["feature", "pearson_with_label"])
	if target_col in numeric_cols:
		corr_with_target = (
			corr_matrix[target_col]
			.drop(labels=[target_col])
			.sort_values(key=lambda s: s.abs(), ascending=False)
			.reset_index()
		)
		corr_with_target.columns = ["feature", "pearson_with_label"]
	return corr_matrix, corr_with_target


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, figures_dir: str) -> None:
	if corr_matrix.empty:
		return
	plt.figure(figsize=(11, 9))
	sns.heatmap(corr_matrix, cmap="coolwarm", center=0, annot=False, square=True)
	plt.title("Matriz de correlacion (Pearson)")
	plt.tight_layout()
	plt.savefig(os.path.join(figures_dir, "05_heatmap_correlaciones.png"), dpi=300, bbox_inches="tight")
	plt.close()


def run_umap_projections(
	df: pd.DataFrame,
	target_col: str,
	figures_dir: str,
	results_dir: str,
	random_state: int = 42,
) -> None:
	"""Calcula UMAP en 2D y 3D y guarda embeddings + figuras."""
	try:
		import umap  # type: ignore
	except ImportError as exc:
		raise ImportError(
			"No se encontro la libreria 'umap-learn'. Instala con: pip install umap-learn"
		) from exc

	feature_cols = [c for c in df.columns if c != target_col]
	X = df[feature_cols].copy()

	# One-hot para categoricas y pipeline simple de imputacion + escalado.
	X = pd.get_dummies(X, drop_first=False)
	X = X.replace([np.inf, -np.inf], np.nan)

	imputer = SimpleImputer(strategy="median")
	scaler = StandardScaler()
	X_imp = imputer.fit_transform(X)
	X_scaled = scaler.fit_transform(X_imp)

	y = df[target_col].astype(int) if target_col in df.columns else pd.Series([0] * len(df))

	umap_2d = umap.UMAP(
		n_components=2,
		n_neighbors=15,
		min_dist=0.1,
		metric="euclidean",
		random_state=random_state,
	)
	Z2 = umap_2d.fit_transform(X_scaled)

	embedding_2d = pd.DataFrame({
		"umap_1": Z2[:, 0],
		"umap_2": Z2[:, 1],
		"label": y.values,
	})
	embedding_2d.to_csv(os.path.join(results_dir, "12_umap_2d_embedding.csv"), index=False)

	plt.figure(figsize=(9, 6))
	sns.scatterplot(
		data=embedding_2d,
		x="umap_1",
		y="umap_2",
		hue="label",
		palette="Set1",
		s=80,
		alpha=0.85,
	)
	plt.title("UMAP 2D - Dataset FIRE UdeA Realista")
	plt.xlabel("UMAP-1")
	plt.ylabel("UMAP-2")
	plt.legend(title="label")
	plt.tight_layout()
	plt.savefig(os.path.join(figures_dir, "07_umap_2d.png"), dpi=300, bbox_inches="tight")
	plt.close()

	umap_3d = umap.UMAP(
		n_components=3,
		n_neighbors=15,
		min_dist=0.1,
		metric="euclidean",
		random_state=random_state,
	)
	Z3 = umap_3d.fit_transform(X_scaled)

	embedding_3d = pd.DataFrame({
		"umap_1": Z3[:, 0],
		"umap_2": Z3[:, 1],
		"umap_3": Z3[:, 2],
		"label": y.values,
	})
	embedding_3d.to_csv(os.path.join(results_dir, "13_umap_3d_embedding.csv"), index=False)

	fig = plt.figure(figsize=(10, 7))
	ax = fig.add_subplot(111, projection="3d")
	colors = {0: "#4E79A7", 1: "#E15759"}
	for class_value in sorted(embedding_3d["label"].unique()):
		subset = embedding_3d[embedding_3d["label"] == class_value]
		ax.scatter(
			subset["umap_1"],
			subset["umap_2"],
			subset["umap_3"],
			label=f"label={class_value}",
			s=60,
			alpha=0.85,
			color=colors.get(int(class_value), "#59A14F"),
		)

	ax.set_title("UMAP 3D - Dataset FIRE UdeA Realista")
	ax.set_xlabel("UMAP-1")
	ax.set_ylabel("UMAP-2")
	ax.set_zlabel("UMAP-3")
	ax.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(figures_dir, "08_umap_3d.png"), dpi=300, bbox_inches="tight")
	plt.close()


def run_eda() -> None:
	script_dir = os.path.dirname(os.path.abspath(__file__))
	dataset_path = os.path.join(script_dir, "dataset_sintetico_FIRE_UdeA_realista.csv")
	output_paths = ensure_dirs(script_dir)

	log_path = os.path.join(output_paths["results"], "00_execution_log.txt")
	log_file = open(log_path, "w", encoding="utf-8")
	original_stdout = sys.stdout
	sys.stdout = Tee(original_stdout, log_file)

	try:
		print_section("INICIO EDA - FIRE UdeA Dataset Realista")
		print(f"[INFO] Ruta dataset: {dataset_path}")

		if not os.path.exists(dataset_path):
			raise FileNotFoundError(f"No existe el dataset en: {dataset_path}")

		df = pd.read_csv(dataset_path)
		print("[OK] Dataset cargado correctamente")
		print(f"[INFO] Forma del dataset: {df.shape}")
		print(f"[INFO] Columnas: {list(df.columns)}")

		target_col = "label"
		year_col = "anio"
		unit_col = "unidad"

		print_section("1) EXPLORACION GENERAL")
		save_dataset_overview(df, output_paths["results"])
		print("[OK] Se guardo info general y estadisticas descriptivas")

		print("\n[INFO] Primeras 5 filas:")
		print(df.head())

		print_section("2) CALIDAD DE DATOS")
		miss = missing_report(df)
		miss_path = os.path.join(output_paths["results"], "05_missing_report.csv")
		miss.to_csv(miss_path, index=False)
		print("[OK] Reporte de nulos guardado")
		print(miss.head(15))

		dup_count = int(df.duplicated().sum())
		print(f"\n[INFO] Filas duplicadas: {dup_count}")

		if target_col in df.columns:
			label_counts = df[target_col].value_counts(dropna=False).rename_axis(target_col).reset_index(name="count")
			label_counts["pct"] = (label_counts["count"] / len(df) * 100).round(2)
			label_counts.to_csv(os.path.join(output_paths["results"], "06_label_distribution.csv"), index=False)
			print("\n[INFO] Distribucion de label:")
			print(label_counts)

		print_section("3) EDA UNIVARIADO")
		numeric_cols = get_numeric_columns(df, exclude=[])
		print(f"[INFO] Variables numericas detectadas ({len(numeric_cols)}): {numeric_cols}")

		plot_target_distribution(df, target_col, output_paths["figures"])
		plot_temporal_patterns(df, year_col, target_col, output_paths["figures"])
		plot_numeric_distributions(df, numeric_cols, output_paths["figures"])
		plot_boxplots(df, numeric_cols, output_paths["figures"])
		print("[OK] Graficos univariados guardados")

		print_section("4) OUTLIERS IQR")
		outliers_df = iqr_outlier_report(df, [c for c in numeric_cols if c != target_col])
		outliers_path = os.path.join(output_paths["results"], "07_outlier_report_iqr.csv")
		outliers_df.to_csv(outliers_path, index=False)
		print("[OK] Reporte de outliers guardado")
		print(outliers_df.head(12))

		print_section("5) EDA BIVARIADO Y CORRELACIONES")
		corr_matrix, corr_target = compute_correlations(df, numeric_cols, target_col)
		corr_matrix.to_csv(os.path.join(output_paths["results"], "08_corr_matrix_pearson.csv"))
		corr_target.to_csv(os.path.join(output_paths["results"], "09_corr_with_label.csv"), index=False)
		plot_correlation_heatmap(corr_matrix, output_paths["figures"])
		print("[OK] Correlaciones calculadas y heatmap guardado")
		if not corr_target.empty:
			print("\n[INFO] Top correlaciones con label:")
			print(corr_target.head(10))

		if year_col in df.columns and unit_col in df.columns and target_col in df.columns:
			prevalence_by_unit = (
				df.groupby(unit_col)[target_col]
				.mean()
				.sort_values(ascending=False)
				.reset_index(name="prevalence_label_1")
			)
			prevalence_by_unit.to_csv(
				os.path.join(output_paths["results"], "10_prevalence_by_unit.csv"),
				index=False,
			)

			pivot_year_unit = df.pivot_table(
				index=unit_col,
				columns=year_col,
				values=target_col,
				aggfunc="mean",
			)
			pivot_year_unit.to_csv(os.path.join(output_paths["results"], "11_prevalence_unit_year.csv"))

			plt.figure(figsize=(10, 5))
			sns.heatmap(pivot_year_unit, cmap="YlOrRd", annot=True, fmt=".2f", cbar_kws={"label": "Prevalencia"})
			plt.title("Prevalencia de label por unidad y anio")
			plt.xlabel("Anio")
			plt.ylabel("Unidad")
			plt.tight_layout()
			plt.savefig(os.path.join(output_paths["figures"], "06_prevalencia_unidad_anio.png"), dpi=300, bbox_inches="tight")
			plt.close()
			print("[OK] Analisis temporal por unidad guardado")

		print_section("6) UMAP 2D Y 3D")
		run_umap_projections(
			df=df,
			target_col=target_col,
			figures_dir=output_paths["figures"],
			results_dir=output_paths["results"],
			random_state=42,
		)
		print("[OK] UMAP 2D y 3D generados y guardados")

		print_section("7) RESUMEN FINAL")
		summary = {
			"rows": int(df.shape[0]),
			"cols": int(df.shape[1]),
			"target_col": target_col,
			"target_prevalence": float(df[target_col].mean()) if target_col in df.columns else None,
			"missing_total": int(df.isnull().sum().sum()),
			"duplicate_rows": dup_count,
			"numeric_features": len(numeric_cols),
			"outputs": output_paths,
		}

		with open(os.path.join(output_paths["results"], "14_summary.json"), "w", encoding="utf-8") as f:
			json.dump(summary, f, indent=2, ensure_ascii=False)

		print("[OK] EDA finalizado correctamente")
		print("[INFO] Carpetas de salida:")
		print(f"  - Figuras: {output_paths['figures']}")
		print(f"  - Reportes: {output_paths['results']}")

	finally:
		sys.stdout = original_stdout
		log_file.close()


if __name__ == "__main__":
	run_eda()
