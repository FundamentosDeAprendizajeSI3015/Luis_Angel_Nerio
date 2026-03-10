"""
Modelo basado en arbol de decision para FIRE-UdeA.

Entrena sobre los archivos procesados en preprocess_outputs/processed,
afina hiperparametros usando validacion temporal y reporta metricas
en train, valid, test y holdout.
"""

import json
import os
import sys
import warnings
from itertools import product
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
	accuracy_score,
	average_precision_score,
	balanced_accuracy_score,
	brier_score_loss,
	classification_report,
	confusion_matrix,
	f1_score,
	log_loss,
	precision_score,
	recall_score,
	roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")

# Modo rapido para ejecutar en menos tiempo.
QUICK_MODE = True


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
		"figures": os.path.join(base_dir, "model_outputs", "figures"),
		"results": os.path.join(base_dir, "model_outputs", "results"),
		"models": os.path.join(base_dir, "model_outputs", "models"),
	}
	for path in paths.values():
		os.makedirs(path, exist_ok=True)
	return paths


def load_split(path: str) -> Tuple[pd.DataFrame, pd.Series]:
	df = pd.read_csv(path)
	if "label" not in df.columns:
		raise ValueError(f"El archivo no contiene la columna 'label': {path}")
	X = df.drop(columns=["label"]).copy()
	y = df["label"].astype(int).copy()
	return X, y


def safe_metric(metric_fn, y_true: pd.Series, values: np.ndarray, default: float = np.nan) -> float:
	try:
		return float(metric_fn(y_true, values))
	except Exception:
		return float(default)


def evaluate_predictions(y_true: pd.Series, prob: np.ndarray, threshold: float) -> Dict[str, float]:
	prob = np.clip(prob, 1e-8, 1 - 1e-8)
	pred = (prob >= threshold).astype(int)

	cm = confusion_matrix(y_true, pred, labels=[0, 1])
	tn, fp, fn, tp = cm.ravel()

	specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
	npv = float(tn / (tn + fn)) if (tn + fn) > 0 else np.nan

	metrics = {
		"n": int(len(y_true)),
		"prevalencia": float(np.mean(y_true)),
		"threshold": float(threshold),
		"accuracy": float(accuracy_score(y_true, pred)),
		"balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
		"precision": float(precision_score(y_true, pred, zero_division=0)),
		"recall": float(recall_score(y_true, pred, zero_division=0)),
		"specificity": specificity,
		"npv": npv,
		"f1": float(f1_score(y_true, pred, zero_division=0)),
		"roc_auc": safe_metric(roc_auc_score, y_true, prob),
		"pr_auc": safe_metric(average_precision_score, y_true, prob),
		"brier": safe_metric(brier_score_loss, y_true, prob),
		"log_loss": safe_metric(log_loss, y_true, prob),
		"tn": int(tn),
		"fp": int(fp),
		"fn": int(fn),
		"tp": int(tp),
	}
	return metrics


def select_threshold(y_valid: pd.Series, prob_valid: np.ndarray, quick_mode: bool = True) -> Tuple[float, pd.DataFrame]:
	if quick_mode:
		base_thresholds = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]
		# Usa pocos cortes representativos de la distribucion de probabilidades.
		q = np.quantile(prob_valid, [0.20, 0.40, 0.60, 0.80]).tolist()
		unique_thresholds = [round(v, 3) for v in q]
	else:
		base_thresholds = np.linspace(0.2, 0.8, 25).tolist()
		unique_thresholds = np.unique(np.round(prob_valid, 3)).tolist()
	midpoints = []
	unique_probs = sorted(set(np.round(prob_valid, 6).tolist()))
	for left, right in zip(unique_probs[:-1], unique_probs[1:]):
		midpoints.append(round((left + right) / 2.0, 6))
	candidates = sorted(set([0.5] + base_thresholds + unique_thresholds + midpoints))

	rows = []
	for threshold in candidates:
		metrics = evaluate_predictions(y_valid, prob_valid, threshold)
		rows.append(metrics)

	table = pd.DataFrame(rows).sort_values(
		by=["balanced_accuracy", "f1", "specificity", "precision", "recall", "pr_auc"],
		ascending=[False, False, False, False, False, False],
	).reset_index(drop=True)
	best_threshold = float(table.loc[0, "threshold"])
	return best_threshold, table


def score_for_selection(metrics: Dict[str, float]) -> float:
	roc = 0.0 if np.isnan(metrics["roc_auc"]) else metrics["roc_auc"]
	pr = 0.0 if np.isnan(metrics["pr_auc"]) else metrics["pr_auc"]
	brier_component = 1.0 - min(max(metrics["brier"], 0.0), 1.0)
	specificity = 0.0 if np.isnan(metrics["specificity"]) else metrics["specificity"]
	balanced_acc = 0.0 if np.isnan(metrics["balanced_accuracy"]) else metrics["balanced_accuracy"]
	return (
		0.28 * metrics["f1"]
		+ 0.22 * balanced_acc
		+ 0.18 * specificity
		+ 0.17 * pr
		+ 0.10 * roc
		+ 0.05 * brier_component
	)


def evaluate_cv_stability(
	X_train: pd.DataFrame,
	y_train: pd.Series,
	params: Dict[str, object],
	quick_mode: bool,
) -> Dict[str, float]:
	if QUICK_MODE:
		n_splits = 4
	else:
		n_splits = 5

	min_class_count = int(y_train.value_counts().min())
	n_splits = max(2, min(n_splits, min_class_count))
	if n_splits < 2:
		return {
			"cv_f1": np.nan,
			"cv_balanced_accuracy": np.nan,
			"cv_specificity": np.nan,
			"cv_pr_auc": np.nan,
			"cv_roc_auc": np.nan,
		}

	skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
	oof_prob = np.zeros(len(y_train), dtype=float)

	for train_idx, val_idx in skf.split(X_train, y_train):
		X_fit = X_train.iloc[train_idx]
		y_fit = y_train.iloc[train_idx]
		X_val = X_train.iloc[val_idx]
		model = DecisionTreeClassifier(random_state=42, **params)
		model.fit(X_fit, y_fit)
		oof_prob[val_idx] = model.predict_proba(X_val)[:, 1]

	threshold, _ = select_threshold(y_train, oof_prob, quick_mode=quick_mode)
	metrics = evaluate_predictions(y_train, oof_prob, threshold)
	return {
		"cv_f1": metrics["f1"],
		"cv_balanced_accuracy": metrics["balanced_accuracy"],
		"cv_specificity": metrics["specificity"],
		"cv_pr_auc": metrics["pr_auc"],
		"cv_roc_auc": metrics["roc_auc"],
	}


def find_best_tree(
	X_train: pd.DataFrame,
	y_train: pd.Series,
	X_valid: pd.DataFrame,
	y_valid: pd.Series,
	class_weight_candidates: List[object],
	quick_mode: bool = True,
) -> Tuple[DecisionTreeClassifier, Dict[str, object], float, pd.DataFrame]:
	if quick_mode:
		grid = {
			"criterion": ["gini", "entropy"],
			"max_depth": [2, 3, 4, 5, 6],
			"min_samples_leaf": [1, 2, 3, 4, 6],
			"min_samples_split": [2, 4, 6],
			"ccp_alpha": [0.0, 0.001, 0.005],
			"class_weight": class_weight_candidates,
		}
	else:
		grid = {
			"criterion": ["gini", "entropy", "log_loss"],
			"max_depth": [3, 4, 5, 6, None],
			"min_samples_leaf": [1, 2, 4, 6],
			"min_samples_split": [2, 4, 8],
			"ccp_alpha": [0.0, 0.001, 0.005, 0.01],
			"class_weight": class_weight_candidates,
		}

	rows = []
	best_model = None
	best_params: Dict[str, object] = {}
	best_threshold = 0.5
	best_score = -np.inf

	total_combinations = int(np.prod([len(v) for v in grid.values()]))
	print(f"[INFO] Combinaciones a evaluar: {total_combinations} (quick_mode={quick_mode})")

	for values in product(*grid.values()):
		params = dict(zip(grid.keys(), values))
		cv_metrics = evaluate_cv_stability(X_train, y_train, params, quick_mode=quick_mode)
		model = DecisionTreeClassifier(random_state=42, **params)
		model.fit(X_train, y_train)
		prob_valid = model.predict_proba(X_valid)[:, 1]
		threshold, threshold_table = select_threshold(y_valid, prob_valid, quick_mode=quick_mode)
		valid_metrics = evaluate_predictions(y_valid, prob_valid, threshold)
		selection_score = 0.55 * score_for_selection(valid_metrics) + 0.45 * score_for_selection({
			"f1": 0.0 if np.isnan(cv_metrics["cv_f1"]) else cv_metrics["cv_f1"],
			"balanced_accuracy": 0.0 if np.isnan(cv_metrics["cv_balanced_accuracy"]) else cv_metrics["cv_balanced_accuracy"],
			"specificity": 0.0 if np.isnan(cv_metrics["cv_specificity"]) else cv_metrics["cv_specificity"],
			"pr_auc": 0.0 if np.isnan(cv_metrics["cv_pr_auc"]) else cv_metrics["cv_pr_auc"],
			"roc_auc": 0.0 if np.isnan(cv_metrics["cv_roc_auc"]) else cv_metrics["cv_roc_auc"],
			"brier": 0.5,
		})

		row = {
			**params,
			"threshold": threshold,
			"selection_score": selection_score,
			**cv_metrics,
			**valid_metrics,
		}
		rows.append(row)

		if selection_score > best_score:
			best_score = selection_score
			best_model = model
			best_params = params
			best_threshold = threshold

	results = pd.DataFrame(rows).sort_values(
		by=["selection_score", "balanced_accuracy", "specificity", "f1", "pr_auc", "roc_auc"],
		ascending=[False, False, False, False, False, False],
	).reset_index(drop=True)

	if best_model is None:
		raise RuntimeError("No fue posible encontrar un modelo de arbol valido.")

	return best_model, best_params, best_threshold, results


def save_confusion_matrix(cm: np.ndarray, split_name: str, figures_dir: str) -> None:
	plt.figure(figsize=(5, 4))
	sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=[0, 1], yticklabels=[0, 1])
	plt.title(f"Matriz de confusion - {split_name}")
	plt.xlabel("Prediccion")
	plt.ylabel("Valor real")
	plt.tight_layout()
	plt.savefig(os.path.join(figures_dir, f"confusion_matrix_{split_name}.png"), dpi=300, bbox_inches="tight")
	plt.close()


def save_tree_plot(model: DecisionTreeClassifier, feature_names: List[str], figures_dir: str) -> None:
	plt.figure(figsize=(24, 12))
	plot_tree(
		model,
		feature_names=feature_names,
		class_names=["0", "1"],
		filled=True,
		rounded=True,
		fontsize=8,
	)
	plt.title("Arbol de decision seleccionado")
	plt.tight_layout()
	plt.savefig(os.path.join(figures_dir, "decision_tree_selected.png"), dpi=300, bbox_inches="tight")
	plt.close()


def run_model() -> None:
	script_dir = os.path.dirname(os.path.abspath(__file__))
	processed_dir = os.path.join(script_dir, "preprocess_outputs", "processed")
	reports_dir = os.path.join(script_dir, "preprocess_outputs", "reports")
	output_dirs = ensure_dirs(script_dir)

	log_path = os.path.join(output_dirs["results"], "model_log.txt")
	log_file = open(log_path, "w", encoding="utf-8")
	original_stdout = sys.stdout
	sys.stdout = Tee(original_stdout, log_file)

	try:
		print_section("MODELO FIRE-UDEA - ARBOL DE DECISION")
		train_path = os.path.join(processed_dir, "02_model_train.csv")
		valid_path = os.path.join(processed_dir, "02_model_valid.csv")
		test_path = os.path.join(processed_dir, "02_model_test.csv")
		holdout_path = os.path.join(processed_dir, "02_model_holdout.csv")

		X_train, y_train = load_split(train_path)
		X_valid, y_valid = load_split(valid_path)
		X_test, y_test = load_split(test_path)
		X_holdout, y_holdout = load_split(holdout_path)

		print(f"[INFO] train:   {X_train.shape} | label={y_train.shape}")
		print(f"[INFO] valid:   {X_valid.shape} | label={y_valid.shape}")
		print(f"[INFO] test:    {X_test.shape} | label={y_test.shape}")
		print(f"[INFO] holdout: {X_holdout.shape} | label={y_holdout.shape}")

		class_weight_candidates: List[object] = [None, "balanced", {0: 1.2, 1: 1.0}, {0: 1.4, 1: 1.0}]
		metadata_path = os.path.join(reports_dir, "preprocessing_metadata.json")
		if os.path.exists(metadata_path):
			with open(metadata_path, "r", encoding="utf-8") as f:
				metadata = json.load(f)
			recommended_weights = metadata.get("recommended_class_weights")
			if recommended_weights:
				class_weight_candidates.append({int(k): float(v) for k, v in recommended_weights.items()})
		class_weight_candidates = list({str(candidate): candidate for candidate in class_weight_candidates}.values())

		print_section("1) BUSQUEDA DE HIPERPARAMETROS")
		best_model, best_params, best_threshold, search_results = find_best_tree(
			X_train=X_train,
			y_train=y_train,
			X_valid=X_valid,
			y_valid=y_valid,
			class_weight_candidates=class_weight_candidates,
			quick_mode=QUICK_MODE,
		)
		search_results.to_csv(os.path.join(output_dirs["results"], "tree_search_results.csv"), index=False)
		print("[OK] Busqueda completada")
		print(f"[INFO] Mejores parametros: {best_params}")
		print(f"[INFO] Mejor threshold: {best_threshold:.3f}")

		print_section("2) EVALUACION POR SPLIT")
		splits = {
			"train": (X_train, y_train),
			"valid": (X_valid, y_valid),
			"test": (X_test, y_test),
			"holdout": (X_holdout, y_holdout),
		}

		metric_rows = []
		for split_name, (X_split, y_split) in splits.items():
			prob = best_model.predict_proba(X_split)[:, 1]
			metrics = evaluate_predictions(y_split, prob, best_threshold)
			metrics["split"] = split_name
			metric_rows.append(metrics)

			scores_df = pd.DataFrame({
				"y_true": y_split.values,
				"prob": np.clip(prob, 1e-8, 1 - 1e-8),
				"pred": (prob >= best_threshold).astype(int),
			})
			scores_df.to_csv(os.path.join(output_dirs["results"], f"scores_{split_name}.csv"), index=False)

			cm = confusion_matrix(y_split, (prob >= best_threshold).astype(int), labels=[0, 1])
			save_confusion_matrix(cm, split_name, output_dirs["figures"])

			print(f"\n[INFO] {split_name.upper()}")
			for key in ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "brier", "log_loss"]:
				value = metrics[key]
				if np.isnan(value):
					print(f"  {key}: nan")
				else:
					print(f"  {key}: {value:.4f}")
			print(f"  confusion: tn={metrics['tn']} fp={metrics['fp']} fn={metrics['fn']} tp={metrics['tp']}")
			print(classification_report(y_split, (prob >= best_threshold).astype(int), zero_division=0))

		metrics_df = pd.DataFrame(metric_rows)
		metrics_df = metrics_df[[
			"split", "n", "prevalencia", "threshold", "accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1",
			"roc_auc", "pr_auc", "brier", "log_loss", "tn", "fp", "fn", "tp",
		]]
		metrics_df.to_csv(os.path.join(output_dirs["results"], "metricas_arbol_decision.csv"), index=False)

		print_section("3) IMPORTANCIA DE VARIABLES")
		importance_df = pd.DataFrame({
			"feature": X_train.columns,
			"importance": best_model.feature_importances_,
		}).sort_values("importance", ascending=False)
		importance_df.to_csv(os.path.join(output_dirs["results"], "feature_importance_tree.csv"), index=False)
		print(importance_df.head(15).to_string(index=False))

		plt.figure(figsize=(10, 6))
		top_n = min(15, len(importance_df))
		importance_df.head(top_n).sort_values("importance").plot(
			kind="barh", x="feature", y="importance", legend=False, color="#4E79A7"
		)
		plt.title("Top variables importantes - Arbol de decision")
		plt.xlabel("Importancia")
		plt.ylabel("Variable")
		plt.tight_layout()
		plt.savefig(os.path.join(output_dirs["figures"], "feature_importance_tree.png"), dpi=300, bbox_inches="tight")
		plt.close()

		print_section("4) VISUALIZACION DEL ARBOL")
		save_tree_plot(best_model, X_train.columns.tolist(), output_dirs["figures"])
		print("[OK] Arbol guardado")

		print_section("5) GUARDADO DEL MODELO")
		joblib.dump(best_model, os.path.join(output_dirs["models"], "decision_tree_model.joblib"))
		training_summary = {
			"best_params": best_params,
			"best_threshold": best_threshold,
			"train_shape": list(X_train.shape),
			"valid_shape": list(X_valid.shape),
			"test_shape": list(X_test.shape),
			"holdout_shape": list(X_holdout.shape),
			"feature_count": int(X_train.shape[1]),
		}
		with open(os.path.join(output_dirs["results"], "training_summary_tree.json"), "w", encoding="utf-8") as f:
			json.dump(training_summary, f, indent=2, ensure_ascii=False)
		print("[OK] Modelo y resumen guardados")

	finally:
		sys.stdout = original_stdout
		log_file.close()


if __name__ == "__main__":
	run_model()
