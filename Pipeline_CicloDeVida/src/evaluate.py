from typing import Dict, Any, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix
)

from .config import RESULTS_DIR, FIGURES_DIR


def ensure_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------- REGRESIÓN ----------
def evaluate_regression(
    y_true: np.ndarray, pred_dict: Dict[str, np.ndarray]
) -> pd.DataFrame:
    rows = []
    for name, y_pred in pred_dict.items():
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        rows.append({"model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
    return pd.DataFrame(rows).sort_values(by="MAE", ascending=True)


def save_regression_pred_plot(y_true: np.ndarray, y_pred: np.ndarray, model_name: str):
    ensure_dirs()
    plt.figure()
    plt.scatter(y_true, y_pred, s=10)
    plt.title(f"y_real vs y_pred ({model_name})")
    plt.xlabel("GPA real")
    plt.ylabel("GPA predicho")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"reg_ytrue_vs_ypred_{model_name}.png", dpi=180)
    plt.close()


# ---------- CLASIFICACIÓN ----------
def evaluate_classification(
    y_true: np.ndarray, pred_dict: Dict[str, np.ndarray], average: str = "weighted"
) -> pd.DataFrame:
    rows = []
    for name, y_pred in pred_dict.items():
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=average, zero_division=0
        )
        rows.append({
            "model": name,
            "Accuracy": acc,
            f"Precision_{average}": prec,
            f"Recall_{average}": rec,
            f"F1_{average}": f1
        })
    # ordenar por F1
    return pd.DataFrame(rows).sort_values(by=f"F1_{average}", ascending=False)


def save_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    class_labels: List[str]
):
    ensure_dirs()
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title(f"Matriz de Confusión ({model_name})")
    plt.xticks(range(len(class_labels)), class_labels, rotation=45, ha="right")
    plt.yticks(range(len(class_labels)), class_labels)
    plt.colorbar()

    # números dentro
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"clf_confusion_{model_name}.png", dpi=180)
    plt.close()
