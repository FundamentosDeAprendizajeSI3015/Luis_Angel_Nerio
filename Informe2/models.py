"""
Modulo final de modelos supervisados para comparar rendimiento antes y despues
 del relabeling.

Entradas:
- data_supervised.csv
- relabeling/data_relabeling_final.csv

Salida en carpeta models/:
- metrics_comparison.csv
- metrics_comparison.png
- confusion_matrices.png
- feature_importance_tree.png
- regression_comparison.png
- final_comparison.png
- conclusions.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


# -----------------------------------------------------------------------------
# Configuracion general
# -----------------------------------------------------------------------------
BASE_PATH = Path(__file__).parent
ORIGINAL_DATA_PATH = BASE_PATH / "data_supervised.csv"
CORRECTED_DATA_PATH = BASE_PATH / "relabeling" / "data_relabeling_final.csv"
OUTPUT_DIR = BASE_PATH / "models"
OUTPUT_LOG_PATH = OUTPUT_DIR / "models_output.txt"
METRICS_PATH = OUTPUT_DIR / "metrics_comparison.csv"
CONCLUSIONS_PATH = OUTPUT_DIR / "conclusions.txt"

SEED = 42
CLASS_NAMES = ["Low", "Moderate", "High"]
sns.set_theme(style="whitegrid")


# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------
def asegurar_directorio() -> None:
    """Crea la carpeta models/ si no existe."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def guardar_log_consola(contenido: str) -> None:
    """Guarda un resumen acumulado del pipeline en models_output.txt."""
    with open(OUTPUT_LOG_PATH, "w", encoding="utf-8") as f:
        f.write(contenido.strip() + "\n")
    print(f"[INFO] Log resumen guardado en: {OUTPUT_LOG_PATH}")


def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga los datasets original y corregido."""
    if not ORIGINAL_DATA_PATH.exists():
        raise FileNotFoundError(f"No existe: {ORIGINAL_DATA_PATH}")
    if not CORRECTED_DATA_PATH.exists():
        raise FileNotFoundError(f"No existe: {CORRECTED_DATA_PATH}")

    original_df = pd.read_csv(ORIGINAL_DATA_PATH)
    corrected_df = pd.read_csv(CORRECTED_DATA_PATH)

    if original_df.empty or corrected_df.empty:
        raise ValueError("Uno de los datasets esta vacio")

    print("=" * 80)
    print("CARGA DE DATOS")
    print("=" * 80)
    print(f"Original shape: {original_df.shape}")
    print(f"Corregido shape: {corrected_df.shape}")

    return original_df.copy(), corrected_df.copy()


def extraer_features_y_labels(
    original_df: pd.DataFrame,
    corrected_df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Alinea features y etiquetas originales/corregidas."""
    original_features = [c for c in original_df.columns if c != "Stress_Level"]

    excluded_corrected = {
        "Student_ID",
        "Stress_Level",
        "Stress_Level_Corrected",
        "Stress_Level_Corrected_Encoded",
        "label_changed",
        "cluster_final",
        "cluster_kmeans",
        "cluster_dbscan",
        "cluster_fuzzy",
        "cluster_subtractive",
    }
    corrected_features = [
        c
        for c in corrected_df.columns
        if c not in excluded_corrected
        and not c.startswith("cluster_")
        and pd.api.types.is_numeric_dtype(corrected_df[c])
    ]

    if set(original_features) != set(corrected_features):
        raise ValueError(
            "Las features no coinciden entre los datasets. "
            f"Original: {original_features} | Corregido: {corrected_features}"
        )

    X = original_df[original_features].copy()
    y_original = original_df["Stress_Level"].astype(int).values

    label_map = {"Low": 0, "Moderate": 1, "High": 2}
    if "Stress_Level_Corrected" not in corrected_df.columns:
        raise ValueError("No existe Stress_Level_Corrected en el dataset corregido")
    y_corrected = corrected_df["Stress_Level_Corrected"].map(label_map)

    if y_corrected.isnull().any():
        invalid = corrected_df.loc[y_corrected.isnull(), "Stress_Level_Corrected"].unique().tolist()
        raise ValueError(f"Se encontraron etiquetas corregidas no validas: {invalid}")

    return X, y_original, y_corrected.astype(int).values


def crear_split_compartido(X: pd.DataFrame, y_ref: np.ndarray):
    """Genera un split comun para ambos conjuntos de etiquetas."""
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=SEED,
        stratify=y_ref,
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    return train_idx, test_idx, X_train, X_test


def preparar_etiquetas_y_metricas(
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_pred_class: np.ndarray,
    y_pred_reg: np.ndarray | None = None,
) -> dict:
    """Calcula metricas de clasificacion y, si aplica, de regresion."""
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred_class),
        "Precision": precision_score(y_test, y_pred_class, average="macro", zero_division=0),
        "Recall": recall_score(y_test, y_pred_class, average="macro", zero_division=0),
        "F1-score": f1_score(y_test, y_pred_class, average="macro", zero_division=0),
    }

    if y_pred_reg is not None:
        metrics["MAE"] = mean_absolute_error(y_test, y_pred_reg)
        metrics["R2"] = r2_score(y_test, y_pred_reg)
    else:
        metrics["MAE"] = np.nan
        metrics["R2"] = np.nan

    return metrics


def evaluar_con_cv(model, X, y, model_name: str, dataset_name: str) -> dict:
    """Evalua un modelo con validacion cruzada estratificada de 5 folds."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    acc_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")

    print(f"[CV] {model_name} ({dataset_name}):")
    print(f"     Accuracy: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
    print(f"     F1-macro: {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")

    return {
        "Modelo": model_name,
        "Dataset": dataset_name,
        "CV_Accuracy_mean": acc_scores.mean(),
        "CV_Accuracy_std": acc_scores.std(),
        "CV_F1_mean": f1_scores.mean(),
        "CV_F1_std": f1_scores.std(),
    }


def evaluar_clasificador(model, X_train, X_test, y_train, y_test, model_name: str, dataset_name: str):
    """Entrena y evalua un clasificador."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = preparar_etiquetas_y_metricas(y_train, y_test, y_pred)
    return {
        "Modelo": model_name,
        "Dataset": dataset_name,
        "Accuracy": metrics["Accuracy"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "F1-score": metrics["F1-score"],
        "MAE": metrics["MAE"],
        "R2": metrics["R2"],
        "y_pred": y_pred,
        "model": model,
    }


def evaluar_regresion_lineal(model, X_train, X_test, y_train, y_test, model_name: str, dataset_name: str):
    """Entrena regresion lineal y evalua como referencia."""
    model.fit(X_train, y_train)
    y_pred_reg = model.predict(X_test)
    y_pred_class = np.clip(np.rint(y_pred_reg), 0, 2).astype(int)
    metrics = preparar_etiquetas_y_metricas(y_train, y_test, y_pred_class, y_pred_reg)
    return {
        "Modelo": model_name,
        "Dataset": dataset_name,
        "Accuracy": metrics["Accuracy"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "F1-score": metrics["F1-score"],
        "MAE": metrics["MAE"],
        "R2": metrics["R2"],
        "y_pred_class": y_pred_class,
        "y_pred_reg": y_pred_reg,
        "model": model,
    }


def guardar_metricas(metricas_df: pd.DataFrame) -> None:
    """Guarda el CSV comparativo principal."""
    cols = ["Modelo", "Dataset", "Accuracy", "F1-score", "Precision", "Recall", "MAE", "R2"]
    metricas_df[cols].to_csv(METRICS_PATH, index=False)


def graficar_metricas_comparativas(metricas_df: pd.DataFrame) -> None:
    """Figura unica con Accuracy y F1-score por modelo, comparando Original vs Corregido."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    palette = {"Original": "#4c78a8", "Corregido": "#f58518"}

    sns.barplot(
        data=metricas_df,
        x="Modelo",
        y="Accuracy",
        hue="Dataset",
        palette=palette,
        ax=axes[0],
    )
    axes[0].set_title("Accuracy por modelo")
    axes[0].set_xlabel("Modelo")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(title="Dataset")
    axes[0].tick_params(axis="x", rotation=15)

    sns.barplot(
        data=metricas_df,
        x="Modelo",
        y="F1-score",
        hue="Dataset",
        palette=palette,
        ax=axes[1],
    )
    axes[1].set_title("F1-score por modelo")
    axes[1].set_xlabel("Modelo")
    axes[1].set_ylabel("F1-score")
    axes[1].legend(title="Dataset")
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "metrics_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def graficar_matrices_confusion(resultados: dict) -> None:
    """Figura unica con 6 matrices de confusion: Tree, Logistica y Lineal, Original y Corregido."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    plots = [
        ("Decision Tree - Original", resultados["Decision Tree"]["Original"]["y_pred"], resultados["y_test_original"], axes[0, 0]),
        ("Decision Tree - Corregido", resultados["Decision Tree"]["Corregido"]["y_pred"], resultados["y_test_corrected"], axes[0, 1]),
        ("Logistic Regression - Original", resultados["Logistic Regression"]["Original"]["y_pred"], resultados["y_test_original"], axes[1, 0]),
        ("Logistic Regression - Corregido", resultados["Logistic Regression"]["Corregido"]["y_pred"], resultados["y_test_corrected"], axes[1, 1]),
        ("Linear Regression - Original", resultados["Linear Regression"]["Original"]["y_pred_class"], resultados["y_test_original"], axes[2, 0]),
        ("Linear Regression - Corregido", resultados["Linear Regression"]["Corregido"]["y_pred_class"], resultados["y_test_corrected"], axes[2, 1]),
    ]

    for title, y_pred, y_true, ax in plots:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close()

def graficar_arbol_decision(tree_original: DecisionTreeClassifier, tree_corrected: DecisionTreeClassifier, feature_names: list[str]) -> None:
    """Figura ampliada del arbol de decision original con mas profundidad."""
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))

    plot_tree(
        tree_original,
        feature_names=feature_names,
        class_names=CLASS_NAMES,
        filled=True,
        rounded=True,
        impurity=False,
        fontsize=9,
        max_depth=5,
        ax=ax,
    )
    ax.set_title("Decision Tree Structure", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "decision_tree_structure.png", dpi=300, bbox_inches="tight")
    plt.close()

def graficar_importancia_arbol(tree_original: DecisionTreeClassifier, tree_corrected: DecisionTreeClassifier, feature_names: list[str]) -> None:
    """Figura con importancias del arbol original."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    importances_original = pd.Series(tree_original.feature_importances_, index=feature_names).sort_values()

    ax.barh(importances_original.index, importances_original.values, color="#4c78a8")
    ax.set_title("Feature Importance", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importancia", fontsize=11)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance_tree.png", dpi=300, bbox_inches="tight")
    plt.close()


def graficar_regresion_comparacion(resultados: dict) -> None:
    """Figura unica con prediccion vs real para regresion lineal antes y despues."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, clave, titulo in [
        (axes[0], "Original", "Regresion lineal - Original"),
        (axes[1], "Corregido", "Regresion lineal - Corregido"),
    ]:
        y_true = resultados["y_test_original"] if clave == "Original" else resultados["y_test_corrected"]
        y_pred_reg = resultados["Linear Regression"][clave]["y_pred_reg"]
        ax.scatter(y_true, y_pred_reg, alpha=0.7, color="#54a24b")
        ax.plot([0, 2], [0, 2], linestyle="--", color="red", linewidth=2)
        ax.set_title(titulo)
        ax.set_xlabel("Valor real")
        ax.set_ylabel("Prediccion")
        ax.set_xlim(-0.1, 2.1)
        ax.set_ylim(-0.1, 2.1)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "regression_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def graficar_comparacion_final(metricas_df: pd.DataFrame, resumen_df: pd.DataFrame) -> None:
    """Figura resumen con accuracy, F1 y mejora corregido-original."""
    models = ["Decision Tree", "Logistic Regression", "Linear Regression"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for metric, ax, title in [
        ("Accuracy", axes[0], "Accuracy por modelo"),
        ("F1-score", axes[1], "F1-score por modelo"),
    ]:
        pivot = metricas_df.pivot(index="Modelo", columns="Dataset", values=metric).reindex(models)
        x = np.arange(len(models))
        width = 0.35
        ax.bar(x - width / 2, pivot["Original"], width=width, label="Original", color="#4c78a8")
        ax.bar(x + width / 2, pivot["Corregido"], width=width, label="Corregido", color="#f58518")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15)
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.legend()

    mejoras = resumen_df.set_index("Modelo").reindex(models)
    x = np.arange(len(models))
    width = 0.2
    axes[2].bar(x - width * 1.5, mejoras["Accuracy_Improvement"], width=width, color="#72b7b2", label="Accuracy")
    axes[2].bar(x - width / 2, mejoras["F1_Improvement"], width=width, color="#e15759", alpha=0.8, label="F1-score")
    axes[2].bar(x + width / 2, mejoras["Precision_Improvement"], width=width, color="#54a24b", alpha=0.8, label="Precision")
    axes[2].bar(x + width * 1.5, mejoras["Recall_Improvement"], width=width, color="#b279a2", alpha=0.8, label="Recall")
    axes[2].axhline(0, color="black", linewidth=1)
    axes[2].set_title("Mejora (Corregido - Original)")
    axes[2].set_ylabel("Diferencia")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=15)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "final_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def generar_conclusiones(metricas_df: pd.DataFrame, resumen_df: pd.DataFrame, tree_feature_names: list[str], tree_corrected: DecisionTreeClassifier) -> None:
    """Genera conclusiones automaticas en TXT."""
    best_row = metricas_df.sort_values("F1-score", ascending=False).iloc[0]
    best_model = best_row["Modelo"]
    best_dataset = best_row["Dataset"]

    importances = pd.Series(tree_corrected.feature_importances_, index=tree_feature_names).sort_values(ascending=False)
    top_features = importances.head(3)

    lines = [
        "Conclusiones del analisis de modelos",
        "=" * 80,
        f"Mejor modelo segun F1-score: {best_model} ({best_dataset})",
        f"Accuracy: {best_row['Accuracy']:.4f}",
        f"F1-score: {best_row['F1-score']:.4f}",
        "",
        "Mejora por reetiquetado:",
    ]

    for _, row in resumen_df.iterrows():
        lines.append(
            f"- {row['Modelo']}: Accuracy {row['Accuracy_Improvement']:+.4f}, F1-score {row['F1_Improvement']:+.4f}"
        )

    lines.extend([
        "",
        "Variables mas importantes segun el arbol corregido:",
    ])
    for feature, value in top_features.items():
        lines.append(f"- {feature}: {value:.4f}")

    lines.extend([
        "",
        "Interpretacion general:",
        "- El reetiquetado se considera util si mejora F1-score y Accuracy en la mayoria de los modelos.",
        "- La regresion lineal se usa como referencia y no como clasificador principal.",
        "- El arbol de decision aporta interpretabilidad mediante importancias de variables.",
    ])

    with open(CONCLUSIONS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("\n" + "=" * 80)
    print("CONCLUSIONES")
    print("=" * 80)
    for line in lines:
        print(line)


# -----------------------------------------------------------------------------
# Pipeline principal
# -----------------------------------------------------------------------------
def pipeline_modelos() -> None:
    """Ejecuta entrenamiento, evaluacion y comparacion antes vs despues."""
    log_lines: list[str] = []

    original_df, corrected_df = cargar_datos()
    X, y_original, y_corrected = extraer_features_y_labels(original_df, corrected_df)
    log_lines.append("REPORTE MODELOS")
    log_lines.append("=" * 80)
    log_lines.append(f"Original shape: {original_df.shape}")
    log_lines.append(f"Corregido shape: {corrected_df.shape}")

    train_idx, test_idx, X_train, X_test = crear_split_compartido(X, y_original)

    y_train_original = y_original[train_idx]
    y_test_original = y_original[test_idx]
    y_train_corrected = y_corrected[train_idx]
    y_test_corrected = y_corrected[test_idx]

    resultados = {
        "y_test_original": y_test_original,
        "y_test_corrected": y_test_corrected,
        "Decision Tree": {},
        "Logistic Regression": {},
        "Linear Regression": {},
    }

    # Decision Tree
    tree_original = DecisionTreeClassifier(random_state=SEED, max_depth=10, class_weight="balanced")
    tree_corrected = DecisionTreeClassifier(random_state=SEED, max_depth=10, class_weight="balanced")
    resultados["Decision Tree"]["Original"] = evaluar_clasificador(
        tree_original, X_train, X_test, y_train_original, y_test_original, "Decision Tree", "Original"
    )
    info = resultados["Decision Tree"]["Original"]
    print(f"[OK] Decision Tree (Original): Acc={info['Accuracy']:.4f}  F1={info['F1-score']:.4f}")
    resultados["Decision Tree"]["Corregido"] = evaluar_clasificador(
        tree_corrected, X_train, X_test, y_train_corrected, y_test_corrected, "Decision Tree", "Corregido"
    )
    info = resultados["Decision Tree"]["Corregido"]
    print(f"[OK] Decision Tree (Corregido): Acc={info['Accuracy']:.4f}  F1={info['F1-score']:.4f}")
    log_lines.append(f"Decision Tree Original: Acc={resultados['Decision Tree']['Original']['Accuracy']:.4f}, F1={resultados['Decision Tree']['Original']['F1-score']:.4f}")
    log_lines.append(f"Decision Tree Corregido: Acc={resultados['Decision Tree']['Corregido']['Accuracy']:.4f}, F1={resultados['Decision Tree']['Corregido']['F1-score']:.4f}")

    # Logistic Regression
    log_original = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED, class_weight="balanced")
    log_corrected = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED, class_weight="balanced")
    resultados["Logistic Regression"]["Original"] = evaluar_clasificador(
        log_original, X_train, X_test, y_train_original, y_test_original, "Logistic Regression", "Original"
    )
    info = resultados["Logistic Regression"]["Original"]
    print(f"[OK] Logistic Regression (Original): Acc={info['Accuracy']:.4f}  F1={info['F1-score']:.4f}")
    resultados["Logistic Regression"]["Corregido"] = evaluar_clasificador(
        log_corrected, X_train, X_test, y_train_corrected, y_test_corrected, "Logistic Regression", "Corregido"
    )
    info = resultados["Logistic Regression"]["Corregido"]
    print(f"[OK] Logistic Regression (Corregido): Acc={info['Accuracy']:.4f}  F1={info['F1-score']:.4f}")
    log_lines.append(f"Logistic Regression Original: Acc={resultados['Logistic Regression']['Original']['Accuracy']:.4f}, F1={resultados['Logistic Regression']['Original']['F1-score']:.4f}")
    log_lines.append(f"Logistic Regression Corregido: Acc={resultados['Logistic Regression']['Corregido']['Accuracy']:.4f}, F1={resultados['Logistic Regression']['Corregido']['F1-score']:.4f}")

    # Linear Regression
    lin_original = LinearRegression()
    lin_corrected = LinearRegression()
    resultados["Linear Regression"]["Original"] = evaluar_regresion_lineal(
        lin_original, X_train, X_test, y_train_original, y_test_original, "Linear Regression", "Original"
    )
    info = resultados["Linear Regression"]["Original"]
    print(f"[OK] Linear Regression (Original): Acc={info['Accuracy']:.4f}  F1={info['F1-score']:.4f}")
    resultados["Linear Regression"]["Corregido"] = evaluar_regresion_lineal(
        lin_corrected, X_train, X_test, y_train_corrected, y_test_corrected, "Linear Regression", "Corregido"
    )
    info = resultados["Linear Regression"]["Corregido"]
    print(f"[OK] Linear Regression (Corregido): Acc={info['Accuracy']:.4f}  F1={info['F1-score']:.4f}")
    log_lines.append(f"Linear Regression Original: Acc={resultados['Linear Regression']['Original']['Accuracy']:.4f}, F1={resultados['Linear Regression']['Original']['F1-score']:.4f}")
    log_lines.append(f"Linear Regression Corregido: Acc={resultados['Linear Regression']['Corregido']['Accuracy']:.4f}, F1={resultados['Linear Regression']['Corregido']['F1-score']:.4f}")

    # DataFrame principal de metricas
    metricas_rows = []
    for modelo in ["Decision Tree", "Logistic Regression", "Linear Regression"]:
        for dataset_name in ["Original", "Corregido"]:
            info = resultados[modelo][dataset_name]
            metricas_rows.append(
                {
                    "Modelo": modelo,
                    "Dataset": dataset_name,
                    "Accuracy": info["Accuracy"],
                    "F1-score": info["F1-score"],
                    "Precision": info["Precision"],
                    "Recall": info["Recall"],
                    "MAE": info["MAE"],
                    "R2": info["R2"],
                }
            )

    metricas_df = pd.DataFrame(metricas_rows)
    guardar_metricas(metricas_df)

    print("\n" + "=" * 80)
    print("CROSS-VALIDATION (5 folds estratificado)")
    print("=" * 80)

    cv_rows = []
    for modelo_name, modelo_orig, modelo_corr in [
        (
            "Decision Tree",
            DecisionTreeClassifier(random_state=SEED, max_depth=10, class_weight="balanced"),
            DecisionTreeClassifier(random_state=SEED, max_depth=10, class_weight="balanced"),
        ),
        (
            "Logistic Regression",
            LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED, class_weight="balanced"),
            LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED, class_weight="balanced"),
        ),
    ]:
        cv_rows.append(evaluar_con_cv(modelo_orig, X, y_original, modelo_name, "Original"))
        cv_rows.append(evaluar_con_cv(modelo_corr, X, y_corrected, modelo_name, "Corregido"))

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(OUTPUT_DIR / "cv_results.csv", index=False)
    print("\n[OK] Resultados CV guardados en cv_results.csv")

    # Resumen de mejoras
    resumen_rows = []
    for modelo in ["Decision Tree", "Logistic Regression", "Linear Regression"]:
        original_row = metricas_df[(metricas_df["Modelo"] == modelo) & (metricas_df["Dataset"] == "Original")].iloc[0]
        corrected_row = metricas_df[(metricas_df["Modelo"] == modelo) & (metricas_df["Dataset"] == "Corregido")].iloc[0]
        resumen_rows.append(
            {
                "Modelo": modelo,
                "Accuracy_Improvement": corrected_row["Accuracy"] - original_row["Accuracy"],
                "F1_Improvement": corrected_row["F1-score"] - original_row["F1-score"],
                "Precision_Improvement": corrected_row["Precision"] - original_row["Precision"],
                "Recall_Improvement": corrected_row["Recall"] - original_row["Recall"],
            }
        )
    resumen_df = pd.DataFrame(resumen_rows)

    # Graficas agrupadas
    graficar_metricas_comparativas(metricas_df)
    graficar_matrices_confusion(resultados)
    graficar_importancia_arbol(
        tree_original=resultados["Decision Tree"]["Original"]["model"],
        tree_corrected=resultados["Decision Tree"]["Corregido"]["model"],
        feature_names=list(X.columns),
    )
    graficar_regresion_comparacion(resultados)
    graficar_comparacion_final(metricas_df, resumen_df)

    graficar_arbol_decision(
        tree_original=resultados["Decision Tree"]["Original"]["model"],
        tree_corrected=resultados["Decision Tree"]["Corregido"]["model"],
        feature_names=list(X.columns),
    )

    # Conclusiones
    generar_conclusiones(
        metricas_df=metricas_df,
        resumen_df=resumen_df,
        tree_feature_names=list(X.columns),
        tree_corrected=resultados["Decision Tree"]["Corregido"]["model"],
    )

    log_lines.append("METRICAS PRINCIPALES")
    log_lines.append(metricas_df.to_string(index=False))
    log_lines.append("RESUMEN DE MEJORAS")
    log_lines.append(resumen_df.to_string(index=False))
    log_lines.append("CV RESULTS")
    log_lines.append(cv_df.to_string(index=False))
    guardar_log_consola("\n".join(log_lines))

    print("\n" + "=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)
    print(metricas_df[["Modelo", "Dataset", "Accuracy", "F1-score"]])
    print(f"\nArchivos guardados en: {OUTPUT_DIR}")


def main() -> None:
    """Ejecuta el pipeline completo y guarda un resumen de salida."""
    asegurar_directorio()

    try:
        pipeline_modelos()
    except Exception as exc:
        print("\n" + "=" * 80)
        print("ERROR EN MODELOS")
        print("=" * 80)
        print(f"Detalle: {exc}")
        raise


if __name__ == "__main__":
    main()
