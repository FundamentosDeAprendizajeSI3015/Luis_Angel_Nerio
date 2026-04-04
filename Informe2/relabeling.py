"""
Modulo de reevaluacion y correccion de etiquetas basado en clustering.

Entradas:
- data_with_clusters.csv
- student_lifestyle_dataset.csv
- clustering/best_model.txt

Salidas en carpeta relabeling/:
- data_relabeling_final.csv
- label_changes_summary.txt
- cluster_label_mapping.csv
- labels_comparison.png
- confusion_matrix_relabeling.png
- pca_before_after.png
- umap_before_after.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import homogeneity_score

try:
    import umap
except ImportError as exc:
    raise ImportError(
        "No se encontro la libreria umap-learn. Instalar con: pip install umap-learn"
    ) from exc


# -----------------------------------------------------------------------------
# Configuracion general
# -----------------------------------------------------------------------------
BASE_PATH = Path(__file__).parent
CLUSTERING_DATA_PATH = BASE_PATH / "data_with_clusters.csv"
ORIGINAL_DATA_PATH = BASE_PATH / "student_lifestyle_dataset.csv"
BEST_MODEL_PATH = BASE_PATH / "clustering" / "best_model.txt"
OUTPUT_DIR = BASE_PATH / "relabeling"

OUTPUT_DATA_PATH = OUTPUT_DIR / "data_relabeling_final.csv"
OUTPUT_SUMMARY_PATH = OUTPUT_DIR / "label_changes_summary.txt"
OUTPUT_MAPPING_PATH = OUTPUT_DIR / "cluster_label_mapping.csv"
OUTPUT_LOG_PATH = OUTPUT_DIR / "relabeling_output.txt"

SEED = 42
sns.set_theme(style="whitegrid")


# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------
def asegurar_directorio() -> None:
    """Crea la carpeta de salida relabeling/."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def guardar_log_consola(contenido: str) -> None:
    """Guarda un resumen acumulado del pipeline en relabeling_output.txt."""
    with open(OUTPUT_LOG_PATH, "w", encoding="utf-8") as f:
        f.write(contenido.strip() + "\n")
    print(f"[INFO] Log resumen guardado en: {OUTPUT_LOG_PATH}")


def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga dataset de clustering y dataset original con etiquetas."""
    if not CLUSTERING_DATA_PATH.exists():
        raise FileNotFoundError(f"No existe: {CLUSTERING_DATA_PATH}")
    if not ORIGINAL_DATA_PATH.exists():
        raise FileNotFoundError(f"No existe: {ORIGINAL_DATA_PATH}")

    clustering_df = pd.read_csv(CLUSTERING_DATA_PATH)
    original_df = pd.read_csv(ORIGINAL_DATA_PATH)

    if clustering_df.empty or original_df.empty:
        raise ValueError("Uno de los datasets esta vacio")

    print("=" * 80)
    print("CARGA DE DATOS")
    print("=" * 80)
    print(f"Clustering shape: {clustering_df.shape}")
    print(f"Original shape: {original_df.shape}")

    if len(clustering_df) != len(original_df):
        raise ValueError(
            "Los datasets no tienen el mismo numero de filas. "
            "No se puede unir por indice de forma segura."
        )

    return clustering_df.copy(), original_df.copy()


def seleccionar_columna_cluster_final(best_model_path: Path) -> tuple[str, str]:
    """Lee best_model.txt y devuelve (nombre_algoritmo, columna_cluster)."""
    if not best_model_path.exists():
        raise FileNotFoundError(f"No existe: {best_model_path}")

    contenido = best_model_path.read_text(encoding="utf-8")
    contenido_lower = contenido.lower()

    if "kmeans" in contenido_lower:
        return "KMeans", "cluster_kmeans"
    if "dbscan" in contenido_lower:
        return "DBSCAN", "cluster_dbscan"
    if "fuzzy" in contenido_lower:
        return "Fuzzy", "cluster_fuzzy"
    if "subtractive" in contenido_lower:
        return "Subtractive", "cluster_subtractive"

    raise ValueError(
        "No se pudo identificar el mejor algoritmo en best_model.txt. "
        "Debe contener KMeans, DBSCAN, Fuzzy o Subtractive."
    )


def unir_datasets(clustering_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
    """Une por indice y conserva variables originales junto con columnas de cluster."""
    required_cluster_cols = [
        "cluster_kmeans",
        "cluster_dbscan",
        "cluster_fuzzy",
        "cluster_subtractive",
    ]
    for col in required_cluster_cols:
        if col not in clustering_df.columns:
            raise ValueError(f"No se encontro la columna requerida: {col}")

    if "Stress_Level" not in original_df.columns:
        raise ValueError("No se encontro la columna Stress_Level en el dataset original")

    # Se hace merge por indice para mantener el orden sin alterar archivos originales.
    merged = original_df.copy()
    merged = merged.join(clustering_df[required_cluster_cols], how="left")

    if "Student_ID" in merged.columns:
        merged = merged.drop(columns=["Student_ID"])
        print("[INFO] Columna Student_ID eliminada del dataset unificado")

    return merged


def mapear_clusters_a_etiquetas(df: pd.DataFrame, cluster_col: str) -> tuple[dict, pd.DataFrame]:
    """
    Mapea clusters a etiquetas garantizando que las 3 clases queden cubiertas.
    Primero intenta mapeo directo por etiqueta dominante. Si alguna clase queda
    sin cobertura, aplica asignacion forzada greedy por mayor proporcion.
    """
    distribucion = pd.crosstab(df[cluster_col], df["Stress_Level"], normalize="index")
    mapping_simple = distribucion.idxmax(axis=1).to_dict()
    clases_cubiertas = set(mapping_simple.values())
    clases_esperadas = {"Low", "Moderate", "High"}
    clases_faltantes = clases_esperadas - clases_cubiertas

    if not clases_faltantes:
        mapping = mapping_simple
        print("[OK] Mapeo directo: todas las clases cubiertas.")
    else:
        print(f"[ADVERTENCIA] Clases sin cobertura en mapeo simple: {clases_faltantes}")
        print("[INFO] Aplicando asignacion forzada greedy...")

        pares = []
        for cluster in distribucion.index:
            for clase in clases_esperadas:
                prop = float(distribucion.loc[cluster, clase]) if clase in distribucion.columns else 0.0
                pares.append((prop, cluster, clase))
        pares.sort(reverse=True)

        mapping = {}
        clases_asignadas = set()
        clusters_asignados_unico = set()

        for prop, cluster, clase in pares:
            if clase not in clases_asignadas and cluster not in clusters_asignados_unico:
                mapping[cluster] = clase
                clases_asignadas.add(clase)
                clusters_asignados_unico.add(cluster)
            if len(clases_asignadas) == len(clases_esperadas):
                break

        for cluster in distribucion.index:
            if cluster not in mapping:
                mapping[cluster] = distribucion.loc[cluster].idxmax()

        print("[OK] Asignacion forzada completada.")
        print(f"[INFO] Clases cubiertas: {set(mapping.values())}")

    mapping_df = (
        pd.DataFrame({"cluster": list(mapping.keys()), "dominant_label": list(mapping.values())})
        .sort_values("cluster")
        .reset_index(drop=True)
    )
    mapping_df.to_csv(OUTPUT_MAPPING_PATH, index=False)

    print("\n" + "=" * 80)
    print("MAPEO DE CLUSTERS A ETIQUETAS")
    print("=" * 80)
    print(f"Columna cluster final: {cluster_col}")
    print("Distribucion de Stress_Level por cluster (proporciones):")
    print(distribucion.round(4))
    print("\nMapeo final cluster -> etiqueta:")
    for c, label in sorted(mapping.items(), key=lambda x: x[0]):
        print(f"  Cluster {c}: {label}")

    return mapping, distribucion


def crear_etiquetas_corregidas(df: pd.DataFrame, cluster_col: str, mapping: dict) -> pd.DataFrame:
    """Crea Stress_Level_Corrected y columna booleana label_changed."""
    out = df.copy()
    out["Stress_Level_Corrected"] = out[cluster_col].map(mapping)

    # Fallback defensivo por si algun cluster no quedo en el mapeo.
    out["Stress_Level_Corrected"] = out["Stress_Level_Corrected"].fillna(out["Stress_Level"])

    mapeo_encoding = {"Low": 0, "Moderate": 1, "High": 2}
    out["Stress_Level_Corrected_Encoded"] = out["Stress_Level_Corrected"].map(mapeo_encoding)

    if out["Stress_Level_Corrected_Encoded"].isnull().sum() > 0:
        print("[ADVERTENCIA] Algunas etiquetas corregidas no pudieron ser codificadas numericamente")

    print("[INFO] Columna Stress_Level_Corrected_Encoded creada (Low=0, Moderate=1, High=2)")

    out["label_changed"] = out["Stress_Level"] != out["Stress_Level_Corrected"]
    return out


def analizar_cambios_por_clase(df: pd.DataFrame) -> pd.DataFrame:
    """Analiza cambios por clase original y guarda el resumen en CSV."""
    filas = []
    for clase in ["Low", "Moderate", "High"]:
        subset = df[df["Stress_Level"] == clase]
        total = len(subset)
        cambiadas_df = subset[subset["label_changed"]]
        cambiadas = len(cambiadas_df)
        pct_cambiado = (cambiadas / total * 100.0) if total > 0 else 0.0

        if cambiadas > 0:
            destino_principal = cambiadas_df["Stress_Level_Corrected"].value_counts().idxmax()
        else:
            destino_principal = "Sin cambios"

        filas.append(
            {
                "clase_original": clase,
                "total": total,
                "cambiadas": cambiadas,
                "pct_cambiado": round(pct_cambiado, 2),
                "destino_principal": destino_principal,
            }
        )

    cambios_por_clase_df = pd.DataFrame(filas)
    cambios_por_clase_df.to_csv(OUTPUT_DIR / "cambios_por_clase.csv", index=False)

    print("\n" + "=" * 80)
    print("CAMBIOS POR CLASE")
    print("=" * 80)
    print(cambios_por_clase_df.to_string(index=False))

    return cambios_por_clase_df


def guardar_resumen_cambios(df: pd.DataFrame, best_algo: str, cluster_col: str) -> None:
    """Guarda resumen textual de cambios de etiquetas."""
    changed_count = int(df["label_changed"].sum())
    total = len(df)
    pct_changed = (changed_count / total) * 100.0

    # Que tan bien los clusters capturan las etiquetas originales/corregidas.
    before_h = homogeneity_score(
        df["Stress_Level"].astype(str),
        df[cluster_col].astype(str),
    )
    after_h = homogeneity_score(
        df["Stress_Level_Corrected"].astype(str),
        df[cluster_col].astype(str),
    )

    if pct_changed <= 30:
        analisis_pct = "[OK] Porcentaje de cambio dentro del rango esperado (~30%)"
    elif pct_changed <= 50:
        analisis_pct = (
            f"[ADVERTENCIA] Porcentaje de cambio elevado ({pct_changed:.1f}%). "
            "Revisar calidad del clustering."
        )
    else:
        analisis_pct = (
            f"[CRITICO] Porcentaje de cambio muy alto ({pct_changed:.1f}%). "
            "El clustering puede no ser representativo de las etiquetas."
        )

    lineas = [
        "Resumen de reevaluacion y correccion de etiquetas",
        "=" * 80,
        f"Mejor algoritmo seleccionado: {best_algo}",
        f"Columna usada como cluster_final: {cluster_col}",
        f"Total de registros: {total}",
        f"Etiquetas modificadas: {changed_count}",
        f"Porcentaje de etiquetas modificadas: {pct_changed:.2f}%",
        analisis_pct,
        f"Homogeneidad (Stress_Level original vs clusters): {before_h:.4f}",
        f"Homogeneidad (Stress_Level_Corrected vs clusters): {after_h:.4f}",
    ]

    with open(OUTPUT_SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas) + "\n")

    print("\n" + "=" * 80)
    print("RESUMEN DE CAMBIOS")
    print("=" * 80)
    for ln in lineas:
        print(ln)


def graficar_distribuciones_etiquetas(df: pd.DataFrame) -> None:
    """Guarda una sola imagen con la comparacion de etiquetas antes y despues."""

    before_counts = df["Stress_Level"].value_counts()
    after_counts = df["Stress_Level_Corrected"].value_counts()

    categorias = sorted(set(before_counts.index).union(set(after_counts.index)))
    before_vals = [before_counts.get(c, 0) for c in categorias]
    after_vals = [after_counts.get(c, 0) for c in categorias]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(x=categorias, y=before_vals, hue=categorias, palette="Set2", ax=axes[0], legend=False)
    axes[0].set_title("Antes")
    axes[0].set_xlabel("Etiqueta")
    axes[0].set_ylabel("Cantidad")

    sns.barplot(x=categorias, y=after_vals, hue=categorias, palette="Set2", ax=axes[1], legend=False)
    axes[1].set_title("Despues")
    axes[1].set_xlabel("Etiqueta")
    axes[1].set_ylabel("Cantidad")

    fig.suptitle("Comparacion de etiquetas antes vs despues", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "labels_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def graficar_matriz_cambios(df: pd.DataFrame) -> None:
    """Genera heatmap de matriz de cambios (original vs corregida)."""
    cm = pd.crosstab(df["Stress_Level"], df["Stress_Level_Corrected"])

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Matriz de cambios: Stress_Level vs Stress_Level_Corrected")
    plt.xlabel("Etiqueta corregida")
    plt.ylabel("Etiqueta original")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix_relabeling.png", dpi=300, bbox_inches="tight")
    plt.close()


def graficar_pca_umap_antes_despues(df: pd.DataFrame) -> None:
    """Genera visualizaciones consolidadas PCA y UMAP (antes vs despues)."""
    # Usar solo features numericas reales, excluyendo etiquetas y cualquier columna de clusters.
    excluir = {
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
    num_cols = [
        c
        for c in df.columns
        if c not in excluir
        and not c.startswith("cluster_")
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    data_num = df[num_cols].copy()

    # PCA 2D
    pca_model = PCA(n_components=2, random_state=SEED)
    pca_array = pca_model.fit_transform(data_num.values)
    pca_df = pd.DataFrame(pca_array, columns=["PCA1", "PCA2"])

    fig_pca, axes_pca = plt.subplots(1, 2, figsize=(16, 6))
    sns.scatterplot(
        data=pca_df,
        x="PCA1",
        y="PCA2",
        hue=df["Stress_Level"].values,
        palette="Set2",
        s=26,
        alpha=0.85,
        ax=axes_pca[0],
    )
    axes_pca[0].set_title("PCA 2D - Etiquetas originales")
    axes_pca[0].set_xlabel("PCA1")
    axes_pca[0].set_ylabel("PCA2")
    axes_pca[0].legend(title="Etiqueta", loc="best")

    sns.scatterplot(
        data=pca_df,
        x="PCA1",
        y="PCA2",
        hue=df["Stress_Level_Corrected"].values,
        palette="Set2",
        s=26,
        alpha=0.85,
        ax=axes_pca[1],
    )
    axes_pca[1].set_title("PCA 2D - Etiquetas corregidas")
    axes_pca[1].set_xlabel("PCA1")
    axes_pca[1].set_ylabel("PCA2")
    axes_pca[1].legend(title="Etiqueta", loc="best")

    fig_pca.suptitle("Comparacion PCA: antes vs despues", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_before_after.png", dpi=300, bbox_inches="tight")
    plt.close()

    # UMAP 2D
    umap_model = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=SEED,
    )
    umap_array = umap_model.fit_transform(data_num.values)
    umap_df = pd.DataFrame(umap_array, columns=["UMAP1", "UMAP2"])

    fig_umap, axes_umap = plt.subplots(1, 2, figsize=(16, 6))
    sns.scatterplot(
        data=umap_df,
        x="UMAP1",
        y="UMAP2",
        hue=df["Stress_Level"].values,
        palette="Set2",
        s=26,
        alpha=0.85,
        ax=axes_umap[0],
    )
    axes_umap[0].set_title("UMAP 2D - Etiquetas originales")
    axes_umap[0].set_xlabel("UMAP1")
    axes_umap[0].set_ylabel("UMAP2")
    axes_umap[0].legend(title="Etiqueta", loc="best")

    sns.scatterplot(
        data=umap_df,
        x="UMAP1",
        y="UMAP2",
        hue=df["Stress_Level_Corrected"].values,
        palette="Set2",
        s=26,
        alpha=0.85,
        ax=axes_umap[1],
    )
    axes_umap[1].set_title("UMAP 2D - Etiquetas corregidas")
    axes_umap[1].set_xlabel("UMAP1")
    axes_umap[1].set_ylabel("UMAP2")
    axes_umap[1].legend(title="Etiqueta", loc="best")

    fig_umap.suptitle("Comparacion UMAP: antes vs despues", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "umap_before_after.png", dpi=300, bbox_inches="tight")
    plt.close()


def guardar_dataset_final(df: pd.DataFrame) -> None:
    """Guarda dataset final de relabeling."""
    columnas_requeridas = [
        "cluster_final",
        "Stress_Level",
        "Stress_Level_Corrected",
        "Stress_Level_Corrected_Encoded",
        "label_changed",
    ]
    for c in columnas_requeridas:
        if c not in df.columns:
            raise ValueError(f"Falta columna requerida en dataset final: {c}")

    df.to_csv(OUTPUT_DATA_PATH, index=False)


# -----------------------------------------------------------------------------
# Pipeline principal
# -----------------------------------------------------------------------------
def pipeline_relabeling() -> None:
    """Ejecuta el flujo completo de relabeling."""
    log_lines: list[str] = []

    clustering_df, original_df = cargar_datos()
    log_lines.append("REPORTE RELABELING")
    log_lines.append("=" * 80)
    log_lines.append(f"Clustering shape: {clustering_df.shape}")
    log_lines.append(f"Original shape: {original_df.shape}")

    df = unir_datasets(clustering_df, original_df)

    best_algo, best_col = seleccionar_columna_cluster_final(BEST_MODEL_PATH)
    df["cluster_final"] = df[best_col]
    log_lines.append(f"Mejor algoritmo detectado: {best_algo}")
    log_lines.append(f"Columna de cluster final: {best_col}")

    print("\n" + "=" * 80)
    print("SELECCION DEL MEJOR ALGORITMO")
    print("=" * 80)
    print(f"Mejor algoritmo detectado: {best_algo}")
    print(f"Columna seleccionada: {best_col}")

    mapping, _ = mapear_clusters_a_etiquetas(df, "cluster_final")
    df = crear_etiquetas_corregidas(df, "cluster_final", mapping)

    clases_resultado = set(df["Stress_Level_Corrected"].unique())
    clases_esperadas = {"Low", "Moderate", "High"}
    clases_perdidas = clases_esperadas - clases_resultado

    if clases_perdidas:
        print(f"[CRITICO] Clases que desaparecieron del reetiquetado: {clases_perdidas}")
        print("[CRITICO] Dataset no valido para entrenamiento multiclase.")
    else:
        print("[OK] Las 3 clases estan presentes en el dataset corregido:")
        for clase in sorted(clases_resultado):
            n = int((df["Stress_Level_Corrected"] == clase).sum())
            pct = n / len(df) * 100
            print(f"  - {clase}: {n} registros ({pct:.1f}%)")

    changed = int(df["label_changed"].sum())
    pct = (changed / len(df)) * 100
    print("\n" + "=" * 80)
    print("RESULTADO DEL REETIQUETADO")
    print("=" * 80)
    print(f"Etiquetas modificadas: {changed} de {len(df)} ({pct:.1f}%)")
    print(f"Etiquetas sin cambio:  {len(df) - changed} ({100 - pct:.1f}%)")

    if pct <= 30:
        estado_pct = "[OK] Porcentaje de cambio dentro del rango esperado (~30%)"
    elif pct <= 50:
        estado_pct = (
            f"[ADVERTENCIA] Porcentaje de cambio elevado ({pct:.1f}%). "
            "Revisar calidad del clustering."
        )
    else:
        estado_pct = (
            f"[CRITICO] Porcentaje de cambio muy alto ({pct:.1f}%). "
            "El clustering puede no ser representativo de las etiquetas."
        )
    print(estado_pct)

    log_lines.append(f"Etiquetas modificadas: {changed} de {len(df)} ({pct:.1f}%)")
    log_lines.append(estado_pct)

    guardar_resumen_cambios(df, best_algo=best_algo, cluster_col="cluster_final")
    cambios_por_clase_df = analizar_cambios_por_clase(df)
    log_lines.append("Cambios por clase (resumen):")
    for _, row in cambios_por_clase_df.iterrows():
        log_lines.append(
            f"- {row['clase_original']}: total={int(row['total'])}, "
            f"cambiadas={int(row['cambiadas'])}, pct={float(row['pct_cambiado']):.2f}%, "
            f"destino_principal={row['destino_principal']}"
        )

    before_h = homogeneity_score(
        df["Stress_Level"].astype(str),
        df["cluster_final"].astype(str),
    )
    after_h = homogeneity_score(
        df["Stress_Level_Corrected"].astype(str),
        df["cluster_final"].astype(str),
    )
    log_lines.append(f"Homogeneidad (Stress_Level original vs clusters): {before_h:.4f}")
    log_lines.append(f"Homogeneidad (Stress_Level_Corrected vs clusters): {after_h:.4f}")

    graficar_distribuciones_etiquetas(df)
    graficar_matriz_cambios(df)
    graficar_pca_umap_antes_despues(df)
    guardar_dataset_final(df)
    guardar_log_consola("\n".join(log_lines))

    print("\n" + "=" * 80)
    print("PROCESO DE RELABELING FINALIZADO")
    print("=" * 80)
    print(f"Salida principal: {OUTPUT_DATA_PATH}")
    print(f"Resumen: {OUTPUT_SUMMARY_PATH}")


def main() -> None:
    """Ejecuta pipeline y guarda resumen de salida en TXT."""
    asegurar_directorio()

    try:
        pipeline_relabeling()
    except Exception as exc:
        print("\n" + "=" * 80)
        print("ERROR EN RELABELING")
        print("=" * 80)
        print(f"Detalle: {exc}")
        raise


if __name__ == "__main__":
    main()
