"""
Modulo de clustering no supervisado para data_clustering.csv.

Incluye:
- KMeans, DBSCAN, Fuzzy C-Means y Subtractive Clustering de Chiu.
- Evaluacion comparativa con metricas.
- Visualizaciones PCA y UMAP (2D y 3D).
- Seleccion automatica del mejor modelo.
- Guardado de resultados en CSV y TXT.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

try:
    import umap
except ImportError as exc:
    raise ImportError(
        "No se encontro la libreria umap-learn. Instalar con: pip install umap-learn"
    ) from exc

try:
    import skfuzzy as fuzz
except ImportError as exc:
    raise ImportError(
        "No se encontro la libreria scikit-fuzzy. Instalar con: pip install scikit-fuzzy"
    ) from exc


# -----------------------------------------------------------------------------
# Configuracion general
# -----------------------------------------------------------------------------
BASE_PATH = Path(__file__).parent
INPUT_PATH = BASE_PATH / "data_clustering.csv"
OUTPUT_DATA_PATH = BASE_PATH / "data_with_clusters.csv"
PLOTS_DIR = BASE_PATH / "clustering"

KMEANS_K = 3
RANDOM_STATE = 42
KMEANS_N_INIT = 10

DBSCAN_EPS = 0.7
DBSCAN_MIN_SAMPLES = 5

FUZZY_C = 3
FUZZY_M = 2.0
FUZZY_ERROR = 0.005
FUZZY_MAXITER = 1000

SUBTRACTIVE_RA = 2.2
SUBTRACTIVE_RB = 3.3

sns.set_theme(style="whitegrid")


# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------
def asegurar_directorio_salida() -> None:
    """Crea la carpeta clustering/ si no existe."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def guardar_log_consola(contenido: str) -> None:
    """Guarda el resumen del pipeline en clustering_output.txt."""
    log_path = PLOTS_DIR / "clustering_output.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(contenido + "\n")
    print(f"[OK] Log guardado en: {log_path}")


def cargar_datos(path: Path) -> pd.DataFrame:
    """Carga y valida el dataset de entrada."""
    if not path.exists():
        raise FileNotFoundError(f"No se encontro el archivo: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("El dataset esta vacio")

    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        raise ValueError(
            "Se encontraron columnas no numericas: "
            f"{non_numeric_cols}. data_clustering.csv debe contener solo numericas"
        )

    print("=" * 80)
    print("PREPARACION")
    print("=" * 80)
    print(f"Ruta de entrada: {path}")
    print(f"Dimensiones: {df.shape}")
    print("Tipos de datos:")
    print(df.dtypes)
    return df


def validar_escalado(df: pd.DataFrame) -> dict:
    """Verifica de forma basica si los datos parecen escalados."""
    means = df.mean()
    stds = df.std(ddof=0)

    mean_ok = bool((means.abs() < 0.25).all())
    std_ok = bool(((stds > 0.5) & (stds < 1.5)).all())

    estado = {
        "media_aproximada_cero": mean_ok,
        "desv_estandar_aproximada_uno": std_ok,
    }

    print("\n" + "=" * 80)
    print("VALIDACION DE ESCALADO")
    print("=" * 80)
    print(f"Media abs maxima: {means.abs().max():.4f}")
    print(f"Desv est minima: {stds.min():.4f}")
    print(f"Desv est maxima: {stds.max():.4f}")
    print(f"Validacion media cercana a 0: {mean_ok}")
    print(f"Validacion desv est cercana a 1: {std_ok}")
    return estado


def aplicar_pca(df: pd.DataFrame, n_components: int = 2) -> tuple[np.ndarray, pd.DataFrame, PCA]:
    """Aplica PCA y retorna coordenadas y modelo."""
    pca_model = PCA(n_components=n_components, random_state=RANDOM_STATE)
    pca_array = pca_model.fit_transform(df.values)
    pca_cols = [f"PCA{i + 1}" for i in range(n_components)]
    pca_df = pd.DataFrame(pca_array, columns=pca_cols)
    return pca_array, pca_df, pca_model


# -----------------------------------------------------------------------------
# Graficas
# -----------------------------------------------------------------------------
def graficar_scatter_2d(
    coords: pd.DataFrame,
    labels: np.ndarray,
    title: str,
    output_name: str,
    x_col: str,
    y_col: str,
) -> None:
    """Genera scatter 2D para etiquetas de clustering."""
    unique_labels = sorted(np.unique(labels))
    palette = sns.color_palette("tab10", max(len(unique_labels), 3))

    plt.figure(figsize=(10, 6))
    for idx, lbl in enumerate(unique_labels):
        mask = labels == lbl
        if lbl == -1:
            plt.scatter(
                coords.loc[mask, x_col],
                coords.loc[mask, y_col],
                s=25,
                alpha=0.7,
                color="lightgray",
                label="Ruido (-1)",
            )
        else:
            plt.scatter(
                coords.loc[mask, x_col],
                coords.loc[mask, y_col],
                s=30,
                alpha=0.8,
                color=palette[idx % len(palette)],
                label=f"Cluster {lbl}",
            )

    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close()


def graficar_scatter_grid_2d(
    coords: pd.DataFrame,
    labels_map: dict,
    x_col: str,
    y_col: str,
    output_name: str,
    titulo_general: str,
) -> None:
    """Genera una sola imagen con subplots 2D para todos los algoritmos."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, (algoritmo, labels) in zip(axes, labels_map.items()):
        unique_labels = sorted(np.unique(labels))
        palette = sns.color_palette("tab10", max(len(unique_labels), 3))

        for idx, lbl in enumerate(unique_labels):
            mask = labels == lbl
            if lbl == -1:
                ax.scatter(
                    coords.loc[mask, x_col],
                    coords.loc[mask, y_col],
                    s=18,
                    alpha=0.65,
                    color="lightgray",
                    label="Ruido (-1)",
                )
            else:
                ax.scatter(
                    coords.loc[mask, x_col],
                    coords.loc[mask, y_col],
                    s=20,
                    alpha=0.8,
                    color=palette[idx % len(palette)],
                    label=f"Cluster {lbl}",
                )

        ax.set_title(algoritmo)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle(titulo_general, fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close()


def graficar_scatter_grid_3d(
    coords_3d: pd.DataFrame,
    labels_map: dict,
    output_name: str,
    titulo_general: str,
) -> None:
    """Genera una sola imagen con subplots 3D para todos los algoritmos."""
    fig = plt.figure(figsize=(18, 12))

    for i, (algoritmo, labels) in enumerate(labels_map.items(), start=1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        unique_labels = sorted(np.unique(labels))
        palette = sns.color_palette("tab10", max(len(unique_labels), 3))

        for idx, lbl in enumerate(unique_labels):
            mask = labels == lbl
            if lbl == -1:
                ax.scatter(
                    coords_3d.loc[mask, "UMAP1"],
                    coords_3d.loc[mask, "UMAP2"],
                    coords_3d.loc[mask, "UMAP3"],
                    s=12,
                    alpha=0.55,
                    color="lightgray",
                    label="Ruido (-1)",
                )
            else:
                ax.scatter(
                    coords_3d.loc[mask, "UMAP1"],
                    coords_3d.loc[mask, "UMAP2"],
                    coords_3d.loc[mask, "UMAP3"],
                    s=14,
                    alpha=0.75,
                    color=palette[idx % len(palette)],
                    label=f"Cluster {lbl}",
                )

        ax.set_title(algoritmo)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_zlabel("UMAP3")
        ax.legend(loc="best", fontsize=7)

    fig.suptitle(titulo_general, fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close()


def graficar_metodo_codo(df: pd.DataFrame, k_min: int = 1, k_max: int = 10) -> list:
    """Calcula inercia por K y guarda grafica del metodo del codo."""
    inertias = []
    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=KMEANS_N_INIT)
        model.fit(df.values)
        inertias.append(model.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(k_min, k_max + 1), inertias, marker="o", linewidth=2)
    plt.title("Metodo del codo para KMeans")
    plt.xlabel("Numero de clusters (K)")
    plt.ylabel("Inercia")
    plt.xticks(range(k_min, k_max + 1))
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "kmeans_elbow.png", dpi=300, bbox_inches="tight")
    plt.close()
    return inertias


def calcular_kdistance_plot(df: pd.DataFrame, k: int = 5) -> float:
    """
    Calcula la curva k-distance para DBSCAN y retorna eps sugerido.

    Args:
        df (pd.DataFrame): Datos numericos escalados
        k (int): Vecino k-esimo

    Returns:
        float: eps sugerido (percentil 90 de distancias k-esimo vecino)
    """
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(df.values)
    distances, _ = nn.kneighbors(df.values)
    k_distances = distances[:, k - 1]

    k_distances_sorted = np.sort(k_distances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(k_distances_sorted) + 1), k_distances_sorted, linewidth=1.6)
    plt.title(f"DBSCAN k-distance plot (k={k})")
    plt.xlabel("Puntos ordenados")
    plt.ylabel("Distancia al k-esimo vecino")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "dbscan_kdistance.png", dpi=300, bbox_inches="tight")
    plt.close()

    eps_sugerido = float(np.percentile(k_distances, 90))
    print(f"[INFO] eps sugerido para DBSCAN (percentil 90): {eps_sugerido:.4f}")
    return eps_sugerido


def graficar_pca_vs_umap_kmeans(pca_df: pd.DataFrame, umap2d_df: pd.DataFrame, labels: np.ndarray) -> None:
    """Compara PCA vs UMAP para KMeans en una sola figura."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(pca_df["PCA1"], pca_df["PCA2"], c=labels, cmap="tab10", s=25, alpha=0.8)
    axes[0].set_title("KMeans en PCA")
    axes[0].set_xlabel("PCA1")
    axes[0].set_ylabel("PCA2")

    axes[1].scatter(umap2d_df["UMAP1"], umap2d_df["UMAP2"], c=labels, cmap="tab10", s=25, alpha=0.8)
    axes[1].set_title("KMeans en UMAP")
    axes[1].set_xlabel("UMAP1")
    axes[1].set_ylabel("UMAP2")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pca_vs_umap_kmeans.png", dpi=300, bbox_inches="tight")
    plt.close()


def graficar_distribucion_clusters(labels: np.ndarray, output_name: str, titulo: str) -> None:
    """Guarda countplot de la distribucion por cluster."""
    series = pd.Series(labels, name="cluster")
    order = sorted(series.unique())
    plt.figure(figsize=(8, 5))
    sns.countplot(x=series, order=order, hue=series, palette="tab10", legend=False)
    plt.title(titulo)
    plt.xlabel("Cluster")
    plt.ylabel("Cantidad de datos")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close()


def graficar_distribucion_clusters_multiples(labels_map: dict, output_name: str) -> None:
    """Genera una sola imagen con distribuciones de clusters para todos los algoritmos."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for ax, (algoritmo, labels) in zip(axes, labels_map.items()):
        series = pd.Series(labels, name="cluster")
        order = sorted(series.unique())
        sns.countplot(x=series, order=order, hue=series, palette="tab10", legend=False, ax=ax)
        ax.set_title(f"Distribucion - {algoritmo}")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Cantidad")

    fig.suptitle("Distribucion de clusters por algoritmo", fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close()


def graficar_umap_sin_clusters(umap_df: pd.DataFrame, output_name: str = "umap_sin_clusters.png") -> None:
    """Genera grafica UMAP 2D sin clusters - todos los puntos en azul."""
    plt.figure(figsize=(10, 8))
    plt.scatter(
        umap_df["UMAP1"],
        umap_df["UMAP2"],
        s=30,
        alpha=0.6,
        color="steelblue"
    )
    plt.title("UMAP sin clusters")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / output_name, dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# Algoritmos
# -----------------------------------------------------------------------------
def ejecutar_kmeans(df: pd.DataFrame, k: int = KMEANS_K) -> tuple[np.ndarray, np.ndarray, KMeans]:
    """Ejecuta KMeans y retorna etiquetas, centroides y modelo."""
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=KMEANS_N_INIT)
    labels = model.fit_predict(df.values)
    return labels, model.cluster_centers_, model


def ejecutar_dbscan(
    df: pd.DataFrame,
    eps: float,
    min_samples: int = DBSCAN_MIN_SAMPLES,
) -> tuple[np.ndarray, DBSCAN]:
    """Ejecuta DBSCAN y retorna etiquetas y modelo."""
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(df.values)
    return labels, model


def ejecutar_fuzzy_cmeans(df: pd.DataFrame, c: int = FUZZY_C) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ejecuta Fuzzy C-Means con skfuzzy y convierte a etiquetas duras."""
    data_t = df.values.T
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        data_t,
        c=c,
        m=FUZZY_M,
        error=FUZZY_ERROR,
        maxiter=FUZZY_MAXITER,
        seed=RANDOM_STATE,
    )
    labels = np.argmax(u, axis=0)
    return labels, cntr, u


def ejecutar_subtractive_chiu(
    df: pd.DataFrame,
    ra: float = SUBTRACTIVE_RA,
    rb: float = SUBTRACTIVE_RB,
    max_centros: int = 50,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Implementa Subtractive Clustering real de Chiu.

    Args:
        df (pd.DataFrame): Datos de entrada escalados
        ra (float): Radio de cluster
        rb (float): Radio de rechazo

    Returns:
        tuple: (labels, centroides)
    """
    X = df.values
    n_samples = X.shape[0]

    if n_samples == 0:
        return np.array([], dtype=int), []

    ra_denom = (ra / 2.0) ** 2
    rb_denom = (rb / 2.0) ** 2

    # Distancias entre todos los pares: vectorizado.
    if n_samples > 500:
        dist2 = cdist(X, X, metric="sqeuclidean")
    else:
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dist2 = np.sum(diff * diff, axis=2)

    potencial = np.sum(np.exp(-dist2 / ra_denom), axis=1)
    idx_max = int(np.argmax(potencial))
    p_max_inicial = float(potencial[idx_max])
    if p_max_inicial <= 0:
        return np.zeros(n_samples, dtype=int), [X[0]]

    centros = [X[idx_max].copy()]
    p_c_star = float(potencial[idx_max])

    while True:
        dist2_c = np.sum((X - centros[-1]) ** 2, axis=1)
        potencial = potencial - p_c_star * np.exp(-dist2_c / rb_denom)
        potencial = np.maximum(potencial, 0.0)

        idx_nuevo = int(np.argmax(potencial))
        p_nuevo = float(potencial[idx_nuevo])
        ratio = p_nuevo / p_max_inicial if p_max_inicial > 0 else 0.0

        if ratio < 0.22:
            break

        candidato = X[idx_nuevo]

        if ratio > 0.5:
            p_c_star = p_nuevo
            centros.append(candidato.copy())
            if len(centros) >= max_centros:
                print(
                    f"[ADVERTENCIA] Subtractive: se alcanzo el limite de {max_centros} centroides. Deteniendo."
                )
                break
            continue

        # Zona gris
        centros_arr = np.array(centros)
        dist_min = float(np.min(np.sqrt(np.sum((centros_arr - candidato) ** 2, axis=1))))
        if dist_min > ra:
            p_c_star = p_nuevo
            centros.append(candidato.copy())
            if len(centros) >= max_centros:
                print(
                    f"[ADVERTENCIA] Subtractive: se alcanzo el limite de {max_centros} centroides. Deteniendo."
                )
                break
        else:
            potencial[idx_nuevo] = 0.0
            if float(np.max(potencial)) <= 0:
                break

    centros_arr = np.array(centros)
    dist_to_centers = cdist(X, centros_arr, metric="euclidean")
    labels = np.argmin(dist_to_centers, axis=1)

    print(f"[INFO] Subtractive Chiu: {len(centros)} centros encontrados (ra={ra}, rb={rb})")

    return labels.astype(int), [c.copy() for c in centros_arr]


# -----------------------------------------------------------------------------
# Evaluacion y reportes
# -----------------------------------------------------------------------------
def _metricas_cluster_valido(data: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    """Calcula silhouette, DBI y CH cuando hay >1 cluster valido."""
    n_clusters = len(np.unique(labels))
    if n_clusters > 1 and len(data) > n_clusters:
        sil = float(silhouette_score(data, labels))
        dbi = float(davies_bouldin_score(data, labels))
        ch = float(calinski_harabasz_score(data, labels))
        return sil, dbi, ch
    return np.nan, np.nan, np.nan


def calcular_metricas_algoritmo(df: pd.DataFrame, labels: np.ndarray, nombre: str) -> dict:
    """Calcula Silhouette, Davies-Bouldin, Calinski-Harabasz, clusters y outliers."""
    if nombre == "DBSCAN":
        n_outliers = int(np.sum(labels == -1))
        labels_validos = labels[labels != -1]
        data_validos = df.values[labels != -1]
        n_clusters = len(np.unique(labels_validos)) if labels_validos.size > 0 else 0
        sil, dbi, ch = _metricas_cluster_valido(data_validos, labels_validos)
    else:
        n_outliers = 0
        n_clusters = len(np.unique(labels))
        sil, dbi, ch = _metricas_cluster_valido(df.values, labels)

    return {
        "Algoritmo": nombre,
        "Silhouette Score": sil,
        "Davies-Bouldin": dbi,
        "Calinski-Harabasz": ch,
        "Numero de clusters": int(n_clusters),
        "Numero de outliers": int(n_outliers),
    }


def _guardar_barplot_metrica(metricas_df: pd.DataFrame, col: str, title: str, out_name: str) -> None:
    """Guarda barplot de una metrica para todos los algoritmos."""
    plt.figure(figsize=(9, 5))
    df_plot = metricas_df.copy()
    df_plot[col] = df_plot[col].fillna(-1)
    sns.barplot(data=df_plot, x="Algoritmo", y=col, palette="Set2", hue="Algoritmo", legend=False)
    plt.title(title)
    plt.xlabel("Algoritmo")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / out_name, dpi=300, bbox_inches="tight")
    plt.close()


def guardar_metricas_y_grafica(metricas_df: pd.DataFrame) -> None:
    """Guarda CSV de metricas y graficas comparativas."""
    metrics_path = PLOTS_DIR / "metrics_comparison.csv"
    metricas_df.to_csv(metrics_path, index=False)

    _guardar_barplot_metrica(
        metricas_df,
        col="Silhouette Score",
        title="Comparacion de Silhouette Score por algoritmo",
        out_name="silhouette_comparison.png",
    )
    _guardar_barplot_metrica(
        metricas_df,
        col="Davies-Bouldin",
        title="Comparacion de Davies-Bouldin por algoritmo (menor es mejor)",
        out_name="davies_bouldin_comparison.png",
    )
    _guardar_barplot_metrica(
        metricas_df,
        col="Calinski-Harabasz",
        title="Comparacion de Calinski-Harabasz por algoritmo (mayor es mejor)",
        out_name="calinski_harabasz_comparison.png",
    )


def analizar_centroides_por_algoritmo(df: pd.DataFrame, labels_map: dict) -> dict:
    """
    Calcula promedios por cluster para cada algoritmo y guarda heatmaps individuales.

    Para DBSCAN se excluye ruido (-1).

    Returns:
        dict: {algoritmo: DataFrame de promedios}
    """
    resultados = {}

    for algoritmo, labels in labels_map.items():
        tmp = df.copy()
        tmp["cluster"] = labels

        if algoritmo == "DBSCAN":
            tmp = tmp[tmp["cluster"] != -1]

        if tmp.empty or tmp["cluster"].nunique() == 0:
            print(f"[ADVERTENCIA] {algoritmo}: sin clusters validos para heatmap")
            resultados[algoritmo] = pd.DataFrame()
            continue

        promedios = tmp.groupby("cluster").mean(numeric_only=True)
        resultados[algoritmo] = promedios

        plt.figure(figsize=(12, 6))
        sns.heatmap(
            promedios,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            cbar_kws={"label": "Valor medio escalado"},
        )
        plt.title(f"Promedio de variables por cluster ({algoritmo})")
        plt.xlabel("Variables")
        plt.ylabel("Cluster")
        plt.tight_layout()

        name = algoritmo.lower().replace(" ", "_").replace("-", "_")
        plt.savefig(PLOTS_DIR / f"centroids_heatmap_{name}.png", dpi=300, bbox_inches="tight")
        plt.close()

    return resultados


def generar_umap(df: pd.DataFrame) -> pd.DataFrame:
    """Genera embedding UMAP en 3D (una sola vez)."""
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=RANDOM_STATE,
    )
    emb = reducer.fit_transform(df.values)
    return pd.DataFrame(emb, columns=["UMAP1", "UMAP2", "UMAP3"])


def puntaje_balance_clusters(labels: np.ndarray) -> float:
    """Calcula puntaje de balance usando entropia normalizada."""
    labels_validos = labels[labels != -1]
    if labels_validos.size == 0:
        return 0.0

    counts = pd.Series(labels_validos).value_counts().values.astype(float)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    max_entropy = np.log(len(counts)) if len(counts) > 1 else 1.0
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def seleccionar_mejor_modelo(metricas_df: pd.DataFrame, labels_map: dict, total_muestras: int) -> tuple[str, str]:
    """Selecciona mejor modelo por silhouette, balance y penalizacion de ruido."""
    candidatos = []

    for _, row in metricas_df.iterrows():
        algoritmo = row["Algoritmo"]
        sil = row["Silhouette Score"]
        sil = float(sil) if pd.notna(sil) else -1.0
        balance = puntaje_balance_clusters(labels_map[algoritmo])
        outliers_ratio = float(row["Numero de outliers"]) / float(total_muestras)

        score = (0.6 * sil) + (0.3 * balance) - (0.1 * outliers_ratio)
        candidatos.append((algoritmo, score, sil, balance, outliers_ratio))

    candidatos.sort(key=lambda x: x[1], reverse=True)
    mejor = candidatos[0]

    explicacion = (
        f"{mejor[0]} seleccionado por mejor puntaje combinado. "
        f"Silhouette={mejor[2]:.4f}, Balance={mejor[3]:.4f}, "
        f"Ruido={mejor[4]:.4f}."
    )
    return mejor[0], explicacion


def guardar_mejor_modelo(nombre_modelo: str, explicacion: str) -> None:
    """Guarda la seleccion del mejor modelo en TXT."""
    best_path = PLOTS_DIR / "best_model.txt"
    with open(best_path, "w", encoding="utf-8") as f:
        f.write(f"Mejor modelo: {nombre_modelo}\n")
        f.write(explicacion + "\n")


def guardar_dataset_final(
    df: pd.DataFrame,
    labels_kmeans: np.ndarray,
    labels_dbscan: np.ndarray,
    labels_fuzzy: np.ndarray,
    labels_subtractive: np.ndarray,
) -> None:
    """Guarda data_with_clusters.csv con etiquetas de todos los algoritmos."""
    out = df.copy()
    out["cluster_kmeans"] = labels_kmeans
    out["cluster_dbscan"] = labels_dbscan
    out["cluster_fuzzy"] = labels_fuzzy
    out["cluster_subtractive"] = labels_subtractive
    out.to_csv(OUTPUT_DATA_PATH, index=False)


# -----------------------------------------------------------------------------
# Pipeline principal
# -----------------------------------------------------------------------------
def pipeline_clustering() -> None:
    """Pipeline principal con clustering, comparacion, visualizacion y seleccion."""
    log_lines: list[str] = []

    df = cargar_datos(INPUT_PATH)
    estado_escalado = validar_escalado(df)
    log_lines.append(f"Validacion escalado: {estado_escalado}")

    resultados = {
        "KMeans": {},
        "DBSCAN": {},
        "Fuzzy C-Means": {},
        "Subtractive": {},
    }

    # PCA baseline
    _, pca_df, pca_model = aplicar_pca(df, n_components=2)
    plt.figure(figsize=(10, 6))
    plt.scatter(pca_df["PCA1"], pca_df["PCA2"], s=25, alpha=0.7, color="#1f77b4")
    plt.title("PCA sin clusters")
    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pca_sin_clusters.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\n" + "=" * 80)
    print("APLICACION DE ALGORITMOS")
    print("=" * 80)

    # KMeans + elbow
    _ = graficar_metodo_codo(df, 1, 10)
    kmeans_labels, kmeans_centroids, _ = ejecutar_kmeans(df, KMEANS_K)
    resultados["KMeans"]["labels"] = kmeans_labels
    resultados["KMeans"]["centroids"] = kmeans_centroids
    log_lines.append(f"KMeans clusters: {len(np.unique(kmeans_labels))}")

    # DBSCAN con eps sugerido por k-distance
    eps_sugerido = calcular_kdistance_plot(df, k=5)
    eps_dbscan = eps_sugerido if eps_sugerido > 0 else DBSCAN_EPS
    print(f"[INFO] eps usado en DBSCAN: {eps_dbscan:.4f}")
    dbscan_labels, _ = ejecutar_dbscan(df, eps=eps_dbscan, min_samples=DBSCAN_MIN_SAMPLES)
    resultados["DBSCAN"]["labels"] = dbscan_labels
    log_lines.append(
        f"DBSCAN eps={eps_dbscan:.4f}, clusters={len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)}, ruido={int(np.sum(dbscan_labels == -1))}"
    )

    # Fuzzy C-Means
    fuzzy_labels, fuzzy_centers, fuzzy_membership = ejecutar_fuzzy_cmeans(df, FUZZY_C)
    resultados["Fuzzy C-Means"]["labels"] = fuzzy_labels
    resultados["Fuzzy C-Means"]["centers"] = fuzzy_centers
    resultados["Fuzzy C-Means"]["membership"] = fuzzy_membership
    log_lines.append(f"Fuzzy C-Means clusters: {len(np.unique(fuzzy_labels))}")

    # Subtractive Chiu real
    subtractive_labels, subtractive_centers = ejecutar_subtractive_chiu(df, ra=SUBTRACTIVE_RA, rb=SUBTRACTIVE_RB)
    resultados["Subtractive"]["labels"] = subtractive_labels
    resultados["Subtractive"]["centers"] = subtractive_centers
    log_lines.append(f"Subtractive (Chiu) clusters: {len(np.unique(subtractive_labels))}")

    # Metricas
    metricas = [
        calcular_metricas_algoritmo(df, resultados["KMeans"]["labels"], "KMeans"),
        calcular_metricas_algoritmo(df, resultados["DBSCAN"]["labels"], "DBSCAN"),
        calcular_metricas_algoritmo(df, resultados["Fuzzy C-Means"]["labels"], "Fuzzy C-Means"),
        calcular_metricas_algoritmo(df, resultados["Subtractive"]["labels"], "Subtractive"),
    ]
    metricas_df = pd.DataFrame(metricas)
    guardar_metricas_y_grafica(metricas_df)

    print("\n" + "=" * 80)
    print("METRICAS COMPARATIVAS")
    print("=" * 80)
    print(metricas_df)
    log_lines.append("Metricas comparativas:")
    log_lines.append(metricas_df.to_string(index=False))

    labels_map = {
        "KMeans": resultados["KMeans"]["labels"],
        "DBSCAN": resultados["DBSCAN"]["labels"],
        "Fuzzy C-Means": resultados["Fuzzy C-Means"]["labels"],
        "Subtractive": resultados["Subtractive"]["labels"],
    }

    # Graficas consolidadas
    graficar_scatter_grid_2d(
        pca_df,
        labels_map,
        x_col="PCA1",
        y_col="PCA2",
        output_name="pca_algoritmos_grid.png",
        titulo_general="Comparacion PCA por algoritmo",
    )

    # UMAP una sola vez en 3D
    umap3d_df = generar_umap(df)
    umap2d_df = umap3d_df[["UMAP1", "UMAP2"]].copy()

    # UMAP sin clusters
    graficar_umap_sin_clusters(umap2d_df, output_name="umap_sin_clusters.png")

    graficar_scatter_grid_2d(
        umap2d_df,
        labels_map,
        x_col="UMAP1",
        y_col="UMAP2",
        output_name="umap_2d_algoritmos_grid.png",
        titulo_general="Comparacion UMAP 2D por algoritmo",
    )

    graficar_scatter_grid_3d(
        umap3d_df,
        labels_map,
        output_name="umap_3d_algoritmos_grid.png",
        titulo_general="Comparacion UMAP 3D por algoritmo",
    )

    graficar_pca_vs_umap_kmeans(pca_df, umap2d_df, resultados["KMeans"]["labels"])
    graficar_distribucion_clusters_multiples(labels_map, "dist_algoritmos_grid.png")

    # Centroides por algoritmo
    centroides_promedio = analizar_centroides_por_algoritmo(df, labels_map)
    for algoritmo, promedios in centroides_promedio.items():
        if not promedios.empty:
            print(f"\nPromedios por cluster ({algoritmo}):")
            print(promedios.round(4))
            log_lines.append(f"Promedios por cluster ({algoritmo}):")
            log_lines.append(promedios.round(4).to_string())

    # Seleccion mejor modelo
    mejor_modelo, explicacion = seleccionar_mejor_modelo(metricas_df, labels_map, total_muestras=len(df))
    guardar_mejor_modelo(mejor_modelo, explicacion)

    print("\n" + "=" * 80)
    print("MEJOR MODELO")
    print("=" * 80)
    print(f"Seleccion: {mejor_modelo}")
    print(explicacion)
    log_lines.append(f"Mejor modelo: {mejor_modelo}")
    log_lines.append(explicacion)

    guardar_dataset_final(
        df,
        resultados["KMeans"]["labels"],
        resultados["DBSCAN"]["labels"],
        resultados["Fuzzy C-Means"]["labels"],
        resultados["Subtractive"]["labels"],
    )

    print("\n" + "=" * 80)
    print("PROCESO FINALIZADO")
    print("=" * 80)
    print(f"Salida de graficas y archivos: {PLOTS_DIR}")
    print(f"Dataset final: {OUTPUT_DATA_PATH}")
    print(f"Modelo PCA usado para baseline: {pca_model}")

    log_lines.append(f"Salida de graficas y archivos: {PLOTS_DIR}")
    log_lines.append(f"Dataset final: {OUTPUT_DATA_PATH}")
    log_lines.append(f"Modelo PCA: {pca_model}")

    guardar_log_consola("\n".join(log_lines))


def main() -> None:
    """Ejecuta pipeline con salida visible en consola y guarda log final."""
    asegurar_directorio_salida()
    try:
        pipeline_clustering()
    except Exception as exc:
        print("\n" + "=" * 80)
        print("ERROR EN EL PIPELINE DE CLUSTERING")
        print("=" * 80)
        print(f"Detalle: {exc}")
        raise


if __name__ == "__main__":
    main()
