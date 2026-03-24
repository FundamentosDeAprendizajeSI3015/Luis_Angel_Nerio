"""
Agrupamiento para el dataset UdeA realista (semana 9), basado en el flujo del notebook
"ejAgrupamiento_kmeans_dbscan.ipynb" y adaptado a datos de mayor dimensionalidad.

Incluye:
- KMeans con K=2 (como en el notebook).
- Metodo del codo (inercia para K=1..10).
- Seleccion automatica de K (por silhouette) para un segundo KMeans.
- DBSCAN.
- Visualizacion de todos los resultados usando UMAP.
- Generacion de HTML interactivos 3D con Plotly.

Salidas:
- Graficas en carpeta "graficas/agrupamiento_realista".
- CSV con etiquetas de cluster en carpeta "resultados/agrupamiento_realista".
- HTML interactivos 3D en carpeta "graficas/agrupamiento_realista".
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import umap.umap_ as umap
except ImportError as exc:
    raise ImportError(
        "No se encontro 'umap-learn'. Instala con: pip install umap-learn"
    ) from exc


RANDOM_STATE = 42


def ensure_output_dirs(base_dir: Path) -> Dict[str, Path]:
    """Crea carpetas de salida para graficas y tablas de resultados."""
    figures_dir = base_dir / "graficas" / "agrupamiento_realista"
    results_dir = base_dir / "resultados" / "agrupamiento_realista"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return {"figures": figures_dir, "results": results_dir}


def load_dataset(base_dir: Path) -> pd.DataFrame:
    """Carga el dataset realista preprocesado y valida formato numerico."""
    dataset_path = (
        base_dir
        / "datos_preprocesados"
        / "dataset_sintetico_FIRE_UdeA_realista_preprocesado.csv"
    )
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"No existe el dataset esperado: {dataset_path}. "
            "Ejecuta primero preprocesamiento.py"
        )

    df = pd.read_csv(dataset_path)
    if df.empty:
        raise ValueError("El dataset preprocesado esta vacio.")

    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        raise ValueError(
            f"Se esperaban solo columnas numericas, pero hay: {non_numeric_cols}"
        )

    if df.isna().sum().sum() > 0:
        raise ValueError(
            "El dataset contiene NaN. Revisa preprocesamiento.py para imputar faltantes."
        )

    return df


def fit_scaler(X: pd.DataFrame) -> np.ndarray:
    """Replica el preprocesamiento del notebook con StandardScaler."""
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    X_scaled = numeric_transformer.fit_transform(X)
    return X_scaled


def compute_umap_embedding(X_scaled: np.ndarray) -> np.ndarray:
    """Proyecta a 2D para visualizacion de clusters en alta dimensionalidad."""
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=RANDOM_STATE,
    )
    embedding = reducer.fit_transform(X_scaled)
    return embedding


def compute_umap_embedding_3d(X_scaled: np.ndarray) -> np.ndarray:
    """Proyecta a 3D para visualizacion interactiva de clusters en alta dimensionalidad."""
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=3,
        metric="euclidean",
        random_state=RANDOM_STATE,
    )
    embedding_3d = reducer.fit_transform(X_scaled)
    return embedding_3d


def plot_umap(
    embedding: np.ndarray,
    labels: np.ndarray | None,
    title: str,
    output_path: Path,
) -> None:
    """Guarda una grafica UMAP, coloreada por clusters cuando existan etiquetas."""
    fig, ax = plt.subplots(figsize=(9, 6))

    if labels is None:
        ax.scatter(embedding[:, 0], embedding[:, 1], s=25, alpha=0.8)
    else:
        scatter = ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=labels,
            s=25,
            alpha=0.85,
            cmap="tab10",
        )
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Etiqueta de cluster")

    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_umap_3d_interactive(
    embedding_3d: np.ndarray,
    labels: np.ndarray | None,
    title: str,
    output_path: Path,
) -> None:
    """Genera un HTML interactivo 3D con Plotly para visualizar y rotar clusters."""
    if labels is None:
        fig = go.Figure(data=[go.Scatter3d(
            x=embedding_3d[:, 0],
            y=embedding_3d[:, 1],
            z=embedding_3d[:, 2],
            mode='markers',
            marker=dict(size=4, color='rgba(31, 119, 180, 0.7)'),
        )])
    else:
        labels_array = np.asarray(labels, dtype=float)
        cmap = plt.get_cmap("tab10")
        vmin = float(np.min(labels_array))
        vmax = float(np.max(labels_array))
        if vmin == vmax:
            normalized = np.zeros_like(labels_array)
        else:
            normalized = (labels_array - vmin) / (vmax - vmin)

        rgba_values = cmap(normalized)
        color_array = [
            f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, 0.8)"
            for r, g, b, _ in rgba_values
        ]
        
        fig = go.Figure(data=[go.Scatter3d(
            x=embedding_3d[:, 0],
            y=embedding_3d[:, 1],
            z=embedding_3d[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color=color_array,
            ),
            text=[f"Cluster: {int(label)}" for label in labels],
            hoverinfo='text',
        )])
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        scene=dict(
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            zaxis_title="UMAP-3",
        ),
        width=1000,
        height=800,
        hovermode='closest',
    )
    
    fig.write_html(str(output_path))


def plot_umap_3d(
    embedding_3d: np.ndarray,
    labels: np.ndarray | None,
    title: str,
    output_path: Path,
) -> None:
    """Guarda una grafica UMAP 3D con multiples vistas (XY, XZ, YZ, isometrica)."""
    fig = plt.figure(figsize=(16, 12))
    
    ax1 = fig.add_subplot(2, 2, 1)
    if labels is None:
        ax1.scatter(embedding_3d[:, 0], embedding_3d[:, 1], s=20, alpha=0.7)
    else:
        ax1.scatter(
            embedding_3d[:, 0],
            embedding_3d[:, 1],
            c=labels,
            s=20,
            alpha=0.8,
            cmap="tab10",
        )
    ax1.set_xlabel("UMAP-1")
    ax1.set_ylabel("UMAP-2")
    ax1.set_title("Vista XY (superior)")
    ax1.grid(alpha=0.25)
    
    ax2 = fig.add_subplot(2, 2, 2)
    if labels is None:
        ax2.scatter(embedding_3d[:, 0], embedding_3d[:, 2], s=20, alpha=0.7)
    else:
        ax2.scatter(
            embedding_3d[:, 0],
            embedding_3d[:, 2],
            c=labels,
            s=20,
            alpha=0.8,
            cmap="tab10",
        )
    ax2.set_xlabel("UMAP-1")
    ax2.set_ylabel("UMAP-3")
    ax2.set_title("Vista XZ (frontal)")
    ax2.grid(alpha=0.25)
    
    ax3 = fig.add_subplot(2, 2, 3)
    if labels is None:
        ax3.scatter(embedding_3d[:, 1], embedding_3d[:, 2], s=20, alpha=0.7)
    else:
        ax3.scatter(
            embedding_3d[:, 1],
            embedding_3d[:, 2],
            c=labels,
            s=20,
            alpha=0.8,
            cmap="tab10",
        )
    ax3.set_xlabel("UMAP-2")
    ax3.set_ylabel("UMAP-3")
    ax3.set_title("Vista YZ (lateral)")
    ax3.grid(alpha=0.25)
    
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    if labels is None:
        ax4.scatter(
            embedding_3d[:, 0],
            embedding_3d[:, 1],
            embedding_3d[:, 2],
            s=20,
            alpha=0.7,
        )
    else:
        scatter = ax4.scatter(
            embedding_3d[:, 0],
            embedding_3d[:, 1],
            embedding_3d[:, 2],
            c=labels,
            s=20,
            alpha=0.8,
            cmap="tab10",
        )
    ax4.set_xlabel("UMAP-1")
    ax4.set_ylabel("UMAP-2")
    ax4.set_zlabel("UMAP-3")
    ax4.set_title("Vista 3D isometrica")
    
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_kmeans_k2(X_scaled: np.ndarray) -> KMeans:
    """Entrena KMeans con K=2 para mantener el flujo original del notebook."""
    model = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=20)
    model.fit(X_scaled)
    return model


def run_elbow(X_scaled: np.ndarray, k_values: List[int]) -> List[float]:
    """Calcula inercia para el metodo del codo."""
    inertias: List[float] = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        model.fit(X_scaled)
        inertias.append(float(model.inertia_))
    return inertias


def plot_elbow(k_values: List[int], inertias: List[float], output_path: Path) -> None:
    """Guarda grafica del metodo del codo."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, inertias, marker="o")
    ax.set_title("Metodo del codo - KMeans")
    ax.set_xlabel("Numero de clusters (K)")
    ax.set_ylabel("Inercia")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def choose_best_k_by_silhouette(X_scaled: np.ndarray, k_values: List[int]) -> int:
    """Selecciona K maximizando silhouette score entre K>=2."""
    best_k = k_values[0]
    best_score = -1.0

    for k in k_values:
        if k < 2:
            continue
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def run_dbscan(X_scaled: np.ndarray) -> DBSCAN:
    """Entrena DBSCAN sobre datos escalados."""
    model = DBSCAN(eps=0.8, min_samples=10)
    model.fit(X_scaled)
    return model


def save_clustered_csv(
    df: pd.DataFrame,
    kmeans2_labels: np.ndarray,
    kmeans_best_labels: np.ndarray,
    dbscan_labels: np.ndarray,
    output_path: Path,
) -> None:
    """Guarda dataset con etiquetas de clustering para analisis posterior."""
    out_df = df.copy()
    out_df["cluster_kmeans_k2"] = kmeans2_labels
    out_df["cluster_kmeans_bestk"] = kmeans_best_labels
    out_df["cluster_dbscan"] = dbscan_labels
    out_df.to_csv(output_path, index=False)


def save_summary(
    kmeans2: KMeans,
    best_k: int,
    kmeans_best: KMeans,
    dbscan: DBSCAN,
    output_path: Path,
) -> None:
    """Guarda resumen numerico de resultados principales."""
    dbscan_labels = dbscan.labels_
    unique, counts = np.unique(dbscan_labels, return_counts=True)
    dbscan_count_map = {int(label): int(count) for label, count in zip(unique, counts)}

    summary_lines = [
        "Resumen de agrupamiento - dataset UdeA realista",
        "",
        f"KMeans K=2 -> inercia: {kmeans2.inertia_:.6f}",
        f"KMeans mejor K (silhouette): {best_k}",
        f"KMeans K={best_k} -> inercia: {kmeans_best.inertia_:.6f}",
        "",
        "DBSCAN distribucion de etiquetas (incluye -1 como ruido):",
        str(dbscan_count_map),
    ]
    output_path.write_text("\n".join(summary_lines), encoding="utf-8")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    outputs = ensure_output_dirs(base_dir)

    print("Cargando dataset realista preprocesado...")
    df = load_dataset(base_dir)
    print(f"Dataset cargado: filas={df.shape[0]}, columnas={df.shape[1]}")

    X_scaled = fit_scaler(df)
    embedding = compute_umap_embedding(X_scaled)
    embedding_3d = compute_umap_embedding_3d(X_scaled)

    plot_umap(
        embedding=embedding,
        labels=None,
        title="UMAP - Dataset realista (sin clusters)",
        output_path=outputs["figures"] / "01_umap_datos_base.png",
    )
    
    plot_umap_3d(
        embedding_3d=embedding_3d,
        labels=None,
        title="UMAP 3D - Dataset realista (sin clusters)",
        output_path=outputs["figures"] / "01b_umap3d_datos_base.png",
    )
    
    plot_umap_3d_interactive(
        embedding_3d=embedding_3d,
        labels=None,
        title="UMAP 3D - Dataset realista (sin clusters) - Interactivo",
        output_path=outputs["figures"] / "01b_umap3d_datos_base.html",
    )

    print("Entrenando KMeans con K=2...")
    kmeans2 = run_kmeans_k2(X_scaled)
    print(f"Inercia K=2: {kmeans2.inertia_:.6f}")

    plot_umap(
        embedding=embedding,
        labels=kmeans2.labels_,
        title="KMeans con K=2 sobre UMAP",
        output_path=outputs["figures"] / "02_umap_kmeans_k2.png",
    )
    
    plot_umap_3d(
        embedding_3d=embedding_3d,
        labels=kmeans2.labels_,
        title="KMeans con K=2 sobre UMAP 3D",
        output_path=outputs["figures"] / "02b_umap3d_kmeans_k2.png",
    )
    
    plot_umap_3d_interactive(
        embedding_3d=embedding_3d,
        labels=kmeans2.labels_,
        title="KMeans con K=2 sobre UMAP 3D - Interactivo",
        output_path=outputs["figures"] / "02b_umap3d_kmeans_k2.html",
    )

    k_values = list(range(1, 11))
    inertias = run_elbow(X_scaled, k_values)
    plot_elbow(k_values, inertias, outputs["figures"] / "03_codo_kmeans.png")

    best_k = choose_best_k_by_silhouette(X_scaled, list(range(2, 11)))
    print(f"Mejor K por silhouette: {best_k}")
    kmeans_best = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
    kmeans_best.fit(X_scaled)

    plot_umap(
        embedding=embedding,
        labels=kmeans_best.labels_,
        title=f"KMeans con K={best_k} (seleccion silhouette)",
        output_path=outputs["figures"] / "04_umap_kmeans_mejor_k.png",
    )
    
    plot_umap_3d(
        embedding_3d=embedding_3d,
        labels=kmeans_best.labels_,
        title=f"KMeans con K={best_k} sobre UMAP 3D (seleccion silhouette)",
        output_path=outputs["figures"] / "04b_umap3d_kmeans_mejor_k.png",
    )
    
    plot_umap_3d_interactive(
        embedding_3d=embedding_3d,
        labels=kmeans_best.labels_,
        title=f"KMeans con K={best_k} sobre UMAP 3D - Interactivo (seleccion silhouette)",
        output_path=outputs["figures"] / "04b_umap3d_kmeans_mejor_k.html",
    )

    print("Entrenando DBSCAN...")
    dbscan = run_dbscan(X_scaled)
    unique_dbscan, counts_dbscan = np.unique(dbscan.labels_, return_counts=True)
    print("Etiquetas DBSCAN (cluster:conteo):")
    print(dict(zip(unique_dbscan.tolist(), counts_dbscan.tolist())))

    plot_umap(
        embedding=embedding,
        labels=dbscan.labels_,
        title="DBSCAN sobre UMAP",
        output_path=outputs["figures"] / "05_umap_dbscan.png",
    )
    
    plot_umap_3d(
        embedding_3d=embedding_3d,
        labels=dbscan.labels_,
        title="DBSCAN sobre UMAP 3D",
        output_path=outputs["figures"] / "05b_umap3d_dbscan.png",
    )
    
    plot_umap_3d_interactive(
        embedding_3d=embedding_3d,
        labels=dbscan.labels_,
        title="DBSCAN sobre UMAP 3D - Interactivo",
        output_path=outputs["figures"] / "05b_umap3d_dbscan.html",
    )

    save_clustered_csv(
        df=df,
        kmeans2_labels=kmeans2.labels_,
        kmeans_best_labels=kmeans_best.labels_,
        dbscan_labels=dbscan.labels_,
        output_path=outputs["results"] / "dataset_realista_con_clusters.csv",
    )

    save_summary(
        kmeans2=kmeans2,
        best_k=best_k,
        kmeans_best=kmeans_best,
        dbscan=dbscan,
        output_path=outputs["results"] / "resumen_agrupamiento.txt",
    )

    print("Proceso finalizado.")
    print(f"Graficas guardadas en: {outputs['figures']}")
    print(f"Resultados guardados en: {outputs['results']}")


if __name__ == "__main__":
    main()
