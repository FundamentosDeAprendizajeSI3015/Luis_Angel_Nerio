# ==========================================================
# Clustering Pipeline: Hierarchical, DBSCAN, HDBSCAN & UMAP
# Author: Academic pipeline for clustering (Clase 1h30)
# ==========================================================

# -----------------------------
# 1. CARGA DE LIBRERÍAS
# -----------------------------

import numpy as np
import pandas as pd

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocesamiento
from sklearn.preprocessing import StandardScaler

# Clustering jerárquico
from scipy.cluster.hierarchy import linkage, dendrogram

# DBSCAN
from sklearn.cluster import DBSCAN

# HDBSCAN (requiere: pip install hdbscan)
import hdbscan

# Reducción de dimensión
from sklearn.decomposition import PCA
import umap.umap_ as umap  # requiere: pip install umap-learn

# Métricas
from sklearn.metrics import silhouette_score

# Estilo gráfico moderno
sns.set(style="whitegrid", context="talk")

# -----------------------------
# 2. CARGA DE DATOS
# -----------------------------

def load_data(csv_path: str) -> pd.DataFrame:
    """
    Carga datos desde un archivo CSV.
    m = número de observaciones
    n = dimensión del espacio
    """
    df = pd.read_csv(csv_path)
    print(f"[INFO] Datos cargados: {df.shape[0]} observaciones, {df.shape[1]} variables")
    return df

# -----------------------------
# 3. PREPROCESAMIENTO
# -----------------------------

def preprocess_data(df: pd.DataFrame) -> np.ndarray:
    """
    Escalamiento estándar (media 0, varianza 1).
    Fundamental para DBSCAN / HDBSCAN.
    """
    scaler = StandardScaler()
    return scaler.fit_transform(df.values)

# -----------------------------
# 4. CLUSTERING JERÁRQUICO
# -----------------------------

def hierarchical_clustering(X: np.ndarray, method: str):
    """
    method ∈ {'single', 'complete', 'average', 'ward'}
    """
    return linkage(X, method=method)


def plot_dendrogram(Z, method: str, truncate_level: int = 40):
    plt.figure(figsize=(14, 6))
    dendrogram(Z, truncate_mode='level', p=truncate_level)
    plt.title(f"Dendrograma jerárquico – {method.upper()}")
    plt.xlabel("Observaciones")
    plt.ylabel("Distancia")
    plt.tight_layout()
    plt.show()

# -----------------------------
# 5. DBSCAN
# -----------------------------

def run_dbscan(X: np.ndarray, eps: float = 0.5, min_samples: int = 5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit_predict(X)

# -----------------------------
# 6. HDBSCAN
# -----------------------------

def run_hdbscan(X: np.ndarray, min_cluster_size: int = 10):
    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = model.fit_predict(X)
    return labels, model

# -----------------------------
# 7. REDUCCIÓN DE DIMENSIÓN
# -----------------------------

def reduce_pca(X: np.ndarray, n_components: int = 2):
    pca = PCA(n_components=n_components)
    X_red = pca.fit_transform(X)
    print(f"[INFO] Varianza explicada PCA: {pca.explained_variance_ratio_.sum():.2%}")
    return X_red


def reduce_umap(X: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1):
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=42
    )
    return reducer.fit_transform(X)

# -----------------------------
# 8. VISUALIZACIÓN
# -----------------------------

def plot_clusters(X_2d: np.ndarray, labels: np.ndarray, title: str):
    df_plot = pd.DataFrame({
        'Dim1': X_2d[:, 0],
        'Dim2': X_2d[:, 1],
        'Cluster': labels.astype(str)
    })

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df_plot,
        x='Dim1',
        y='Dim2',
        hue='Cluster',
        palette='tab10',
        s=70,
        alpha=0.85
    )
    plt.title(title)
    plt.legend(title='Clúster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# -----------------------------
# 9. EVALUACIÓN
# -----------------------------

def silhouette(X: np.ndarray, labels: np.ndarray):
    mask = labels != -1
    if len(np.unique(labels[mask])) > 1:
        score = silhouette_score(X[mask], labels[mask])
        print(f"[INFO] Silhouette score: {score:.3f}")
    else:
        print("[WARNING] Silhouette no definido")

# -----------------------------
# 10. PIPELINE PRINCIPAL
# -----------------------------

def main():
    csv_path = 'datos.csv'

    df = load_data(csv_path)
    X = preprocess_data(df)

    # Reducciones de dimensión
    X_pca = reduce_pca(X)
    X_umap = reduce_umap(X)

    # Clustering jerárquico
    for method in ['single', 'complete', 'average', 'ward']:
        Z = hierarchical_clustering(X, method)
        plot_dendrogram(Z, method)

    # DBSCAN
    labels_db = run_dbscan(X, eps=0.5, min_samples=5)
    plot_clusters(X_pca, labels_db, 'DBSCAN + PCA')
    plot_clusters(X_umap, labels_db, 'DBSCAN + UMAP')
    silhouette(X, labels_db)

    # HDBSCAN
    labels_hdb, _ = run_hdbscan(X, min_cluster_size=10)
    plot_clusters(X_pca, labels_hdb, 'HDBSCAN + PCA')
    plot_clusters(X_umap, labels_hdb, 'HDBSCAN + UMAP')
    silhouette(X, labels_hdb)


if __name__ == '__main__':
    main()
