# ==========================================================
# Clustering Pipeline EXTENDED
# Hierarchical, DBSCAN, HDBSCAN, UMAP
# + k-distance plot
# + Clustering stability analysis
# ==========================================================

# -----------------------------
# 1. LIBRARIES
# -----------------------------
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

from scipy.cluster.hierarchy import linkage, dendrogram
import hdbscan
import umap.umap_ as umap

sns.set(style="whitegrid", context="talk")

# -----------------------------
# 2. DATA LOADING
# -----------------------------

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    print(f"[INFO] Loaded data: {df.shape[0]} samples, {df.shape[1]} features")
    return df

# -----------------------------
# 3. PREPROCESSING
# -----------------------------

def preprocess_data(df: pd.DataFrame) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(df.values)

# -----------------------------
# 4. k-DISTANCE PLOT (DBSCAN)
# -----------------------------

def k_distance_plot(X: np.ndarray, k: int = 5):
    """
    Automatic k-distance plot for DBSCAN parameter selection.
    Typical choice: k = min_samples.
    """
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    k_distances = np.sort(distances[:, k-1])

    plt.figure(figsize=(10, 6))
    plt.plot(k_distances)
    plt.xlabel('Sorted observations')
    plt.ylabel(f'{k}-distance')
    plt.title('k-distance plot (DBSCAN)')
    plt.tight_layout()
    plt.show()

# -----------------------------
# 5. HIERARCHICAL CLUSTERING
# -----------------------------

def hierarchical_clustering(X: np.ndarray, method: str):
    return linkage(X, method=method)


def plot_dendrogram(Z, method: str, truncate_level: int = 40):
    plt.figure(figsize=(14, 6))
    dendrogram(Z, truncate_mode='level', p=truncate_level)
    plt.title(f'Dendrogram – {method.upper()}')
    plt.xlabel('Observations')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.show()

# -----------------------------
# 6. DBSCAN
# -----------------------------

def run_dbscan(X: np.ndarray, eps: float, min_samples: int):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit_predict(X)

# -----------------------------
# 7. HDBSCAN
# -----------------------------

def run_hdbscan(X: np.ndarray, min_cluster_size: int):
    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = model.fit_predict(X)
    return labels

# -----------------------------
# 8. DIMENSIONALITY REDUCTION
# -----------------------------

def reduce_pca(X: np.ndarray):
    pca = PCA(n_components=2)
    X_red = pca.fit_transform(X)
    print(f"[INFO] PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    return X_red


def reduce_umap(X: np.ndarray):
    reducer = umap.UMAP(n_components=2, random_state=42)
    return reducer.fit_transform(X)

# -----------------------------
# 9. VISUALIZATION
# -----------------------------

def plot_clusters(X_2d, labels, title):
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
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# -----------------------------
# 10. CLUSTERING STABILITY
# -----------------------------

def clustering_stability(X: np.ndarray, cluster_func, n_bootstrap: int = 20):
    """
    Estimates clustering stability using bootstrap
    and Adjusted Rand Index (ARI).
    """
    labels_ref = cluster_func(X)
    ari_scores = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(len(X), size=len(X), replace=True)
        X_sample = X[idx]
        labels_sample = cluster_func(X_sample)

        # Align sizes for ARI
        ari = adjusted_rand_score(labels_ref[idx], labels_sample)
        ari_scores.append(ari)

    print(f"[INFO] Mean ARI stability: {np.mean(ari_scores):.3f}")
    print(f"[INFO] Std ARI stability: {np.std(ari_scores):.3f}")

# -----------------------------
# 11. PIPELINE
# -----------------------------

def main():
    csv_path = 'datos.csv'

    # Load & preprocess
    df = load_data(csv_path)
    X = preprocess_data(df)

    # k-distance plot
    k_distance_plot(X, k=5)

    # Dimensionality reduction
    X_pca = reduce_pca(X)
    X_umap = reduce_umap(X)

    # Hierarchical
    for method in ['single', 'complete', 'average', 'ward']:
        Z = hierarchical_clustering(X, method)
        plot_dendrogram(Z, method)

    # DBSCAN
    db_labels = run_dbscan(X, eps=0.5, min_samples=5)
    plot_clusters(X_pca, db_labels, 'DBSCAN + PCA')
    plot_clusters(X_umap, db_labels, 'DBSCAN + UMAP')

    # Stability DBSCAN
    clustering_stability(
        X,
        lambda X_: run_dbscan(X_, eps=0.5, min_samples=5)
    )

    # HDBSCAN
    hdb_labels = run_hdbscan(X, min_cluster_size=10)
    plot_clusters(X_pca, hdb_labels, 'HDBSCAN + PCA')
    plot_clusters(X_umap, hdb_labels, 'HDBSCAN + UMAP')

    # Stability HDBSCAN
    clustering_stability(
        X,
        lambda X_: run_hdbscan(X_, min_cluster_size=10)
    )


if __name__ == '__main__':
    main()
