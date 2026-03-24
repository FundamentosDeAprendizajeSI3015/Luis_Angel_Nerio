"""
DBSCAN sensitivity analysis for dataset_sintetico_FIRE_UdeA (large dataset).

Goal:
- Run DBSCAN with multiple eps (radio) values.
- Plot all results in one single image (subplots) to compare how clusters change.
"""

from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


RANDOM_STATE = 42
BASE_DIR = Path(__file__).parent
DATA_PREP_PATH = BASE_DIR / "datos_preprocesados" / "dataset_sintetico_FIRE_UdeA_preprocesado.csv"
DATA_ORIG_PATH = BASE_DIR / "dataset_sintetico_FIRE_UdeA.csv"
OUT_DIR = BASE_DIR / "figuras_dbscan_sensibilidad"
OUT_DIR.mkdir(exist_ok=True)


def load_data() -> tuple[pd.DataFrame, np.ndarray]:
    x_df = pd.read_csv(DATA_PREP_PATH)
    y_true = pd.read_csv(DATA_ORIG_PATH)["label"].to_numpy()
    return x_df, y_true


def project_2d(x_scaled: np.ndarray) -> np.ndarray:
    if UMAP_AVAILABLE:
        reducer = umap.UMAP(
            n_neighbors=20,
            min_dist=0.08,
            n_components=2,
            random_state=RANDOM_STATE,
        )
        return reducer.fit_transform(x_scaled)

    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    return pca.fit_transform(x_scaled)


def estimate_eps_values(
    x_scaled: np.ndarray,
    min_samples: int,
    quantiles: List[float],
) -> Tuple[np.ndarray, List[float]]:
    """Estimate candidate eps values from sorted k-distance curve."""
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(x_scaled)
    distances, _ = nn.kneighbors(x_scaled)

    kth_distances = np.sort(distances[:, -1])
    eps_values = sorted({round(float(np.quantile(kth_distances, q)), 3) for q in quantiles})
    return kth_distances, eps_values


def run_dbscan_sweep(x_scaled: np.ndarray, eps_values: List[float], min_samples: int) -> List[Dict]:
    results: List[Dict] = []

    for eps in eps_values:
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(x_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))

        results.append(
            {
                "eps": eps,
                "min_samples": min_samples,
                "labels": labels,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "noise_pct": 100.0 * n_noise / len(labels),
            }
        )

    return results


def plot_results(
    x_2d: np.ndarray,
    y_true: np.ndarray,
    kth_distances: np.ndarray,
    eps_values: List[float],
    min_samples: int,
    results: List[Dict],
    title: str,
    out_file: Path,
) -> None:
    total_panels = len(results) + 2
    cols = 3
    rows = int(np.ceil(total_panels / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.array(axes).reshape(-1)

    # Panel 1: labels originales
    ax0 = axes[0]
    s0 = ax0.scatter(x_2d[:, 0], x_2d[:, 1], c=y_true, cmap="Set1", s=18, alpha=0.8)
    ax0.set_title("Original labels")
    ax0.set_xlabel("Dim 1")
    ax0.set_ylabel("Dim 2")
    fig.colorbar(s0, ax=ax0, fraction=0.046, pad=0.04)

    # Panel 2: k-distance curve to justify eps choices
    ax1 = axes[1]
    ax1.plot(np.arange(len(kth_distances)), kth_distances, color="black", linewidth=1.3)
    for eps in eps_values:
        ax1.axhline(y=eps, linestyle="--", alpha=0.35)
    ax1.set_title(f"k-distance curve (k={min_samples})")
    ax1.set_xlabel("Sorted points")
    ax1.set_ylabel("Distance to k-th neighbor")
    ax1.grid(alpha=0.25)

    # Paneles DBSCAN
    for i, result in enumerate(results, start=2):
        ax = axes[i]
        labels = result["labels"]

        mask = labels != -1
        if np.any(mask):
            sc = ax.scatter(
                x_2d[mask, 0],
                x_2d[mask, 1],
                c=labels[mask],
                cmap="tab10",
                s=18,
                alpha=0.8,
            )
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

        if np.any(~mask):
            ax.scatter(
                x_2d[~mask, 0],
                x_2d[~mask, 1],
                c="gray",
                marker="x",
                s=24,
                alpha=0.7,
                label="noise",
            )
            ax.legend(loc="best")

        ax.set_title(
            f"eps={result['eps']}, min_samples={result['min_samples']}\n"
            f"clusters={result['n_clusters']}, noise={result['n_noise']} ({result['noise_pct']:.1f}%)"
        )
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")

    # Ocultar ejes vacios
    for j in range(total_panels, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(out_file, dpi=220)
    plt.close(fig)


def save_summary(results: List[Dict], out_file: Path) -> None:
    rows = []
    for r in results:
        rows.append(
            {
                "eps": r["eps"],
                "min_samples": r["min_samples"],
                "n_clusters": r["n_clusters"],
                "n_noise": r["n_noise"],
                "noise_pct": r["noise_pct"],
            }
        )

    pd.DataFrame(rows).to_csv(out_file, index=False)


def main() -> None:
    min_samples = 5
    quantiles = [0.45, 0.60, 0.72, 0.80, 0.88, 0.93, 0.97, 0.99]

    x_df, y_true = load_data()
    x_scaled = StandardScaler().fit_transform(x_df)
    x_2d = project_2d(x_scaled)
    kth_distances, eps_values = estimate_eps_values(
        x_scaled=x_scaled,
        min_samples=min_samples,
        quantiles=quantiles,
    )

    results = run_dbscan_sweep(x_scaled, eps_values, min_samples=min_samples)

    fig_path = OUT_DIR / "dbscan_sensibilidad_grande.png"
    csv_path = OUT_DIR / "dbscan_sensibilidad_grande_resumen.csv"

    plot_results(
        x_2d=x_2d,
        y_true=y_true,
        kth_distances=kth_distances,
        eps_values=eps_values,
        min_samples=min_samples,
        results=results,
        title="DBSCAN sensitivity - UdeA grande",
        out_file=fig_path,
    )
    save_summary(results, csv_path)

    print("Done")
    print(f"Figure: {fig_path}")
    print(f"Summary: {csv_path}")


if __name__ == "__main__":
    main()
