from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .config import (
    FIGURES_DIR,
    COL_GPA,
    COL_STRESS,
    COL_HOURS,
)

# UMAP (requiere umap-learn)
try:
    import umap.umap_ as umap
except Exception:
    umap = None

# Plotly (opcional para HTML interactivo)
try:
    import plotly.express as px
except Exception:
    px = None


def ensure_figures_dir() -> None:
    """Crea directorio de figuras si no existe."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    """Retorna lista de columnas con tipo numérico.
    
    Args:
        df: DataFrame a inspeccionar.
    
    Returns:
        Lista de nombres de columnas numéricas.
    """
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def save_histograms(df: pd.DataFrame) -> None:
    """Genera histogramas de todas las variables numéricas.
    
    Incluye líneas de media y mediana en cada histograma.
    Guarda archivo 'histogramas_todas_variables.png'.
    
    Args:
        df: DataFrame a visualizar.
    """
    ensure_figures_dir()
    num_cols = _numeric_columns(df)
    
    if not num_cols:
        return
    
    # Calcular grid para subplots
    n_cols = min(3, len(num_cols))
    n_rows = (len(num_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()  # Convertir a 1D para iteración
    
    for idx, col in enumerate(num_cols):
        data = df[col].dropna()
        axes[idx].hist(data, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        
        # Agregar líneas de media y mediana
        media = data.mean()
        mediana = data.median()
        
        axes[idx].axvline(media, color='red', linestyle='--', linewidth=2, label=f'Media: {media:.2f}')
        axes[idx].axvline(mediana, color='green', linestyle='--', linewidth=2, label=f'Mediana: {mediana:.2f}')
        
        axes[idx].set_title(f"Histograma - {col}", fontsize=10, fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel("Frecuencia")
        axes[idx].grid(axis='y', alpha=0.3)
        axes[idx].legend(fontsize=8)
    
    # Ocultar subplots vacíos
    for idx in range(len(num_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "histogramas_todas_variables.png", dpi=180, bbox_inches='tight')
    plt.close()


def save_outlier_boxplots(df: pd.DataFrame) -> None:
    """
    Genera box plots para detectar y visualizar outliers en variables numéricas.
    Utiliza la regla 1.5 × IQR para identificar outliers.
    """
    ensure_figures_dir()
    num_cols = _numeric_columns(df)
    
    if not num_cols:
        return
    
    # Calcular grid para subplots
    n_cols = min(3, len(num_cols))
    n_rows = (len(num_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(num_cols):
        data = df[col].dropna()
        
        # Calcular IQR y límites de outliers
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Box plot
        bp = axes[idx].boxplot(data, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        
        # Agregar información de outliers
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        num_outliers = len(outliers)
        pct_outliers = (num_outliers / len(data) * 100) if len(data) > 0 else 0
        
        axes[idx].set_title(f"{col}\n({num_outliers} outliers, {pct_outliers:.1f}%)", 
                           fontsize=10, fontweight='bold')
        axes[idx].set_ylabel(col)
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Agregar líneas de límites
        axes[idx].axhline(lower_bound, color='red', linestyle=':', alpha=0.5, label='Límite inferior')
        axes[idx].axhline(upper_bound, color='red', linestyle=':', alpha=0.5, label='Límite superior')
    
    # Ocultar subplots vacíos
    for idx in range(len(num_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "boxplots_outliers.png", dpi=180, bbox_inches='tight')
    plt.close()


def save_outlier_comparison(df: pd.DataFrame) -> None:
    """
    Genera comparación visual de estadísticas antes y después de eliminar outliers.
    """
    ensure_figures_dir()
    num_cols = _numeric_columns(df)
    
    if not num_cols:
        return
    
    # Crear gráficos de comparación (media y mediana antes/después)
    n_cols_plot = min(3, len(num_cols))
    n_rows_plot = (len(num_cols) + n_cols_plot - 1) // n_cols_plot
    
    fig, axes = plt.subplots(n_rows_plot, n_cols_plot, figsize=(15, 4*n_rows_plot))
    axes = axes.flatten()
    
    for idx, col in enumerate(num_cols):
        data = df[col].dropna()
        
        # Calcular límites de outliers
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Datos sin outliers
        data_clean = data[(data >= lower_bound) & (data <= upper_bound)]
        
        # Estadísticas
        stats_before = {
            'Media': data.mean(),
            'Mediana': data.median(),
            'Desv. Est.': data.std()
        }
        
        stats_after = {
            'Media': data_clean.mean(),
            'Mediana': data_clean.median(),
            'Desv. Est.': data_clean.std()
        }
        
        # Gráfico de barras comparativo
        x_pos = np.arange(len(stats_before))
        width = 0.35
        
        values_before = list(stats_before.values())
        values_after = list(stats_after.values())
        
        axes[idx].bar(x_pos - width/2, values_before, width, label='Antes', alpha=0.8, color='skyblue')
        axes[idx].bar(x_pos + width/2, values_after, width, label='Después', alpha=0.8, color='lightcoral')
        
        axes[idx].set_title(f"{col}", fontsize=10, fontweight='bold')
        axes[idx].set_ylabel("Valor")
        axes[idx].set_xticks(x_pos)
        axes[idx].set_xticklabels(stats_before.keys(), rotation=45, ha='right', fontsize=8)
        axes[idx].legend(fontsize=8)
        axes[idx].grid(axis='y', alpha=0.3)
    
    # Ocultar subplots vacíos
    for idx in range(len(num_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "comparacion_antes_despues_outliers.png", dpi=180, bbox_inches='tight')
    plt.close()


def save_scatter_habits_vs_gpa(df: pd.DataFrame) -> None:
    """Genera scatter plots de cada hábito vs GPA para visualizar relaciones.
    
    Crea un gráfico por cada variable de hábitos disponible.
    Guarda archivo 'scatters_habitos_vs_gpa.png'.
    
    Args:
        df: DataFrame con hábitos y GPA.
    """
    ensure_figures_dir()
    if COL_GPA not in df.columns:
        return

    scatter_cols = [col for col in sorted(COL_HOURS) if col in df.columns]
    if not scatter_cols:
        return
    
    # Calcular grid para subplots
    n_cols = min(3, len(scatter_cols))
    n_rows = (len(scatter_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()  # Convertir a 1D para iteración
    
    for idx, col in enumerate(scatter_cols):
        axes[idx].scatter(df[col], df[COL_GPA], s=10, alpha=0.6)
        axes[idx].set_title(f"{col} vs {COL_GPA}", fontsize=10, fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel(COL_GPA)
        axes[idx].grid(alpha=0.3)
    
    # Ocultar subplots vacíos
    for idx in range(len(scatter_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "scatters_habitos_vs_gpa.png", dpi=180, bbox_inches='tight')
    plt.close()


def save_boxplots_by_stress(df: pd.DataFrame) -> None:
    """Genera box plots para visualizar distribución de variables por nivel de estrés.
    
    Crea un gráfico por cada hábito y GPA, agrupados por Stress Level.
    Guarda archivo 'boxplots_por_stress.png'.
    
    Args:
        df: DataFrame con variables y Stress Level.
    """
    ensure_figures_dir()
    if COL_STRESS not in df.columns:
        return

    # Recopilar todas las columnas para boxplots
    boxplot_cols = []
    if COL_GPA in df.columns:
        boxplot_cols.append(COL_GPA)
    
    boxplot_cols.extend([col for col in sorted(COL_HOURS) if col in df.columns])
    
    if not boxplot_cols:
        return
    
    # Calcular grid para subplots
    n_cols = min(3, len(boxplot_cols))
    n_rows = (len(boxplot_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()  # Convertir a 1D para iteración
    
    for idx, col in enumerate(boxplot_cols):
        # Preparar datos para boxplot
        data_by_stress = [df[df[COL_STRESS] == stress][col].dropna().values 
                          for stress in sorted(df[COL_STRESS].unique())]

        
        axes[idx].boxplot(data_by_stress, labels=sorted(df[COL_STRESS].unique()))
        axes[idx].set_title(f"{col} por {COL_STRESS}", fontsize=10, fontweight='bold')
        axes[idx].set_xlabel(COL_STRESS)
        axes[idx].set_ylabel(col)
        axes[idx].grid(axis='y', alpha=0.3)
    
    # Ocultar subplots vacíos
    for idx in range(len(boxplot_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "boxplots_por_stress.png", dpi=180, bbox_inches='tight')
    plt.close()


def save_correlation_heatmap(df: pd.DataFrame, method: str = "pearson") -> None:
    """Genera heatmap de matriz de correlación de variables numéricas.
    
    Visualiza relaciones lineales entre variables.
    Guarda archivo 'corr_heatmap_<method>.png'.
    
    Args:
        df: DataFrame a analizar.
        method: Método de correlación ('pearson' o 'spearman').
    """
    ensure_figures_dir()
    num_cols = _numeric_columns(df)
    if len(num_cols) < 2:
        return

    corr = df[num_cols].corr(method=method, numeric_only=True)

    plt.figure(figsize=(8, 6))
    plt.imshow(corr.values)
    plt.xticks(range(len(num_cols)), num_cols, rotation=45, ha="right")
    plt.yticks(range(len(num_cols)), num_cols)
    plt.title(f"Matriz de correlacion ({method})")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"corr_heatmap_{method}.png", dpi=180)
    plt.close()


def save_log_transform_comparison(
    original_df: pd.DataFrame,
    transformed_df: pd.DataFrame,
    log_cols: List[str],
) -> None:
    ensure_figures_dir()
    if not log_cols:
        return

    n_rows = len(log_cols)
    fig_hist, axes_hist = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
    fig_box, axes_box = plt.subplots(n_rows, 2, figsize=(10, 4 * n_rows))

    if n_rows == 1:
        axes_hist = np.array([axes_hist])
        axes_box = np.array([axes_box])

    for idx, col in enumerate(log_cols):
        log_col = f"{col}_log10"
        if log_col not in transformed_df.columns or col not in original_df.columns:
            continue

        orig = pd.to_numeric(original_df[col], errors="coerce").dropna()
        log_vals = pd.to_numeric(transformed_df[log_col], errors="coerce").dropna()

        axes_hist[idx, 0].hist(orig, bins=30, edgecolor="black", alpha=0.7, color="skyblue")
        axes_hist[idx, 0].set_title(f"{col} (original)")
        axes_hist[idx, 0].grid(axis="y", alpha=0.3)

        axes_hist[idx, 1].hist(log_vals, bins=30, edgecolor="black", alpha=0.7, color="lightcoral")
        axes_hist[idx, 1].set_title(f"{log_col}")
        axes_hist[idx, 1].grid(axis="y", alpha=0.3)

        axes_box[idx, 0].boxplot(orig, vert=True, patch_artist=True)
        axes_box[idx, 0].set_title(f"{col} (original)")
        axes_box[idx, 0].grid(axis="y", alpha=0.3)

        axes_box[idx, 1].boxplot(log_vals, vert=True, patch_artist=True)
        axes_box[idx, 1].set_title(f"{log_col}")
        axes_box[idx, 1].grid(axis="y", alpha=0.3)

    fig_hist.tight_layout()
    fig_hist.savefig(FIGURES_DIR / "log_transform_hist_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig_hist)

    fig_box.tight_layout()
    fig_box.savefig(FIGURES_DIR / "log_transform_box_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig_box)


def _prepare_umap_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Usa solo columnas numéricas (hábitos + GPA si quieres).
    Para UMAP exploratorio, normalmente usamos X = hábitos (sin targets),
    así que aquí usamos SOLO variables de hábitos numéricas.
    """
    feature_cols = [c for c in sorted(COL_HOURS) if c in df.columns]
    if not feature_cols:
        raise ValueError("No encontré columnas de hábitos para UMAP.")

    X = df[feature_cols].copy()

    # Convertir a numérico por seguridad
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # Si hubiera nulos (por conversión), los eliminamos solo para UMAP
    X = X.dropna(axis=0, how="any")
    return X.values, feature_cols


def save_pca_2d(df: pd.DataFrame) -> None:
    """Genera proyección PCA 2D coloreada por GPA y Stress Level.
    
    Reduce dimensionalidad con PCA y visualiza primeros 2 componentes.
    Genera dos gráficos: uno por GPA (escala) y otro por Stress (categorías).
    Guarda archivos 'pca_2d_gpa.png' y 'pca_2d_stress.png'.
    
    Args:
        df: DataFrame con hábitos, GPA y Stress Level.
    """
    ensure_figures_dir()

    X, feature_cols = _prepare_umap_matrix(df)

    df_aligned = df[feature_cols + [c for c in [COL_GPA, COL_STRESS] if c in df.columns]].copy()
    df_aligned = df_aligned.dropna(axis=0, how="any")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    emb2 = pca.fit_transform(Xs)

    if COL_GPA in df_aligned.columns:
        plt.figure()
        sc = plt.scatter(emb2[:, 0], emb2[:, 1], c=df_aligned[COL_GPA].values, s=10)
        plt.title("PCA 2D (habitos) coloreado por GPA")
        plt.xlabel("PCA-1")
        plt.ylabel("PCA-2")
        plt.colorbar(sc, label="GPA")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "pca_2d_gpa.png", dpi=180)
        plt.close()

    if COL_STRESS in df_aligned.columns:
        stress = df_aligned[COL_STRESS].astype(str).values
        classes = sorted(np.unique(stress))
        class_to_int = {c: i for i, c in enumerate(classes)}
        stress_int = np.array([class_to_int[s] for s in stress])

        plt.figure()
        sc = plt.scatter(emb2[:, 0], emb2[:, 1], c=stress_int, s=10)
        plt.title("PCA 2D (habitos) coloreado por Stress Level")
        plt.xlabel("PCA-1")
        plt.ylabel("PCA-2")
        cb = plt.colorbar(sc)
        cb.set_ticks(range(len(classes)))
        cb.set_ticklabels(classes)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "pca_2d_stress.png", dpi=180)
        plt.close()


def save_tsne_2d(df: pd.DataFrame, random_state: int = 42) -> None:
    """
    Genera t-SNE 2D coloreado por GPA y por Stress Level.
    """
    ensure_figures_dir()

    X, feature_cols = _prepare_umap_matrix(df)
    n_samples = X.shape[0]
    if n_samples < 5:
        return

    df_aligned = df[feature_cols + [c for c in [COL_GPA, COL_STRESS] if c in df.columns]].copy()
    df_aligned = df_aligned.dropna(axis=0, how="any")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    max_perplexity = max(2, (n_samples - 1) // 3)
    perplexity = min(30, max_perplexity)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=random_state,
    )
    emb2 = tsne.fit_transform(Xs)

    if COL_GPA in df_aligned.columns:
        plt.figure()
        sc = plt.scatter(emb2[:, 0], emb2[:, 1], c=df_aligned[COL_GPA].values, s=10)
        plt.title("t-SNE 2D (habitos) coloreado por GPA")
        plt.xlabel("t-SNE-1")
        plt.ylabel("t-SNE-2")
        plt.colorbar(sc, label="GPA")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "tsne_2d_gpa.png", dpi=180)
        plt.close()

    if COL_STRESS in df_aligned.columns:
        stress = df_aligned[COL_STRESS].astype(str).values
        classes = sorted(np.unique(stress))
        class_to_int = {c: i for i, c in enumerate(classes)}
        stress_int = np.array([class_to_int[s] for s in stress])

        plt.figure()
        sc = plt.scatter(emb2[:, 0], emb2[:, 1], c=stress_int, s=10)
        plt.title("t-SNE 2D (habitos) coloreado por Stress Level")
        plt.xlabel("t-SNE-1")
        plt.ylabel("t-SNE-2")
        cb = plt.colorbar(sc)
        cb.set_ticks(range(len(classes)))
        cb.set_ticklabels(classes)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "tsne_2d_stress.png", dpi=180)
        plt.close()


def save_umap_2d_3d(df: pd.DataFrame, random_state: int = 42) -> None:
    """
    Genera:
    - UMAP 2D y 3D coloreado por GPA
    - UMAP 2D y 3D coloreado por Stress Level
    """
    ensure_figures_dir()

    if umap is None:
        raise ImportError(
            "UMAP no está instalado. Instala con: pip install umap-learn"
        )

    X, feature_cols = _prepare_umap_matrix(df)

    # Necesitamos alinear targets con las filas que quedaron (por dropna)
    # Para eso repetimos el dropna en el df con las mismas cols
    df_aligned = df[feature_cols + [c for c in [COL_GPA, COL_STRESS] if c in df.columns]].copy()
    df_aligned = df_aligned.dropna(axis=0, how="any")

    # Escalamiento recomendado para UMAP
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # --- UMAP 2D ---
    reducer2 = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=random_state,
    )
    emb2 = reducer2.fit_transform(Xs)

    # 2D coloreado por GPA
    if COL_GPA in df_aligned.columns:
        plt.figure()
        sc = plt.scatter(emb2[:, 0], emb2[:, 1], c=df_aligned[COL_GPA].values, s=10)
        plt.title("UMAP 2D (hábitos) coloreado por GPA")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.colorbar(sc, label="GPA")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "umap_2d_gpa.png", dpi=180)
        plt.close()

    # 2D coloreado por Stress (categorías)
    if COL_STRESS in df_aligned.columns:
        stress = df_aligned[COL_STRESS].astype(str).values
        classes = sorted(np.unique(stress))
        class_to_int = {c: i for i, c in enumerate(classes)}
        stress_int = np.array([class_to_int[s] for s in stress])

        plt.figure()
        sc = plt.scatter(emb2[:, 0], emb2[:, 1], c=stress_int, s=10)
        plt.title("UMAP 2D (hábitos) coloreado por Stress Level")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        cb = plt.colorbar(sc)
        cb.set_ticks(range(len(classes)))
        cb.set_ticklabels(classes)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "umap_2d_stress.png", dpi=180)
        plt.close()

    # --- UMAP 3D ---
    reducer3 = umap.UMAP(
        n_components=3,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=random_state,
    )
    emb3 = reducer3.fit_transform(Xs)

    # 3D plots requieren proyección 3D
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # 3D coloreado por GPA
    if COL_GPA in df_aligned.columns:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        p = ax.scatter(emb3[:, 0], emb3[:, 1], emb3[:, 2], c=df_aligned[COL_GPA].values, s=10)
        ax.set_title("UMAP 3D (hábitos) coloreado por GPA")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_zlabel("UMAP-3")
        fig.colorbar(p, ax=ax, label="GPA", shrink=0.6)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "umap_3d_gpa.png", dpi=180)
        plt.close()

        if px is None:
            raise ImportError("Plotly no está instalado. Instala con: pip install plotly")

        fig_html = px.scatter_3d(
            x=emb3[:, 0],
            y=emb3[:, 1],
            z=emb3[:, 2],
            color=df_aligned[COL_GPA].values,
            title="UMAP 3D (habitos) coloreado por GPA",
            labels={"x": "UMAP-1", "y": "UMAP-2", "z": "UMAP-3"},
        )
        fig_html.write_html(FIGURES_DIR / "umap_3d_gpa.html")

    # 3D coloreado por Stress
    if COL_STRESS in df_aligned.columns:
        stress = df_aligned[COL_STRESS].astype(str).values
        classes = sorted(np.unique(stress))
        class_to_int = {c: i for i, c in enumerate(classes)}
        stress_int = np.array([class_to_int[s] for s in stress])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        p = ax.scatter(emb3[:, 0], emb3[:, 1], emb3[:, 2], c=stress_int, s=10)
        ax.set_title("UMAP 3D (hábitos) coloreado por Stress Level")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_zlabel("UMAP-3")
        cb = fig.colorbar(p, ax=ax, shrink=0.6)
        cb.set_ticks(range(len(classes)))
        cb.set_ticklabels(classes)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "umap_3d_stress.png", dpi=180)
        plt.close()

        if px is None:
            raise ImportError("Plotly no está instalado. Instala con: pip install plotly")

        fig_html = px.scatter_3d(
            x=emb3[:, 0],
            y=emb3[:, 1],
            z=emb3[:, 2],
            color=stress_int,
            title="UMAP 3D (habitos) coloreado por Stress Level",
            labels={"x": "UMAP-1", "y": "UMAP-2", "z": "UMAP-3"},
        )
        fig_html.update_traces(marker={"size": 3})
        fig_html.write_html(FIGURES_DIR / "umap_3d_stress.html")


def run_all_visuals(df: pd.DataFrame) -> None:
    """Función principal: ejecuta todas las visualizaciones.
    
    Genera:
    - Histogramas de todas las variables
    - Scatter plots hábitos vs GPA
    - Box plots por Stress Level
    - Heatmaps de correlación (Pearson y Spearman)
    - Box plots de outliers
    - Comparación de estadísticas antes/después de remover outliers
    - Proyecciones dimensionales: PCA 2D, t-SNE 2D, UMAP 2D/3D
    
    Args:
        df: DataFrame a visualizar (completo, con características originales).
    """
    save_histograms(df)
    save_scatter_habits_vs_gpa(df)
    save_boxplots_by_stress(df)
    save_correlation_heatmap(df, method="pearson")
    save_correlation_heatmap(df, method="spearman")
    save_outlier_boxplots(df)
    save_outlier_comparison(df)
    save_pca_2d(df)
    save_tsne_2d(df)
    save_umap_2d_3d(df)
