"""
================================================================================
VALIDACIÓN DE CLUSTERING: Análisis de concordancia entre clusters y etiquetas
================================================================================

Objetivo:
    Verificar si los datos etiquetados en el dataset UdeA realista están 
    verdaderamente agrupados según sus características. Se realiza clustering
    sin usar las etiquetas y se comparan los clusters resultantes con las
    etiquetas originales usando múltiples métricas.

Proceso:
    1. Carga el dataset preprocesado (sin etiquetas)
    2. Obtiene las etiquetas originales del dataset sin preprocesar
    3. Aplica múltiples algoritmos de clustering (KMeans, DBSCAN, HDBSCAN)
    4. Compara cada algoritmo con las etiquetas originales
    5. Calcula métricas de concordancia y pureza
    6. Genera visualizaciones y reportes

Salidas:
    - Gráficos de clustering (UMAP 2D)
    - Matriz de confusión
    - CSV con resultados detallados
    - Resumen de métricas de validación
"""

from pathlib import Path
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix,
    homogeneity_completeness_v_measure
)

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

RANDOM_STATE = 42
WORKSPACE_DIR = Path(__file__).parent
DATA_DIR = WORKSPACE_DIR / "datos_preprocesados"
OUTPUT_DIR = WORKSPACE_DIR / "resultados_validacion"
FIGURES_DIR = WORKSPACE_DIR / "figuras_validacion"

# Crear directorios de salida
OUTPUT_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Estilos gráficos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# 1. CARGA DE DATOS
# ============================================================================

def load_data() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Carga los datos preprocesados y las etiquetas originales.
    
    Returns:
        - X: DataFrame con características preprocesadas (80, 21)
        - y: Array con etiquetas originales (80,)
        - feature_names: Nombres de las características
    """
    print("[INFO] Cargando datos...")
    
    # Dataset preprocesado (sin etiquetas)
    processed_path = DATA_DIR / "dataset_sintetico_FIRE_UdeA_realista_preprocesado.csv"
    X = pd.read_csv(processed_path)
    print(f"  ✓ Dataset preprocesado cargado: {X.shape}")
    
    # Dataset original (para obtener etiquetas)
    original_path = WORKSPACE_DIR / "dataset_sintetico_FIRE_UdeA_realista.csv"
    df_original = pd.read_csv(original_path)
    y = df_original['label'].values
    print(f"  ✓ Etiquetas originales cargadas: {y.shape}")
    
    # Distribución de clases
    unique_labels, counts = np.unique(y, return_counts=True)
    print(f"\n  Distribución de etiquetas:")
    for label, count in zip(unique_labels, counts):
        pct = (count / len(y)) * 100
        print(f"    - Clase {label}: {count} muestras ({pct:.1f}%)")
    
    return X, y, X.columns.tolist(), df_original


# ============================================================================
# 2. PREPROCESAMIENTO
# ============================================================================

def preprocess_features(X: pd.DataFrame) -> np.ndarray:
    """
    Normaliza las características usando StandardScaler.
    
    Args:
        X: Features sin normalizar
        
    Returns:
        X_scaled: Features normalizadas
    """
    print("\n[INFO] Normalizando características...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"  ✓ Características normalizadas: media={X_scaled.mean():.4f}, std={X_scaled.std():.4f}")
    return X_scaled


# ============================================================================
# 3. REDUCCIÓN DE DIMENSIÓN (para visualización)
# ============================================================================

def compute_umap_embedding(X_scaled: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Computa embeddings UMAP para visualización.
    
    Args:
        X_scaled: Datos normalizados
        n_components: 2 o 3 dimensiones
        
    Returns:
        Embeddings UMAP
    """
    if not UMAP_AVAILABLE:
        print("⚠ UMAP no disponible. Usando PCA como alternativa.")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        return pca.fit_transform(X_scaled)
    
    print(f"\n[INFO] Computando UMAP (n_components={n_components})...")
    
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=n_components,
        metric='euclidean',
        random_state=RANDOM_STATE,
        verbose=0
    )
    embedding = reducer.fit_transform(X_scaled)
    print(f"  ✓ UMAP embedding computado: {embedding.shape}")
    return embedding


# ============================================================================
# 4. CLUSTERING - KMeans
# ============================================================================

def run_kmeans(X_scaled: np.ndarray, n_clusters: int = 2) -> Tuple[np.ndarray, float]:
    """
    Aplica KMeans con validación automática de K.
    
    Args:
        X_scaled: Datos normalizados
        n_clusters: Número de clusters esperado
        
    Returns:
        labels: Etiquetas de cluster
        silhouette: Score de silhouette
    """
    print(f"\n[INFO] Ejecutando KMeans con K={n_clusters}...")
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init=20,
        max_iter=300
    )
    labels = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, labels)
    
    print(f"  ✓ KMeans completado")
    print(f"  ✓ Silhouette Score: {sil_score:.4f}")
    
    return labels, sil_score


def find_optimal_clusters(X_scaled: np.ndarray, k_range: List[int] = None) -> Tuple[int, np.ndarray, List[float]]:
    """
    Encuentra el número óptimo de clusters usando silhouette score.
    
    Args:
        X_scaled: Datos normalizados
        k_range: Rango de K a probar
        
    Returns:
        optimal_k: Número óptimo de clusters
        labels: Etiquetas con K óptimo
        scores: Silhouette scores para cada K
    """
    if k_range is None:
        k_range = range(2, 8)
    
    print(f"\n[INFO] Evaluando K óptimo (rango: {k_range.start}-{k_range.stop})...")
    
    scores = []
    cluster_results = {}
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append(score)
        cluster_results[k] = labels
        print(f"  K={k}: Silhouette={score:.4f}")
    
    optimal_k = k_range[np.argmax(scores)]
    optimal_labels = cluster_results[optimal_k]
    
    print(f"\n  ✓ K óptimo encontrado: {optimal_k} (Silhouette={max(scores):.4f})")
    
    return optimal_k, optimal_labels, scores


# ============================================================================
# 5. CLUSTERING - DBSCAN
# ============================================================================

def run_dbscan(X_scaled: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> Tuple[np.ndarray, int, float]:
    """
    Aplica DBSCAN para clustering.
    
    Args:
        X_scaled: Datos normalizados
        eps: Radio de vecindad
        min_samples: Mínimo de muestras por cluster
        
    Returns:
        labels: Etiquetas de cluster (-1 para ruido)
        n_clusters: Número de clusters encontrados
        silhouette: Score de silhouette (solo si hay clusters válidos)
    """
    print(f"\n[INFO] Ejecutando DBSCAN (eps={eps}, min_samples={min_samples})...")
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"  ✓ DBSCAN completado")
    print(f"  ✓ Clusters encontrados: {n_clusters}")
    print(f"  ✓ Puntos de ruido: {n_noise}")
    
    # Calcular silhouette score
    if n_clusters > 1:
        # Filtrar puntos de ruido para el cálculo
        mask = labels != -1
        if mask.sum() > 0:
            sil_score = silhouette_score(X_scaled[mask], labels[mask])
            print(f"  ✓ Silhouette Score (excluyendo ruido): {sil_score:.4f}")
        else:
            sil_score = -1
    else:
        sil_score = -1
    
    return labels, n_clusters, sil_score


# ============================================================================
# 6. CLUSTERING - HDBSCAN
# ============================================================================

def run_hdbscan(X_scaled: np.ndarray, min_cluster_size: int = 5) -> Tuple[np.ndarray, int, float]:
    """
    Aplica HDBSCAN para clustering jerárquico robusto.
    
    Args:
        X_scaled: Datos normalizados
        min_cluster_size: Tamaño mínimo de cluster
        
    Returns:
        labels: Etiquetas de cluster (-1 para ruido)
        n_clusters: Número de clusters encontrados
        silhouette: Score de silhouette
    """
    if not HDBSCAN_AVAILABLE:
        print("⚠ HDBSCAN no disponible. Saltando...")
        return None, 0, -1
    
    print(f"\n[INFO] Ejecutando HDBSCAN (min_cluster_size={min_cluster_size})...")
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(X_scaled)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"  ✓ HDBSCAN completado")
    print(f"  ✓ Clusters encontrados: {n_clusters}")
    print(f"  ✓ Puntos de ruido: {n_noise}")
    
    if n_clusters > 1:
        mask = labels != -1
        if mask.sum() > 0:
            sil_score = silhouette_score(X_scaled[mask], labels[mask])
            print(f"  ✓ Silhouette Score: {sil_score:.4f}")
        else:
            sil_score = -1
    else:
        sil_score = -1
    
    return labels, n_clusters, sil_score


# ============================================================================
# 7. MÉTRICAS DE VALIDACIÓN
# ============================================================================

def calculate_validation_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                 algorithm_name: str) -> Dict:
    """
    Calcula múltiples métricas de validación entre etiquetas verdaderas y predichas.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Etiquetas predichas por clustering
        algorithm_name: Nombre del algoritmo
        
    Returns:
        Dictionary con todas las métricas
    """
    print(f"\n[INFO] Calculando métricas de validación para {algorithm_name}...")
    
    # Filtrar puntos de ruido (DBSCAN/HDBSCAN pueden tener -1)
    valid_mask = y_pred != -1
    n_noise_points = np.sum(~valid_mask)
    
    if valid_mask.sum() == 0:
        print(f"  ✗ No hay clusters válidos. Saltando métricas.")
        return None
    
    y_true_filtered = y_true[valid_mask]
    y_pred_filtered = y_pred[valid_mask]
    
    # 1. Adjusted Rand Index (ARI)
    ari = adjusted_rand_score(y_true_filtered, y_pred_filtered)
    
    # 2. Normalized Mutual Information (NMI)
    nmi = normalized_mutual_info_score(y_true_filtered, y_pred_filtered)
    
    # 3. Homogeneidad, Completitud y V-measure
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
        y_true_filtered, y_pred_filtered
    )
    
    # 4. Pureza de clusters
    # Para cada cluster predicho, encontrar la etiqueta verdadera más común
    n_clusters_pred = len(np.unique(y_pred_filtered))
    purity_scores = []
    
    for cluster_id in np.unique(y_pred_filtered):
        mask = y_pred_filtered == cluster_id
        if mask.sum() > 0:
            # Encontrar la etiqueta más común en este cluster
            cluster_labels = y_true_filtered[mask]
            most_common_count = np.bincount(cluster_labels).max()
            purity = most_common_count / mask.sum()
            purity_scores.append(purity)
    
    average_purity = np.mean(purity_scores) if purity_scores else 0
    
    # 5. Matriz de confusión
    cm = confusion_matrix(y_true_filtered, y_pred_filtered)
    
    metrics = {
        'algorithm': algorithm_name,
        'n_samples': len(y_true),
        'n_noise_points': n_noise_points,
        'n_clusters': n_clusters_pred,
        'ari': ari,
        'nmi': nmi,
        'homogeneity': homogeneity,
        'completeness': completeness,
        'v_measure': v_measure,
        'average_purity': average_purity,
        'confusion_matrix': cm
    }
    
    # Imprimir resumen
    print(f"\n  ╔════════════════════════════════════╗")
    print(f"  ║ MÉTRICAS - {algorithm_name:25} ║")
    print(f"  ╚════════════════════════════════════╝")
    print(f"  • Muestras analizadas:     {len(y_true_filtered)}/{len(y_true)}")
    print(f"  • Puntos de ruido:         {n_noise_points}")
    print(f"  • Clusters encontrados:    {n_clusters_pred}")
    print(f"  ┌─ Índices de Similitud ─────────────┐")
    print(f"  │ Adjusted Rand Index (ARI):   {ari:7.4f}  │")
    print(f"  │ Mutual Information (NMI):    {nmi:7.4f}  │")
    print(f"  │ V-measure Score:             {v_measure:7.4f}  │")
    print(f"  └────────────────────────────────────┘")
    print(f"  ┌─ Análisis de Coherencia ───────────┐")
    print(f"  │ Homogeneidad:                {homogeneity:7.4f}  │")
    print(f"  │ Completitud:                 {completeness:7.4f}  │")
    print(f"  │ Pureza Promedio:             {average_purity:7.4f}  │")
    print(f"  │ Pureza (porcentaje):         {average_purity*100:6.2f}%  │")
    print(f"  └────────────────────────────────────┘")
    
    return metrics


# ============================================================================
# 8. VISUALIZACIÓN
# ============================================================================

def plot_clustering_results(X_2d: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                            algorithm_name: str, save_prefix: str = ""):
    """
    Crea visualización comparativa del clustering vs etiquetas reales.
    
    Args:
        X_2d: Datos en 2D (UMAP/PCA)
        y_true: Etiquetas verdaderas
        y_pred: Etiquetas predichas
        algorithm_name: Nombre del algoritmo
        save_prefix: Prefijo para guardar figura
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gráfico 1: Etiquetas verdaderas
    scatter1 = axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, 
                               cmap='Set1', s=100, alpha=0.7, edgecolors='k', linewidth=0.5)
    axes[0].set_title('Etiquetas Originales (Ground Truth)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('UMAP-1')
    axes[0].set_ylabel('UMAP-2')
    plt.colorbar(scatter1, ax=axes[0], label='Clase')
    axes[0].grid(alpha=0.3)
    
    # Gráfico 2: Clusters predichos
    # Manejar puntos de ruido si existen
    mask = y_pred != -1
    scatter2 = axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1], c=y_pred[mask], 
                               cmap='tab10', s=100, alpha=0.7, edgecolors='k', linewidth=0.5)
    
    if not mask.all():
        # Graficar puntos de ruido en gris
        axes[1].scatter(X_2d[~mask, 0], X_2d[~mask, 1], c='gray', 
                       marker='X', s=150, alpha=0.5, label='Ruido')
        axes[1].legend()
    
    axes[1].set_title(f'Clusters Predichos ({algorithm_name})', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('UMAP-1')
    axes[1].set_ylabel('UMAP-2')
    plt.colorbar(scatter2, ax=axes[1], label='Cluster')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    filepath = FIGURES_DIR / f"{save_prefix}_{algorithm_name.replace(' ', '_').lower()}.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n  ✓ Figura guardada: {filepath.name}")
    plt.close()


def plot_confusion_matrices(metrics_list: List[Dict]):
    """
    Grafica matrices de confusión para todos los algoritmos.
    
    Args:
        metrics_list: Lista de diccionarios con métricas
    """
    n_algorithms = len(metrics_list)
    fig, axes = plt.subplots(1, n_algorithms, figsize=(6*n_algorithms, 5))
    
    if n_algorithms == 1:
        axes = [axes]
    
    for idx, (ax, metrics) in enumerate(zip(axes, metrics_list)):
        if metrics is None:
            continue
        
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_title(f"{metrics['algorithm']}\nMatriz de Confusión", fontweight='bold')
        ax.set_ylabel('Etiqueta Verdadera')
        ax.set_xlabel('Cluster Predicho')
    
    plt.tight_layout()
    filepath = FIGURES_DIR / "confusion_matrices.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n  ✓ Matriz de confusión guardada: {filepath.name}")
    plt.close()


def plot_metrics_comparison(metrics_list: List[Dict]):
    """
    Crea gráficos comparativos de las métricas entre algoritmos.
    
    Args:
        metrics_list: Lista de diccionarios con métricas
    """
    algorithms = [m['algorithm'] for m in metrics_list if m is not None]
    ari_scores = [m['ari'] for m in metrics_list if m is not None]
    nmi_scores = [m['nmi'] for m in metrics_list if m is not None]
    v_measure_scores = [m['v_measure'] for m in metrics_list if m is not None]
    purity_scores = [m['average_purity'] for m in metrics_list if m is not None]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ARI
    axes[0, 0].bar(algorithms, ari_scores, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Adjusted Rand Index (ARI)', fontweight='bold')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_ylim([-1, 1])
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].grid(alpha=0.3)
    
    # NMI
    axes[0, 1].bar(algorithms, nmi_scores, color='mediumseagreen', alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Normalized Mutual Information (NMI)', fontweight='bold')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(alpha=0.3)
    
    # V-measure
    axes[1, 0].bar(algorithms, v_measure_scores, color='coral', alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('V-Measure Score', fontweight='bold')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(alpha=0.3)
    
    # Pureza (porcentaje)
    purity_percentages = [p * 100 for p in purity_scores]
    axes[1, 1].bar(algorithms, purity_percentages, color='mediumpurple', alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Pureza Promedio de Clusters (%)', fontweight='bold')
    axes[1, 1].set_ylabel('Pureza (%)')
    axes[1, 1].set_ylim([0, 100])
    axes[1, 1].grid(alpha=0.3)
    
    for idx in range(2):
        for jdx in range(2):
            axes[idx, jdx].tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    filepath = FIGURES_DIR / "metrics_comparison.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n  ✓ Comparación de métricas guardada: {filepath.name}")
    plt.close()


def map_clusters_to_majority_class(y_true: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:
    """Mapea cada cluster a la clase real mayoritaria para convertirlo en predicción de clase."""
    y_mapped = np.zeros_like(y_true)
    for cluster_id in np.unique(cluster_labels):
        idx = np.where(cluster_labels == cluster_id)[0]
        if len(idx) == 0:
            continue
        majority_class = np.bincount(y_true[idx]).argmax()
        y_mapped[idx] = majority_class
    return y_mapped


def build_error_table_by_unit(
    df_original: pd.DataFrame,
    y_true: np.ndarray,
    cluster_labels: np.ndarray,
) -> pd.DataFrame:
    """Construye tabla de error por unidad para un modelo de clustering."""
    y_pred = map_clusters_to_majority_class(y_true, cluster_labels)
    error = (y_pred != y_true).astype(int)

    df_eval = pd.DataFrame({
        'unidad': df_original['unidad'].astype(str),
        'error': error
    })

    summary = (
        df_eval
        .groupby('unidad', as_index=False)
        .agg(n=('error', 'size'), errores=('error', 'sum'))
    )
    summary['error_pct'] = 100 * summary['errores'] / summary['n']
    return summary.sort_values('error_pct', ascending=False)


def plot_error_percentage_by_unit(
    df_original: pd.DataFrame,
    y_true: np.ndarray,
    labels_kmeans_k2: np.ndarray,
    labels_kmeans_kopt: np.ndarray,
    k_opt: int,
):
    """
    Calcula y grafica el porcentaje de error por unidad/facultad comparando:
    - KMeans K=2
    - KMeans K-óptimo
    También genera una tabla de consenso (promedio de error entre ambos modelos).
    """
    if 'unidad' not in df_original.columns:
        print("⚠ No se encontró la columna 'unidad'. Se omite gráfica por unidad.")
        return None

    summary_k2 = build_error_table_by_unit(df_original, y_true, labels_kmeans_k2)
    summary_kopt = build_error_table_by_unit(df_original, y_true, labels_kmeans_kopt)

    csv_k2 = OUTPUT_DIR / 'error_por_unidad_kmeans_k2.csv'
    csv_kopt = OUTPUT_DIR / 'error_por_unidad_kmeans_koptimo.csv'
    summary_k2.to_csv(csv_k2, index=False)
    summary_kopt.to_csv(csv_kopt, index=False)

    # Tabla de consenso para reducir sensibilidad a un solo modelo
    consensus = summary_k2[['unidad', 'n', 'errores', 'error_pct']].rename(
        columns={'errores': 'errores_k2', 'error_pct': 'error_pct_k2'}
    ).merge(
        summary_kopt[['unidad', 'errores', 'error_pct']].rename(
            columns={'errores': 'errores_kopt', 'error_pct': 'error_pct_kopt'}
        ),
        on='unidad',
        how='inner'
    )
    consensus['error_pct_consenso'] = (consensus['error_pct_k2'] + consensus['error_pct_kopt']) / 2
    consensus = consensus.sort_values('error_pct_consenso', ascending=False)
    csv_consensus = OUTPUT_DIR / 'error_por_unidad_consenso.csv'
    consensus.to_csv(csv_consensus, index=False)

    # Figura comparativa K=2 vs K-optimo
    plot_df = consensus[['unidad', 'error_pct_k2', 'error_pct_kopt']].melt(
        id_vars='unidad',
        value_vars=['error_pct_k2', 'error_pct_kopt'],
        var_name='modelo',
        value_name='error_pct'
    )
    plot_df['modelo'] = plot_df['modelo'].map({
        'error_pct_k2': 'KMeans K=2',
        'error_pct_kopt': f'KMeans K-optimo (K={k_opt})'
    })

    plt.figure(figsize=(13, 6))
    ax = sns.barplot(
        data=plot_df,
        x='unidad',
        y='error_pct',
        hue='modelo',
        palette='Set2'
    )
    ax.set_title('Porcentaje de error por unidad: comparacion entre modelos', fontweight='bold')
    ax.set_xlabel('Unidad / Facultad')
    ax.set_ylabel('Error (%)')
    ax.set_ylim(0, 100)
    ax.tick_params(axis='x', rotation=25)
    ax.legend(title='Modelo')

    plt.tight_layout()
    fig_compare = FIGURES_DIR / 'error_por_unidad_comparacion_kmeans.png'
    plt.savefig(fig_compare, dpi=300, bbox_inches='tight')
    plt.close()

    # Figura de consenso (una sola barra por unidad)
    plt.figure(figsize=(12, 6))
    ax2 = sns.barplot(
        data=consensus,
        x='unidad',
        y='error_pct_consenso',
        palette='Reds_r'
    )
    ax2.set_title('Porcentaje de error por unidad (consenso KMeans)', fontweight='bold')
    ax2.set_xlabel('Unidad / Facultad')
    ax2.set_ylabel('Error consenso (%)')
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='x', rotation=25)

    for idx, value in enumerate(consensus['error_pct_consenso']):
        ax2.text(idx, value + 1.0, f"{value:.1f}%", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig_consensus = FIGURES_DIR / 'error_por_unidad_consenso.png'
    plt.savefig(fig_consensus, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n  ✓ Tabla error K=2 guardada: {csv_k2.name}")
    print(f"  ✓ Tabla error K-óptimo guardada: {csv_kopt.name}")
    print(f"  ✓ Tabla error consenso guardada: {csv_consensus.name}")
    print(f"  ✓ Gráfica comparativa guardada: {fig_compare.name}")
    print(f"  ✓ Gráfica consenso guardada: {fig_consensus.name}")

    return consensus


# ============================================================================
# 9. REPORTE FINAL
# ============================================================================

def generate_report(X, y, metrics_list: List[Dict], X_umap: np.ndarray):
    """
    Genera reporte completo en CSV y texto.
    
    Args:
        X: Features originales
        y: Etiquetas verdaderas
        metrics_list: Lista de métricas de cada algoritmo
        X_umap: Embeddings UMAP para visualización
    """
    print("\n" + "="*70)
    print("GENERANDO REPORTE FINAL")
    print("="*70)
    
    # 1. Reporte en CSV
    metrics_data = []
    for metrics in metrics_list:
        if metrics is not None:
            metrics_data.append({
                'Algoritmo': metrics['algorithm'],
                'N_Clusters': metrics['n_clusters'],
                'Puntos_Ruido': metrics['n_noise_points'],
                'ARI': metrics['ari'],
                'NMI': metrics['nmi'],
                'V_Measure': metrics['v_measure'],
                'Homogeneidad': metrics['homogeneity'],
                'Completitud': metrics['completeness'],
                'Pureza_Promedio': metrics['average_purity'],
                'Pureza_Porcentaje': f"{metrics['average_purity']*100:.2f}%"
            })
    
    metrics_df = pd.DataFrame(metrics_data)
    csv_path = OUTPUT_DIR / "metricas_validacion.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"\n✓ Métricas guardadas en: {csv_path.name}")
    print("\n" + metrics_df.to_string())
    
    # 2. Reporte detallado en texto
    report_path = OUTPUT_DIR / "reporte_validacion.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("REPORTE DE VALIDACIÓN DE CLUSTERING\n")
        f.write("Dataset UdeA Realista - Semana 10\n")
        f.write("="*70 + "\n\n")
        
        f.write("INFORMACIÓN DEL DATASET\n")
        f.write("-"*70 + "\n")
        f.write(f"• Número de muestras: {len(y)}\n")
        f.write(f"• Número de características: {X.shape[1]}\n")
        unique_labels, counts = np.unique(y, return_counts=True)
        f.write(f"• Clases en etiquetas originales: {len(unique_labels)}\n")
        f.write(f"  - Clase 0: {counts[0]} muestras ({counts[0]/len(y)*100:.1f}%)\n")
        f.write(f"  - Clase 1: {counts[1]} muestras ({counts[1]/len(y)*100:.1f}%)\n\n")
        
        f.write("RESULTADOS DE CLUSTERING\n")
        f.write("-"*70 + "\n")
        for metrics in metrics_list:
            if metrics is not None:
                f.write(f"\n{metrics['algorithm'].upper()}\n")
                f.write(f"  • Clusters encontrados: {metrics['n_clusters']}\n")
                f.write(f"  • Puntos de ruido: {metrics['n_noise_points']}\n")
                f.write(f"  • Adjusted Rand Index: {metrics['ari']:.4f}\n")
                f.write(f"  • Normalized Mutual Information: {metrics['nmi']:.4f}\n")
                f.write(f"  • V-measure: {metrics['v_measure']:.4f}\n")
                f.write(f"  • Homogeneidad: {metrics['homogeneity']:.4f}\n")
                f.write(f"  • Completitud: {metrics['completeness']:.4f}\n")
                f.write(f"  • PUREZA PROMEDIO: {metrics['average_purity']:.4f} ({metrics['average_purity']*100:.2f}%)\n")
                f.write(f"  → Verificación: {metrics['average_purity']*100:.1f}% de los datos\n")
                f.write(f"     están correctamente agrupados según sus etiquetas\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("INTERPRETACIÓN DE RESULTADOS\n")
        f.write("="*70 + "\n\n")
        
        f.write("MÉTRICAS EXPLICADAS:\n")
        f.write("-"*70 + "\n")
        f.write("• ARI (Adjusted Rand Index): Mide similitud entre dos particiones.\n")
        f.write("  Rango: [-1, 1]. Valores cercanos a 1 indican buena concordancia.\n\n")
        
        f.write("• NMI (Normalized Mutual Information): Mide información compartida.\n")
        f.write("  Rango: [0, 1]. Valores cercanos a 1 indican buena concordancia.\n\n")
        
        f.write("• V-measure: Promedio armónico de homogeneidad y completitud.\n")
        f.write("  Rango: [0, 1]. Mejor valor: 1.\n\n")
        
        f.write("• PUREZA: Porcentaje de muestras en clusters correctos.\n")
        f.write("  Rango: [0, 1]. Mejor valor: 1. ← MÁS DIRECTO E INTERPRETABLE\n\n")
        
        best_metrics = max([m for m in metrics_list if m is not None], 
                          key=lambda x: x['average_purity'])
        
        f.write("\nRESULTADO PRINCIPAL:\n")
        f.write("-"*70 + "\n")
        f.write(f"Mejor algoritmo: {best_metrics['algorithm']}\n")
        f.write(f"Pureza alcanzada: {best_metrics['average_purity']*100:.2f}%\n")
        f.write(f"\n{'✓' if best_metrics['average_purity'] > 0.7 else '✗'} CONCLUSIÓN:\n")
        if best_metrics['average_purity'] > 0.8:
            f.write("Los datos SÍ están bien etiquetados. El clustering sin etiquetar\n")
            f.write("logra recuperar la estructura original en >80% de los casos.\n")
        elif best_metrics['average_purity'] > 0.6:
            f.write("Los datos tienen ESTRUCTURA PARCIALMENTE CLARA. El clustering\n")
            f.write("recupera la estructura en 60-80% de los casos.\n")
        else:
            f.write("Los datos NO están bien etiquetados. El clustering sin etiquetar\n")
            f.write("recupera la estructura en <60% de los casos.\n")
            f.write("Posibles causas: etiquetado incorrecto, características inadecuadas,\n")
            f.write("o clases muy superpuestas.\n")
    
    print(f"✓ Reporte detallado guardado en: {report_path.name}")
    
    return report_path


# ============================================================================
# 10. FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Orquesta todo el pipeline de validación de clustering."""
    
    print("\n" + "="*70)
    print("VALIDACIÓN DE CLUSTERING - DATASET UDEA REALISTA")
    print("="*70)
    
    # 1. Cargar datos
    X, y, feature_names, df_original = load_data()
    
    # 2. Preprocesar
    X_scaled = preprocess_features(X)
    
    # 3. Reducir dimensión para visualización
    X_umap = compute_umap_embedding(X_scaled, n_components=2)
    print(f"  ✓ UMAP 2D computado: {X_umap.shape}")
    
    # 4. Ejecutar clustering con múltiples algoritmos
    results_list = []
    
    # 4a. KMeans K=2 (conocemos que hay 2 clases)
    print("\n" + "-"*70)
    print("CLUSTERING: KMeans (K=2)")
    print("-"*70)
    labels_kmeans_k2, sil_kmeans_k2 = run_kmeans(X_scaled, n_clusters=2)
    results_list.append(('KMeans K=2', labels_kmeans_k2, sil_kmeans_k2))
    
    # 4b. KMeans con K óptimo
    print("\n" + "-"*70)
    print("CLUSTERING: KMeans (K óptimo automático)")
    print("-"*70)
    optimal_k, labels_kmeans_opt, silhouette_scores = find_optimal_clusters(X_scaled)
    results_list.append(('KMeans K-óptimo', labels_kmeans_opt, max(silhouette_scores)))
    
    # 4c. DBSCAN
    print("\n" + "-"*70)
    print("CLUSTERING: DBSCAN")
    print("-"*70)
    labels_dbscan, n_clusters_dbscan, sil_dbscan = run_dbscan(X_scaled, eps=0.3, min_samples=3)
    results_list.append(('DBSCAN', labels_dbscan, sil_dbscan))
    
    # 4d. HDBSCAN
    print("\n" + "-"*70)
    print("CLUSTERING: HDBSCAN")
    print("-"*70)
    labels_hdbscan, n_clusters_hdbscan, sil_hdbscan = run_hdbscan(X_scaled, min_cluster_size=5)
    if labels_hdbscan is not None:
        results_list.append(('HDBSCAN', labels_hdbscan, sil_hdbscan))
    
    # 5. Calcular métricas de validación
    print("\n" + "="*70)
    print("CÁLCULO DE MÉTRICAS DE VALIDACIÓN")
    print("="*70)
    
    metrics_list = []
    for algorithm_name, labels, silhouette in results_list:
        metrics = calculate_validation_metrics(y, labels, algorithm_name)
        metrics_list.append(metrics)
    
    # 6. Crear visualizaciones
    print("\n" + "="*70)
    print("CREANDO VISUALIZACIONES")
    print("="*70)
    
    for (algorithm_name, labels, _), metrics in zip(results_list, metrics_list):
        if metrics is not None:
            plot_clustering_results(X_umap, y, labels, algorithm_name, save_prefix="clustering")
    
    plot_confusion_matrices(metrics_list)
    plot_metrics_comparison(metrics_list)
    plot_error_percentage_by_unit(
        df_original=df_original,
        y_true=y,
        labels_kmeans_k2=labels_kmeans_k2,
        labels_kmeans_kopt=labels_kmeans_opt,
        k_opt=optimal_k,
    )
    
    # 7. Generar reporte
    report_path = generate_report(X, y, metrics_list, X_umap)
    
    print("\n" + "="*70)
    print("VALIDACIÓN COMPLETADA")
    print("="*70)
    print(f"\n📁 Resultados guardados en: {OUTPUT_DIR}")
    print(f"📊 Figuras guardadas en: {FIGURES_DIR}")
    print("\nArchivos generados:")
    print(f"  • metricas_validacion.csv")
    print(f"  • reporte_validacion.txt")
    print(f"  • clustering_*.png (comparativas)")
    print(f"  • confusion_matrices.png")
    print(f"  • metrics_comparison.png")
    print(f"  • error_por_unidad_comparacion_kmeans.png")
    print(f"  • error_por_unidad_consenso.png")
    print(f"  • error_por_unidad_kmeans_k2.csv")
    print(f"  • error_por_unidad_kmeans_koptimo.csv")
    print(f"  • error_por_unidad_consenso.csv")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
