"""
Modulo de Analisis Exploratorio de Datos (EDA)

Incluye:
- Estadisticas descriptivas completas
- Deteccion de outliers con IQR
- Visualizaciones: histogramas, boxplots, dispersiones
- Matriz de correlacion y heatmap
- Analisis de variables vs Stress_Level
- Generacion de conclusiones automaticas

Dataset: student_lifestyle_dataset.csv (2000 registros, 7 features)
Salida: carpeta eda/ con graficas, CSV y conclusiones
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


# ============================================================================
# Configuracion general
# ============================================================================

BASE_PATH = Path(__file__).parent
INPUT_PATH = BASE_PATH / "student_lifestyle_dataset.csv"
OUTPUT_PATH = BASE_PATH / "eda"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10


# ============================================================================
# Carga de datos
# ============================================================================

def load_data():
    """Carga dataset y retorna DataFrame."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {INPUT_PATH}")
    
    df = pd.read_csv(INPUT_PATH)
    print(f"Dataset cargado: {df.shape[0]} registros, {df.shape[1]} features")
    return df


# ============================================================================
# Estadisticas descriptivas
# ============================================================================

def estadisticas_descriptivas(df):
    """Genera estadisticas descriptivas y las guarda."""
    print("\n--- Estadisticas Descriptivas ---")
    stats = df.describe()
    print(stats)
    
    csv_path = OUTPUT_PATH / "estadisticas_descriptivas.csv"
    stats.to_csv(csv_path)
    print(f"Guardado en: {csv_path}")
    
    return stats


def detalle_columnas(df):
    """Imprime detalles de cada columna."""
    print("\n--- Detalle Columnas ---")
    for col in df.columns:
        unique = df[col].nunique()
        missing = df[col].isnull().sum()
        dtype = df[col].dtype
        print(f"{col}: {dtype} | unicos={unique} | nulos={missing}")
    
    return df.info()


# ============================================================================
# Deteccion de outliers
# ============================================================================

def detectar_outliers_iqr(df):
    """Detecta outliers usando IQR para cada variable numerica."""
    print("\n--- Deteccion de Outliers (IQR) ---")
    
    outliers_dict = {}
    
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < limite_inferior) | (df[col] > limite_superior)]
        outliers_dict[col] = len(outliers)
        
        if len(outliers) > 0:
            print(f"{col}: {len(outliers)} outliers detectados")
            print(f"  Rango: [{limite_inferior:.2f}, {limite_superior:.2f}]")
    
    return outliers_dict


# ============================================================================
# Visualizaciones basicas
# ============================================================================

def graficar_histogramas(df):
    """Grafica histogramas de variables numericas."""
    print("\n--- Generando histogramas ---")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_cols = len(numeric_cols)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 4))
    axes = axes.flatten()
    
    for idx, col in enumerate(numeric_cols):
        axes[idx].hist(df[col], bins=30, color="steelblue", edgecolor="black", alpha=0.7)
        axes[idx].set_title(f"Distribucion: {col}")
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel("Frecuencia")
    
    for idx in range(n_cols, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_file = OUTPUT_PATH / "histogramas.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Guardado: {output_file}")
    plt.close()


def graficar_boxplots(df):
    """Grafica boxplots de variables numericas."""
    print("\n--- Generando boxplots ---")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_cols = len(numeric_cols)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 4))
    axes = axes.flatten()
    
    for idx, col in enumerate(numeric_cols):
        axes[idx].boxplot(df[col])
        axes[idx].set_title(f"Boxplot: {col}")
        axes[idx].set_ylabel(col)
    
    for idx in range(n_cols, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_file = OUTPUT_PATH / "boxplots.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Guardado: {output_file}")
    plt.close()


def graficar_dispersiones(df):
    """Grafica dispersiones de pares de variables."""
    print("\n--- Generando dispersiones ---")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        print("Insuficientes variables para dispersiones")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    pairs = [
        (numeric_cols[0], numeric_cols[1]),
        (numeric_cols[0], numeric_cols[2]) if len(numeric_cols) > 2 else (numeric_cols[0], numeric_cols[1]),
        (numeric_cols[1], numeric_cols[2]) if len(numeric_cols) > 2 else (numeric_cols[0], numeric_cols[1]),
    ]
    
    for idx, (col1, col2) in enumerate(pairs[:6]):
        axes[idx].scatter(df[col1], df[col2], alpha=0.5, s=30)
        axes[idx].set_xlabel(col1)
        axes[idx].set_ylabel(col2)
        axes[idx].set_title(f"Dispersi\u00f3n: {col1} vs {col2}")
    
    for idx in range(len(pairs), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_file = OUTPUT_PATH / "dispersiones.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Guardado: {output_file}")
    plt.close()


# ============================================================================
# Correlacion y heatmap
# ============================================================================

def graficar_correlacion(df):
    """Grafica matriz de correlacion."""
    print("\n--- Generando matriz de correlacion ---")
    
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                center=0, ax=ax, cbar_kws={"label": "Correlacion"})
    ax.set_title("Matriz de Correlacion")
    
    plt.tight_layout()
    output_file = OUTPUT_PATH / "correlacion_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Guardado: {output_file}")
    plt.close()
    
    # Guardar matriz como CSV
    csv_path = OUTPUT_PATH / "correlacion_matriz.csv"
    corr_matrix.to_csv(csv_path)
    print(f"Matriz guardada en: {csv_path}")
    
    return corr_matrix


# ============================================================================
# Analisis vs Stress_Level
# ============================================================================

def analizar_stress_level(df):
    """Analiza variables en funcion de Stress_Level."""
    print("\n--- Analisis vs Stress_Level ---")
    
    if "Stress_Level" not in df.columns:
        print("Columna 'Stress_Level' no encontrada")
        return
    
    target_col = "Stress_Level"
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    n_cols = len(numeric_cols)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 4))
    axes = axes.flatten()
    
    for idx, col in enumerate(numeric_cols):
        for stress_level in sorted(df[target_col].unique()):
            data = df[df[target_col] == stress_level][col]
            axes[idx].hist(data, alpha=0.5, label=f"{target_col}={stress_level}", bins=20)
        
        axes[idx].set_title(f"{col} vs {target_col}")
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel("Frecuencia")
        axes[idx].legend()
    
    for idx in range(n_cols, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_file = OUTPUT_PATH / "analisis_stress_level.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Guardado: {output_file}")
    plt.close()


# ============================================================================
# Conclusiones
# ============================================================================

def generar_conclusiones(df):
    """Genera conclusiones automaticas del EDA."""
    print("\n--- Conclusiones del EDA ---")
    
    conclusiones = []
    conclusiones.append("ANALISIS EXPLORATORIO DE DATOS - CONCLUSIONES\n")
    conclusiones.append(f"Dataset: {df.shape[0]} registros, {df.shape[1]} features\n")
    
    # Variables y tipos
    conclusiones.append("\nVARIABLES:")
    for col in df.columns:
        dtype = df[col].dtype
        unique = df[col].nunique()
        conclusiones.append(f"  - {col}: {dtype} ({unique} unicos)")
    
    # Valores faltantes
    conclusiones.append("\nVALORES FALTANTES:")
    missing_count = df.isnull().sum().sum()
    if missing_count == 0:
        conclusiones.append("  - No hay valores faltantes")
    else:
        conclusiones.append(f"  - Total: {missing_count} valores faltantes")
    
    # Estadisticas numericas
    conclusiones.append("\nESTADISTICAS NUMERICAS:")
    stats = df.describe()
    for col in stats.columns:
        conclusiones.append(f"  - {col}: min={stats[col]['min']:.2f}, max={stats[col]['max']:.2f}, media={stats[col]['mean']:.2f}")
    
    # Outliers
    conclusiones.append("\nOUTLIERS (IQR):")
    outliers_dict = detectar_outliers_iqr(df)
    total_outliers = sum(outliers_dict.values())
    conclusiones.append(f"  - Total outliers detectados: {total_outliers}")
    
    # Correlaciones
    conclusiones.append("\nCORRELACIONES DESTACADAS:")
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr().unstack().sort_values(ascending=False)
    corr = corr[(corr != 1.0) & (corr != 0.0)]
    
    if len(corr) > 0:
        top_corr = corr.head(5)
        for (col1, col2), value in top_corr.items():
            if col1 != col2:
                conclusiones.append(f"  - {col1} vs {col2}: {value:.3f}")
    
    conclusiones_text = "\n".join(conclusiones)
    print(conclusiones_text)
    
    # Guardar en archivo
    output_file = OUTPUT_PATH / "conclusiones_eda.txt"
    with open(output_file, "w") as f:
        f.write(conclusiones_text)
    print(f"\nConcluciones guardadas en: {output_file}")


# ============================================================================
# Ejecucion principal
# ============================================================================

def main():
    """Ejecuta pipeline completo de EDA."""
    print("Iniciando Analisis Exploratorio de Datos...")
    print(f"Salida: {OUTPUT_PATH}\n")
    
    # Cargar datos
    df = load_data()
    
    # Analisis
    print("\n" + "=" * 80)
    estadisticas_descriptivas(df)
    detalle_columnas(df)
    
    print("\n" + "=" * 80)
    detectar_outliers_iqr(df)
    
    # Visualizaciones
    print("\n" + "=" * 80)
    graficar_histogramas(df)
    graficar_boxplots(df)
    graficar_dispersiones(df)
    
    print("\n" + "=" * 80)
    graficar_correlacion(df)
    
    print("\n" + "=" * 80)
    analizar_stress_level(df)
    
    print("\n" + "=" * 80)
    generar_conclusiones(df)
    
    print("\n" + "=" * 80)
    print("EDA completado exitosamente")
    print(f"Resultados en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
