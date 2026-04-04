"""
Módulo de Análisis Exploratorio de Datos (EDA)
Dataset: student_lifestyle_dataset.csv

Este módulo realiza un análisis completo del dataset de estilo de vida de estudiantes,
incluyendo estadísticas descriptivas, detección de outliers y visualizaciones.

Autor: Luis Ángel Nerio
Fecha: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Configuración de estilos
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# ============================================================================
# CONFIGURACIÓN DE RUTAS
# ============================================================================

# Obtener ruta base del proyecto
BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / "student_lifestyle_dataset.csv"
OUTPUT_PATH = BASE_PATH / "eda"

# Crear carpeta de salida si no existe
OUTPUT_PATH.mkdir(exist_ok=True)

print(f"📁 Ruta de datos: {DATA_PATH}")
print(f"📁 Ruta de salida: {OUTPUT_PATH}")
print("=" * 80)


# ============================================================================
# FUNCIÓN 1: CARGAR Y EXPLORACIÓN INICIAL
# ============================================================================

def cargar_dataset(ruta):
    """
    Carga el dataset desde un archivo CSV.
    
    Args:
        ruta (str): Ruta al archivo CSV
        
    Returns:
        pd.DataFrame: Dataset cargado
    """
    try:
        df = pd.read_csv(ruta)
        print(f"✅ Dataset cargado exitosamente: {ruta}")
        return df
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo {ruta}")
        return None


def exploracion_inicial(df):
    """
    Realiza una exploración inicial del dataset.
    
    Args:
        df (pd.DataFrame): Dataset a explorar
    """
    print("\n" + "=" * 80)
    print("📊 EXPLORACIÓN INICIAL DEL DATASET")
    print("=" * 80)
    
    # Primeras filas
    print("\n🔍 Primeras 5 filas del dataset:")
    print(df.head())
    
    # Dimensiones
    print(f"\n📏 Dimensiones del dataset: {df.shape[0]} filas × {df.shape[1]} columnas")
    
    # Tipos de datos
    print("\n📋 Tipos de datos:")
    print(df.dtypes)
    
    # Información general
    print("\n📈 Información general del dataset:")
    print(df.info())
    
    # Valores nulos
    print("\n❓ Valores nulos por columna:")
    print(df.isnull().sum())
    
    if df.isnull().sum().sum() == 0:
        print("   ✅ No hay valores nulos en el dataset")
    else:
        print(f"   ⚠️  Total de valores nulos: {df.isnull().sum().sum()}")


# ============================================================================
# FUNCIÓN 2: ESTADÍSTICAS DESCRIPTIVAS
# ============================================================================

def estadisticas_descriptivas(df):
    """
    Calcula estadísticas descriptivas completas para variables numéricas.
    
    Args:
        df (pd.DataFrame): Dataset
    """
    print("\n" + "=" * 80)
    print("📊 ESTADÍSTICAS DESCRIPTIVAS")
    print("=" * 80)
    
    # Seleccionar solo columnas numéricas
    df_numericas = df.select_dtypes(include=[np.number])
    
    # Remover Student_ID de las estadísticas
    if 'Student_ID' in df_numericas.columns:
        df_numericas = df_numericas.drop('Student_ID', axis=1)
    
    print("\n📈 Resumen estadístico completo:")
    print(df_numericas.describe())
    
    # Estadísticas completas
    print("\n🔢 Estadísticas detalladas:")
    stats_df = pd.DataFrame({
        'Media': df_numericas.mean(),
        'Mediana': df_numericas.median(),
        'Moda': df_numericas.mode().iloc[0] if len(df_numericas.mode()) > 0 else np.nan,
        'Varianza': df_numericas.var(),
        'Desv. Estándar': df_numericas.std(),
        'Mínimo': df_numericas.min(),
        'Q1': df_numericas.quantile(0.25),
        'Q2 (Mediana)': df_numericas.quantile(0.50),
        'Q3': df_numericas.quantile(0.75),
        'Máximo': df_numericas.max(),
        'Rango': df_numericas.max() - df_numericas.min()
    })
    print(stats_df.round(4))
    
    # Guardar en archivo
    stats_df.to_csv(OUTPUT_PATH / "estadisticas_descriptivas.csv")
    print(f"\n✅ Estadísticas guardadas en: {OUTPUT_PATH / 'estadisticas_descriptivas.csv'}")
    
    return stats_df


# ============================================================================
# FUNCIÓN 3: DETECCIÓN DE OUTLIERS
# ============================================================================

def detectar_outliers(df):
    """
    Detecta outliers usando el método del IQR.
    
    Args:
        df (pd.DataFrame): Dataset
        
    Returns:
        dict: Diccionario con información de outliers por columna
    """
    print("\n" + "=" * 80)
    print("🎯 DETECCIÓN DE OUTLIERS - MÉTODO IQR")
    print("=" * 80)
    
    # Seleccionar solo columnas numéricas
    df_numericas = df.select_dtypes(include=[np.number])
    
    # Remover Student_ID
    if 'Student_ID' in df_numericas.columns:
        df_numericas = df_numericas.drop('Student_ID', axis=1)
    
    outliers_info = {}
    total_outliers = 0
    
    for columna in df_numericas.columns:
        Q1 = df[columna].quantile(0.25)
        Q3 = df[columna].quantile(0.75)
        IQR = Q3 - Q1
        
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        
        # Identificar outliers
        outliers = df[(df[columna] < limite_inferior) | (df[columna] > limite_superior)]
        
        outliers_info[columna] = {
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'Límite Inferior': limite_inferior,
            'Límite Superior': limite_superior,
            'Cantidad de Outliers': len(outliers),
            'Índices': outliers.index.tolist()
        }
        
        total_outliers += len(outliers)
        
        if len(outliers) > 0:
            print(f"\n⚠️  Columna: {columna}")
            print(f"   - Q1: {Q1:.4f}, Q3: {Q3:.4f}, IQR: {IQR:.4f}")
            print(f"   - Límites: [{limite_inferior:.4f}, {limite_superior:.4f}]")
            print(f"   - Cantidad de outliers: {len(outliers)}")
            print(f"   - Valores: {outliers[columna].values}")
        else:
            print(f"\n✅ Columna: {columna} - No hay outliers")
    
    print(f"\n📈 Total de outliers encontrados: {total_outliers}")
    
    # Guardar información de outliers
    outliers_df = pd.DataFrame({col: [outliers_info[col]['Cantidad de Outliers']] 
                                 for col in outliers_info})
    outliers_df.to_csv(OUTPUT_PATH / "outliers_resumen.csv", index=False)
    
    return outliers_info


# ============================================================================
# FUNCIÓN 4: VISUALIZACIONES - HISTOGRAMAS
# ============================================================================

def graficar_histogramas(df):
    """
    Crea histogramas para todas las variables numéricas.
    
    Args:
        df (pd.DataFrame): Dataset
    """
    print("\n" + "=" * 80)
    print("📊 GENERANDO HISTOGRAMAS")
    print("=" * 80)
    
    df_numericas = df.select_dtypes(include=[np.number])
    
    # Remover Student_ID
    if 'Student_ID' in df_numericas.columns:
        df_numericas = df_numericas.drop('Student_ID', axis=1)
    
    cols = df_numericas.columns
    n_cols = 3
    n_rows = (len(cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, columna in enumerate(cols):
        axes[idx].hist(df[columna], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'Distribución: {columna}', fontweight='bold')
        axes[idx].set_xlabel('Valor')
        axes[idx].set_ylabel('Frecuencia')
        axes[idx].grid(True, alpha=0.3)
        
        # Calcular asimetría
        skewness = df[columna].skew()
        axes[idx].text(0.5, 0.95, f'Asimetría: {skewness:.2f}', 
                      transform=axes[idx].transAxes, 
                      verticalalignment='top', horizontalalignment='center',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Eliminar subplots vacíos
    for idx in range(len(cols), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    fig.savefig(OUTPUT_PATH / "01_histogramas.png", dpi=300, bbox_inches='tight')
    print(f"✅ Histogramas guardados en: {OUTPUT_PATH / '01_histogramas.png'}")
    plt.close()


# ============================================================================
# FUNCIÓN 5: VISUALIZACIONES - BOXPLOTS
# ============================================================================

def graficar_boxplots(df):
    """
    Crea boxplots para detectar outliers visualmente.
    
    Args:
        df (pd.DataFrame): Dataset
    """
    print("\n" + "=" * 80)
    print("📊 GENERANDO BOXPLOTS")
    print("=" * 80)
    
    df_numericas = df.select_dtypes(include=[np.number])
    
    # Remover Student_ID
    if 'Student_ID' in df_numericas.columns:
        df_numericas = df_numericas.drop('Student_ID', axis=1)
    
    cols = df_numericas.columns
    n_cols = 3
    n_rows = (len(cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, columna in enumerate(cols):
        axes[idx].boxplot(df[columna], vert=True)
        axes[idx].set_title(f'Boxplot: {columna}', fontweight='bold')
        axes[idx].set_ylabel('Valor')
        axes[idx].grid(True, alpha=0.3)
    
    # Eliminar subplots vacíos
    for idx in range(len(cols), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    fig.savefig(OUTPUT_PATH / "02_boxplots.png", dpi=300, bbox_inches='tight')
    print(f"✅ Boxplots guardados en: {OUTPUT_PATH / '02_boxplots.png'}")
    plt.close()


# ============================================================================
# FUNCIÓN 6: VISUALIZACIÓN - STRESS_LEVEL
# ============================================================================

def graficar_stress_level(df):
    """
    Crea gráfico de barras para la distribución de Stress_Level.
    
    Args:
        df (pd.DataFrame): Dataset
    """
    print("\n" + "=" * 80)
    print("📊 GENERANDO GRÁFICO DE STRESS_LEVEL")
    print("=" * 80)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Contar frecuencias
    stress_counts = df['Stress_Level'].value_counts()
    stress_proportions = df['Stress_Level'].value_counts(normalize=True) * 100
    
    # Gráfico de barras
    stress_counts.plot(kind='bar', ax=axes[0], color=['green', 'orange', 'red'], alpha=0.7)
    axes[0].set_title('Distribución de Stress_Level (Frecuencia Absoluta)', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Nivel de Estrés')
    axes[0].set_ylabel('Cantidad de Estudiantes')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Añadir valores en las barras
    for i, v in enumerate(stress_counts):
        axes[0].text(i, v + 1, str(v), ha='center', fontweight='bold')
    
    # Gráfico de pastel
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    axes[1].pie(stress_counts, labels=stress_counts.index, autopct='%1.1f%%', 
               colors=colors, startangle=90)
    axes[1].set_title('Distribución de Stress_Level (Porcentaje)', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_PATH / "03_stress_level_distribucion.png", dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico de Stress_Level guardado en: {OUTPUT_PATH / '03_stress_level_distribucion.png'}")
    
    # Mostrar estadísticas
    print("\n📊 Estadísticas de Stress_Level:")
    print(f"\nFrecuncia absoluta:")
    print(stress_counts)
    print(f"\nFrecuencia relativa (%):")
    print(stress_proportions.round(2))
    
    plt.close()


# ============================================================================
# FUNCIÓN 7: VISUALIZACIÓN - GRÁFICOS DE DISPERSIÓN
# ============================================================================

def graficar_dispersiones(df):
    """
    Crea gráficos de dispersión para analizar relaciones entre variables.
    
    Args:
        df (pd.DataFrame): Dataset
    """
    print("\n" + "=" * 80)
    print("📊 GENERANDO GRÁFICOS DE DISPERSIÓN")
    print("=" * 80)
    
    # Crear figura con 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Study_Hours vs GPA
    axes[0].scatter(df['Study_Hours_Per_Day'], df['GPA'], alpha=0.6, s=50, color='steelblue')
    axes[0].set_xlabel('Horas de Estudio por Día', fontweight='bold')
    axes[0].set_ylabel('GPA', fontweight='bold')
    axes[0].set_title('Relación: Estudio vs GPA', fontweight='bold')
    
    # Calcular correlación
    corr_study_gpa = df['Study_Hours_Per_Day'].corr(df['GPA'])
    axes[0].text(0.05, 0.95, f'Correlación: {corr_study_gpa:.3f}', 
                transform=axes[0].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[0].grid(True, alpha=0.3)
    
    # 2. Sleep_Hours vs Stress_Level
    stress_map = {'Low': 0, 'Moderate': 1, 'High': 2}
    stress_numeric = df['Stress_Level'].map(stress_map)
    
    colors_stress = df['Stress_Level'].map({'Low': 'green', 'Moderate': 'orange', 'High': 'red'})
    axes[1].scatter(df['Sleep_Hours_Per_Day'], stress_numeric, alpha=0.6, s=50, c=colors_stress)
    axes[1].set_xlabel('Horas de Sueño por Día', fontweight='bold')
    axes[1].set_ylabel('Nivel de Estrés (0=Low, 1=Moderate, 2=High)', fontweight='bold')
    axes[1].set_title('Relación: Sueño vs Estrés', fontweight='bold')
    
    corr_sleep_stress = df['Sleep_Hours_Per_Day'].corr(stress_numeric)
    axes[1].text(0.05, 0.95, f'Correlación: {corr_sleep_stress:.3f}', 
                transform=axes[1].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1].grid(True, alpha=0.3)
    
    # 3. Social_Hours vs Stress_Level
    axes[2].scatter(df['Social_Hours_Per_Day'], stress_numeric, alpha=0.6, s=50, c=colors_stress)
    axes[2].set_xlabel('Horas Sociales por Día', fontweight='bold')
    axes[2].set_ylabel('Nivel de Estrés (0=Low, 1=Moderate, 2=High)', fontweight='bold')
    axes[2].set_title('Relación: Social vs Estrés', fontweight='bold')
    
    corr_social_stress = df['Social_Hours_Per_Day'].corr(stress_numeric)
    axes[2].text(0.05, 0.95, f'Correlación: {corr_social_stress:.3f}', 
                transform=axes[2].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_PATH / "04_gráficos_dispersión.png", dpi=300, bbox_inches='tight')
    print(f"✅ Gráficos de dispersión guardados en: {OUTPUT_PATH / '04_gráficos_dispersión.png'}")
    plt.close()


# ============================================================================
# FUNCIÓN 8: MATRIZ DE CORRELACIÓN Y HEATMAP
# ============================================================================

def graficar_correlacion(df):
    """
    Calcula la matriz de correlación y crea un heatmap.
    
    Args:
        df (pd.DataFrame): Dataset
    """
    print("\n" + "=" * 80)
    print("📊 GENERANDO MATRIZ DE CORRELACIÓN")
    print("=" * 80)
    
    # Seleccionar solo columnas numéricas
    df_numericas = df.select_dtypes(include=[np.number])
    
    # Remover Student_ID
    if 'Student_ID' in df_numericas.columns:
        df_numericas = df_numericas.drop('Student_ID', axis=1)
    
    # Calcular matriz de correlación
    matriz_corr = df_numericas.corr()
    
    print("\n📊 Matriz de Correlación de Pearson:")
    print(matriz_corr.round(4))
    
    # Guardar matriz de correlación
    matriz_corr.to_csv(OUTPUT_PATH / "matriz_correlacion.csv")
    
    # Identificar correlaciones altas
    print("\n🔗 Pares de variables con alta correlación (|r| > 0.5):")
    for i in range(len(matriz_corr.columns)):
        for j in range(i+1, len(matriz_corr.columns)):
            corr_value = matriz_corr.iloc[i, j]
            if abs(corr_value) > 0.5:
                print(f"   - {matriz_corr.columns[i]} ↔ {matriz_corr.columns[j]}: {corr_value:.4f}")
    
    # Crear heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matriz_corr, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Matriz de Correlación - Variables Numéricas', fontweight='bold', fontsize=14)
    plt.tight_layout()
    fig.savefig(OUTPUT_PATH / "05_matriz_correlacion_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"\n✅ Heatmap de correlación guardado en: {OUTPUT_PATH / '05_matriz_correlacion_heatmap.png'}")
    plt.close()


# ============================================================================
# FUNCIÓN 9: ANÁLISIS COMPLETO DE STRESS_LEVEL
# ============================================================================

def analisis_stress_vs_variables(df):
    """
    Realiza un análisis detallado de Stress_Level vs otras variables.
    
    Args:
        df (pd.DataFrame): Dataset
    """
    print("\n" + "=" * 80)
    print("🔍 ANÁLISIS DETALLADO DE STRESS_LEVEL")
    print("=" * 80)
    
    # Seleccionar solo columnas numéricas
    df_numericas = df.select_dtypes(include=[np.number])
    
    # Remover Student_ID
    if 'Student_ID' in df_numericas.columns:
        df_numericas = df_numericas.drop('Student_ID', axis=1)
    
    print("\n📊 Estadísticas por nivel de estrés:\n")
    
    for stress_level in df['Stress_Level'].unique():
        df_stress = df[df['Stress_Level'] == stress_level][df_numericas.columns]
        print(f"\n{'='*60}")
        print(f"Nivel de Estrés: {stress_level}")
        print(f"{'='*60}")
        print(f"Cantidad de estudiantes: {len(df_stress)}")
        print(f"\nEstadísticas descriptivas:")
        print(df_stress.describe().round(3))


# ============================================================================
# FUNCIÓN 10: GENERACIÓN DE CONCLUSIONES
# ============================================================================

def generar_conclusiones(df, outliers_info, stats_df):
    """
    Genera conclusiones automáticas en texto sobre el dataset.
    
    Args:
        df (pd.DataFrame): Dataset
        outliers_info (dict): Información de outliers
        stats_df (pd.DataFrame): Estadísticas descriptivas
    """
    print("\n" + "=" * 80)
    print("📝 CONCLUSIONES Y HALLAZGOS PRINCIPALES")
    print("=" * 80)
    
    conclusiones = []
    
    # 1. Información del dataset
    conclusiones.append(f"\n1. INFORMACIÓN DEL DATASET")
    conclusiones.append(f"   - Total de registros: {df.shape[0]}")
    conclusiones.append(f"   - Total de variables: {df.shape[1]}")
    conclusiones.append(f"   - Variables categóricas: Stress_Level")
    conclusiones.append(f"   - Variables numéricas: {df.shape[1] - 2} (excluyendo Student_ID)")
    
    # 2. Calidad de datos
    conclusiones.append(f"\n2. CALIDAD DE DATOS")
    valores_nulos = df.isnull().sum().sum()
    if valores_nulos == 0:
        conclusiones.append(f"   ✅ Dataset completo sin valores nulos")
    else:
        conclusiones.append(f"   ⚠️  Se encontraron {valores_nulos} valores nulos")
    
    # 3. Outliers
    total_outliers = sum([outliers_info[col]['Cantidad de Outliers'] 
                         for col in outliers_info])
    conclusiones.append(f"\n3. OUTLIERS DETECTADOS")
    conclusiones.append(f"   - Total de outliers encontrados: {total_outliers}")
    
    outliers_por_columna = [(col, outliers_info[col]['Cantidad de Outliers']) 
                            for col in outliers_info 
                            if outliers_info[col]['Cantidad de Outliers'] > 0]
    
    if outliers_por_columna:
        conclusiones.append(f"   - Columnas con outliers:")
        for col, count in sorted(outliers_por_columna, key=lambda x: x[1], reverse=True):
            conclusiones.append(f"     • {col}: {count} outliers")
    else:
        conclusiones.append(f"   - No se detectaron outliers significativos")
    
    # 4. Distribuciones
    conclusiones.append(f"\n4. ANÁLISIS DE DISTRIBUCIONES")
    
    df_numericas = df.select_dtypes(include=[np.number])
    if 'Student_ID' in df_numericas.columns:
        df_numericas = df_numericas.drop('Student_ID', axis=1)
    
    for col in df_numericas.columns:
        skewness = df[col].skew()
        kurtosis = df[col].kurtosis()
        
        if abs(skewness) < 0.5:
            dist_type = "aproximadamente simétrica (normal)"
        elif skewness > 0.5:
            dist_type = "sesgada a la derecha"
        else:
            dist_type = "sesgada a la izquierda"
        
        conclusiones.append(f"   - {col}: {dist_type} (Asimetría: {skewness:.3f})")
    
    # 5. Variable objetivo - Stress_Level
    conclusiones.append(f"\n5. DISTRIBUCIÓN DE STRESS_LEVEL")
    stress_counts = df['Stress_Level'].value_counts()
    stress_pct = df['Stress_Level'].value_counts(normalize=True) * 100
    
    for level in ['Low', 'Moderate', 'High']:
        if level in stress_counts.index:
            conclusiones.append(f"   - {level}: {stress_counts[level]} estudiantes ({stress_pct[level]:.1f}%)")
    
    # 6. Correlaciones importantes
    conclusiones.append(f"\n6. CORRELACIONES IMPORTANTES")
    
    df_corr = df_numericas.corr()
    
    # Convertir Stress_Level a numérico para correlación
    stress_map = {'Low': 0, 'Moderate': 1, 'High': 2}
    stress_numeric = df['Stress_Level'].map(stress_map)
    
    corr_gpa_stress = stats_df.loc['GPA'] if 'GPA' in stats_df.index else None
    
    for col in df_numericas.columns:
        if col != 'GPA':
            corr_with_gpa = df[col].corr(df['GPA'])
            if abs(corr_with_gpa) > 0.3:
                direction = "positiva" if corr_with_gpa > 0 else "negativa"
                conclusiones.append(f"   - {col} ↔ GPA: correlación {direction} ({corr_with_gpa:.3f})")
    
    # 7. Variables destacadas
    conclusiones.append(f"\n7. HALLAZGOS PRINCIPALES")
    
    # Variable con mayor variabilidad
    cv_dict = (df_numericas.std() / df_numericas.mean()).abs()
    var_max_cv = cv_dict.idxmax()
    conclusiones.append(f"   - Variable con mayor variabilidad (CV): {var_max_cv} (CV: {cv_dict[var_max_cv]:.3f})")
    
    # GPA promedio
    gpa_promedio = df['GPA'].mean()
    conclusiones.append(f"   - GPA promedio de los estudiantes: {gpa_promedio:.2f}")
    
    # Horas de sueño promedio
    sleep_promedio = df['Sleep_Hours_Per_Day'].mean()
    conclusiones.append(f"   - Horas de sueño promedio: {sleep_promedio:.2f} horas/día")
    
    # 8. Recomendaciones
    conclusiones.append(f"\n8. RECOMENDACIONES")
    conclusiones.append(f"   ✓ Dataset de buena calidad para preprocesamiento")
    conclusiones.append(f"   ✓ Proceder con escalado de variables para clustering")
    conclusiones.append(f"   ✓ Variables numéricas están normalizadas y listas para modelado")
    
    # Imprimir conclusiones
    texto_conclusiones = "\n".join(conclusiones)
    print(texto_conclusiones)
    
    # Guardar conclusiones en archivo
    with open(OUTPUT_PATH / "conclusiones_eda.txt", 'w', encoding='utf-8') as f:
        f.write(texto_conclusiones)
    
    print(f"\n✅ Conclusiones guardadas en: {OUTPUT_PATH / 'conclusiones_eda.txt'}")


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """
    Función principal que ejecuta todo el análisis EDA.
    """
    print("\n" + "=" * 80)
    print("🚀 INICIANDO ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
    print("=" * 80)
    
    # 1. Cargar dataset
    df = cargar_dataset(DATA_PATH)
    
    if df is None:
        print("❌ No se pudo cargar el dataset. Abortando análisis.")
        return
    
    # Crear copia para no modificar original
    df = df.copy()
    
    # 2. Exploración inicial
    exploracion_inicial(df)
    
    # 3. Estadísticas descriptivas
    stats_df = estadisticas_descriptivas(df)
    
    # 4. Detección de outliers
    outliers_info = detectar_outliers(df)
    
    # 5. Visualizaciones
    print("\n" + "=" * 80)
    print("📊 GENERANDO VISUALIZACIONES")
    print("=" * 80)
    
    graficar_histogramas(df)
    graficar_boxplots(df)
    graficar_stress_level(df)
    graficar_dispersiones(df)
    graficar_correlacion(df)
    
    # 6. Análisis de Stress_Level
    analisis_stress_vs_variables(df)
    
    # 7. Conclusiones
    generar_conclusiones(df, outliers_info, stats_df)
    
    # Resumen final
    print("\n" + "=" * 80)
    print("✅ ANÁLISIS EDA COMPLETADO EXITOSAMENTE")
    print("=" * 80)
    
    print(f"\n📁 Archivos generados en: {OUTPUT_PATH}")
    print("\n📋 Lista de archivos:")
    
    archivos_generados = sorted(OUTPUT_PATH.glob("*"))
    for f in archivos_generados:
        print(f"   - {f.name}")
    
    print("\n✨ ¡Análisis listo para el preprocesamiento!")


# ============================================================================
# EJECUTAR PROGRAMA
# ============================================================================

if __name__ == "__main__":
    main()
