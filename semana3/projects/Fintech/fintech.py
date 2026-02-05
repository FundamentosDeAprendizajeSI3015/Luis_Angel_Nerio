# Análisis UMAP para dataset Fintech sintético

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Instala UMAP si no está disponible
try:
    import umap
except ImportError:
    print("UMAP no está instalado. Instalando...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'umap-learn'])
    import umap

# 1) Carga de datos
print("\n" + "="*70)
print("1) CARGANDO DATOS DE FINTECH")
print("="*70)

data_path = Path('Fintech/fintech_top_sintetico_2025.csv')

if not data_path.exists():
    raise FileNotFoundError(f"No se encontró {data_path}. Verifica la ruta.")

df = pd.read_csv(data_path)
print(f"\nDataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
print("\nPrimeras filas:")
print(df.head())

# 2) Exploración inicial
print("\n" + "="*70)
print("2) EXPLORACIÓN DE DATOS")
print("="*70)

print("\nInfo del dataset:")
print(df.info())

print("\nValores nulos por columna:")
print(df.isna().sum())

# Identifica tipos de columnas
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"\nColumnas numéricas ({len(numeric_cols)}): {numeric_cols[:5]}...")
print(f"Columnas categóricas ({len(categorical_cols)}): {categorical_cols}")

# 3) Limpieza de datos
print("\n" + "="*70)
print("3) LIMPIEZA Y PREPARACIÓN")
print("="*70)

df_clean = df.copy()

# Rellena nulos: mediana para numéricos, "MISSING" para categóricos
for col in numeric_cols:
    if df_clean[col].isna().any():
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)
        print(f"  - Rellenados {df_clean[col].isna().sum()} nulos en '{col}' con mediana: {median_val:.2f}")

for col in categorical_cols:
    if df_clean[col].isna().any():
        df_clean[col] = df_clean[col].fillna("MISSING")
        print(f"  - Rellenados nulos en columna categórica '{col}'")

print("\nDatos después de limpieza:")
print(f"Nulos totales: {df_clean.isna().sum().sum()}")

# 4) Preparación para UMAP
print("\n" + "="*70)
print("4) PREPARACIÓN PARA UMAP")
print("="*70)

# Selecciona solo columnas numéricas
X = df_clean[numeric_cols].copy()

print(f"\nDatos para UMAP: {X.shape[0]} muestras, {X.shape[1]} características")

# Escala datos a media=0 y desv.est.=1
print("\nEscalando datos (StandardScaler)...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(X_scaled, columns=numeric_cols)

print("Datos escalados - Primeras filas:")
print(X_scaled.head())

# 5) Aplicar UMAP 2D
print("\n" + "="*70)
print("5) APLICANDO UMAP (PROYECCIÓN A 2D)")
print("="*70)

# Configura UMAP: 2 dimensiones, 15 vecinos, dist mínima 0.1
reducer_2d = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric='euclidean',
    random_state=42,
    verbose=True
)

print("\nCalculando proyección UMAP 2D (puede tomar un momento)...")
umap_2d = reducer_2d.fit_transform(X_scaled)
print("✓ Proyección 2D completada")

umap_2d_df = pd.DataFrame(
    umap_2d,
    columns=['UMAP-1', 'UMAP-2']
)

print(f"\nForma de datos UMAP 2D: {umap_2d_df.shape}")
print("Primeras filas:")
print(umap_2d_df.head())

# 6) Visualización UMAP 2D
print("\n" + "="*70)
print("6) VISUALIZACIÓN UMAP 2D")
print("="*70)

# Prepara colores: convierte categorías a números para matplotlib
colors_numeric = None
color_categories = None
color_label = 'Índice'

if 'Segment' in df_clean.columns:
    colors_numeric, color_categories = pd.factorize(df_clean['Segment'])
    color_label = 'Segment'
    print(f"Coloreando por '{color_label}'")
    print(f"  - Categorías encontradas: {len(color_categories)}")
    print(f"    {list(color_categories[:5])}...")
elif 'Country' in df_clean.columns:
    colors_numeric, color_categories = pd.factorize(df_clean['Country'])
    color_label = 'Country'
    print(f"Coloreando por '{color_label}'")
    print(f"  - Categorías encontradas: {len(color_categories)}")
else:
    colors_numeric = np.arange(len(umap_2d_df))
    print(f"Coloreando por índice")

fig, ax = plt.subplots(figsize=(12, 8))

# Dibuja scatter plot con puntos coloreados
scatter = ax.scatter(
    umap_2d_df['UMAP-1'],
    umap_2d_df['UMAP-2'],
    c=colors_numeric,
    s=50,
    alpha=0.6,
    cmap='tab20',
    edgecolors='black',
    linewidth=0.5
)

# Configura ejes y título
ax.set_xlabel('UMAP-1 (Primera Dimensión)', fontsize=12, fontweight='bold')
ax.set_ylabel('UMAP-2 (Segunda Dimensión)', fontsize=12, fontweight='bold')
ax.set_title(f'UMAP 2D - Dataset Fintech\nColoreado por {color_label}', 
             fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')

# Añade leyenda o colorbar según número de categorías
if color_categories is not None and len(color_categories) <= 20:
    from matplotlib.patches import Patch
    cmap = plt.cm.get_cmap('tab20')
    legend_elements = [
        Patch(facecolor=cmap(i), edgecolor='black', label=cat)
        for i, cat in enumerate(color_categories)
    ]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), 
              fontsize=9, title=color_label, title_fontsize=10)
else:
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(color_label, fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('umap_2d_fintech.png', dpi=300, bbox_inches='tight')
print("\n✓ Gráfica 2D guardada como 'umap_2d_fintech.png'")
plt.show()

# 7) Aplicar UMAP 3D
print("\n" + "="*70)
print("7) APLICANDO UMAP (PROYECCIÓN A 3D)")
print("="*70)

# Configura UMAP para 3 dimensiones
reducer_3d = umap.UMAP(
    n_components=3,
    n_neighbors=15,
    min_dist=0.1,
    metric='euclidean',
    random_state=42,
    verbose=False
)

print("Calculando proyección UMAP 3D...")
umap_3d = reducer_3d.fit_transform(X_scaled)
print("✓ Proyección 3D completada")

umap_3d_df = pd.DataFrame(
    umap_3d,
    columns=['UMAP-1', 'UMAP-2', 'UMAP-3']
)

print(f"Forma de datos UMAP 3D: {umap_3d_df.shape}")

# 8) Visualización UMAP 3D
print("\n" + "="*70)
print("8) VISUALIZACIÓN UMAP 3D")
print("="*70)

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Dibuja puntos en 3D con los mismos colores que 2D
scatter = ax.scatter(
    umap_3d_df['UMAP-1'],
    umap_3d_df['UMAP-2'],
    umap_3d_df['UMAP-3'],
    c=colors_numeric,
    s=40,
    alpha=0.6,
    cmap='tab20',
    edgecolors='black',
    linewidth=0.5
)

# Configura ejes y título 3D
ax.set_xlabel('UMAP-1', fontsize=11, fontweight='bold')
ax.set_ylabel('UMAP-2', fontsize=11, fontweight='bold')
ax.set_zlabel('UMAP-3', fontsize=11, fontweight='bold')
ax.set_title(f'UMAP 3D - Dataset Fintech\nColoreado por {color_label}', 
             fontsize=14, fontweight='bold', pad=20)

cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
cbar.set_label(color_label, fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('umap_3d_fintech.png', dpi=300, bbox_inches='tight')
print("\n✓ Gráfica 3D guardada como 'umap_3d_fintech.png'")
plt.show()

# 9) Análisis de resultados
print("\n" + "="*70)
print("9) ANÁLISIS DE RESULTADOS UMAP")
print("="*70)

print("\nEstadísticas UMAP 2D:")
print(umap_2d_df.describe())

print("\nEstadísticas UMAP 3D:")
print(umap_3d_df.describe())

# Calcula puntos más alejados del origen
print("\nPuntos más extremos (esquinas) en UMAP 2D:")
umap_2d_df['distancia'] = np.sqrt(umap_2d_df['UMAP-1']**2 + umap_2d_df['UMAP-2']**2)
extreme_points = umap_2d_df.nlargest(5, 'distancia')
print(extreme_points)

# 10) Exportación
print("\n" + "="*70)
print("10) EXPORTACIÓN DE DATOS")
print("="*70)

output_dir = Path('./umap_results')
output_dir.mkdir(exist_ok=True)

# Elimina columna auxiliar de distancia
umap_2d_export = umap_2d_df.drop(columns=['distancia'], errors='ignore')

# Exporta proyecciones UMAP
umap_2d_export.to_csv(output_dir / 'umap_2d_projections.csv', index=False)
umap_3d_df.to_csv(output_dir / 'umap_3d_projections.csv', index=False)

# Combina datos originales con proyecciones
results_2d = pd.concat([df_clean.reset_index(drop=True), umap_2d_export], axis=1)
results_2d.to_csv(output_dir / 'fintech_with_umap_2d.csv', index=False)

results_3d = pd.concat([df_clean.reset_index(drop=True), umap_3d_df], axis=1)
results_3d.to_csv(output_dir / 'fintech_with_umap_3d.csv', index=False)

print(f"\nArchivos guardados en directorio '{output_dir}/':")
print(f"  - umap_2d_projections.csv (solo coordenadas UMAP 2D)")
print(f"  - umap_3d_projections.csv (solo coordenadas UMAP 3D)")
print(f"  - fintech_with_umap_2d.csv (datos originales + UMAP 2D)")
print(f"  - fintech_with_umap_3d.csv (datos originales + UMAP 3D)")

print("\n" + "="*70)
print("✓ ANÁLISIS UMAP COMPLETADO")
print("="*70)
print("\nResumen:")
print(f"  - Datos procesados: {df_clean.shape[0]} muestras")
print(f"  - Características originales: {len(numeric_cols)}")
print(f"  - Proyección UMAP 2D: generada ✓")
print(f"  - Proyección UMAP 3D: generada ✓")
print(f"  - Visualizaciones: umap_2d_fintech.png, umap_3d_fintech.png")
print("\nTips para interpretación:")
print("  - Puntos cercanos = muestras similares")
print("  - Clusters = grupos naturales en los datos")
print("  - Distancia relativa preservada = estructura de datos respetada")
print("="*70 + "\n")
