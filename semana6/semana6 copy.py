"""
Script completo de árboles de decisión para Student Lifestyle Dataset
Adaptado del notebook ej_arboles.ipynb
Incluye: limpieza, transformación, división, entrenamiento y evaluación de modelos
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import time
import os
import json
import warnings
import sys
warnings.filterwarnings('ignore')

# ==================== CLASE PARA DUPLICAR SALIDA ====================
class Tee:
    """Clase para escribir la salida tanto en archivo como en pantalla"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()

# ==================== IMPORTAR LIBRERÍAS DE SKLEARN ====================
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# ==================== CONFIGURACIÓN ====================
random_state = 42
plt.rc('font', family='serif', size=12)
np.random.seed(random_state)

# Crear carpeta para guardar datos procesados y resultados
script_dir = os.path.dirname(os.path.abspath(__file__))
output_folder = os.path.join(script_dir, 'datos_procesados')
figures_folder = os.path.join(script_dir, 'figures')
results_folder = os.path.join(script_dir, 'results')

for folder in [output_folder, figures_folder, results_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Crear archivo de log para guardar toda la salida
log_path = os.path.join(results_folder, 'execution_log.txt')
log_file = open(log_path, 'w', encoding='utf-8')
original_stdout = sys.stdout

# Redirigir stdout para escribir en archivo y pantalla
sys.stdout = Tee(original_stdout, log_file)

print("="*70)
print("INICIALIZACIÓN DEL SISTEMA")
print("="*70)
print(f"[OK] Carpetas creadas exitosamente")
print(f"[INFO] Datos: {output_folder}")
print(f"[INFO] Figuras: {figures_folder}")
print(f"[INFO] Resultados: {results_folder}")

# ==================== 1. CARGAR EL DATASET ====================
print("\n" + "="*70)
print("PASO 1: CARGANDO LOS DATOS")
print("="*70)

# Buscar el archivo en el mismo directorio que el script
dataset_path = os.path.join(script_dir, 'student_lifestyle_dataset.csv')
print(f"[INFO] Buscando dataset en: {dataset_path}")

data = pd.read_csv(dataset_path)
print("[OK] Dataset cargado con éxito")
print(f"[INFO] Forma del dataset: {data.shape}")
print("\n[INFO] Primeras filas:")
print(data.head())

# ==================== ANÁLISIS INICIAL DEL DATASET ====================
print("\n" + "="*70)
print("EXPLORACIÓN DEL DATASET")
print("="*70)

print("\n[INFO] Información general del dataset:")
print(data.info())

print("\n[INFO] Estadísticas descriptivas:")
print(data.describe(include='all'))

print("\n[INFO] Valores faltantes:")
print(data.isnull().sum())

print("\n[INFO] Distribución de la columna objetivo (Stress_Level):")
print(data['Stress_Level'].value_counts())

# ==================== HISTOGRAMAS DE DISTRIBUCIÓN ====================
print("\n[INFO] Generando histogramas de distribución...")
numeric_cols = data.select_dtypes(include=np.number).columns
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, col in enumerate(numeric_cols):
    if idx < len(axes):
        axes[idx].hist(data[col], bins=50, color='skyblue', edgecolor='black')
        axes[idx].set_title(f'Distribucion de {col}')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frecuencia')

plt.tight_layout()
plt.savefig(os.path.join(figures_folder, 'distribucion_variables.png'), dpi=300, bbox_inches='tight')
print("[OK] Grafica de distribuciones guardada")
plt.close()

# ==================== 2. LIMPIEZA Y TRANSFORMACIÓN ====================
print("\n" + "="*70)
print("PASO 2: LIMPIEZA Y TRANSFORMACIÓN DEL DATASET")
print("="*70)

# Verificar valores anómalos en columnas numéricas
print("\n[INFO] Análisis de valores anomalos en columnas numéricas:")
for col in numeric_cols:
    print(f"\n{col}:")
    print(f"  Minimo: {data[col].min()}")
    print(f"  Maximo: {data[col].max()}")
    print(f"  Valores nulos: {data[col].isnull().sum()}")

# Eliminar Student_ID (es solo un identificador)
print("\n[INFO] Eliminando columna Student_ID (es solo un identificador)...")
data = data.drop(columns=['Student_ID'])

# Eliminar filas con valores nulos
print("[INFO] Eliminando filas con valores nulos...")
data = data.dropna()
print(f"[INFO] Forma despues de eliminar nulos: {data.shape}")

# Verificar duplicados
duplicados = data.duplicated().sum()
print(f"[INFO] Filas duplicadas: {duplicados}")
if duplicados > 0:
    data = data.drop_duplicates()
    print(f"[INFO] Forma despues de eliminar duplicados: {data.shape}")

print("\n[INFO] Dataset despues de limpieza:")
print(data.head())
print(f"[INFO] Forma final del dataset limpio: {data.shape}")

# ==================== 3. DEFINIR CARACTERÍSTICAS Y OBJETIVO ====================
print("\n" + "="*70)
print("PASO 3: DEFINIENDO CARACTERÍSTICAS Y OBJETIVO")
print("="*70)

# Definir columna objetivo
target_column = 'Stress_Level'

# Separar features (X) y target (y)
y = data[target_column]
X = data.drop(columns=[target_column])

# Identificar columnas categóricas y numéricas
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

print(f"\n[INFO] Columnas categoricas: {categorical_cols}")
print(f"[INFO] Columnas numericas: {numerical_cols}")
print(f"[INFO] Columna objetivo: {target_column}")

print(f"\n[INFO] Caracteristicas (X):")
print(X.head())
print(f"[INFO] Forma de X: {X.shape}")

print(f"\n[INFO] Objetivo (y):")
print(y.head())
print(f"[INFO] Forma de y: {y.shape}")
print(f"\n[INFO] Distribucion de clases en el objetivo:")
print(y.value_counts())

# ==================== VISUALIZAR DISTRIBUCIÓN DE CLASES ====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Grafico de barras
y.value_counts().plot(kind='bar', ax=axes[0], color=['#ff9999', '#66b3ff', '#99ff99'])
axes[0].set_title('Distribucion de Stress_Level', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Stress_Level')
axes[0].set_ylabel('Frecuencia')
axes[0].tick_params(axis='x', rotation=0)

# Grafico de pastel
y.value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%', 
                      colors=['#ff9999', '#66b3ff', '#99ff99'])
axes[1].set_title('Proporcion de Stress_Level', fontsize=14, fontweight='bold')
axes[1].set_ylabel('')

plt.tight_layout()
plt.savefig(os.path.join(figures_folder, 'distribucion_clases.png'), dpi=300, bbox_inches='tight')
print("[OK] Grafica de distribucion de clases guardada")
plt.close()

# ==================== 4. TRANSFORMACIÓN DE DATOS ====================
print("\n" + "="*70)
print("TRANSFORMACIÓN DE DATOS - ORDINAL ENCODING")
print("="*70)

if len(categorical_cols) > 0:
    print(f"\n[INFO] Aplicando Ordinal Encoding a columnas categoricas: {categorical_cols}")
    
    # Crear transformador para variables categóricas
    categorical_transformer = Pipeline(
        steps=[("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]
    )
    
    # Crear ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'
    )
    
    # Aplicar transformación
    X_transformed = preprocessor.fit_transform(X)
    
    # Obtener nombres de columnas transformadas
    transformed_columns = preprocessor.get_feature_names_out()
    cleaned_columns = [col.replace('cat__', '').replace('remainder__', '') for col in transformed_columns]
    
    # Crear DataFrame con datos transformados
    X = pd.DataFrame(X_transformed, columns=cleaned_columns)
    print("[OK] Datos transformados exitosamente")
    print(X.head())
else:
    print("\n[INFO] No hay columnas categoricas para transformar")

# ==================== CODIFICAR VARIABLE OBJETIVO ====================
print("\n" + "="*70)
print("CODIFICACIÓN DE VARIABLE OBJETIVO")
print("="*70)

# Codificar clases
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = label_encoder.classes_

print(f"\n[INFO] Clases: {class_names}")
print(f"[INFO] Encoding: {dict(zip(class_names, range(len(class_names))))}")

# ==================== 5. DIVISION EN ENTRENAMIENTO, VALIDACIÓN Y PRUEBA ====================
print("\n" + "="*70)
print("DIVISION EN ENTRENAMIENTO (10%), VALIDACIÓN (45%) Y PRUEBA (45%)")
print("="*70)

# Primer split: Separar 10% entrenamiento del 90% validación+prueba
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_encoded, test_size=0.9, random_state=random_state, stratify=y_encoded
)

# Segundo split: Dividir el 90% en 50-50 para obtener 45% validación y 45% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
)

print(f"\n[INFO] Tamaño del conjunto de entrenamiento: {X_train.shape[0]} muestras ({X_train.shape[0]/X.shape[0]*100:.1f}%)")
print(f"[INFO] Tamaño del conjunto de validación: {X_val.shape[0]} muestras ({X_val.shape[0]/X.shape[0]*100:.1f}%)")
print(f"[INFO] Tamaño del conjunto de prueba: {X_test.shape[0]} muestras ({X_test.shape[0]/X.shape[0]*100:.1f}%)")

print(f"\n[INFO] Distribucion de clases en entrenamiento:")
unique, counts = np.unique(y_train, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Clase {class_names[u]}: {c}")

print(f"\n[INFO] Distribucion de clases en validación:")
unique, counts = np.unique(y_val, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Clase {class_names[u]}: {c}")

print(f"\n[INFO] Distribucion de clases en prueba:")
unique, counts = np.unique(y_test, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Clase {class_names[u]}: {c}")

# ==================== GUARDAR DATOS PROCESADOS ====================
print("\n" + "="*70)
print("GUARDANDO DATOS PROCESADOS")
print("="*70)

# Guardar conjunto de entrenamiento
train_data = pd.concat([X_train.reset_index(drop=True), 
                        pd.Series(y_train, name=target_column)], axis=1)
train_path = os.path.join(output_folder, 'student_lifestyle_train.csv')
train_data.to_csv(train_path, index=False)
print(f"[OK] Archivo '{train_path}' guardado ({train_data.shape[0]} muestras)")

# Guardar conjunto de validación
val_data = pd.concat([X_val.reset_index(drop=True), 
                      pd.Series(y_val, name=target_column)], axis=1)
val_path = os.path.join(output_folder, 'student_lifestyle_validation.csv')
val_data.to_csv(val_path, index=False)
print(f"[OK] Archivo '{val_path}' guardado ({val_data.shape[0]} muestras)")

# Guardar conjunto de prueba
test_data = pd.concat([X_test.reset_index(drop=True), 
                       pd.Series(y_test, name=target_column)], axis=1)
test_path = os.path.join(output_folder, 'student_lifestyle_test.csv')
test_data.to_csv(test_path, index=False)
print(f"[OK] Archivo '{test_path}' guardado ({test_data.shape[0]} muestras)")

# ==================== ENTRENAMIENTO: RANDOM FOREST ====================
print("\n" + "="*70)
print("ENTRENAMIENTO - RANDOM FOREST CLASSIFIER")
print("="*70)

print("\n[INFO] Configurando modelo Random Forest...")
rf_base = RandomForestClassifier(random_state=random_state, n_jobs=-1)

# Definir parametros para busqueda
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [6, 8, 10, 12],
    'min_samples_leaf': [2, 5, 10]
}

print("[INFO] Iniciando GridSearchCV para Random Forest...")
print(f"[INFO] Parametros a explorar: {param_grid_rf}")

start_time = time.time()
rf = GridSearchCV(rf_base, cv=3, param_grid=param_grid_rf, n_jobs=-1, verbose=1)
rf.fit(X_train, y_train)
rf_time = time.time() - start_time

print(f"\n[OK] Entrenamiento completado en {rf_time:.2f} segundos")
print(f"[INFO] Mejores parametros: {rf.best_params_}")
print(f"[INFO] Mejor score (CV): {rf.best_score_:.4f}")

# ==================== ENTRENAMIENTO: GRADIENT BOOSTING ====================
print("\n" + "="*70)
print("ENTRENAMIENTO - GRADIENT BOOSTING CLASSIFIER")
print("="*70)

print("\n[INFO] Configurando modelo Gradient Boosting...")
gb_base = GradientBoostingClassifier(random_state=random_state)

# Definir parametros para busqueda
param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1]
}

print("[INFO] Iniciando GridSearchCV para Gradient Boosting...")
print(f"[INFO] Parametros a explorar: {param_grid_gb}")

start_time = time.time()
gb = GridSearchCV(gb_base, cv=3, param_grid=param_grid_gb, n_jobs=-1, verbose=1)
gb.fit(X_train, y_train)
gb_time = time.time() - start_time

print(f"\n[OK] Entrenamiento completado en {gb_time:.2f} segundos")
print(f"[INFO] Mejores parametros: {gb.best_params_}")
print(f"[INFO] Mejor score (CV): {gb.best_score_:.4f}")

# ==================== EVALUACIÓN EN CONJUNTOS DE ENTRENAMIENTO Y PRUEBA ====================
print("\n" + "="*70)
print("EVALUACIÓN DE MODELOS")
print("="*70)

models = {'Random Forest': rf, 'Gradient Boosting': gb}
results_dict = {}

for model_name, model in models.items():
    print(f"\n{'='*70}")
    print(f"MODELO: {model_name}")
    print(f"{'='*70}")
    
    # Predicciones
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Métricas para entrenamiento
    print(f"\n[INFO] CONJUNTO DE ENTRENAMIENTO:")
    train_acc = accuracy_score(y_train, y_train_pred)
    train_prec = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
    train_rec = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
    
    print(f"  Accuracy:  {train_acc:.4f}")
    print(f"  Precision: {train_prec:.4f}")
    print(f"  Recall:    {train_rec:.4f}")
    print(f"  F1-Score:  {train_f1:.4f}")
    
    # Métricas para validación
    print(f"\n[INFO] CONJUNTO DE VALIDACIÓN:")
    val_acc = accuracy_score(y_val, y_val_pred)
    val_prec = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)
    val_rec = recall_score(y_val, y_val_pred, average='weighted', zero_division=0)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
    
    print(f"  Accuracy:  {val_acc:.4f}")
    print(f"  Precision: {val_prec:.4f}")
    print(f"  Recall:    {val_rec:.4f}")
    print(f"  F1-Score:  {val_f1:.4f}")
    
    # Métricas para prueba
    print(f"\n[INFO] CONJUNTO DE PRUEBA:")
    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_rec = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall:    {test_rec:.4f}")
    print(f"  F1-Score:  {test_f1:.4f}")
    
    # Reporte de clasificación en prueba
    print(f"\n[INFO] REPORTE DE CLASIFICACIÓN (CONJUNTO DE PRUEBA):")
    print(classification_report(y_test, y_test_pred, target_names=class_names))
    
    # Matrices de confusión para los tres conjuntos
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_val = confusion_matrix(y_val, y_val_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)
    
    print(f"\n[INFO] Matriz de confusión (Entrenamiento):\n{cm_train}")
    print(f"\n[INFO] Matriz de confusión (Validación):\n{cm_val}")
    print(f"\n[INFO] Matriz de confusión (Prueba):\n{cm_test}")
    
    # Guardar resultados
    results_dict[model_name] = {
        'train': {'accuracy': train_acc, 'precision': train_prec, 'recall': train_rec, 'f1': train_f1},
        'val': {'accuracy': val_acc, 'precision': val_prec, 'recall': val_rec, 'f1': val_f1},
        'test': {'accuracy': test_acc, 'precision': test_prec, 'recall': test_rec, 'f1': test_f1},
        'best_params': model.best_params_
    }
    
    # Graficar matrices de confusión para los tres conjuntos
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Entrenamiento
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, 
                yticklabels=class_names, ax=axes[0], cbar_kws={'label': 'Cantidad'})
    axes[0].set_title(f'Matriz de Confusión - {model_name}\n(Entrenamiento)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Etiqueta Verdadera')
    axes[0].set_xlabel('Etiqueta Predicha')
    
    # Validación
    sns.heatmap(cm_val, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, 
                yticklabels=class_names, ax=axes[1], cbar_kws={'label': 'Cantidad'})
    axes[1].set_title(f'Matriz de Confusión - {model_name}\n(Validación)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Etiqueta Verdadera')
    axes[1].set_xlabel('Etiqueta Predicha')
    
    # Prueba
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Oranges', xticklabels=class_names, 
                yticklabels=class_names, ax=axes[2], cbar_kws={'label': 'Cantidad'})
    axes[2].set_title(f'Matriz de Confusión - {model_name}\n(Prueba)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Etiqueta Verdadera')
    axes[2].set_xlabel('Etiqueta Predicha')
    
    plt.tight_layout()
    filename = os.path.join(figures_folder, f'confusion_matrices_{model_name.lower().replace(" ", "_")}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"[OK] Matrices de confusión guardadas: {filename}")
    plt.close()

# ==================== COMPARACIÓN DE MODELOS ====================
print("\n" + "="*70)
print("COMPARACIÓN DE MODELOS")
print("="*70)

# Crear DataFrame de comparación
comparison_data = []
for model_name in models.keys():
    for dataset in ['train', 'val', 'test']:
        row = {
            'Modelo': model_name,
            'Conjunto': dataset,
            'Accuracy': results_dict[model_name][dataset]['accuracy'],
            'Precision': results_dict[model_name][dataset]['precision'],
            'Recall': results_dict[model_name][dataset]['recall'],
            'F1-Score': results_dict[model_name][dataset]['f1']
        }
        comparison_data.append(row)

comparison_df = pd.DataFrame(comparison_data)
print("\n[INFO] Tabla de comparación de métricas:")
print(comparison_df.to_string(index=False))

# Visualizar comparación - MEJORADA CON BARRAS AGRUPADAS
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Comparación de Modelos: Random Forest vs Gradient Boosting', fontsize=16, fontweight='bold', y=0.995)

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metric_keys = ['accuracy', 'precision', 'recall', 'f1']

# Colores para modelos y conjuntos
model_colors = {'Random Forest': ['#1f77b4', '#1f77b4'], 'Gradient Boosting': ['#ff7f0e', '#ff7f0e']}
dataset_colors = {'Train': 0.5, 'Val': 0.7, 'Test': 1.0}  # Para variar intensidad
datasets = ['Train', 'Val', 'Test']

for idx, (metric, metric_key) in enumerate(zip(metrics, metric_keys)):
    ax = axes[idx // 2, idx % 2]
    
    x_pos = np.arange(len(datasets))
    bar_width = 0.35
    
    # Preparar datos para cada modelo
    for model_idx, model_name in enumerate(models.keys()):
        values = [results_dict[model_name][ds.lower()][metric_key] for ds in datasets]
        
        # Color base del modelo
        if model_name == 'Random Forest':
            color = '#1f77b4'
        else:
            color = '#ff7f0e'
        
        # Dibujar barras
        bars = ax.bar(x_pos + (model_idx * bar_width), values, bar_width, 
                     label=model_name, color=color, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Añadir valores sobre las barras
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Configuración del gráfico
    ax.set_xlabel('Conjunto de Datos', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric}', fontsize=13, fontweight='bold', pad=10)
    ax.set_xticks(x_pos + bar_width / 2)
    ax.set_xticklabels(datasets, fontsize=11, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(os.path.join(figures_folder, 'comparacion_modelos.png'), dpi=300, bbox_inches='tight')
print("\n[OK] Gráfica de comparación de modelos (MEJORADA) guardada con éxito")
plt.close()

# ==================== IMPORTANCIA DE CARACTERÍSTICAS ====================
print("\n" + "="*70)
print("IMPORTANCIA DE CARACTERÍSTICAS")
print("="*70)

# Random Forest
print("\n[INFO] Top 10 caracteristicas mas importantes (Random Forest):")
rf_importances = rf.best_estimator_.feature_importances_
rf_feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_importances
}).sort_values('importance', ascending=False)

print(rf_feature_importance.head(10).to_string(index=False))

# Gradient Boosting
print("\n[INFO] Top 10 caracteristicas mas importantes (Gradient Boosting):")
gb_importances = gb.best_estimator_.feature_importances_
gb_feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': gb_importances
}).sort_values('importance', ascending=False)

print(gb_feature_importance.head(10).to_string(index=False))

# Graficar importancia de características
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Random Forest
top_n = 10
rf_feature_importance.head(top_n).sort_values('importance').plot(
    kind='barh', x='feature', y='importance', ax=axes[0], color='steelblue', legend=False)
axes[0].set_title('Top 10 Features - Random Forest', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Importancia')

# Gradient Boosting
gb_feature_importance.head(top_n).sort_values('importance').plot(
    kind='barh', x='feature', y='importance', ax=axes[1], color='darkorange', legend=False)
axes[1].set_title('Top 10 Features - Gradient Boosting', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Importancia')

plt.tight_layout()
plt.savefig(os.path.join(figures_folder, 'feature_importance.png'), dpi=300, bbox_inches='tight')
print("[OK] Grafica de importancia de caracteristicas guardada")
plt.close()

# ==================== GUARDAR RESULTADOS ====================
print("\n" + "="*70)
print("GUARDANDO RESULTADOS")
print("="*70)

# Guardar información del procesamiento
info_dict = {
    'dataset_info': {
        'total_samples': X.shape[0],
        'n_features': X.shape[1],
        'target': target_column,
        'target_classes': class_names.tolist(),
        'class_distribution': y.value_counts().to_dict()
    },
    'train_val_test_split': {
        'train_size': X_train.shape[0],
        'train_percentage': 60,
        'val_size': X_val.shape[0],
        'val_percentage': 20,
        'test_size': X_test.shape[0],
        'test_percentage': 20
    },
    'features': list(X.columns),
    'model_results': results_dict
}

info_path = os.path.join(results_folder, 'training_results.json')
with open(info_path, 'w') as f:
    # Convertir numpy types para JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    json.dump(info_dict, f, indent=4, default=convert_to_serializable)

print(f"[OK] Archivo '{info_path}' guardado")

# Guardar comparación de modelos
comparison_csv_path = os.path.join(results_folder, 'model_comparison.csv')
comparison_df.to_csv(comparison_csv_path, index=False)
print(f"[OK] Archivo '{comparison_csv_path}' guardado")

# ==================== RESUMEN FINAL ====================
print("\n" + "="*70)
print("RESUMEN FINAL")
print("="*70)

print(f"\n[INFO] Dataset original: {data.shape[0]} filas, {data.shape[1] + 1} columnas")
print(f"[INFO] Dataset procesado: {X.shape[0]} filas, {X.shape[1]} caracteristicas + 1 objetivo")
print(f"\n[INFO] Division de datos:")
print(f"  - Entrenamiento (60%): {X_train.shape}")
print(f"  - Validación (20%):    {X_val.shape}")
print(f"  - Prueba (20%):        {X_test.shape}")

print(f"\n[INFO] Caracteristicas utilizadas:")
for i, col in enumerate(X.columns, 1):
    print(f"  {i}. {col}")

print(f"\n[INFO] Variables objetivo (Stress_Level):")
for i, class_name in enumerate(class_names):
    print(f"  {i}. {class_name}")

print(f"\n[INFO] Mejores parametros:")
print(f"  Random Forest: {rf.best_params_}")
print(f"  Gradient Boosting: {gb.best_params_}")

print(f"\n[INFO] Mejor performance en conjunto de PRUEBA:")
best_model_name = max(models.keys(), 
                      key=lambda x: results_dict[x]['test']['accuracy'])
best_accuracy = results_dict[best_model_name]['test']['accuracy']
print(f"  Modelo: {best_model_name}")
print(f"  Accuracy: {best_accuracy:.4f}")

print(f"\n[INFO] Archivos guardados en:")
print(f"  - Datos procesados: {output_folder}/")
print(f"  - Figuras: {figures_folder}/")
print(f"  - Resultados: {results_folder}/")

print("\n" + "="*70)
print("[OK] PROCESAMIENTO COMPLETADO EXITOSAMENTE")
print("="*70)

# Cerrar archivo de log
sys.stdout = original_stdout
log_file.close()

print(f"\n[OK] Archivo de log guardado en: {log_path}")
print(f"[INFO] Puedes revisar la salida completa en ese archivo.")

