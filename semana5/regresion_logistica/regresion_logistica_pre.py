import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# ======================================================
# PREPROCESAMIENTO DE DATOS - HEART DISEASE DATASET
# ======================================================

# Obtener la ruta del directorio donde está el script
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'Heart_disease_statlog.csv')
output_path = os.path.join(script_dir, 'Heart_disease_statlog_processed.csv')

# 1. Cargar el dataset
print("=" * 70)
print("CARGANDO DATASET...")
print("=" * 70)
df = pd.read_csv(data_path)
print(f"Tamaño original del dataset: {df.shape}")
print(f"Columnas: {list(df.columns)}")
print("\nPrimeras filas:")
print(df.head())

# 2. Información inicial
print("\n" + "=" * 70)
print("INFORMACIÓN INICIAL")
print("=" * 70)
print("\nTipos de datos:")
print(df.dtypes)
print("\nValores nulos por columna:")
print(df.isnull().sum())
print("\nEstadísticas descriptivas:")
print(df.describe())

# 3. Analizar distribución del target
print("\n" + "=" * 70)
print("ANÁLISIS DE LA VARIABLE OBJETIVO (TARGET)")
print("=" * 70)
print(f"\nDistribución del target:")
print(df['target'].value_counts())
print(f"\nProporción (%):")
print(df['target'].value_counts(normalize=True) * 100)
print("\n✅ Target balanceado: 55.6% sin enfermedad, 44.4% con enfermedad")

# 4. Identificar variables categóricas y numéricas
print("\n" + "=" * 70)
print("IDENTIFICANDO VARIABLES CATEGÓRICAS Y NUMÉRICAS")
print("=" * 70)

categorical_features = ['cp', 'restecg', 'slope', 'ca', 'thal']
binary_features = ['sex', 'fbs', 'exang']
numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

print(f"\nVariables categóricas (a codificar con OneHotEncoder): {categorical_features}")
print(f"Variables binarias (mantener como están): {binary_features}")
print(f"Variables numéricas (mantener como están): {numeric_features}")

# Mostrar valores únicos
print("\nValores únicos por columna categórica:")
for col in categorical_features:
    print(f"  {col}: {sorted(df[col].unique())}")

# 5. Separar features y target
print("\n" + "=" * 70)
print("SEPARANDO FEATURES Y TARGET")
print("=" * 70)

X = df.drop('target', axis=1)
y = df['target']

print(f"Shape de X: {X.shape}")
print(f"Shape de y: {y.shape}")
print(f"\nFeature names:\n{list(X.columns)}")

# 6. Codificar variables categóricas usando OneHotEncoder y normalizar numéricas
print("\n" + "=" * 70)
print("CODIFICANDO CATEGÓRICAS CON ONEHOTENCODER Y NORMALIZANDO NUMÉRICAS")
print("=" * 70)

print("\nESTADÍSTICAS ANTES DE NORMALIZACIÓN:")
print(f"\nVariables numéricas (edad, presión, colesterol, etc.):")
print(X[numeric_features].describe())

# Crear transformer que codifica categorías y normaliza variables numéricas
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features),
        ('num', StandardScaler(), numeric_features),
        ('bin', 'passthrough', binary_features)
    ])

# Ajustar y transformar
X_transformed = preprocessor.fit_transform(X)

# Obtener nombres de las nuevas columnas
categorical_params = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = list(categorical_params) + numeric_features + binary_features

# Crear DataFrame con las nuevas características
X_processed = pd.DataFrame(X_transformed, columns=all_feature_names)

print(f"\n✅ OneHotEncoder aplicado a: {categorical_features}")
print(f"✅ StandardScaler aplicado a: {numeric_features}")
print(f"✅ Variables binarias mantenidas sin cambios: {binary_features}")

print("\nESTADÍSTICAS DESPUÉS DE NORMALIZACIÓN:")
print(f"Variables numéricas (ahora normalizadas con media≈0 y std≈1):")
print(X_processed[numeric_features].describe())

print(f"\nShape después de transformación: {X_processed.shape}")
print(f"\nNuevas columnas ({len(all_feature_names)} total):")
print(all_feature_names)

# 7. Verificar el resultado
print("\n" + "=" * 70)
print("VERIFICANDO DATOS TRANSFORMADOS")
print("=" * 70)
print(f"\nPrimeras filas de X procesado:")
print(X_processed.head())
print(f"\nTipos de datos:")
print(X_processed.dtypes)
print(f"\nValores nulos totales: {X_processed.isnull().sum().sum()}")

# 8. Crear el dataset procesado completo
print("\n" + "=" * 70)
print("CREANDO DATASET PROCESADO COMPLETO")
print("=" * 70)

df_processed = X_processed.copy()
df_processed['target'] = y.values

print(f"Dimensiones finales: {df_processed.shape}")
print(f"Columnas finales: {list(df_processed.columns)}")

# 9. Guardar el dataset procesado
df_processed.to_csv(output_path, index=False)
print(f"\n✅ Dataset procesado guardado en: {output_path}")

# 10. Resumen final
print("\n" + "=" * 70)
print("RESUMEN FINAL DEL PREPROCESAMIENTO")
print("=" * 70)
print(f"✅ Dataset original: {df.shape[0]} filas × {df.shape[1]} columnas")
print(f"✅ Dataset procesado: {df_processed.shape[0]} filas × {df_processed.shape[1]} columnas")
print(f"✅ Variables categóricas codificadas: {categorical_features}")
print(f"✅ Variables numéricas conservadas: {numeric_features + binary_features}")
print(f"✅ Variable objetivo: target (0=Sin enfermedad, 1=Con enfermedad)")
print(f"✅ Valores faltantes: {df_processed.isnull().sum().sum()}")
print(f"✅ Todas las columnas son numéricas: {all(df_processed.dtypes != 'object')}")

print("\n" + "=" * 70)
print("¡PREPROCESAMIENTO COMPLETADO CON ÉXITO!")
print("=" * 70)
print("\nPróximos pasos:")
print("1. Cargar Heart_disease_statlog_processed.csv")
print("2. Dividir en train/test")
print("3. Aplicar normalización con StandardScaler en el pipeline")
print("4. Definir regresión logística con RandomizedSearchCV")
print("5. Entrenar y evaluar el modelo")
print("\nÚltimas filas del dataset procesado:")
print(df_processed.tail(10))
