# Laboratorio de preprocesamiento del dataset Iris (150 muestras, 4 características, 3 clases)
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# 1) Carga del dataset Iris desde scikit-learn
print("Cargando dataset Iris...")
iris = load_iris()

# Convierte a DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

print("Primeras filas:")
print(df.head())

# 2) Exploración inicial del dataset
print("Información del dataset:")
print(df.info())

print("Descripción estadística:")
print(df.describe())

print("Valores únicos por columna:")
print(df.nunique())

# 3) Verificación de valores nulos
print("Verificando valores nulos:")
print(df.isna().sum())

# Imputa con media si hay nulos
if df.isna().sum().sum() > 0:
    df = df.fillna(df.mean(numeric_only=True))

# Detección de outliers con z-score (>3 desviaciones estándar)
from scipy import stats
z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
outlier_mask = (z_scores > 3).any(axis=1)
print(f"Registros detectados como posibles outliers: {outlier_mask.sum()}")

# 4) Separación de variables predictoras y objetivo
X = df.drop(columns=['target'])
y = df['target']

# Escalado a media 0 y desviación estándar 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("Ejemplo de datos escalados:")
print(X_scaled.head())

# 5) Partición estratificada 80/20 train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Tamaños de los conjuntos:")
print("Train:", X_train.shape, " Test:", X_test.shape)

# 6) Exportación a formato Parquet
output_dir = Path("./data_output")
output_dir.mkdir(exist_ok=True)

train_path = output_dir / "iris_train.parquet"
test_path = output_dir / "iris_test.parquet"

# Combina X e y antes de exportar
X_train.assign(target=y_train).to_parquet(train_path, index=False)
X_test.assign(target=y_test).to_parquet(test_path, index=False)

print("Archivos exportados:")
print(train_path)
print(test_path)

print("Laboratorio finalizado correctamente.")
