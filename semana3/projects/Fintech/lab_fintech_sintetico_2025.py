# LAB FINTECH - Preprocesamiento de datos sintéticos 2025
# Ejecutar: python lab_fintech_sintetico_2025.py

import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configuración de archivos y rutas
DATA_CSV = 'fintech_top_sintetico_2025.csv'
DATA_DICT = 'fintech_top_sintetico_dictionary.json'
OUTDIR = Path('./data_output_finanzas_sintetico')
SPLIT_DATE = '2025-09-01'

# Definición de columnas del dataset
DATE_COL = 'Month'
ID_COLS = ['Company']
CAT_COLS = ['Country', 'Region', 'Segment', 'Subsegment', 'IsPublic', 'Ticker']
NUM_COLS = [
    'Users_M','NewUsers_K','TPV_USD_B','TakeRate_pct','Revenue_USD_M',
    'ARPU_USD','Churn_pct','Marketing_Spend_USD_M','CAC_USD','CAC_Total_USD_M',
    'Close_USD','Private_Valuation_USD_B'
]
PRICE_COLS = ['Close_USD']

# Carga el diccionario de datos JSON
print("\n=== 0) Cargando diccionario de datos ===")
dict_path = Path(DATA_DICT)
if not dict_path.exists():
    raise FileNotFoundError(f"No se encontró {DATA_DICT}. Asegúrate de tener el archivo en la misma carpeta.")

with open(dict_path, 'r', encoding='utf-8') as f:
    data_dict = json.load(f)
print("Descripción:", data_dict.get('description', '(sin descripción)'))
print("Periodo:", data_dict.get('period', '(desconocido)'))

# Carga el CSV y ordena por fecha
print("\n=== 1) Cargando CSV sintético ===")
csv_path = Path(DATA_CSV)
if not csv_path.exists():
    raise FileNotFoundError(f"No se encontró {DATA_CSV}. Asegúrate de tener el archivo en la misma carpeta.")

df = pd.read_csv(csv_path)
print("Shape:", df.shape)

# Convierte columna de fecha y ordena cronológicamente
if DATE_COL not in df.columns:
    raise KeyError(f"La columna de fecha '{DATE_COL}' no existe en el CSV.")

df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
df = df.sort_values([DATE_COL] + ID_COLS).reset_index(drop=True)

print("Primeras filas:")
print(df.head(3))

# Análisis exploratorio básico
print("\n=== 2) EDA rápido ===")
print("Info:")
print(df.info())
print("\nNulos por columna (top 15):")
print(df.isna().sum().sort_values(ascending=False).head(15))

# Limpia valores nulos: numéricos con mediana, categóricos con marcador
print("\n=== 3) Limpieza ===")
for c in NUM_COLS:
    if c in df.columns and df[c].isna().any():
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c] = df[c].fillna(df[c].median())

for c in CAT_COLS:
    if c in df.columns and df[c].isna().any():
        df[c] = df[c].fillna('__MISSING__')

# Calcula retornos simples y logarítmicos del precio por empresa
print("\n=== 4) Ingeniería de rasgos (retornos) ===")
if all([pc in df.columns for pc in PRICE_COLS]):
    for pc in PRICE_COLS:
        df[pc + '_ret'] = (
            df.sort_values([ID_COLS[0], DATE_COL])
              .groupby(ID_COLS)[pc]
              .pct_change()
        )
        df[pc + '_logret'] = np.log1p(df[pc + '_ret'])
        df[pc + '_ret'] = df[pc + '_ret'].fillna(0.0)
        df[pc + '_logret'] = df[pc + '_logret'].fillna(0.0)
else:
    print("[INFO] Columnas de precio no disponibles; se omite cálculo de retornos.")

# Actualiza lista de columnas numéricas con las nuevas creadas
extra_num = [c for c in [pc + '_ret' for pc in PRICE_COLS] + [pc + '_logret' for pc in PRICE_COLS] if c in df.columns]
NUM_USED = [c for c in NUM_COLS if c in df.columns] + extra_num

# Prepara matriz X: elimina identificadores y fecha
print("\n=== 5) Preparación de X: codificación one-hot y escalado ===")
X = df.drop(columns=[DATE_COL] + ID_COLS, errors='ignore').copy()

# Codifica variables categóricas con one-hot (drop_first evita multicolinealidad)
cat_in_X = [c for c in CAT_COLS if c in X.columns]
X = pd.get_dummies(X, columns=cat_in_X, drop_first=True)

# Partición temporal: divide datos antes y después de la fecha de corte
cutoff = pd.to_datetime(SPLIT_DATE)
idx_train = df[DATE_COL] < cutoff
idx_test = df[DATE_COL] >= cutoff

X_train, X_test = X.loc[idx_train].copy(), X.loc[idx_test].copy()

# Escala variables numéricas (fit en train, transform en test)
num_in_X = [c for c in NUM_USED if c in X_train.columns]
scaler = StandardScaler()
if num_in_X:
    X_train[num_in_X] = scaler.fit_transform(X_train[num_in_X])
    X_test[num_in_X] = scaler.transform(X_test[num_in_X])
else:
    print("[INFO] No se encontraron columnas numéricas para escalar.")

print("Shapes -> X_train:", X_train.shape, " X_test:", X_test.shape)

# Exporta conjuntos procesados en formato Parquet
print("\n=== 6) Exportación ===")
OUTDIR.mkdir(parents=True, exist_ok=True)
train_path = OUTDIR / 'fintech_train.parquet'
test_path = OUTDIR / 'fintech_test.parquet'

X_train.to_parquet(train_path, index=False)
X_test.to_parquet(test_path, index=False)

# Guarda esquema con metadatos del procesamiento
processed_schema = {
    'source_csv': str(csv_path.resolve()),
    'source_dict': str(dict_path.resolve()),
    'date_col': DATE_COL,
    'id_cols': ID_COLS,
    'categorical_cols_used': cat_in_X,
    'numeric_cols_used': num_in_X,
    'engineered_cols': extra_num,
    'split': {
        'type': 'time_split',
        'cutoff': SPLIT_DATE,
        'train_rows': int(idx_train.sum()),
        'test_rows': int(idx_test.sum()),
    },
    'X_train_shape': list(X_train.shape),
    'X_test_shape': list(X_test.shape),
    'notes': [
        'Dataset 100% SINTÉTICO con fines académicos; no refleja métricas reales.',
        'Evitar fuga de datos: el escalador se ajusta en TRAIN y se aplica a TEST.'
    ]
}

with open(OUTDIR / 'processed_schema.json', 'w', encoding='utf-8') as f:
    json.dump(processed_schema, f, ensure_ascii=False, indent=2)

# Guarda lista de columnas finales
with open(OUTDIR / 'features_columns.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(X_train.columns))

print("\nArchivos exportados:")
print(" -", train_path)
print(" -", test_path)
print(" -", OUTDIR / 'processed_schema.json')
print(" -", OUTDIR / 'features_columns.txt')

print("\n✔ Listo. Recuerda: este dataset es sintético para práctica académica.")
