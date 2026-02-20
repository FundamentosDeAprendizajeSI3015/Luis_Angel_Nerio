import pandas as pd
import numpy as np

# ============================================
# PREPROCESAMIENTO DE DATOS - SPORT CAR PRICE
# ============================================

# 1. Cargar el dataset
print("=" * 60)
print("CARGANDO DATASET...")
print("=" * 60)
df = pd.read_csv('Sport car price (1).csv')
print(f"Tamaño original del dataset: {df.shape}")
print(f"Columnas: {list(df.columns)}")
print("\nPrimeras filas:")
print(df.head())

# 2. Información inicial
print("\n" + "=" * 60)
print("INFORMACIÓN INICIAL")
print("=" * 60)
print("\nTipos de datos:")
print(df.dtypes)
print("\nValores nulos por columna:")
print(df.isnull().sum())

# 3. Limpiar la columna Price (remover comas y convertir a numérico)
print("\n" + "=" * 60)
print("LIMPIANDO COLUMNA PRICE...")
print("=" * 60)
print(f"Tipo de dato antes: {df['Price (in USD)'].dtype}")
print(f"Ejemplo antes: {df['Price (in USD)'].iloc[0]}")

df['Price (in USD)'] = df['Price (in USD)'].str.replace(',', '').astype(float)

print(f"Tipo de dato después: {df['Price (in USD)'].dtype}")
print(f"Ejemplo después: {df['Price (in USD)'].iloc[0]}")
print(f"Rango de precios: ${df['Price (in USD)'].min():,.0f} - ${df['Price (in USD)'].max():,.0f}")

# 4. Manejar Engine Size (eliminar valores no numéricos como "Electric")
print("\n" + "=" * 60)
print("LIMPIANDO COLUMNA ENGINE SIZE...")
print("=" * 60)
print(f"Valores únicos de Engine Size: {df['Engine Size (L)'].unique()}")
print(f"Filas antes: {len(df)}")

# Eliminar filas con "Electric" o valores no numéricos
df_before = len(df)
df = df[~df['Engine Size (L)'].str.contains('Electric', na=False, case=False)]
print(f"Filas eliminadas (Electric): {df_before - len(df)}")

# Convertir a numérico (coerce convierte errores a NaN)
df['Engine Size (L)'] = pd.to_numeric(df['Engine Size (L)'], errors='coerce')
print(f"Valores nulos después de conversión: {df['Engine Size (L)'].isnull().sum()}")

# 5. Convertir Horsepower a numérico
print("\n" + "=" * 60)
print("CONVIRTIENDO HORSEPOWER A NUMÉRICO...")
print("=" * 60)
print(f"Tipo antes: {df['Horsepower'].dtype}")
df['Horsepower'] = pd.to_numeric(df['Horsepower'], errors='coerce')
print(f"Tipo después: {df['Horsepower'].dtype}")
print(f"Valores nulos: {df['Horsepower'].isnull().sum()}")

# 6. Convertir Torque a numérico
print("\n" + "=" * 60)
print("CONVIRTIENDO TORQUE A NUMÉRICO...")
print("=" * 60)
print(f"Tipo antes: {df['Torque (lb-ft)'].dtype}")
df['Torque (lb-ft)'] = pd.to_numeric(df['Torque (lb-ft)'], errors='coerce')
print(f"Tipo después: {df['Torque (lb-ft)'].dtype}")
print(f"Valores nulos: {df['Torque (lb-ft)'].isnull().sum()}")

# 7. Convertir 0-60 MPH Time a numérico
print("\n" + "=" * 60)
print("CONVIRTIENDO 0-60 MPH TIME A NUMÉRICO...")
print("=" * 60)
print(f"Tipo antes: {df['0-60 MPH Time (seconds)'].dtype}")
df['0-60 MPH Time (seconds)'] = pd.to_numeric(df['0-60 MPH Time (seconds)'], errors='coerce')
print(f"Tipo después: {df['0-60 MPH Time (seconds)'].dtype}")
print(f"Valores nulos: {df['0-60 MPH Time (seconds)'].isnull().sum()}")

# 8. Eliminar filas con valores nulos
print("\n" + "=" * 60)
print("ELIMINANDO VALORES NULOS...")
print("=" * 60)
print(f"Filas antes: {len(df)}")
print(f"Total de valores nulos por columna:")
print(df.isnull().sum())

df = df.dropna()
print(f"\nFilas después de eliminar nulos: {len(df)}")
print(f"Filas eliminadas: {df_before - len(df)}")

# 9. Descartar columnas Car Make y Car Model (variables categóricas)
print("\n" + "=" * 60)
print("ELIMINANDO COLUMNAS CATEGÓRICAS...")
print("=" * 60)
print(f"Columnas antes: {list(df.columns)}")

df_clean = df.drop(['Car Make', 'Car Model'], axis=1)

print(f"Columnas después: {list(df_clean.columns)}")

# 10. Resumen final
print("\n" + "=" * 60)
print("RESUMEN FINAL DEL DATASET LIMPIO")
print("=" * 60)
print(f"Dimensiones: {df_clean.shape}")
print("\nInformación:")
print(df_clean.info())
print("\nEstadísticas descriptivas:")
print(df_clean.describe())

# 11. Verificar que no hay valores nulos
print("\n" + "=" * 60)
print("VERIFICACIÓN FINAL")
print("=" * 60)
print(f"Valores nulos totales: {df_clean.isnull().sum().sum()}")
print(f"Todas las columnas son numéricas: {all(df_clean.dtypes != 'object')}")

# 12. Guardar el dataset limpio
output_file = 'Sport_car_price_clean.csv'
df_clean.to_csv(output_file, index=False)
print(f"\n Dataset limpio guardado en: {output_file}")

# 13. Mostrar las primeras filas del dataset limpio
print("\n" + "=" * 60)
print("PRIMERAS FILAS DEL DATASET LIMPIO")
print("=" * 60)
print(df_clean.head(10))

print("\n" + "=" * 60)
print("¡PREPROCESAMIENTO COMPLETADO CON ÉXITO!")
print("=" * 60)
print(f" Dataset original: 1007 filas")
print(f" Dataset limpio: {len(df_clean)} filas")
print(f" Filas perdidas: {1007 - len(df_clean)} ({((1007 - len(df_clean))/1007*100):.1f}%)")
print(f" Columnas finales: {len(df_clean.columns)}")
print(f" Variable objetivo: Price (in USD)")
print(f" Variables predictoras: {list(df_clean.columns[:-1])}")
