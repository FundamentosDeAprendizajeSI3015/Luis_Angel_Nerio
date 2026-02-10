import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# Cargar el DataFrame del Titanic
df = pd.read_csv('Titanic-Dataset.csv')

print("DataFrame Original - Primeras 5 filas:")
print(df.head())

print("\n--- Información del Dataset ---")
print(f"Dimensiones: {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"\nColumnas: {df.columns.tolist()}")

print("\n--- Media (Promedio) de Variables Numéricas ---")

# Media de la columna 'Age'
media_age = df['Age'].mean()
print(f"Media de Edad: {media_age:.2f} años")

# Media de la columna 'Fare'
media_fare = df['Fare'].mean()
print(f"Media de Tarifa: {media_fare:.2f}")

# Media de la columna 'SibSp' (Hermanos/Cónyuges a bordo)
media_sibsp = df['SibSp'].mean()
print(f"Media de Hermanos/Cónyuges: {media_sibsp:.2f}")

# Media de la columna 'Parch' (Padres/Hijos a bordo)
media_parch = df['Parch'].mean()
print(f"Media de Padres/Hijos: {media_parch:.2f}")

# También puedes calcular la media de todas las columnas numéricas a la vez
media_todas = df[['Age', 'Fare', 'SibSp', 'Parch']].mean()
print("\nMedia de todas las variables numéricas:")
print(media_todas)

print("\n--- Mediana de Variables Numéricas ---")

# Mediana de la columna 'Age'
mediana_age = df['Age'].median()
print(f"Mediana de Edad: {mediana_age:.2f} años")

# Mediana de la columna 'Fare'
mediana_fare = df['Fare'].median()
print(f"Mediana de Tarifa: {mediana_fare:.2f}")

# Mediana de la columna 'Pclass'
mediana_pclass = df['Pclass'].median()
print(f"Mediana de Clase: {mediana_pclass:.2f}")

# También puedes calcular la mediana de todas las columnas numéricas a la vez
mediana_todas = df[['Age', 'Fare', 'SibSp', 'Parch', 'Pclass']].median()
print("\nMediana de todas las variables numéricas:")
print(mediana_todas)

print("\n--- Moda de Variables Categóricas y Numéricas ---")

# Moda de la columna 'Survived'
moda_survived = df['Survived'].mode()
print(f"Moda de Supervivencia: {moda_survived.tolist()} (0=No sobrevivió, 1=Sobrevivió)")

# Moda de la columna 'Pclass'
moda_pclass = df['Pclass'].mode()
print(f"Moda de Clase: {moda_pclass.tolist()}")

# Moda de la columna 'Sex'
moda_sex = df['Sex'].mode()
print(f"Moda de Género: {moda_sex.tolist()}")

# Moda de la columna 'Embarked'
moda_embarked = df['Embarked'].mode()
print(f"Moda de Puerto de Embarque: {moda_embarked.tolist()} (C=Cherbourg, Q=Queenstown, S=Southampton)")

# Moda de la columna 'SibSp'
moda_sibsp = df['SibSp'].mode()
print(f"Moda de Hermanos/Cónyuges: {moda_sibsp.tolist()}")

print("\n--- Estadísticas Adicionales ---")
print(f"Tasa de Supervivencia: {df['Survived'].mean():.2%}")
print(f"Distribución por Género:")
print(df['Sex'].value_counts())

# --- Calcular Q1, Q3 y el IQR para identificar outliers (usando Age) ---


# Nota: La columna 'Age' tiene valores nulos, los eliminamos para este análisis
df_age = df['Age'].dropna()

print("\n--- Análisis de Cuartiles y Outliers (Edad) ---")
print(f"Número de observaciones (sin valores nulos): {len(df_age)}")

Q1 = df_age.quantile(0.25)
Q2 = df_age.quantile(0.5)
Q3 = df_age.quantile(0.75)

IQR = Q3 - Q1

lower_bound_outlier = Q1 - 1.5 * IQR #outlier por debajo.
upper_bound_outlier = Q3 + 1.5 * IQR #outlier por encima.

print(f"\nQ1 (25%): {Q1:.2f}")
print(f"Q2 (50%): {Q2:.2f}")
print(f"Q3 (75%): {Q3:.2f}")
print(f"IQR: {IQR:.2f}")
print(f"Límite inferior para outliers: {lower_bound_outlier:.2f}")
print(f"Límite superior para outliers: {upper_bound_outlier:.2f}")

# Identificar outliers
outliers = df[df['Age'].notna() & ((df['Age'] < lower_bound_outlier) | (df['Age'] > upper_bound_outlier))]

print(f"\nNúmero de outliers encontrados: {len(outliers)}")
print("\nPrimeros outliers según la regla del 1.5 * IQR:")
print(outliers[['PassengerId', 'Name', 'Age', 'Pclass']].head())

# --- Visualizar con un Box Plot para confirmar ---
plt.figure(figsize=(8, 6))
sns.boxplot(y=df_age)
plt.title('Box Plot de Edad en el Titanic - Detección de Outliers')
plt.ylabel('Edad')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('01_boxplot_edad_outliers.png', dpi=300, bbox_inches='tight')
plt.show()

# --- Análisis de Percentiles (usando Age) ---
print("\n--- Análisis de Percentiles (Edad) ---")
print(f"Número total de datos (sin valores nulos): {len(df_age)}")

# --- Usando Pandas ---
# El método .quantile() espera la fracción (0.0 a 1.0)
# Calcular el percentil 70 (0.70)
percentil_70_pandas = df_age.quantile(0.70)
print(f"\nPercentil 70 (Pandas): {percentil_70_pandas:.2f} años")

# Calcular múltiples percentiles a la vez (ej. P10, P50 (mediana), P90)
multi_percentiles_pandas = df_age.quantile([0.10, 0.50, 0.90])
print("\nMúltiples percentiles (Pandas):")
print(multi_percentiles_pandas)

# --- Usando NumPy ---
# El método np.percentile() espera el percentil como un número entero (0 a 100)
percentil_70_numpy = np.percentile(df_age, 70)
print(f"\nPercentil 70 (NumPy): {percentil_70_numpy:.2f} años")

# Calcular múltiples percentiles con NumPy
multi_percentiles_numpy = np.percentile(df_age, [10, 50, 90])
print("\nMúltiples percentiles (NumPy):")
print(multi_percentiles_numpy)


# ============================================
# MEDIDAS DE POSICIÓN
# ============================================

print("\n" + "="*60)
print("MEDIDAS DE POSICIÓN")
print("="*60)

# --- Deciles ---
print("\n--- Deciles (Edad) ---")
print("Los deciles dividen los datos en 10 partes iguales")
deciles = df_age.quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
for i, valor in enumerate(deciles, start=1):
    print(f"D{i} (Decil {i}): {valor:.2f} años")

# --- Quintiles ---
print("\n--- Quintiles (Edad) ---")
print("Los quintiles dividen los datos en 5 partes iguales")
quintiles = df_age.quantile([0.2, 0.4, 0.6, 0.8])
for i, valor in enumerate(quintiles, start=1):
    print(f"Q{i} (Quintil {i}): {valor:.2f} años")

# --- Medidas de Posición para Tarifa (Fare) ---
print("\n--- Medidas de Posición para Tarifa (Fare) ---")
df_fare = df['Fare'].dropna()

print(f"\nCuartiles de Tarifa:")
Q1_fare = df_fare.quantile(0.25)
Q2_fare = df_fare.quantile(0.50)
Q3_fare = df_fare.quantile(0.75)
print(f"Q1 (25%): ${Q1_fare:.2f}")
print(f"Q2 (50%): ${Q2_fare:.2f}")
print(f"Q3 (75%): ${Q3_fare:.2f}")

print(f"\nDeciles de Tarifa:")
deciles_fare = df_fare.quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
for i, valor in enumerate(deciles_fare, start=1):
    print(f"D{i}: ${valor:.2f}")


# ============================================
# ELIMINACIÓN DE OUTLIERS
# ============================================

print("\n" + "="*60)
print("ELIMINACIÓN DE OUTLIERS")
print("="*60)

# Crear una copia del dataframe original
df_sin_outliers = df.copy()

print(f"\n--- Dataset Original ---")
print(f"Total de registros: {len(df)}")
print(f"Total de valores de Edad (no nulos): {df['Age'].notna().sum()}")

# --- Eliminar outliers de Age ---
print("\n--- Eliminando Outliers de Edad ---")
print(f"Límites para Age: [{lower_bound_outlier:.2f}, {upper_bound_outlier:.2f}]")

# Filtrar el dataframe: mantener solo las filas donde Age está dentro de los límites
# o donde Age es nulo (para no perder esos registros)
df_sin_outliers = df_sin_outliers[
    (df_sin_outliers['Age'].isna()) | 
    ((df_sin_outliers['Age'] >= lower_bound_outlier) & (df_sin_outliers['Age'] <= upper_bound_outlier))
]

print(f"Registros eliminados por outliers en Age: {len(df) - len(df_sin_outliers)}")
print(f"Total de registros después de eliminar outliers: {len(df_sin_outliers)}")

# --- Eliminar outliers de Fare ---
print("\n--- Eliminando Outliers de Tarifa (Fare) ---")
Q1_fare = df_sin_outliers['Fare'].quantile(0.25)
Q3_fare = df_sin_outliers['Fare'].quantile(0.75)
IQR_fare = Q3_fare - Q1_fare

lower_bound_fare = Q1_fare - 1.5 * IQR_fare
upper_bound_fare = Q3_fare + 1.5 * IQR_fare

print(f"Límites para Fare: [{lower_bound_fare:.2f}, {upper_bound_fare:.2f}]")

outliers_fare = df_sin_outliers[
    (df_sin_outliers['Fare'].notna()) & 
    ((df_sin_outliers['Fare'] < lower_bound_fare) | (df_sin_outliers['Fare'] > upper_bound_fare))
]
print(f"Outliers detectados en Fare: {len(outliers_fare)}")

df_sin_outliers = df_sin_outliers[
    (df_sin_outliers['Fare'].isna()) | 
    ((df_sin_outliers['Fare'] >= lower_bound_fare) & (df_sin_outliers['Fare'] <= upper_bound_fare))
]

print(f"Registros eliminados por outliers en Fare: {len(df) - len(df_sin_outliers) - len(outliers)}")
print(f"Total de registros después de eliminar todos los outliers: {len(df_sin_outliers)}")

# --- Comparación antes y después ---
print("\n--- Comparación de Estadísticas: Original vs Sin Outliers ---")

print("\nEdad (Age):")
print(f"  Original - Media: {df['Age'].mean():.2f}, Mediana: {df['Age'].median():.2f}, Std: {df['Age'].std():.2f}")
print(f"  Sin Outliers - Media: {df_sin_outliers['Age'].mean():.2f}, Mediana: {df_sin_outliers['Age'].median():.2f}, Std: {df_sin_outliers['Age'].std():.2f}")

print("\nTarifa (Fare):")
print(f"  Original - Media: {df['Fare'].mean():.2f}, Mediana: {df['Fare'].median():.2f}, Std: {df['Fare'].std():.2f}")
print(f"  Sin Outliers - Media: {df_sin_outliers['Fare'].mean():.2f}, Mediana: {df_sin_outliers['Fare'].median():.2f}, Std: {df_sin_outliers['Fare'].std():.2f}")

# --- Visualización comparativa ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Box plot Age - Original
axes[0, 0].boxplot(df['Age'].dropna())
axes[0, 0].set_title('Age - Original')
axes[0, 0].set_ylabel('Edad')
axes[0, 0].grid(axis='y', alpha=0.3)

# Box plot Age - Sin Outliers
axes[0, 1].boxplot(df_sin_outliers['Age'].dropna())
axes[0, 1].set_title('Age - Sin Outliers')
axes[0, 1].set_ylabel('Edad')
axes[0, 1].grid(axis='y', alpha=0.3)

# Box plot Fare - Original
axes[1, 0].boxplot(df['Fare'].dropna())
axes[1, 0].set_title('Fare - Original')
axes[1, 0].set_ylabel('Tarifa ($)')
axes[1, 0].grid(axis='y', alpha=0.3)

# Box plot Fare - Sin Outliers
axes[1, 1].boxplot(df_sin_outliers['Fare'].dropna())
axes[1, 1].set_title('Fare - Sin Outliers')
axes[1, 1].set_ylabel('Tarifa ($)')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.suptitle('Comparación: Dataset Original vs Sin Outliers', y=1.02, fontsize=14, fontweight='bold')
plt.savefig('02_comparacion_outliers_eliminados.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n--- Dataset limpio guardado en variable 'df_sin_outliers' ---")
print(f"Forma final: {df_sin_outliers.shape}")


# ============================================
# ANÁLISIS DE DISTRIBUCIÓN DE SALARIOS
# ============================================
# Nota: %reset -f es un comando de IPython/Jupyter, no aplicable en scripts .py

print("\n" + "="*60)
print("ANÁLISIS DE DISTRIBUCIÓN DE SALARIOS")
print("="*60)

# Generar un DataFrame de salarios anuales (reutilizando el ejemplo anterior)
np.random.seed(42) # Para reproducibilidad

salarios_bajos = np.random.normal(loc=40000, scale=8000, size=900)
salarios_altos = np.random.normal(loc=120000, scale=25000, size=100)
salarios_super_altos = np.random.normal(loc=300000, scale=50000, size=10)

salarios = np.concatenate([salarios_bajos, salarios_altos, salarios_super_altos])

#Evitar valores negativos o irrealmente bajos:
salarios = np.maximum(salarios, 15000)

df_salarios = pd.DataFrame({'Salario_Anual': salarios})

print("\nDataFrame de Salarios (primeras 5 filas):")
print(df_salarios.head())
print(f"\nNúmero total de empleados: {len(df_salarios)}")

# Calcular la media y la mediana para visualizarlas en el histograma
media_salario = df_salarios['Salario_Anual'].mean()
mediana_salario = df_salarios['Salario_Anual'].median()

# --- Crear el Histograma ---
plt.figure(figsize=(12, 6)) # Define el tamaño de la figura

# data: el DataFrame
# x: la columna que queremos graficar
# bins: Número de intervalos o una secuencia de bordes de bin
# color: Color de las barras
#sns.histplot(data=df_salarios, x='Salario_Anual', kde=True, bins=50, color='skyblue')
sns.histplot(data=df_salarios, x='Salario_Anual', bins=50, color='skyblue')

# Añadir líneas para la media y la mediana (opcional, pero útil para EDA)
plt.axvline(media_salario, color='red', linestyle='--', label=f'Media: ${media_salario:,.0f}')
plt.axvline(mediana_salario, color='green', linestyle='-', label=f'Mediana: ${mediana_salario:,.0f}')

# Añadir título y etiquetas a los ejes
plt.title('Distribución de Salarios Anuales (Histograma)', fontsize=16)
plt.xlabel('Salario Anual ($)', fontsize=12)
plt.ylabel('Frecuencia de Empleados', fontsize=12)

# Añadir leyenda y cuadrícula
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Mostrar el gráfico
plt.savefig('03_histograma_salarios.png', dpi=300, bbox_inches='tight')
plt.show()


# ============================================
# GRÁFICOS DE DISPERSIÓN
# ============================================
# Nota: %reset -f es un comando de IPython/Jupyter, no aplicable en scripts .py

print("\n" + "="*60)
print("GRÁFICOS DE DISPERSIÓN - ANÁLISIS DE RELACIONES")
print("="*60)

# Usaremos el DataFrame del Titanic (df) para analizar relaciones entre variables
# Vamos a crear varios gráficos de dispersión para diferentes pares de variables

# --- Gráfico 1: Age vs Fare ---
print("\n--- Análisis 1: Relación entre Edad y Tarifa ---")

# Crear un DataFrame sin valores nulos para estas columnas
df_age_fare = df[['Age', 'Fare']].dropna()
print(f"Número de registros válidos (Age y Fare sin nulos): {len(df_age_fare)}")

# Calcular la correlación
correlacion_age_fare = df_age_fare['Age'].corr(df_age_fare['Fare'])
print(f"Coeficiente de Correlación entre Age y Fare: {correlacion_age_fare:.3f}")

# Crear el gráfico de dispersión
plt.figure(figsize=(10, 7))
sns.scatterplot(x='Age', y='Fare', data=df_age_fare, color='teal', alpha=0.6, s=80)

# Añadir línea de regresión para visualizar la tendencia
sns.regplot(x='Age', y='Fare', data=df_age_fare, 
            scatter=False, color='red', line_kws={'linestyle':'--', 'linewidth':2}, 
            label=f'Línea de Tendencia (r={correlacion_age_fare:.3f})')

plt.title('Relación entre Edad y Tarifa Pagada - Titanic', fontsize=16)
plt.xlabel('Edad (años)', fontsize=12)
plt.ylabel('Tarifa Pagada ($)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('04_dispersion_age_vs_fare.png', dpi=300, bbox_inches='tight')
plt.show()


# --- Gráfico 2: Age vs Pclass (con color por clase) ---
print("\n--- Análisis 2: Relación entre Edad y Clase de Pasajero ---")

df_age_pclass = df[['Age', 'Pclass', 'Survived']].dropna()
print(f"Número de registros válidos: {len(df_age_pclass)}")

# Crear el gráfico de dispersión con colores por Pclass
plt.figure(figsize=(10, 7))
sns.scatterplot(x='Pclass', y='Age', data=df_age_pclass, 
                hue='Survived', palette={0: 'red', 1: 'green'}, 
                alpha=0.6, s=80)

plt.title('Relación entre Clase de Pasajero y Edad (por Supervivencia)', fontsize=16)
plt.xlabel('Clase de Pasajero (1=Primera, 2=Segunda, 3=Tercera)', fontsize=12)
plt.ylabel('Edad (años)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Sobrevivió', labels=['No', 'Sí'])
plt.tight_layout()
plt.savefig('05_dispersion_age_vs_pclass.png', dpi=300, bbox_inches='tight')
plt.show()


# --- Gráfico 3: SibSp vs Parch ---
print("\n--- Análisis 3: Relación entre Hermanos/Cónyuges y Padres/Hijos ---")

df_family = df[['SibSp', 'Parch', 'Survived']].dropna()
print(f"Número de registros válidos: {len(df_family)}")

# Calcular la correlación
correlacion_family = df_family['SibSp'].corr(df_family['Parch'])
print(f"Coeficiente de Correlación entre SibSp y Parch: {correlacion_family:.3f}")

# Crear el gráfico de dispersión
plt.figure(figsize=(10, 7))
sns.scatterplot(x='SibSp', y='Parch', data=df_family,
                hue='Survived', palette={0: 'red', 1: 'green'},
                alpha=0.6, s=100)

plt.title('Relación entre Hermanos/Cónyuges y Padres/Hijos a Bordo', fontsize=16)
plt.xlabel('Número de Hermanos/Cónyuges a Bordo', fontsize=12)
plt.ylabel('Número de Padres/Hijos a Bordo', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='Sobrevivió', labels=['No', 'Sí'])
plt.tight_layout()
plt.savefig('06_dispersion_sibsp_vs_parch.png', dpi=300, bbox_inches='tight')
plt.show()


# --- Gráfico 4: Fare vs Pclass ---
print("\n--- Análisis 4: Relación entre Tarifa y Clase de Pasajero ---")

df_fare_pclass = df[['Fare', 'Pclass']].dropna()
print(f"Número de registros válidos: {len(df_fare_pclass)}")

# Calcular la correlación
correlacion_fare_pclass = df_fare_pclass['Pclass'].corr(df_fare_pclass['Fare'])
print(f"Coeficiente de Correlación entre Pclass y Fare: {correlacion_fare_pclass:.3f}")

# Crear el gráfico de dispersión con jitter para ver mejor los puntos
plt.figure(figsize=(10, 7))
sns.stripplot(x='Pclass', y='Fare', data=df_fare_pclass, 
              alpha=0.5, jitter=True, size=4, color='steelblue')

# Añadir box plot superpuesto para ver la distribución
sns.boxplot(x='Pclass', y='Fare', data=df_fare_pclass, 
            width=0.3, showfliers=False, color='lightcoral', 
            boxprops=dict(alpha=0.5), whiskerprops=dict(alpha=0.5),
            capprops=dict(alpha=0.5), medianprops=dict(color='red', linewidth=2))

plt.title('Relación entre Clase de Pasajero y Tarifa Pagada', fontsize=16)
plt.xlabel('Clase de Pasajero (1=Primera, 2=Segunda, 3=Tercera)', fontsize=12)
plt.ylabel('Tarifa Pagada ($)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6, axis='y')
plt.tight_layout()
plt.savefig('07_dispersion_fare_vs_pclass.png', dpi=300, bbox_inches='tight')
plt.show()


# --- Matriz de Correlación para Variables Numéricas ---
print("\n--- Matriz de Correlación entre Variables Numéricas ---")

# Seleccionar solo columnas numéricas
columnas_numericas = ['Age', 'Fare', 'SibSp', 'Parch', 'Pclass', 'Survived']
df_correlacion = df[columnas_numericas].dropna()

# Calcular la matriz de correlación
matriz_correlacion = df_correlacion.corr()
print("\nMatriz de Correlación:")
print(matriz_correlacion)

# Visualizar con un heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(matriz_correlacion, annot=True, cmap='coolwarm', center=0,
            fmt='.3f', linewidths=0.5, square=True, cbar_kws={'label': 'Correlación'})
plt.title('Matriz de Correlación - Variables del Titanic', fontsize=16)
plt.tight_layout()
plt.savefig('08_matriz_correlacion.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n--- Análisis de Gráficos de Dispersión Completado ---")


# ============================================
# TRANSFORMACIÓN DE COLUMNAS - ENCODING
# ============================================

print("\n" + "="*60)
print("TRANSFORMACIÓN DE COLUMNAS - ENCODING")
print("="*60)

# Trabajaremos con una copia del DataFrame original del Titanic
df_encoding = df.copy()

# --- ONE HOT ENCODING ---
print("\n--- 1. ONE HOT ENCODING ---")
print("Transforma variables categóricas en columnas binarias (0 o 1)")
print("\nColumnas categóricas originales: Sex, Embarked")

# Mostrar distribución original
print("\nDistribución de 'Sex':")
print(df_encoding['Sex'].value_counts())
print("\nDistribución de 'Embarked':")
print(df_encoding['Embarked'].value_counts())

# Aplicar One-Hot Encoding a las columnas Sex y Embarked
df_one_hot = pd.get_dummies(df_encoding, columns=['Sex', 'Embarked'], prefix=['Sex', 'Embarked'])

print("\nDataFrame después de One-Hot Encoding (primeras 5 filas):")
print(df_one_hot[['PassengerId', 'Name', 'Sex_female', 'Sex_male', 
                   'Embarked_C', 'Embarked_Q', 'Embarked_S']].head())

print(f"\nColumnas agregadas: {[col for col in df_one_hot.columns if col.startswith(('Sex_', 'Embarked_'))]}")
print(f"Forma del DataFrame Original: {df_encoding.shape}")
print(f"Forma del DataFrame con One-Hot: {df_one_hot.shape}")


# --- LABEL ENCODING ---
print("\n" + "="*60)
print("--- 2. LABEL ENCODING ---")
print("Asigna un número entero único a cada categoría")

from sklearn.preprocessing import LabelEncoder

# Crear una copia para Label Encoding
df_label = df_encoding.copy()

# Aplicar Label Encoding a la columna 'Sex'
le_sex = LabelEncoder()
df_label['Sex_Encoded'] = le_sex.fit_transform(df_label['Sex'])

print("\nLabel Encoding para 'Sex':")
print(f"Categorías originales: {le_sex.classes_}")
print(f"Mapeo: {dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))}")
print("\nComparación:")
print(df_label[['PassengerId', 'Name', 'Sex', 'Sex_Encoded']].head(10))

# Aplicar Label Encoding a la columna 'Embarked' (eliminando nulos primero)
df_label_embarked = df_label[df_label['Embarked'].notna()].copy()
le_embarked = LabelEncoder()
df_label_embarked['Embarked_Encoded'] = le_embarked.fit_transform(df_label_embarked['Embarked'])

print("\nLabel Encoding para 'Embarked':")
print(f"Categorías originales: {le_embarked.classes_}")
print(f"Mapeo: {dict(zip(le_embarked.classes_, le_embarked.transform(le_embarked.classes_)))}")
print("\nComparación:")
print(df_label_embarked[['PassengerId', 'Name', 'Embarked', 'Embarked_Encoded']].head(10))


# --- BINARY ENCODING ---
print("\n" + "="*60)
print("--- 3. BINARY ENCODING ---")
print("Convierte categorías a representación binaria (más eficiente para muchas categorías)")

# Verificar si category_encoders está instalado, si no, intentar instalarlo
try:
    import category_encoders as ce
except ImportError:
    print("\nInstalando category_encoders...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "category_encoders", "-q"])
    import category_encoders as ce

# Crear una copia para Binary Encoding
df_binary = df_encoding.copy()

# Aplicar Binary Encoding a la columna 'Embarked'
# Primero llenamos los nulos con 'Unknown'
df_binary['Embarked'] = df_binary['Embarked'].fillna('Unknown')

encoder_embarked = ce.BinaryEncoder(cols=['Embarked'])
df_binary_embarked = encoder_embarked.fit_transform(df_binary['Embarked'])

print("\nBinary Encoding para 'Embarked':")
print(f"Categorías originales: C, Q, S, Unknown")
print("\nColumnas generadas por Binary Encoding:")
print(df_binary_embarked.head(10))

# Concatenar con el dataframe original
df_binary = pd.concat([df_binary, df_binary_embarked], axis=1)

print("\nDataFrame con Binary Encoding (muestra):")
print(df_binary[['PassengerId', 'Name', 'Embarked'] + list(df_binary_embarked.columns)].head(10))


# Aplicar Binary Encoding a la columna 'Pclass' (aunque es numérica, la trataremos como categórica)
encoder_pclass = ce.BinaryEncoder(cols=['Pclass'])
df_binary_pclass = encoder_pclass.fit_transform(df_binary[['Pclass']])

print("\nBinary Encoding para 'Pclass':")
print(f"Categorías originales: 1, 2, 3")
print("\nColumnas generadas:")
print(df_binary_pclass.head(10))


# --- COMPARACIÓN DE TÉCNICAS ---
print("\n" + "="*60)
print("COMPARACIÓN DE TÉCNICAS DE ENCODING")
print("="*60)

print("\n1. ONE-HOT ENCODING:")
print("   ✓ Ventajas: Fácil de interpretar, no asume orden entre categorías")
print("   ✗ Desventajas: Aumenta mucho las dimensiones con muchas categorías")
print(f"   Ejemplo: 'Embarked' (3 categorías) → 3 columnas binarias")

print("\n2. LABEL ENCODING:")
print("   ✓ Ventajas: Muy simple, no aumenta dimensiones")
print("   ✗ Desventajas: Asume orden entre categorías (puede confundir al modelo)")
print(f"   Ejemplo: 'Embarked' (3 categorías) → 1 columna con valores 0, 1, 2")

print("\n3. BINARY ENCODING:")
print("   ✓ Ventajas: Más eficiente que One-Hot, mantiene baja dimensionalidad")
print("   ✗ Desventajas: Menos interpretable, requiere librería adicional")
print(f"   Ejemplo: 'Embarked' (4 categorías con Unknown) → 2 columnas binarias")

print("\n" + "="*60)
print("RESUMEN DE SHAPES")
print("="*60)
print(f"DataFrame Original: {df_encoding.shape}")
print(f"Con One-Hot Encoding: {df_one_hot.shape}")
print(f"Con Label Encoding: {df_label.shape} (mismo tamaño, solo nuevas columnas)")
print(f"Con Binary Encoding: {df_binary.shape}")

print("\n--- Transformación de Columnas Completada ---")


# ============================================
# ANÁLISIS DE CORRELACIÓN Y SELECCIÓN DE COLUMNAS
# ============================================

print("\n" + "="*60)
print("ANÁLISIS DE CORRELACIÓN ENTRE COLUMNAS")
print("="*60)

# Seleccionar solo las columnas numéricas del DataFrame original
columnas_numericas = ['Age', 'Fare', 'SibSp', 'Parch', 'Pclass', 'Survived']
df_correlacion = df[columnas_numericas].copy()

# Eliminar filas con valores nulos para el análisis
print(f"\nFilas originales: {len(df)}")
df_correlacion = df_correlacion.dropna()
print(f"Filas después de eliminar nulos: {len(df_correlacion)}")

# --- Correlación entre dos columnas específicas ---
print("\n--- Correlación entre columnas específicas ---")

# Correlación Age vs Fare
corr_age_fare = df_correlacion['Age'].corr(df_correlacion['Fare'])
print(f"Correlación entre Age y Fare: {corr_age_fare:.4f}")

# Correlación SibSp vs Parch
corr_sibsp_parch = df_correlacion['SibSp'].corr(df_correlacion['Parch'])
print(f"Correlación entre SibSp y Parch: {corr_sibsp_parch:.4f}")

# Correlación Pclass vs Fare
corr_pclass_fare = df_correlacion['Pclass'].corr(df_correlacion['Fare'])
print(f"Correlación entre Pclass y Fare: {corr_pclass_fare:.4f}")

# Correlación Pclass vs Survived
corr_pclass_survived = df_correlacion['Pclass'].corr(df_correlacion['Survived'])
print(f"Correlación entre Pclass y Survived: {corr_pclass_survived:.4f}")


# --- Matriz de Correlación Completa ---
print("\n--- Matriz de Correlación Completa ---")
matriz_corr = df_correlacion.corr()
print("\nMatriz de correlación:")
print(matriz_corr.round(3))


# --- Visualización: Mapa de Calor de Correlaciones ---
plt.figure(figsize=(10, 8))
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', center=0,
            fmt='.3f', linewidths=1, square=True, 
            cbar_kws={'label': 'Coeficiente de Correlación'},
            vmin=-1, vmax=1)
plt.title('Mapa de Calor de Correlaciones - Dataset Titanic', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('09_heatmap_correlaciones.png', dpi=300, bbox_inches='tight')
plt.show()


# --- Análisis de Correlación con Pearson y Spearman ---
print("\n" + "="*60)
print("COMPARACIÓN: CORRELACIÓN PEARSON VS SPEARMAN")
print("="*60)

print("\nPearson: Mide relaciones LINEALES")
print("Spearman: Mide relaciones MONOTÓNICAS (lineales o no)")

# Comparar ambos métodos para algunas variables clave
print("\n--- Age vs Fare ---")
pearson_af = df_correlacion['Age'].corr(df_correlacion['Fare'], method='pearson')
spearman_af = df_correlacion['Age'].corr(df_correlacion['Fare'], method='spearman')
print(f"Pearson:  {pearson_af:.4f}")
print(f"Spearman: {spearman_af:.4f}")

print("\n--- Pclass vs Fare ---")
pearson_pf = df_correlacion['Pclass'].corr(df_correlacion['Fare'], method='pearson')
spearman_pf = df_correlacion['Pclass'].corr(df_correlacion['Fare'], method='spearman')
print(f"Pearson:  {pearson_pf:.4f}")
print(f"Spearman: {spearman_pf:.4f}")

print("\n--- SibSp vs Parch ---")
pearson_sp = df_correlacion['SibSp'].corr(df_correlacion['Parch'], method='pearson')
spearman_sp = df_correlacion['SibSp'].corr(df_correlacion['Parch'], method='spearman')
print(f"Pearson:  {pearson_sp:.4f}")
print(f"Spearman: {spearman_sp:.4f}")


# --- Identificar columnas altamente correlacionadas ---
print("\n" + "="*60)
print("IDENTIFICACIÓN DE COLUMNAS ALTAMENTE CORRELACIONADAS")
print("="*60)

# Definir umbral para correlación alta (típicamente > 0.7 o < -0.7)
umbral_correlacion = 0.7

print(f"\nUmbral de correlación alta: ±{umbral_correlacion}")
print("\nPares de variables con correlación alta:")

# Buscar correlaciones altas (excluyendo la diagonal)
encontrado = False
for i in range(len(matriz_corr.columns)):
    for j in range(i+1, len(matriz_corr.columns)):
        corr_value = matriz_corr.iloc[i, j]
        if abs(corr_value) >= umbral_correlacion:
            var1 = matriz_corr.columns[i]
            var2 = matriz_corr.columns[j]
            print(f"  • {var1} vs {var2}: {corr_value:.4f}")
            encontrado = True

if not encontrado:
    print(f"  No se encontraron pares de variables con correlación >= {umbral_correlacion}")


# --- Recomendaciones para eliminación de columnas ---
print("\n" + "="*60)
print("RECOMENDACIONES PARA ELIMINACIÓN DE COLUMNAS")
print("="*60)

print("\n📊 CRITERIOS DE DECISIÓN:")
print("  • Correlación alta (|r| > 0.7): Considerar eliminar una de las dos")
print("  • Correlación moderada (0.5 < |r| < 0.7): Evaluar según contexto")
print("  • Correlación baja (|r| < 0.5): Mantener ambas columnas")

print("\n🔍 ANÁLISIS DEL DATASET TITANIC:")

# Evaluar cada par de correlaciones
correlaciones_importantes = []
for i in range(len(matriz_corr.columns)):
    for j in range(i+1, len(matriz_corr.columns)):
        corr_value = abs(matriz_corr.iloc[i, j])
        if corr_value >= 0.5:  # Correlación moderada o alta
            var1 = matriz_corr.columns[i]
            var2 = matriz_corr.columns[j]
            correlaciones_importantes.append((var1, var2, matriz_corr.iloc[i, j]))

if correlaciones_importantes:
    print("\nPares con correlación moderada o alta:")
    for var1, var2, corr_val in sorted(correlaciones_importantes, key=lambda x: abs(x[2]), reverse=True):
        if abs(corr_val) >= 0.7:
            nivel = "⚠️ ALTA"
        else:
            nivel = "⚡ MODERADA"
        print(f"  {nivel} - {var1} vs {var2}: {corr_val:.4f}")
else:
    print("\nNo hay pares con correlación moderada o alta.")

print("\n💡 RECOMENDACIONES ESPECÍFICAS:")

# Analizar correlaciones específicas del Titanic
if abs(matriz_corr.loc['Pclass', 'Fare']) >= 0.5:
    print(f"\n  1. Pclass vs Fare (r={matriz_corr.loc['Pclass', 'Fare']:.3f}):")
    print("     → Correlación moderada/alta NEGATIVA")
    print("     → A mayor clase (3), menor tarifa")
    print("     → RECOMENDACIÓN: MANTENER ambas")
    print("       • Pclass: Variable categórica ordinal importante")
    print("       • Fare: Variable continua con información única")

if abs(matriz_corr.loc['SibSp', 'Parch']) >= 0.3:
    print(f"\n  2. SibSp vs Parch (r={matriz_corr.loc['SibSp', 'Parch']:.3f}):")
    print("     → Correlación baja/moderada POSITIVA")
    print("     → Ambas relacionadas con tamaño de familia")
    print("     → RECOMENDACIÓN: MANTENER ambas")
    print("       • Representan relaciones familiares diferentes")
    print("       • O crear una nueva variable 'FamilySize' = SibSp + Parch + 1")

if abs(matriz_corr.loc['Age', 'Fare']) < 0.2:
    print(f"\n  3. Age vs Fare (r={matriz_corr.loc['Age', 'Fare']:.3f}):")
    print("     → Correlación MUY BAJA")
    print("     → Variables independientes entre sí")
    print("     → RECOMENDACIÓN: MANTENER ambas")

print("\n✅ CONCLUSIÓN FINAL:")
print("  En el dataset del Titanic, NO hay columnas con correlación")
print("  lo suficientemente alta como para recomendar su eliminación.")
print("  Todas las variables aportan información única y valiosa.")

# --- Crear DataFrame con variables seleccionadas (ejemplo) ---
print("\n" + "="*60)
print("CREACIÓN DE DATASET CON FEATURE ENGINEERING")
print("="*60)

# Crear una nueva variable combinando SibSp y Parch
df_final = df_correlacion.copy()
df_final['FamilySize'] = df_final['SibSp'] + df_final['Parch'] + 1
df_final['IsAlone'] = (df_final['FamilySize'] == 1).astype(int)

print("\nNuevas variables creadas:")
print("  • FamilySize: SibSp + Parch + 1")
print("  • IsAlone: 1 si viajaba solo, 0 si con familia")

print("\nDataFrame final (primeras 10 filas):")
print(df_final[['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize', 'IsAlone', 'Survived']].head(10))

# Correlación de las nuevas variables con Survived
print("\nCorrelación de nuevas variables con Survived:")
print(f"  FamilySize vs Survived: {df_final['FamilySize'].corr(df_final['Survived']):.4f}")
print(f"  IsAlone vs Survived:   {df_final['IsAlone'].corr(df_final['Survived']):.4f}")

print("\n--- Análisis de Correlación Completado ---")

# ============================================
# TRANSFORMACIÓN LOGARÍTMICA
# ============================================

print("\n" + "="*60)
print("TRANSFORMACIÓN LOGARÍTMICA")
print("="*60)

print("\n📌 ¿CUÁNDO APLICAR TRANSFORMACIÓN LOGARÍTMICA?")
print("  • Datos con distribución muy sesgada (asimétrica)")
print("  • Valores muy dispersos (outliers extremos)")
print("  • Diferencias de escala de varios órdenes de magnitud")
print("  • Para estabilizar varianza y hacer datos más 'normales'")

# --- Análisis de la distribución de Fare ---
print("\n" + "="*60)
print("ANÁLISIS: ¿FARE NECESITA TRANSFORMACIÓN LOGARÍTMICA?")
print("="*60)

df_fare_analysis = df[df['Fare'] > 0].copy()  # Eliminar Fare = 0 para poder aplicar log
print(f"\nRegistros con Fare > 0: {len(df_fare_analysis)}")
print(f"Registros con Fare = 0: {len(df[df['Fare'] == 0])}")

# Estadísticas de Fare original
print("\n📊 Estadísticas de 'Fare' (Original):")
print(f"  Mínimo:  ${df_fare_analysis['Fare'].min():.2f}")
print(f"  Q1:      ${df_fare_analysis['Fare'].quantile(0.25):.2f}")
print(f"  Mediana: ${df_fare_analysis['Fare'].median():.2f}")
print(f"  Q3:      ${df_fare_analysis['Fare'].quantile(0.75):.2f}")
print(f"  Máximo:  ${df_fare_analysis['Fare'].max():.2f}")
print(f"  Media:   ${df_fare_analysis['Fare'].mean():.2f}")
print(f"  Desv. Std: ${df_fare_analysis['Fare'].std():.2f}")

# Calcular asimetría (skewness)
skewness_fare = df_fare_analysis['Fare'].skew()
print(f"\n  Asimetría (Skewness): {skewness_fare:.3f}")
print("    → Skewness > 1: Altamente sesgada a la derecha ✓")
print("    → ¡NECESITA TRANSFORMACIÓN LOGARÍTMICA!")

# --- Aplicar Transformación Logarítmica ---
print("\n--- Aplicando Transformación Logarítmica (Log10) ---")

# Aplicar log10 a Fare (solo valores > 0)
df_fare_analysis['Fare_Log10'] = np.log10(df_fare_analysis['Fare'])

# Estadísticas después de la transformación
print("\n📊 Estadísticas de 'Fare_Log10' (Transformado):")
print(f"  Mínimo:  {df_fare_analysis['Fare_Log10'].min():.3f}")
print(f"  Q1:      {df_fare_analysis['Fare_Log10'].quantile(0.25):.3f}")
print(f"  Mediana: {df_fare_analysis['Fare_Log10'].median():.3f}")
print(f"  Q3:      {df_fare_analysis['Fare_Log10'].quantile(0.75):.3f}")
print(f"  Máximo:  {df_fare_analysis['Fare_Log10'].max():.3f}")
print(f"  Media:   {df_fare_analysis['Fare_Log10'].mean():.3f}")
print(f"  Desv. Std: {df_fare_analysis['Fare_Log10'].std():.3f}")

skewness_fare_log = df_fare_analysis['Fare_Log10'].skew()
print(f"\n  Asimetría (Skewness): {skewness_fare_log:.3f}")
print("    → Skewness reducida significativamente ✓")
print("    → Distribución más simétrica y 'normal'")

# --- Visualización: Antes y Después ---
print("\n--- Generando Visualización Comparativa ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Histograma Fare Original
sns.histplot(df_fare_analysis['Fare'], bins=50, kde=True, ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('Distribución Original de Fare', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Tarifa ($)', fontsize=11)
axes[0, 0].set_ylabel('Frecuencia', fontsize=11)
axes[0, 0].axvline(df_fare_analysis['Fare'].mean(), color='red', linestyle='--', 
                   label=f'Media: ${df_fare_analysis["Fare"].mean():.2f}')
axes[0, 0].axvline(df_fare_analysis['Fare'].median(), color='green', linestyle='-', 
                   label=f'Mediana: ${df_fare_analysis["Fare"].median():.2f}')
axes[0, 0].legend()
axes[0, 0].grid(axis='y', alpha=0.3)

# 2. Histograma Fare Logarítmico
sns.histplot(df_fare_analysis['Fare_Log10'], bins=30, kde=True, ax=axes[0, 1], color='salmon')
axes[0, 1].set_title('Distribución Transformada (Log10)', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Log10(Tarifa)', fontsize=11)
axes[0, 1].set_ylabel('Frecuencia', fontsize=11)
axes[0, 1].axvline(df_fare_analysis['Fare_Log10'].mean(), color='red', linestyle='--', 
                   label=f'Media: {df_fare_analysis["Fare_Log10"].mean():.3f}')
axes[0, 1].axvline(df_fare_analysis['Fare_Log10'].median(), color='green', linestyle='-', 
                   label=f'Mediana: {df_fare_analysis["Fare_Log10"].median():.3f}')
axes[0, 1].legend()
axes[0, 1].grid(axis='y', alpha=0.3)

# 3. Box Plot Fare Original
axes[1, 0].boxplot(df_fare_analysis['Fare'], vert=True)
axes[1, 0].set_title('Box Plot - Fare Original', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Tarifa ($)', fontsize=11)
axes[1, 0].grid(axis='y', alpha=0.3)

# 4. Box Plot Fare Logarítmico
axes[1, 1].boxplot(df_fare_analysis['Fare_Log10'], vert=True)
axes[1, 1].set_title('Box Plot - Fare Transformado (Log10)', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Log10(Tarifa)', fontsize=11)
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.suptitle('Comparación: Transformación Logarítmica de Fare', 
             y=1.01, fontsize=16, fontweight='bold')
plt.savefig('10_transformacion_logaritmica_fare.png', dpi=300, bbox_inches='tight')
plt.show()


# --- Análisis de Age (opcional) ---
print("\n" + "="*60)
print("ANÁLISIS: ¿AGE NECESITA TRANSFORMACIÓN LOGARÍTMICA?")
print("="*60)

df_age_analysis = df[df['Age'].notna()].copy()
print(f"\nRegistros con Age válida: {len(df_age_analysis)}")

# Estadísticas de Age
print("\n📊 Estadísticas de 'Age' (Original):")
print(f"  Mínimo:  {df_age_analysis['Age'].min():.2f} años")
print(f"  Q1:      {df_age_analysis['Age'].quantile(0.25):.2f} años")
print(f"  Mediana: {df_age_analysis['Age'].median():.2f} años")
print(f"  Q3:      {df_age_analysis['Age'].quantile(0.75):.2f} años")
print(f"  Máximo:  {df_age_analysis['Age'].max():.2f} años")
print(f"  Media:   {df_age_analysis['Age'].mean():.2f} años")

skewness_age = df_age_analysis['Age'].skew()
print(f"\n  Asimetría (Skewness): {skewness_age:.3f}")
if abs(skewness_age) < 0.5:
    print("    → Skewness < 0.5: Distribución relativamente simétrica")
    print("    → NO NECESITA transformación logarítmica")
else:
    print("    → Skewness >= 0.5: Distribución moderadamente sesgada")
    print("    → Transformación logarítmica podría ser beneficiosa")


# --- Comparación con correlaciones ---
print("\n" + "="*60)
print("IMPACTO EN CORRELACIONES")
print("="*60)

# Crear DataFrame con Fare transformado
df_compare = df[['Age', 'Fare', 'Pclass', 'Survived']].dropna()
df_compare_log = df_compare.copy()
df_compare_log['Fare_Log10'] = np.log10(df_compare_log['Fare'].replace(0, 0.01))  # Evitar log(0)

print("\n--- Correlación con 'Survived' ---")
print(f"  Fare (Original):     {df_compare['Fare'].corr(df_compare['Survived']):.4f}")
print(f"  Fare_Log10:          {df_compare_log['Fare_Log10'].corr(df_compare_log['Survived']):.4f}")
print(f"\n--- Correlación con 'Pclass' ---")
print(f"  Fare (Original):     {df_compare['Fare'].corr(df_compare['Pclass']):.4f}")
print(f"  Fare_Log10:          {df_compare_log['Fare_Log10'].corr(df_compare_log['Pclass']):.4f}")


