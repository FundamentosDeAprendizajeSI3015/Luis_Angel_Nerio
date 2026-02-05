import pandas as pd
import matplotlib.pyplot as plt
#carga de datos
df = pd.read_csv('opcion1/Titanic-Dataset.csv')

#inpseccion rapida de los datos

# Ver las primeras 5 filas
df.head()
# Ver las ultimas 5 filas
df.tail()
# Dimensiones (filas, columnas)
df.shape
# Tipos de datos y memoria usada
df.info()
# Ver los tipos de datos de cada columna.
df.dtypes
# Estadisticas descriptivas (media, min, max, percentiles)
df.describe()
# Conteo de valores unicos por columna
df.nunique()
# Conteo de valores unicos en una columna especifica
# Reemplaza "columna" por una columna real del dataset, por ejemplo "Sex".
df["Sex"].nunique()
# Elementos unicos por columna
df["Sex"].unique()

#Depuracoon de datos segun valores nulos 

# Detectar nulos por columna (suma total)
df.isnull().sum()
# Total de valores nulos en todo el DataFrame
df.isnull().sum().sum()
# Porcentaje de NaN por columna
df.isnull().sum() / len(df) * 100
# Eliminar filas que tengan AL MENOS un nulo
df_limpio = df.dropna()
# Eliminar columnas que tengan AL MENOS un nulo
df_limpio = df.dropna(axis=1)
# Deja filas que tienen N cantidad de valores no nulos en adelante.

df_limpio = df.dropna(thresh=5)
# Eliminar fila donde TODAS los valores sean nulos.
df_limpio = df.dropna(how='all')
# Rellenar nulos con un valor constante.
df.fillna(0, inplace=True)
# Rellenar nulos con un valor constante en una columna especifica
# Usa columnas reales del dataset Titanic: "Age" (numérica) y "Embarked" (categórica)
df.fillna({"Age": 0}, inplace=True)
# Imputacion inteligente: usar la media
df.fillna({"Age": df["Age"].mean()}, inplace=True)
# Imputacion inteligente: usar la mediana
df.fillna({"Age": df["Age"].median()}, inplace=True)
# Imputacion inteligente: usar la moda
df.fillna({"Age": df["Age"].mode()[0]}, inplace=True)
# Rellena con el valor anterior.
df["Age"] = df["Age"].ffill()
# Rellena con el valor posterior.
df["Age"] = df["Age"].bfill()
# Rellena con interpolacion.
df["Age"] = df["Age"].interpolate()
# Rellenar NaN en la columna categórica con "Desconocido"
df["Embarked"].fillna('Desconocido')



# #manipiulacion de filas y columnas

# # Eliminar columnas especificas
# >>> df.drop([’ID’, ’Direccion’], axis=1, inplace=True)
# # Eliminar filas por indice
# >>> df.drop([0, 1, 2], axis=0)
# # Renombrar columnas
# >>> df.rename(columns={’old_name’: ’new_name’}, inplace=True)
# # Renombrar todas las columnas.
# >>> df.columns = [’nueva_col1’, ’nueva_col2’, ...]
# # Reordenar columnas.
# >>> df = df[[’col_ordenada1’, ’col_ordenada2’, ...]]
# # Crear una nueva columna basada en otras
# >>> df[’total’] = df[’precio’] * df[’cantidad’]


#limpieza de texto y duplicados 
# Devuelve una Serie booleana indicando filas duplicadas (la primera ocurrencia no se marca).
df.duplicated()
# Busca duplicados solo en un subconjunto de columnas.
df.duplicated(subset=['Name', 'Ticket']).sum()
# Convertir a minusculas y quitar espacios en blanco
df['Embarked'] = df['Embarked'].astype('string').str.lower().str.strip()
# Reemplazar valores de texto
df['Embarked'] = df['Embarked'].replace('s ', 's')
# Cuenta el numero total de filas duplicadas.
df.duplicated().sum()
# Eliminar filas duplicadas
df.drop_duplicates(inplace=True)
# Elimina duplicados basandose en un subconjunto de columnas, manteniendo la ultima ocurrencia.
df = df.drop_duplicates(subset=['Name', 'Ticket'], keep='last')


#consistencia y validacion de datos
# Conservamos las filas que cumplan con que la edad sea mayor a 17 y la tarifa menor a 5000.
df = df[(df['Age'] > 17) & (df['Fare'] < 5000)]
# La tarifa debe ser siempre mayor o igual que 0.
df = df[df['Fare'] >= 0]
# Eliminar columnas que tienen un unico valor en todas las filas, ya que no aportan informacion.
df.loc[:, df.nunique() > 1]



#transformacion de tipo  + filtrado 

# conversiones y filtros (Titanic)
# Cambiar tipo de dato (ej: float a int)
df['Age'] = df['Age'].round().astype('Int64')
# Convertir columna a formato Fecha (extraemos fecha del Ticket si existiera; aqui usamos el índice como ejemplo)
df['Fecha_Registro'] = pd.to_datetime(df.index, unit='D', origin='1899-12-31')
# Filtrar datos bajo una condicion
df_filtrado = df[df['Age'] > 18]
# Filtrado complejo (Multiple condicion)
df_vip = df[(df['Fare'] > 100) & (df['Embarked'] == 'C')]

#agregacion y agrupamiento 

# AGRUPAR POR CATEGORIA Y HACER CALCULO (Titanic)
# Promedio de tarifa por clase y sexo
promedio_fare = df.groupby(['Pclass', 'Sex'])['Fare'].mean()

# CREAR TABLA PIVOTE Y HACER CALCULO (similar a Excel)
# Tabla pivote: promedio de tarifa por clase y puerto de embarque
pivote_fare = df.pivot_table(
	index='Pclass',
	columns='Embarked',
	values='Fare',
	aggfunc='mean'
)


#transformacion one hot encoding

# Convertimos columnas categóricas en variables binarias
df_encoded = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=False)

# visualizacion de datos finales con matplotlib
plt.figure(figsize=(12, 8))

# 1) Distribucion de edades
plt.subplot(2, 2, 1)
df_encoded['Age'].dropna().hist(bins=30, color='skyblue', edgecolor='black')
plt.title('Distribucion de Edad')
plt.xlabel('Edad')
plt.ylabel('Frecuencia')

# 2) Supervivencia
plt.subplot(2, 2, 2)
df_encoded['Survived'].value_counts().plot(kind='bar', color=['salmon', 'seagreen'])
plt.title('Supervivencia')
plt.xlabel('Sobrevivio (0/1)')
plt.ylabel('Cantidad')

# 3) Tarifa por clase
plt.subplot(2, 2, 3)
(df_encoded.groupby('Pclass')['Fare'].mean()).plot(kind='bar', color='orange')
plt.title('Tarifa Promedio por Clase')
plt.xlabel('Clase')
plt.ylabel('Tarifa Promedio')

# 4) Correlacion basica
plt.subplot(2, 2, 4)
correlacion = df_encoded[['Age', 'Fare', 'Survived', 'Pclass']].corr()
plt.imshow(correlacion, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar()
plt.xticks(range(len(correlacion.columns)), correlacion.columns, rotation=45)
plt.yticks(range(len(correlacion.index)), correlacion.index)
plt.title('Correlacion (Variables Seleccionadas)')

plt.tight_layout()
plt.show()




#graficar con umap 

import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Patch

try:
    import umap
except ImportError:
    print("UMAP no está instalado. Instálalo con: pip install umap-learn")
    umap = None

# Preparar datos para UMAP (solo columnas numéricas)
numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
X = df_encoded[numeric_cols].dropna()

# Normalizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

if umap is not None:
    # UMAP 2D
    reducer_2d = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding_2d = reducer_2d.fit_transform(X_scaled)
    
    # UMAP 3D
    reducer_3d = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding_3d = reducer_3d.fit_transform(X_scaled)
    
    # Visualización UMAP 2D
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # UMAP 2D coloreado por Supervivencia
    scatter1 = axes[0].scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                               c=X['Survived'], cmap='RdYlGn', alpha=0.6, s=50)
    axes[0].set_xlabel('UMAP 1')
    axes[0].set_ylabel('UMAP 2')
    axes[0].set_title('UMAP 2D - Coloreado por Supervivencia')
    plt.colorbar(scatter1, ax=axes[0], label='Supervivencia (0/1)')
    
    # UMAP 2D coloreado por Clase
    scatter2 = axes[1].scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                               c=X['Pclass'], cmap='viridis', alpha=0.6, s=50)
    axes[1].set_xlabel('UMAP 1')
    axes[1].set_ylabel('UMAP 2')
    axes[1].set_title('UMAP 2D - Coloreado por Clase')
    plt.colorbar(scatter2, ax=axes[1], label='Clase')
    
    plt.tight_layout()
    plt.show()
    
    # Visualización UMAP 3D
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(16, 6))
    
    # UMAP 3D coloreado por Supervivencia
    ax1 = fig.add_subplot(121, projection='3d')
    scatter3d_1 = ax1.scatter(embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2],
                              c=X['Survived'], cmap='RdYlGn', alpha=0.6, s=30)
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    ax1.set_zlabel('UMAP 3')
    ax1.set_title('UMAP 3D - Coloreado por Supervivencia')
    plt.colorbar(scatter3d_1, ax=ax1, label='Supervivencia (0/1)')
    
    # UMAP 3D coloreado por Clase
    ax2 = fig.add_subplot(122, projection='3d')
    scatter3d_2 = ax2.scatter(embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2],
                              c=X['Pclass'], cmap='viridis', alpha=0.6, s=30)
    ax2.set_xlabel('UMAP 1')
    ax2.set_ylabel('UMAP 2')
    ax2.set_zlabel('UMAP 3')
    ax2.set_title('UMAP 3D - Coloreado por Clase')
    plt.colorbar(scatter3d_2, ax=ax2, label='Clase')
    
    plt.tight_layout()
    plt.show()
else:
    print("No se puede visualizar UMAP sin la librería instalada.")




