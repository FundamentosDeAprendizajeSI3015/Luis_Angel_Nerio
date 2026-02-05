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

