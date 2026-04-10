# Semana 9: Aprendizaje No Supervisado - K-means y DBSCAN

Introducción a técnicas de aprendizaje no supervisado enfocadas en agrupamiento (clustering). Implementación de K-means y DBSCAN con análisis comparativo.

## Descripción General

El aprendizaje no supervisado permite descubrir estructuras ocultas en datos sin etiquetas predefinidas. Este módulo cubre técnicas fundamentales de clustering: K-means para particionar datos y DBSCAN para encontrar clusters basados en densidad.

## Contenido

### Scripts Principales

#### 1. agrupamientodatasetudea.py
Implementación de K-means en dataset sintético grande:
- Carga del dataset FIRE UdeA completo
- Preprocesamiento y escalado
- Búsqueda del número óptimo de clusters (método del codo)
- Ajuste del modelo K-means
- Generación de visualizaciones
- Cálculo de métricas de evaluación

#### 2. agrupamientodatasetudearealista.py
Análisis de K-means en versión realista:
- Mismos pasos que versión grande
- Dataset con características más realistas
- Comparación de resultados entre versiones
- Análisis de impacto de características

#### 3. preprocesamiento.py
Preparación de datos específica para clustering:
- Carga del dataset
- Manejo de valores faltantes
- Escalado de variables (StandardScaler o similar)
- Transformaciones de distribuciones
- Exportación de datos procesados

#### 4. ejAgrupamiento_kmeans_dbscan.ipynb
Notebook interactivo con ejercicios y comparativas:
- Implementación paso a paso de K-means
- Implementación de DBSCAN
- Comparación visual de resultados
- Explicación de parámetros
- Ejercicios prácticos interactivos

### Datasets

- **dataset_sintetico_FIRE_UdeA.csv**: Dataset sintético original
  - Datos generados con características específicas
  - Tamaño mayor para análisis a escala
  
- **dataset_sintetico_FIRE_UdeA_realista.csv**: Versión con realismo
  - Distribuaciones más complejas
  - Mayor variabilidad
  - Clusters con forma irregular

### Estructura de Carpetas

#### datos_preprocesados/
Datasets preparados lista para clustering:
- Variables escaladas
- Valores faltantes imputados
- Transformaciones aplicadas
- Listos para uso en algoritmos

#### graficas/
Visualizaciones del proceso de clustering:
- Gráficos del método del codo (K óptimo)
- Scatter plots de clusters en 2D
- Proyecciones con PCA o TSNE
- Siluetas de clusters
- Evolución de centroides
- Comparativa K-means vs DBSCAN

#### resultados/
Métricas y resultados numéricos:
- Inercia vs número de clusters
- Coeficiente de silhueta
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Tabla de pertenencia a clusters
- Análisis estadístico por grupo

## Algoritmos Cubiertos

### K-means

**Algoritmo**:
1. Inicializa k centroides aleatoriamente
2. Asigna cada punto al centroide más cercano
3. Recalcula centroides como promedio de puntos
4. Repite pasos 2-3 hasta convergencia

**Parámetros**:
- k: Número de clusters (debe predefinirse)
- n_init: Número de inicializaciones
- max_iter: Máximas iteraciones
- random_state: Reproducibilidad

**Ventajas**:
- Rápido y escalable
- Fácil interpretación
- Funciona bien con clusters esféricos

**Desventajas**:
- Requiere definir k a priori
- Sensible a outliers
- Asume clusters de tamaño similar

### DBSCAN

**Algoritmo**:
1. Para cada punto, busca vecinos dentro de eps
2. Si hay al menos min_samples vecinos, es core point
3. Forma clusters conectando core points
4. Puntos aislados se etiquetan como ruido

**Parámetros**:
- eps: Radio de vecindad
- min_samples: Mínimo de puntos para core
- metric: Métrica de distancia

**Ventajas**:
- No requiere definir k
- Detecta clusters de formas arbitrarias
- Maneja outliers como ruido

**Desventajas**:
- Sensible a parámetros eps y min_samples
- Lento con datasets muy grandes
- Dificultad con densidades variadas

## Métricas de Evaluación

### Métodos Internos (sin etiquetas verdaderas)

1. **Silhueta (Silhouette Score)**
   - Rango: [-1, 1]
   - Valida 1: Clusters bien separados
   - Cercano a 0: Clusters solapados
   - Negativo: Puntos mal asignados

2. **Inercia (Within-cluster sum of squares)**
   - Suma distancias intra-cluster
   - Menor = Clusters más compactos

3. **Davies-Bouldin Index**
   - Ratio de distancia inter/intra-cluster
   - Menor = Mejor separación

4. **Calinski-Harabasz Index**
   - Ratio varianza entre/dentro clusters
   - Mayor = Mejor definición

### Método del Codo
- Grafica inercia vs k
- Punto de inflexión sugiere k óptimo

## Objetivos de Aprendizaje

### Conceptuales
- Aprendizaje no supervisado vs supervisado
- Concepto de agrupamiento (clustering)
- K-means: Algoritmo y propiedades
- DBSCAN: Clustering basado en densidad
- Métrica de distancia y similaridad

### Prácticos
- Implementación de K-means
- Determinación de k óptimo
- Interpretacion de silhuetas
- Comparación de algoritmos
- Evaluación de clusters

### Analíticos
- Descubramiento de patrones en datos
- Segmentación de usuarios/items
- Análisis de comunidades
- Generación de hipótesis

## Flujo de Trabajo

1. Carga y explora el dataset
2. Preprocesa datos (escala, normaliza)
3. Para K-means: Busca k óptimo usando codo
4. Entrena K-means con k seleccionado
5. Entrena DBSCAN con parámetros apropiados
6. Compara resultados visuales y métricas
7. Interpreta clusters en contexto del dominio
8. Documenta hallazgos


