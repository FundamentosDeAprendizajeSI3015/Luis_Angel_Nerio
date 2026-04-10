# Semana 10: Técnicas Avanzadas de Clustering

Técnicas sofisticadas de reducción de dimensionalidad y clustering basado en densidad. Incluye análisis de sensibilidad de parámetros y comparativas entre métodos.

## Descripción General

Este módulo avanza más allá de K-means tradicional, introduciendo UMAP para visualización y reducción de dimensionalidad, y explorando en profundidad la sensibilidad de DBSCAN. Se incluyen análisis sistemáticos de parámetros y escalabilidad a datos grandes.

## Contenido

### Scripts Principales

#### Pipeline Base
- **ClusterUdea.py**: Pipeline de clustering completo (dataset original)
  - Carga y preprocesamiento
  - Múltiples técnicas de clustering
  - Generación de métricas
  - Comparativas visuales

- **clusterUdeaRealista.py**: Pipeline con dataset realista
  - Mismos pasos con datos más complejos
  - Análisis de robustez
  - Resultados en contexto realista

#### Pipelines Especializados

- **clustering_pipeline_umap.py**: Implementación de UMAP
  - Reducción a 2D/3D para visualización
  - Parámetros optimizados
  - Proyecciones interpretables
  - Preservación de estructura local y global

- **clustering_pipeline_umap_stability.py**: Análisis de estabilidad de UMAP
  - Múltiples ejecuciones con aleatorización
  - Variabilidad en proyecciones
  - Robustez de patrones descubiertos
  - Reproducibilidad under randomness

#### Análisis de Sensibilidad

- **dbscan_sensibilidad_grande.py**: Análisis de parámetros en dataset grande
  - Variación sistemática de eps
  - Variación de min_samples
  - Generación de matriz de resultados
  - Identificación de rangos óptimos

- **dbscan_sensibilidad_realista.py**: Análisis en dataset realista
  - Parámetros específicos para datos complejos
  - Visualización de sensitivity
  - Recomendaciones basadas en análisis

#### Análisis a Escala

- **semana10_dataset_grande.py**: Evaluación con datos de mayor escala
  - Rendimiento computacional
  - Escalabilidad de algoritmos
  - Trade-offs tiempo vs calidad
  - Recomendaciones para producción

### Datasets

- **dataset_sintetico_FIRE_UdeA.csv**: Dataset original y educativo
- **dataset_sintetico_FIRE_UdeA_realista.csv**: Datos con distribuciones complejas

### Estructura de Carpetas

#### datos_preprocesados/
Datasets preparados:
- Variables con escalado aplicado
- Transformaciones para mejorar clustering
- Múltiples versiones (con/sin outliers)

#### figuras_dbscan_sensibilidad/
Visualizaciones de análisis de parámetros:
- Gráficos de eps vs min_samples
- Heatmaps de número de clusters
- Ratio de puntos ruido
- Tamaño menor cluster vs parámetros
- Características de clusters por parámetro

#### figuras_validacion / figuras_validacion_grande
Comparativas entre técnicas:
- K-means vs DBSCAN vs UMAP
- Proyecciones UMAP de clusters
- Dendrogramas de clustering jerárquico
- Métricas visuales comparativas
- Variabilidad en UMAP (stability analysis)

#### resultados_validacion / resultados_validacion_grande
Métricas y análisis numéricos:
- Tablas de silhueta por algoritmo
- Tiempos de ejecución
- Número de clusters encontrados
- Puntos clasificados como ruido
- Davies-Bouldin Index comparativo
- Recomendaciones de parámetros

## Técnicas Avanzadas

### UMAP (Uniform Manifold Approximation and Projection)

**Ventajas**:
- Mantiene estructura local y global
- Escalable a datasets grandes
- Interpretable en 2D/3D
- Rápido comparado con t-SNE

**Parámetros clave**:
- n_neighbors: Locales vs globales (default 15)
- min_dist: Separación mínima (default 0.1)
- metric: Distancia (euclidean, cosine, etc)

**Aplicaciones**:
- Visualización de datos multidimensionales
- Entrada a clustering
- Exploración interactiva de datos

### DBSCAN Avanzado

**Análisis de sensibilidad**:
- Eps: Define radio de vecindad
  - Pequeño: Muchos clusters, muchos ruidos
  - Grande: Pocos clusters grandes
  
- min_samples: Umbral de densidad
  - Bajo: Clusters frágiles
  - Alto: Muchos puntos ruido

**Estrategias de selección**:
- K-distance graph para eps
- Análisis de densidad local
- Validación cruzada de parámetros

## Análisis Comparativo

### Matriz de Decisión

| Aspecto | K-means | DBSCAN | UMAP+Clustering |
|---------|---------|--------|-----------------|
| K predefinido | Requerido | No necesario | No |
| Formas | Esféricas | Arbitrarias | Arbitrarias |
| Outliers | Sensible | Robusto | Depende |
| Escala | Excelente | Buena | Buena |
| Interpretación | Fácil | Media | Media |
| Visualización | Limitada | Limitada | Excelente |

## Objetivos de Aprendizaje

### Conceptuales
- Principios de reducción de dimensionalidad
- UMAP y teoría manifold
- Análisis de sensibilidad de parámetros
- Escalabilidad y rendimiento
- Comparativas sistemáticas entre métodos

### Prácticos
- Implementación de UMAP
- Análisis robusto de parámetros en DBSCAN
- Benchmarking de algoritmos
- Visualización avanzada
- Generación de reportes comparativos

### Aplicados
- Selección de técnica según contexto
- Tuning de parámetros data-driven
- Evaluación de stability
- Recomendaciones reproducibles

## Flujo de Trabajo

1. Carga datos y preprocesa (ejecutar pipeline base)
2. Aplica UMAP para visualización (clustering_pipeline_umap.py)
3. Analiza estabilidad de UMAP (clustering_pipeline_umap_stability.py)
4. Realiza análisis de sensibilidad DBSCAN
5. Compara resultados entre técnicas
6. Evalúa escalabilidad con datos grandes
7. Documenta recomendaciones de parámetros
8. Selecciona enfoque según métricas

## Cómo Utilizar Este Módulo

1. **Exploración inicial**:
   - Ejecuta ClusterUdea.py o clusterUdeaRealista.py
   - Revisa gráficas en figuras_validacion/

2. **Análisis de UMAP**:
   - Ejecuta clustering_pipeline_umap.py
   - Examina proyecciones 2D
   - Ejecuta análisis de estabilidad

3. **Sensibilidad de DBSCAN**:
   - Ejecuta dbscan_sensibilidad_*.py
   - Interpreta heatmaps de parámetros
   - Identifica rangos recomendados

4. **Escalabilidad**:
   - Ejecuta semana10_dataset_grande.py
   - Analiza rendimiento
   - Anota tiempos y limitaciones

5. **Síntesis**:
   - Compara todos los resultados
   - Redacta recomendaciones
   - Selecciona enfoque final


