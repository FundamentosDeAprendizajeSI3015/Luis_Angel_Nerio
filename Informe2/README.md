# Informe 2: Análisis Integral de Estrés Estudiantil

## Descripción General

Este project implementa un **pipeline completo de Machine Learning** para analizar y predecir el nivel de estrés de estudiantes basándose en factores de su estilo de vida. El análisis combina técnicas de **análisis exploratorio de datos (EDA)**, **clustering no supervisado**, **preprocesamiento avanzado** y **modelado supervisado**.

### Dataset Principal
- **Dataset**: `student_lifestyle_dataset.csv`
- **Registros**: 2,000 estudiantes
- **Variables**: 8 features (1 ID + 7 características de estilo de vida)
- **Target**: Stress_Level (3 categorías: Low, Medium, High)

---

## Características del Dataset

### Variables de Entrada
1. **Study_Hours_Per_Day**: Horas de estudio diarias (5-10 horas)
2. **Extracurricular_Hours_Per_Day**: Horas en actividades extracurriculares (0-4 horas)
3. **Sleep_Hours_Per_Day**: Horas de sueño diarias (5-10 horas)
4. **Social_Hours_Per_Day**: Horas sociales diarias (0-6 horas)
5. **Physical_Activity_Hours_Per_Day**: Horas de actividad física (0-13 horas)
6. **GPA**: Promedio de calificaciones (2.17-3.98)

### Estadísticas Clave
- **Correlación más fuerte**: Study_Hours_Per_Day ↔ GPA (r=0.693)
- **Valores faltantes**: 0 (dataset limpio)
- **Outliers detectados**: 8 registros
- **Balance de datos**: Bien distribuido

---

## Pipeline de Ejecución

```
1. PREPROCESAMIENTO (preprocessing.py)
   ├─ Carga y limpieza de datos
   ├─ Codificación de variables categóricas
   ├─ Escalado standardizado (StandardScaler)
   └─ Generación de 2 datasets: clustering y supervisado

2. ANÁLISIS EXPLORATORIO (eda.py)
   ├─ Estadísticas descriptivas
   ├─ Detección de outliers (IQR)
   ├─ Visualizaciones: histogramas, boxplots, dispersiones
   ├─ Matriz de correlación
   └─ Conclusiones automatizadas

3. CLUSTERING NO SUPERVISADO (clustering.py)
   ├─ Algoritmos: KMeans, DBSCAN, Fuzzy C-Means, Subtractive Clustering
   ├─ Evaluación con métricas: Silhouette, Davies-Bouldin, Calinski-Harabasz
   ├─ Visualizaciones: PCA (2D/3D), UMAP (2D/3D)
   └─ Selección automática del mejor modelo (KMeans)

4. RELABELING DE ETIQUETAS (relabeling.py)
   ├─ Mapeo de clusters a etiquetas de estrés
   ├─ Corrección de inconsistencias
   └─ Generación de dataset mejorado

5. MODELADO SUPERVISADO (models.py)
   ├─ Clasificación: LogisticRegression, DecisionTreeClassifier
   ├─ Regresión: LinearRegression
   ├─ Validación cruzada estratificada
   ├─ Comparación antes/después relabeling
   └─ Análisis de importancia de features
```

---

## Estructura de Carpetas

```
Informe2/
├── README.md                      # Este archivo
├── preprocessing.py               # Preprocesamiento de datos
├── eda.py                         # Análisis exploratorio
├── clustering.py                  # Algoritmos de clustering
├── models.py                      # Modelos supervisados
├── relabeling.py                  # Corrección de etiquetas
├── student_lifestyle_dataset.csv  # Dataset original
├── data_clustering.csv            # Datos procesados para clustering
├── data_supervised.csv            # Datos procesados para modelos
├── mapeo_stress.json              # Mapeo cluster → estrés
│
├── eda/                           # Resultados de EDA
│   ├── conclusiones_eda.txt
│   ├── estadisticas_descriptivas.csv
│   └── correlacion_matriz.csv
│
├── clustering/                    # Resultados de clustering
│   ├── best_model.txt
│   ├── metrics_comparison.csv
│   └── clustering_output.txt
│
├── models/                        # Resultados de modelos
│   ├── conclusions.txt
│   ├── metrics_comparison.csv
│   └── models_output.txt
│
└── relabeling/                    # Resultados de relabeling
    ├── data_relabeling_final.csv
    ├── cambios_por_clase.csv
    ├── cluster_label_mapping.csv
    └── relabeling_output.txt
```

---

## Instrucciones de Uso

### 1. Instalación de Dependencias

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy scikit-fuzzy umap-learn joblib
```

### 2. Ejecución del Pipeline Completo

```bash
# Paso 1: Preprocesamiento
python preprocessing.py

# Paso 2: Análisis Exploratorio
python eda.py

# Paso 3: Clustering
python clustering.py

# Paso 4: Relabeling
python relabeling.py

# Paso 5: Modelos Supervisados
python models.py
```

### 3. Ejecución Individual

Cada módulo puede ejecutarse de forma independiente siempre que los datos de entrada estén disponibles.

---

## Resultados Principales

### Clustering
- **Mejor modelo**: KMeans
- **Métrica Silhouette**: 0.2042
- **Balance de clusters**: 0.9915
- **Ruido detectado**: 0.0000

### Modelos Supervisados
- Los modelos se evalúan antes y después del relabeling
- Se comparan clasificación, regresión y validación cruzada
- Se generan matrices de confusión e importancia de features

### Análisis de Correlación
- Correlación fuerte entre **Study_Hours** y **GPA** (0.693)
- Baja correlación entre Sleep_Hours y Extracurricular_Hours
- No hay multicolinealidad significativa

---

## Detalles de los Módulos

### **preprocessing.py**
- Carga de dataset original
- Codificación de variables categóricas (Stress_Level)
- Escalado standardizado para clustering
- Preservación de valores originales para modelos supervisados
- Guardado de scaler para reproducibilidad

### **eda.py**
- Estadísticas descriptivas completas
- Detección de outliers mediante IQR
- 15+ visualizaciones (histogramas, boxplots, dispersiones, heatmap)
- Matriz de correlación de Pearson
- Conclusiones automatizadas sobre datos

### **clustering.py**
- **Algoritmos implementados**:
  - KMeans (k=2 a 10)
  - DBSCAN (eps, min_samples variados)
  - Fuzzy C-Means (c=2 a 10)
  - Subtractive Clustering de Chiu
  
- **Métricas de evaluación**:
  - Silhouette Score (más alto es mejor)
  - Davies-Bouldin Index (más bajo es mejor)
  - Calinski-Harabasz Score (más alto es mejor)
  
- **Visualizaciones**:
  - PCA 2D y 3D
  - UMAP 2D y 3D
  - Comparación de modelos

### **relabeling.py**
- Mapeo automático de clusters a etiquetas de estrés
- Detección y corrección de inconsistencias
- Generación de reportes de cambios
- Validación de mapeo

### **models.py**
- **Clasificación**:
  - LogisticRegression con validación cruzada
  - DecisionTreeClassifier con análisis de importancia
  
- **Regresión**:
  - LinearRegression para predicción continua
  
- **Evaluación**:
  - Accuracy, Precision, Recall, F1-Score
  - Matrices de confusión
  - Validación cruzada estratificada (5-fold)
  
- **Comparativa**:
  - Antes vs después de relabeling
  - Impacto de corrección de etiquetas

---

## Interpretación de Resultados

### ¿Qué nos dice el análisis?

1. **EDA**: Identifica patrones y relaciones entre variables de estilo de vida
2. **Clustering**: Agrupa estudiantes por similitud en comportamiento
3. **Relabeling**: Mejora la calidad de las etiquetas de estrés usando patrones encontrados
4. **Modelos**: Predicen estrés futuro con base en comportamiento actual

### Insights Clave

- El **GPA correlaciona fuertemente con horas de estudio**, sugiriendo disciplina académica
- Los **clusters identificados representan diferentes arquetipos de estudiantes**
- La **corrección de etiquetas mejora la precisión de los modelos**
- Hay **pocos outliers**, indicando datos de buena calidad

---

## Archivos de Salida

### Datos
- `data_clustering.csv`: Datos escalados para clustering
- `data_supervised.csv`: Datos preparados para modelos supervisa
- `data_with_clusters.csv`: Datos originales con asignación de clusters

### Reportes
- `eda/conclusiones_eda.txt`: Resumen del análisis exploratorio
- `clustering/best_model.txt`: Información del mejor modelo de clustering
- `relabeling/relabeling_output.txt`: Log de cambios en etiquetas
- `models/conclusions.txt`: Conclusiones de modelos supervisados

### Datos Analíticos
- `eda/estadisticas_descriptivas.csv`: Estadísticas por variable
- `eda/correlacion_matriz.csv`: Matriz de correlación
- `clustering/metrics_comparison.csv`: Comparativa de métricas de clustering
- `models/metrics_comparison.csv`: Comparativa de métricas de modelos
- `relabeling/cambios_por_clase.csv`: Cambios de etiquetas por clase
- `mapeo_stress.json`: Mapeo de clusters a niveles de estrés

---

## Requisitos Técnicos

- **Python**: >= 3.8
- **Sistemas operativos**: Windows, macOS, Linux
- **Memoria**: ~500 MB para dataset completo
- **Tiempo de ejecución**: ~5-10 minutos (pipeline completo)

### Dependencias Principales
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
scikit-fuzzy>=0.4.2
umap-learn>=0.5.3
joblib>=1.1.0
```

---

## Autor

**Luis Ángel Nerio**

---

## Notas Importantes

1. **Reproducibilidad**: Todos los modelos usan `SEED=42` para resultados consistentes
2. **Preprocesamiento**: El scaler se guarda para aplicar a datos nuevos
3. **Validación**: Se usa validación cruzada estratificada para datos desbalanceados
4. **Interpretación**: Las métricas de clustering pueden variar según el algoritmo y parámetros

---



