#  Pipeline de Ciclo de Vida de Datos - Machine Learning

Un pipeline completo y profesional para **análisis exploratorio, limpieza, transformación y preparación de datos** para proyectos de machine learning. 

Este proyecto implementa un flujo automatizado que toma un CSV crudo y lo transforma en datasets listos para entrenamiento, generando reportes estadísticos y visualizaciones en el proceso.

---

##  ¿Qué hace este pipeline?

El pipeline ejecuta **6 etapas principales** de forma automática:

1. **Ingesta y Perfilado** → Carga el CSV, valida integridad y genera resumen de características
2. **Análisis Exploratorio (EDA)** → Estadísticas, outliers, correlaciones, distribuciones
3. **Visualizaciones** → Histogramas, scatter plots, box plots, heatmaps, PCA, t-SNE, UMAP
4. **Transformaciones** → Feature engineering, transformación logarítmica, encoding (one-hot, label, binary)
5. **Preprocesamiento** → Normalización con StandardScaler, codificación de targets, limpieza
6. **Generación de Reportes** → JSON con análisis, gráficos PNG y CSV procesados

### Salida del Pipeline

Después de ejecutar, obtendrás:

```
reports/
├─ results/
│  ├─ eda_report.json               # Análisis estadístico completo
│  ├─ transform_report.json         # Detalle de transformaciones
│  ├─ stress_mapping.json           # Mapeo de variables categóricas
│  ├─ data_overview.json            # Perfil del dataset original
│  └─ execution_log_TIMESTAMP.txt   # Log de ejecución
│
└─ figures/
   ├─ histogramas_todas_variables.png
   ├─ scatters_habitos_vs_gpa.png
   ├─ boxplots_por_stress.png
   ├─ corr_heatmap_pearson.png
   ├─ corr_heatmap_spearman.png
   ├─ pca_2d_gpa.png
   ├─ tsne_2d_gpa.png
   └─ umap_2d_3d_gpa.html

data/processed/
├─ dataset_transformado.csv         # Datos con transformaciones básicas
├─ dataset_transformado_onehot.csv  # One-hot encoding
├─ dataset_transformado_label.csv   # Label encoding
├─ dataset_transformado_binary.csv  # Binary encoding
└─ dataset_processed.csv            # Features normalizadas + targets
```

---

##  Requisitos

- **Python 3.9+**
- Librerías listadas en `requirements.txt` (pandas, scikit-learn, matplotlib, plotly, umap-learn, etc.)

---
##  Cómo ejecutar

### 1. Preparación (primera vez)

```bash
# Clonar o descargar el proyecto
cd Pipeline_CicloDeVida

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Preparar los datos

Coloca tu archivo CSV en:
```
data/raw/tu_dataset.csv
```

El pipeline automáticamente encontrará y cargará el primer CSV en esa carpeta.

### 3. Ejecutar el pipeline

```bash
python main.py
```

Eso es todo. El pipeline hará el resto (ingesta → EDA → visualizaciones → transformaciones → preprocesamiento).

---

##  Estructura del Proyecto

```
Pipeline_CicloDeVida/
├─ data/
│  ├─ raw/                 # Coloca aquí tu CSV crudo
│  ├─ interim/             # (Opcional) datos intermedios
│  └─ processed/           # Datasets transformados (generados automáticamente)
│
├─ src/
│  ├─ config.py            # Configuración centralizada (rutas, nombres de columnas)
│  ├─ ingest.py            # Carga y validación inicial de datos
│  ├─ eda.py               # Análisis exploratorio detallado
│  ├─ visualize.py         # Generación de 10+ visualizaciones
│  ├─ preprocess.py        # Limpieza y normalización
│  ├─ transform.py         # Transformaciones y encodings
│  └─ utils.py             # Funciones auxiliares y logging
│
├─ reports/
│  ├─ figures/             # Gráficos PNG e HTML (generados automáticamente)
│  └─ results/             # Reportes JSON y logs (generados automáticamente)
│
├─ main.py                 # Punto de entrada (orquesta todo el flujo)
├─ requirements.txt        # Dependencias del proyecto
└─ README.md               # Este archivo
```

---

##  Configuración

Edita `src/config.py` para personalizar:

- **Rutas**: dónde buscar datos crudos, dónde guardar procesados
- **Nombres de columnas**: `COL_GPA`, `COL_STRESS`, `COL_HOURS` 
- **Rangos esperados**: límites válidos para validaciones
- **Umbrales**: asimetría para transformaciones logarítmicas, correlación mínima, etc.

### Ejemplo de adaptación a tu dataset:

```python
# src/config.py
COL_GPA = "tu_columna_gpa"           
COL_STRESS = "tu_columna_estres"     
COL_HOURS = {"hora_estudio", "hora_sueno", ...}  # Tus columnas de hábitos
```

---

##  Análisis que genera

El pipeline calcula y documenta:

- **Estadísticas básicas**: media, mediana, moda, desv. estándar, varianza
- **Cuartiles e IQR**: para detección de outliers
- **Percentiles y deciles**: distribución de datos
- **Análisis de outliers**: cantidad, porcentaje, impacto de remoción
- **Correlaciones**: matrices Pearson y Spearman + pares altos
- **Asimetrías**: skewness de cada variable
- **Validaciones**: rangos esperados, valores únicos, identificación de anomalías

---

##  Visualizaciones incluidas

-  **Histogramas** (todas las variables con media/mediana)
-  **Scatter plots** (relaciones bivariadas)
-  **Box plots** (distribuciones y outliers)
-  **Heatmaps** (correlaciones Pearson y Spearman)
-  **Comparación outliers** (antes/después de remover)
- **PCA 2D** (reducción de dimensionalidad)
-  **t-SNE 2D** (separación no-lineal)
-  **UMAP 2D/3D** (exploración interactiva en HTML)

---

##  Stack tecnológico

| Componente | Librería |
|-----------|----------|
| Procesamiento de datos | pandas, numpy |
| Estadísticas | scipy |
| Machine Learning | scikit-learn |
| Visualización | matplotlib, plotly |
| Reducción dimensional | PCA, t-SNE, UMAP |
| Encoding | category_encoders |

---


---

##  Notas

- Todos los outputs se guardan automáticamente con timestamps
- El log de ejecución captura todas las operaciones realizadas
- Maneja datasets con celdas vacías (NaN) de forma robusta
- Compatible con Windows, macOS y Linux

---

##  Autor

Luis Angel Nerio | Aprendizaje Automático

---

**¡Listo! Tu pipeline está configurado y listo para analizar datos.** 🎉
