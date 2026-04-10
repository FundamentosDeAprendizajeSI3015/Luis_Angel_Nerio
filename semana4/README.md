# Semana 4: Análisis Integral del Dataset Titanic

Análisis detallado y preparación de datos del dataset Titanic. Esta semana profundiza en técnicas de limpieza, transformación y visualización avanzada.

## Descripción General

El dataset Titanic es un conjunto clásico en machine learning que contiene información sobre pasajeros del barco Titanic. Este módulo cubre el ciclo completo: desde exploración inicial hasta transformación de variables para preparar los datos para modelado posterior.

## Contenido

### Scripts de Procesamiento
- **actividadpart1.py**: Carga inicial, exploración y análisis descriptivo
- **actividadpart2.py**: Transformaciones avanzadas y preparación para modelado

### Datasets
- **Titanic-Dataset.csv**: Dataset original sin procesar
- **Titanic_Transformado.csv**: Dataset después de transformaciones y limpieza

### Visualizaciones Generadas
Análisis exploratorio visual sistematizado:
- **Boxplots (boxplot_edad_outliers.png)**: Detección y análisis de outliers en edad
- **Comparación de outliers (comparacion_outliers_eliminados.png)**: Impacto de su remoción
- **Histogramas (histograma_salarios.png)**: Distribución univariada de fares
- **Gráficos de dispersión (dispersion_*.png)**: Relaciones bivariadas entre variables
- **Matrices de correlación (matriz_correlacion.png, heatmap_correlaciones.png)**: Asociaciones entre variables
- **Transformaciones logarítmicas (transformacion_logaritmica_fare.png)**: Normalización de distribuciones

## Fases del Análisis

### Fase 1: Exploración Inicial
- Carga del dataset
- Inspección de dimensiones y tipos de datos
- Estadísticas descriptivas
- Identifición de valores faltantes

### Fase 2: Limpieza de Datos
- Manejo de valores faltantes (imputación, eliminación)
- Detección y tratamiento de outliers
- Análisis de distribuciones

### Fase 3: Transformación de Variables
- Transformaciones logarítmicas para normalizar distribuciones
- Codificación de variables categóricas
- Escalado de variables numéricas
- Ingeniería de características

### Fase 4: Análisis de Correlaciones
- Cálculo de matrices de correlación
- Identificación de multicolinealidad
- Análisis de dependencias entre variables

## Objetivos de Aprendizaje

- Detección and tratamiento robusto de datos faltantes y outliers
- Transformación de variables para mejorar distribuciones
- Análisis de correlaciones y multicolinealidad
- Visualización avanzada para comunicar hallazgos
- Preparación de datos de alta calidad para modelado
- Documentación reproducible de procesos

## Conceptos Clave

- Valores faltantes: Mecanismos de ausencia (MCAR, MAR, MNAR)
- Outliers: Detección (IQR, Z-score) y tratamiento
- Transformaciones: Log, raíz cuadrada, estandarización
- Correlación de Pearson y Spearman
- Matriz de correlación y heatmaps

## Archivo Adicional
- **541.g**: Archivo de configuración o complementario

## Cómo Utilizar Este Módulo

1. Carga el dataset original (Titanic-Dataset.csv)
2. Ejecuta actividadpart1.py para exploración inicial
3. Analiza las visualizaciones generadas
4. Ejecuta actividadpart2.py para transformaciones
5. Compara el dataset original con el transformado
6. Utiliza el dataset transformado como entrada para modelado
