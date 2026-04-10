# Semana 8: Pipeline Completo de Machine Learning

Proyecto integral que implementa un flujo completo de machine learning desde exploración inicial hasta construcción y evaluación de modelos predictivos.

## Descripción General

Este proyecto demuestra cómo integrar todas las fases de un proyecto de machine learning en un flujo automatizado y reproducible. Utiliza un dataset sintético realista con características del contexto UdeA.

## Contenido

### Scripts Principales

#### 1. EDA.py - Análisis Exploratorio de Datos
Realiza exploración completa del dataset:
- Carga del dataset
- Estadísticas descriptivas (media, mediana, desviación estándar)
- Análisis de valores faltantes
- Identificación de outliers
- Distribuciones univariadas
- Correlaciones y análisis multivariado
- Generación de visualizaciones informativas

Salida: Figuras y archivo de resultados con hallazgos clave

#### 2. Preprocesamineto.py - Transformación de Datos
Preparación de datos para modelado:
- Imputación de valores faltantes
- Tratamiento de outliers
- Escalado y normalización de variables
- Codificación de variables categóricas
- Ingeniería de characteristics (si aplica)
- División entrenamiento-prueba
- Balanceo de clases (si es problema desbalanceado)

Salida: Dataset limpio y preparado para modelado

#### 3. Modelo.py - Construcción de Modelos
Entrenamiento y evaluación de modelos:
- Selección de algoritmos apropiados
- Ajuste de hiperparámetros
- Entrenamiento de múltiples modelos
- Validación cruzada
- Comparación de rendimiento
- Evaluación en datos de prueba
- Análisis de importancia de features

Salida: Modelos entrenados y métricas de evaluación

### Dataset

- **dataset_sintetico_FIRE_UdeA_realista.csv**: Dataset sintético con características realistas
  - Representa un dominio similar al contexto UdeA
  - Contiene variables cuantitativas y categóricas
  - Incluye desafíos prácticos (valores faltantes, outliers, desbalance)

### Estructura de Carpetas de Salida

#### eda_figures/
Visualizaciones del análisis exploratorio:
- Histogramas y distribuciones
- Boxplots y análisis de outliers
- Matrices de correlación y heatmaps
- Gráficos de dispersión
- Análisis por grupos

#### eda_results/
Resultados numéricos del EDA:
- Estadísticas descriptivas
- Matriz de correlación
- Resumen de valores faltantes
- Análisis de outliers

#### preprocess_outputs/
Artefactos del preprocesamiento:
- Dataset transformado
- Escalado y transformaciones aplicadas
- Codificación de variables categóricas
- Conjuntos entrenamiento-prueba

#### model_outputs/
Resultados del modelado:
- Modelos entrenados (serializados)
- Métricas de evaluación comparativas
- matriz de confusión (para clasificación)
- Curvas de aprendizaje
- Importancia de features
- Predicciones en datos de prueba

## Flujo de Trabajo Automático

1. **Fase 1**: EDA.py explora y visualiza los datos
2. **Fase 2**: Preprocesamineto.py limpia y transforma
3. **Fase 3**: Modelo.py entrena y evalúa modelos
4. **Repetición**: Ciclo iterativo para mejora

## Objetivos de Aprendizaje

### Conceptuales
- Comprensión de flujo completo de ML
- Integración de fases independientes
- Reproducibilidad en proyectos de datos
- Buenas prácticas en organización

### Prácticos
- Automatización de procesos repetitivos
- Generación eficiente de reportes
- Comparación sistemática de enfoques
- Documentación de decisiones

### Técnicos
- Manejo de múltiples archivos y módulos
- Importación entre scripts
- Gestión de archivos de entrada/salida
- Control de versiones

## Conceptos Clave

- Reproducibilidad: Mismo resultado con mismos datos
- Modularidad: Fases independientes pero integradas
- Automatización: Minimiza intervención manual
- Documentación: Cada paso es explicable y auditable
- Iteración: Mejora continua basada en resultados

## Mejores Prácticas Implementadas

1. **Separación de Preocupaciones**: Cada script una fase
2. **Control de Parámetros**: Fácil ajuste sin modificar código
3. **Registro de Resultados**: Todos los saltos guardados
4. **Visualización**: Múltiples gráficos para interpretación
5. **Reproducibilidad**: Seeds fijas para resultados consistentes

## Cómo Utilizar Este Proyecto

1. Coloca tu dataset en la carpeta raíz (o ajusta rutas)
2. Ejecuta EDA.py para exploración inicial
3. Revisa las visualizaciones y resultados en carpetas eda_*
4. Ejecuta Preprocesamineto.py para preparar datos
5. Ejecuta Modelo.py para entrenar y evaluar modelos
6. Analiza los resultados en carpetas model_outputs
7. Itera: Ajusta parámetros y repite según necesario

## Extensiones Posibles

- Agregar más algoritmos de machine learning
- Implementar ensemble methods (Random Forest, Gradient Boosting)
- Agregar técnicas de selección de features
- Implementar hyperparameter tuning automático
- Generar reportes automáticos en HTML
- Agregar validación cruzada anidada
