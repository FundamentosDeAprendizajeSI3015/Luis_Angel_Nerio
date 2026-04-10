# Semana 5: Regresión Lineal y Logística

Introducción profunda a modelos de regresión lineal y logística. Fundamentos del aprendizaje supervisado aplicados a problemas de predicción continua y clasificación binaria.

## Descripción General

Este módulo cubre dos pilares fundamentales del machine learning supervisado: la regresión lineal para predecir variables continuas y la regresión logística para clasificación binaria. Se incluye tanto fundamentación teórica como implementación práctica.

## Contenido

### Carpetas Principales

- **regresion_lineal/**: Modelos, ejercicios y análisis de regresión lineal
  - Implementación de algoritmo
  - Ejercicios progresivos
  - Datasets para práctica
  - Evaluación de modelos
  
- **regresion_logistica/**: Modelos, ejercicios y análisis de regresión logística
  - Regresión logística binaria
  - Interpretación de probabilidades
  - Matrices de confusión
  - Curves ROC

### Recursos Teóricos

- **semana5_regresionLineal_logistica.pdf**: Fundamentos teóricos detallados
- **semana5_Ejercicio práctico.pdf**: Descripción de ejercicios y casos de estudio

## Componentes de Cada Modelo

### Regresión Lineal

**Teoría**:
- Formulación matemática: y = β₀ + β₁x₁ + ... + βₙxₙ + ε
- Método de mínimos cuadrados ordinarios (OLS)
- Supuestos del modelo lineal

**Práctica**:
- Ajuste del modelo a datos
- Estimación de coeficientes
- Predicciones en nuevos datos

### Regresión Logística

**Teoría**:
- Función logística (sigmoid)
- Formulación para probabilidades: P(y=1|x)
- Relación con regresión lineal

**Práctica**:
- Clasificación binaria
- Cálculo de probabilidades
- Optimización de parámetros

## Objetivos de Aprendizaje

### Conceptos Teóricos
- Fundamentos de regresión lineal y supuestos
- Interpretación de coeficientes y su significancia
- Regresión logística y probabilidades
- Frontera de decisión en clasificación

### Habilidades Prácticas
- Implementación de modelos de regresión
- Evaluación con múltiples métricas
- Validación cruzada y prueba-entrenamiento
- Interpretación de resultados

### Evaluación de Modelos
- **Regresión Lineal**: R², MSE, RMSE, MAE
- **Regresión Logística**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- Matriz de confusión
- Curvas de aprendizaje

## Conceptos Clave

- Coeficientes y su interpretación
- Hipótesis nula e intervalo de confianza
- Valor p (p-value) y significancia estadística
- Función de pérdida (loss function)
- Optimización de parámetros
- Sesgo-Varianza (Bias-Variance Tradeoff)
- Sobrejuste y subfagajuste

## Flujo de Trabajo

1. Carga y preparación del dataset
2. División entrenamiento-prueba
3. Ajuste del modelo
4. Visualización de resultados
5. Evaluación con métricas apropiadas
6. Interpretación de coeficientes
7. Realización de predicciones en nuevos datos

## Cómo Utilizar Este Módulo

1. Revisa los documentos PDF para entender la teoría
2. Explora la carpeta de regresión lineal:
   - Implementa el modelo
   - Evalúa con R² y errores
   - Visualiza la recta de regresión
3. Explora la carpeta de regresión logística:
   - Implementa model binario
   - Calcula la matriz de confusión
   - Genera curva ROC
4. Completa los ejercicios prácticos propuestos
5. Interpreta los resultados en el contexto del problema
