# Clasificación de Niveles de Estrés - Semana 6

## Descripción del Proyecto

Implementación de modelos de **árboles de decisión** (Random Forest y Gradient Boosting) para clasificar el nivel de estrés de estudiantes en tres categorías: **High**, **Moderate** y **Low**.

**Dataset**: Student Lifestyle (2,000 estudiantes, 6 características)

---

## Estructura

```
semana6/
├── semana6.py                    # Script principal
├── student_lifestyle_dataset.csv # Dataset original
├── datos_procesados/             # Train/Val/Test splits
├── figures/                      # Gráficas generadas
└── results/                      # Métricas y logs
```

---

## Pipeline Implementado

1. **Carga de datos**: 2,000 estudiantes con 6 características
2. **Limpieza**: Eliminación de ID y verificación de nulos
3. **Balanceo de clases**: Undersampling a 297 muestras por clase
4. **División**: 60% train / 20% validation / 20% test
5. **Entrenamiento**: GridSearchCV para ambos modelos
6. **Evaluación**: Métricas y matrices de confusión

---

## Resultados

### Métricas (Ambos Modelos)

| Conjunto | Accuracy | Precision | Recall | F1-Score |
|----------|----------|-----------|--------|----------|
| Entrenamiento | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| Validación | **1.0000** | 1.0000 | 1.0000 | 1.0000 |
| Prueba | **1.0000** | 1.0000 | 1.0000 | 1.0000 |

### Matrices de Confusión

**Entrenamiento (178 por clase):**
```
[[178   0   0]
 [  0 178   0]
 [  0   0 178]]
```

**Validación y Prueba:**
```
[[59-60   0     0  ]
 [  0   59-60   0  ]
 [  0     0   59-60]]
```

**Cero errores en todas las predicciones** (matrices perfectamente diagonales)

---

## ¿Por Qué Accuracy = 1.0?

### Explicación Principal

El accuracy perfecto (100%) se debe a que este es un **dataset sintético con reglas determinísticas muy simples**:

**Importancia de características:**
- `Study_Hours_Per_Day`: **62-77%** → Variable dominante
- `Sleep_Hours_Per_Day`: **17-23%** → Variable secundaria
- Otras 4 variables: **<5%** → Casi irrelevantes

**Patrón identificado**: El modelo básicamente usa solo 2 variables para decidir:
```
• Muchas horas de estudio + Poco sueño → High Stress
• Pocas horas de estudio + Mucho sueño → Low Stress
• Valores intermedios → Moderate Stress
```

### ¿Es Overfitting?

**No**. El modelo generaliza perfectamente en datos no vistos (validación y prueba también tienen 100% accuracy). Esto indica que:
- Las clases están **perfectamente separables**
- El dataset tiene **reglas consistentes sin excepciones**
- Los árboles de decisión son **ideales** para este tipo de patrones

### Impacto del Balanceo

**Antes del balanceo** (desbalanceado):
```
High: 1029 (51%) | Moderate: 674 (34%) | Low: 297 (15%)
```
→ Accuracy alto podría ser engañoso (sesgo hacia clase mayoritaria)

**Después del balanceo** (undersampling):
```
High: 297 (33%) | Moderate: 297 (33%) | Low: 297 (33%)
```
→ Accuracy de 1.0 ahora es **confiable**: clasifica perfectamente TODAS las clases por igual

**Conclusión**: El balanceo asegura que el 100% accuracy significa que el modelo predice correctamente cada clase, no solo la mayoritaria.

---

## Conclusiones

1. **Balanceo exitoso**: Todas las clases tienen igual representación
2. **Modelos perfectos**: Ambos logran 100% accuracy en todos los conjuntos
3. **Dataset demasiado simple**: Las reglas son determinísticas sin ruido
4. **Métricas confiables**: Gracias al balanceo, el accuracy refleja rendimiento real
5. **2 características dominan**: Study_Hours y Sleep_Hours explican casi todo

**Lección importante**: En datasets reales, accuracy = 1.0 es extremadamente raro. Este resultado refleja la naturaleza sintética del dataset, no un logro extraordinario del modelo.

---

## Ejecución

```bash
python semana6.py
```

Genera automáticamente todas las gráficas, métricas y logs en las carpetas correspondientes.

---

**Autor**: Luis Ángel Nerio | Fecha: Febrero 2026