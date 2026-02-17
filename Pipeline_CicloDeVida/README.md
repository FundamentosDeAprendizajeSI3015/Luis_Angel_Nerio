# Proyecto de ML: Análisis de Hábitos

Proyecto de Machine Learning para predecir **GPA** (regresión) y nivel de **estrés** (clasificación) basado en hábitos de estudiantes.

## Estructura del Proyecto

```
proyecto_habitos_ml/
├─ data/
│  ├─ raw/                 # Datos originales sin modificar
│  ├─ interim/             # Datos intermedios (opcional)
│  └─ processed/           # Datos limpios y listos para modelar
├─ src/
│  ├─ config.py            # Rutas y configuración
│  ├─ ingest.py            # Carga de datos
│  ├─ eda.py               # Análisis exploratorio
│  ├─ visualize.py         # Generación de gráficas
│  ├─ preprocess.py        # Limpieza y preparación
│  ├─ train_regression.py  # Modelos para GPA
│  ├─ train_classif.py     # Modelos para estrés
│  ├─ evaluate.py          # Evaluación y métricas
│  └─ utils.py             # Funciones auxiliares
├─ reports/
│  ├─ figures/             # Gráficas
│  └─ results/             # Métricas y reportes
├─ models/                 # Modelos guardados
├─ main.py                 # Script principal
├─ requirements.txt        # Dependencias
└─ README.md               # Este archivo
```

## Instalación

1. Clonar o descargar el proyecto
2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

Ejecutar el pipeline completo:
```bash
python main.py
```

## Autores
- [Tu nombre]

## Licencia
MIT
