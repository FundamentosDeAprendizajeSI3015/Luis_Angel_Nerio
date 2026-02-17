# Proyecto: Análisis de Hábitos de Estudiantes

Proyecto de análisis exploratorio y preparación de datos de hábitos de estudiantes.

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
│  ├─ transform.py         # Transformación de datos
│  └─ utils.py             # Funciones auxiliares
├─ reports/
│  ├─ figures/             # Gráficas
│  └─ results/             # Reportes
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

Ejecutar el análisis completo:
```bash
python main.py
```

