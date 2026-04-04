"""
Modulo de Preprocesamiento de Datos
Dataset: student_lifestyle_dataset.csv

Este modulo prepara los datos para algoritmos de clustering y modelos supervisados.
Incluye carga de datos, limpieza, codificacion, escalado y generacion de dos datasets
procesados (clustering y supervisado).

Autor: Luis Angel Nerio

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
import json
import joblib
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACION DE RUTAS
# ============================================================================

BASE_PATH = Path(__file__).parent
DATA_RAW_PATH = BASE_PATH / "student_lifestyle_dataset.csv"
DATA_CLUSTERING_PATH = BASE_PATH / "data_clustering.csv"
DATA_SUPERVISED_PATH = BASE_PATH / "data_supervised.csv"
SCALER_PATH = BASE_PATH / "scaler.pkl"
MAPEO_PATH = BASE_PATH / "mapeo_stress.json"

print("=" * 80)
print("CONFIGURACION DE RUTAS")
print("=" * 80)
print(f"Ruta datos raw: {DATA_RAW_PATH}")
print(f"Ruta datos clustering: {DATA_CLUSTERING_PATH}")
print(f"Ruta datos supervisado: {DATA_SUPERVISED_PATH}")


# ============================================================================
# FUNCION 1: CARGAR DATASET
# ============================================================================

def cargar_dataset(ruta):
    """
    Carga el dataset desde un archivo CSV.
    
    Args:
        ruta (str o Path): Ruta al archivo CSV
        
    Returns:
        pd.DataFrame: Dataset cargado o None si hay error
    """
    try:
        df = pd.read_csv(ruta)
        print(f"[OK] Dataset cargado: {ruta}")
        return df
    except FileNotFoundError:
        print(f"[ERROR] No se encontro el archivo: {ruta}")
        return None
    except Exception as e:
        print(f"[ERROR] Error al cargar archivo: {e}")
        return None


# ============================================================================
# FUNCION 2: EXPLORACION INICIAL
# ============================================================================

def exploracion_inicial(df):
    """
    Muestra informacion general del dataset.
    
    Args:
        df (pd.DataFrame): Dataset a explorar
    """
    print("\n" + "=" * 80)
    print("EXPLORACION INICIAL DEL DATASET")
    print("=" * 80)
    
    print("\n[PRIMERAS FILAS DEL DATASET]")
    print(df.head())
    
    print(f"\n[DIMENSIONES DEL DATASET]")
    print(f"Filas: {df.shape[0]}")
    print(f"Columnas: {df.shape[1]}")
    
    print(f"\n[TIPOS DE DATOS]")
    print(df.dtypes)
    
    print(f"\n[INFORMACION GENERAL DEL DATASET]")
    df.info()
    
    print(f"\n[VALORES NULOS]")
    valores_nulos = df.isnull().sum()
    print(valores_nulos)
    
    if valores_nulos.sum() == 0:
        print("[OK] No hay valores nulos en el dataset")
    else:
        print(f"[ADVERTENCIA] Se encontraron {valores_nulos.sum()} valores nulos")


# ============================================================================
# FUNCION 3: ELIMINAR COLUMNA STUDENT_ID
# ============================================================================

def eliminar_student_id(df):
    """
    Elimina la columna Student_ID que no aporta valor predictivo.
    
    Args:
        df (pd.DataFrame): Dataset
        
    Returns:
        pd.DataFrame: Dataset sin Student_ID
    """
    print("\n" + "=" * 80)
    print("ELIMINANDO COLUMNA STUDENT_ID")
    print("=" * 80)
    
    if 'Student_ID' in df.columns:
        df = df.drop('Student_ID', axis=1)
        print("[OK] Columna Student_ID eliminada")
        print(f"Nuevas columnas: {df.columns.tolist()}")
    else:
        print("[ADVERTENCIA] Columna Student_ID no encontrada")
    
    return df


# ============================================================================
# FUNCION 4: MANEJAR VALORES NULOS
# ============================================================================

def manejar_valores_nulos(df):
    """
    Verifica y maneja valores nulos. Si existen, imputa con la media.
    
    Args:
        df (pd.DataFrame): Dataset
        
    Returns:
        pd.DataFrame: Dataset sin valores nulos
    """
    print("\n" + "=" * 80)
    print("MANEJO DE VALORES NULOS")
    print("=" * 80)
    
    valores_nulos = df.isnull().sum()
    
    if valores_nulos.sum() == 0:
        print("[OK] No hay valores nulos. Dataset limpio.")
        return df
    
    print(f"[ADVERTENCIA] Se encontraron {valores_nulos.sum()} valores nulos")
    
    # Imputar valores nulos con la media para columnas numericas
    columnas_numericas = df.select_dtypes(include=[np.number]).columns
    
    for columna in columnas_numericas:
        if df[columna].isnull().sum() > 0:
            media = df[columna].mean()
            df[columna] = df[columna].fillna(media)
            print(f"[INFO] Columna '{columna}': {df[columna].isnull().sum()} valores imputados con media = {media:.2f}")
    
    # Imputar valores nulos en columnas categoricas con la moda
    columnas_categoricas = df.select_dtypes(include=['object']).columns
    
    for columna in columnas_categoricas:
        if df[columna].isnull().sum() > 0:
            moda = df[columna].mode()[0]
            df[columna] = df[columna].fillna(moda)
            print(f"[INFO] Columna '{columna}': {df[columna].isnull().sum()} valores imputados con moda = {moda}")
    
    print(f"[OK] Valores nulos manejados correctamente")
    
    return df


# ============================================================================
# FUNCION 4.5: DETECTAR OUTLIERS (IQR)
# ============================================================================

def detectar_outliers(df):
    """
    Detecta outliers por columna usando el metodo IQR.

    Args:
        df (pd.DataFrame): DataFrame con solo columnas numericas

    Returns:
        dict: Diccionario con {columna: cantidad_outliers}
    """
    print("\n" + "=" * 80)
    print("DETECCION DE OUTLIERS (METODO IQR)")
    print("=" * 80)

    if df.empty:
        print("[ADVERTENCIA] DataFrame numerico vacio. No se detectan outliers.")
        return {}

    outliers_por_columna = {}
    total_outliers = 0
    total_celdas = len(df) * len(df.columns)

    for columna in df.columns:
        Q1 = df[columna].quantile(0.25)
        Q3 = df[columna].quantile(0.75)
        IQR = Q3 - Q1

        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR

        mascara_outlier = (df[columna] < limite_inferior) | (df[columna] > limite_superior)
        cantidad_outliers = int(mascara_outlier.sum())
        porcentaje = (cantidad_outliers / len(df)) * 100 if len(df) > 0 else 0.0

        outliers_por_columna[columna] = cantidad_outliers
        total_outliers += cantidad_outliers

        print(
            f"[INFO] Columna '{columna}': {cantidad_outliers} outliers "
            f"({porcentaje:.2f}% de registros)"
        )

    porcentaje_total = (total_outliers / total_celdas) * 100 if total_celdas > 0 else 0.0
    print(f"\n[INFO] Total outliers detectados en el dataset: {total_outliers}")
    print(f"[INFO] Proporcion total de celdas atipicas: {porcentaje_total:.2f}%")
    print("[OK] Deteccion de outliers completada (sin eliminar registros)")

    return outliers_por_columna


# ============================================================================
# FUNCION 4.6: VALIDAR RANGOS LOGICOS
# ============================================================================

def validar_rangos_logicos(df):
    """
    Valida rangos logicos de variables de horas y GPA, sin modificar los datos.

    Reglas:
    - Horas por dia en [0, 24]
    - GPA en [0.0, 4.0]
    - Suma de horas diarias <= 24

    Args:
        df (pd.DataFrame): DataFrame con columnas originales antes de escalar
    """
    print("\n" + "=" * 80)
    print("VALIDACION DE RANGOS LOGICOS")
    print("=" * 80)

    reglas = {
        "Study_Hours_Per_Day": (0, 24),
        "Sleep_Hours_Per_Day": (0, 24),
        "Extracurricular_Hours_Per_Day": (0, 24),
        "Social_Hours_Per_Day": (0, 24),
        "Physical_Activity_Hours_Per_Day": (0, 24),
        "GPA": (0.0, 4.0),
    }

    for columna, (min_val, max_val) in reglas.items():
        if columna not in df.columns:
            print(f"[ADVERTENCIA] Columna '{columna}' no encontrada para validacion")
            continue

        fuera_rango = ((df[columna] < min_val) | (df[columna] > max_val)).sum()
        if fuera_rango == 0:
            print(f"[OK] Columna '{columna}' dentro del rango [{min_val}, {max_val}]")
        else:
            porcentaje = (fuera_rango / len(df)) * 100 if len(df) > 0 else 0.0
            print(
                f"[ADVERTENCIA] Columna '{columna}' con {fuera_rango} valores fuera de "
                f"rango [{min_val}, {max_val}] ({porcentaje:.2f}%)"
            )

    columnas_horas = [
        "Study_Hours_Per_Day",
        "Sleep_Hours_Per_Day",
        "Extracurricular_Hours_Per_Day",
        "Social_Hours_Per_Day",
        "Physical_Activity_Hours_Per_Day",
    ]
    columnas_horas_disponibles = [c for c in columnas_horas if c in df.columns]

    if len(columnas_horas_disponibles) == len(columnas_horas):
        suma_horas = df[columnas_horas_disponibles].sum(axis=1)
        registros_exceso = int((suma_horas > 24).sum())
        porcentaje_exceso = (registros_exceso / len(df)) * 100 if len(df) > 0 else 0.0

        if registros_exceso > 0:
            print(
                f"[ADVERTENCIA] Registros con suma de horas > 24: {registros_exceso} "
                f"({porcentaje_exceso:.2f}%)"
            )
        else:
            print("[OK] No hay registros con suma de horas diaria mayor a 24")
    else:
        print("[ADVERTENCIA] No se pudo validar la suma de horas por falta de columnas")


# ============================================================================
# FUNCION 5: SEPARAR FEATURES Y LABEL
# ============================================================================

def separar_features_label(df):
    """
    Separa las features (todas excepto Stress_Level) del label (Stress_Level).
    
    Args:
        df (pd.DataFrame): Dataset completo
        
    Returns:
        tuple: (features_df, label_series)
    """
    print("\n" + "=" * 80)
    print("SEPARACION DE FEATURES Y LABEL")
    print("=" * 80)
    
    if 'Stress_Level' not in df.columns:
        print("[ERROR] Columna 'Stress_Level' no encontrada")
        return None, None
    
    # Separar features (todo excepto Stress_Level)
    X = df.drop('Stress_Level', axis=1)
    
    # Separar label
    y = df['Stress_Level']
    
    print(f"[OK] Features shape: {X.shape}")
    print(f"[OK] Features columnas: {X.columns.tolist()}")
    print(f"[OK] Label shape: {y.shape}")
    print(f"[OK] Valores unicos en label: {y.unique()}")
    
    return X, y


# ============================================================================
# FUNCION 6: CODIFICAR VARIABLE CATEGORICA
# ============================================================================

def codificar_stress_level(y):
    """
    Codifica la variable categorica Stress_Level a valores numericos.
    
    Mapeo:
    - "Low" -> 0
    - "Moderate" -> 1
    - "High" -> 2
    
    Args:
        y (pd.Series): Serie con valores categoricos
        
    Returns:
        pd.Series: Serie codificada numericamente
        dict: Diccionario con el mapeo de codificacion
    """
    print("\n" + "=" * 80)
    print("CODIFICACION DE VARIABLE CATEGORICA: STRESS_LEVEL")
    print("=" * 80)
    
    # Definir mapeo personalizado
    mapeo_stress = {
        'Low': 0,
        'Moderate': 1,
        'High': 2
    }
    
    print(f"[INFO] Mapeo de codificacion:")
    for categoria, valor in mapeo_stress.items():
        print(f"  - '{categoria}' -> {valor}")
    
    # Aplicar mapeo
    y_codificado = y.map(mapeo_stress)
    
    # Verificar si hay valores no mapeados
    if y_codificado.isnull().sum() > 0:
        print(f"[ADVERTENCIA] Se encontraron {y_codificado.isnull().sum()} valores no mapeados")
    
    print(f"[OK] Valores originales: {y.unique()}")
    print(f"[OK] Valores codificados: {y_codificado.unique()}")
    print(f"[OK] Distribucion de valores:")
    print(y_codificado.value_counts().sort_index())
    
    return y_codificado, mapeo_stress


# ============================================================================
# FUNCION 6.5: VERIFICAR BALANCE DE CLASES
# ============================================================================

def verificar_balance_clases(y, y_codificado, mapeo):
    """
    Verifica el balance de clases en la variable objetivo.

    Args:
        y (pd.Series): Serie original de etiquetas
        y_codificado (pd.Series): Serie codificada de etiquetas
        mapeo (dict): Diccionario de mapeo de etiquetas

    Returns:
        float: Ratio entre la clase mas frecuente y la menos frecuente
    """
    print("\n" + "=" * 80)
    print("VERIFICACION DE BALANCE DE CLASES")
    print("=" * 80)

    conteo_original = y.value_counts().sort_index()
    conteo_codificado = y_codificado.value_counts().sort_index()

    print("[INFO] Distribucion absoluta (original):")
    print(conteo_original)

    porcentaje_original = (conteo_original / len(y)) * 100 if len(y) > 0 else conteo_original * 0
    print("\n[INFO] Distribucion porcentual (original):")
    print(porcentaje_original.round(2))

    print("\n[INFO] Distribucion absoluta (codificada):")
    print(conteo_codificado)

    print("\n[INFO] Mapeo aplicado:")
    for etiqueta, codigo in mapeo.items():
        print(f"  - '{etiqueta}' -> {codigo}")

    max_frecuencia = int(conteo_original.max()) if len(conteo_original) > 0 else 0
    min_frecuencia = int(conteo_original.min()) if len(conteo_original) > 0 else 0

    if min_frecuencia == 0:
        ratio = float("inf")
    else:
        ratio = max_frecuencia / min_frecuencia

    print(f"\n[INFO] Ratio clase mayor / clase menor: {ratio:.4f}")

    if ratio > 1.5:
        print(
            "[ADVERTENCIA] Dataset desbalanceado. Considerar tecnicas como "
            "SMOTE o ajuste de class_weight en modelos supervisados."
        )
    else:
        print("[OK] Dataset balanceado.")

    return ratio


# ============================================================================
# FUNCION 7: ESCALAR DATOS NUMERICOS
# ============================================================================

def escalar_datos(X, metodo='standard'):
    """
    Escala los datos numericos usando StandardScaler o MinMaxScaler.
    
    Args:
        X (pd.DataFrame): Features a escalar
        metodo (str): 'standard' (StandardScaler) o 'minmax' (MinMaxScaler)
        
    Returns:
        tuple: (datos_escalados_array, datos_escalados_df, scaler_object)
    """
    print("\n" + "=" * 80)
    print("ESCALADO DE DATOS NUMERICOS")
    print("=" * 80)
    
    print(f"[INFO] Metodo de escalado: {metodo}")
    
    # Mostrar estadisticas ANTES de escalar
    print("\n[ANTES DEL ESCALADO]")
    print(f"Media: {X.mean().values}")
    print(f"Minimo: {X.min().values}")
    print(f"Maximo: {X.max().values}")
    print(f"Desviacion Estandar: {X.std().values}")
    
    # Crear scaler segun metodo
    if metodo.lower() == 'standard':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        print("[INFO] Usando StandardScaler (media=0, desv.est=1)")
    elif metodo.lower() == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        print("[INFO] Usando MinMaxScaler (rango 0-1)")
    else:
        print(f"[ERROR] Metodo '{metodo}' no reconocido")
        return None, None, None
    
    # Aplicar escalado
    X_escalado_array = scaler.fit_transform(X)

    # Guardar scaler entrenado para uso futuro en inferencia
    joblib.dump(scaler, SCALER_PATH)
    print(f"[OK] Scaler guardado en: {SCALER_PATH}")
    
    # Convertir a DataFrame para mantener nombres de columnas
    X_escalado_df = pd.DataFrame(X_escalado_array, columns=X.columns)
    
    # Mostrar estadisticas DESPUES de escalar
    print("\n[DESPUES DEL ESCALADO]")
    print(f"Media: {X_escalado_df.mean().values}")
    print(f"Minimo: {X_escalado_df.min().values}")
    print(f"Maximo: {X_escalado_df.max().values}")
    print(f"Desviacion Estandar: {X_escalado_df.std().values}")
    
    print(f"\n[OK] Datos escalados correctamente")
    print(f"[OK] Forma original: {X.shape}")
    print(f"[OK] Forma escalada: {X_escalado_df.shape}")
    
    return X_escalado_array, X_escalado_df, scaler


# ============================================================================
# FUNCION 8: CREAR DATASET PARA CLUSTERING
# ============================================================================

def crear_dataset_clustering(X_escalado):
    """
    Crea dataset sin Stress_Level para algoritmos de clustering.
    
    Args:
        X_escalado (pd.DataFrame): Features escaladas
        
    Returns:
        pd.DataFrame: Dataset para clustering
    """
    print("\n" + "=" * 80)
    print("CREANDO DATASET PARA CLUSTERING")
    print("=" * 80)
    
    # El dataset de clustering es simplemente las features escaladas
    data_clustering = X_escalado.copy()
    
    print(f"[OK] Dataset de clustering creado")
    print(f"[OK] Forma: {data_clustering.shape}")
    print(f"[OK] Columnas: {data_clustering.columns.tolist()}")
    print(f"[OK] Primeras filas:")
    print(data_clustering.head())
    
    return data_clustering


# ============================================================================
# FUNCION 9: CREAR DATASET PARA MODELOS SUPERVISADOS
# ============================================================================

def crear_dataset_supervisado(X_escalado, y_codificado):
    """
    Crea dataset con Stress_Level para modelos supervisados.
    
    Args:
        X_escalado (pd.DataFrame): Features escaladas
        y_codificado (pd.Series): Label codificado
        
    Returns:
        pd.DataFrame: Dataset completo para modelos supervisados
    """
    print("\n" + "=" * 80)
    print("CREANDO DATASET PARA MODELOS SUPERVISADOS")
    print("=" * 80)
    
    # Combinar features con label
    data_supervisado = X_escalado.copy()
    data_supervisado['Stress_Level'] = y_codificado.values
    
    print(f"[OK] Dataset supervisado creado")
    print(f"[OK] Forma: {data_supervisado.shape}")
    print(f"[OK] Columnas: {data_supervisado.columns.tolist()}")
    print(f"[OK] Primeras filas:")
    print(data_supervisado.head())
    
    return data_supervisado


# ============================================================================
# FUNCION 10: GUARDAR DATASETS PROCESADOS
# ============================================================================

def guardar_datasets(data_clustering, data_supervisado, ruta_clustering, ruta_supervisado):
    """
    Guarda los datasets procesados en archivos CSV.
    
    Args:
        data_clustering (pd.DataFrame): Dataset para clustering
        data_supervisado (pd.DataFrame): Dataset para modelos supervisados
        ruta_clustering (str o Path): Ruta destino para dataset clustering
        ruta_supervisado (str o Path): Ruta destino para dataset supervisado
        
    Returns:
        tuple: (exito_clustering, exito_supervisado)
    """
    print("\n" + "=" * 80)
    print("GUARDANDO DATASETS PROCESADOS")
    print("=" * 80)
    
    exito_clustering = False
    exito_supervisado = False
    
    try:
        data_clustering.to_csv(ruta_clustering, index=False)
        print(f"[OK] Dataset clustering guardado: {ruta_clustering}")
        print(f"     Filas: {len(data_clustering)}, Columnas: {len(data_clustering.columns)}")
        exito_clustering = True
    except Exception as e:
        print(f"[ERROR] No se pudo guardar dataset clustering: {e}")
    
    try:
        data_supervisado.to_csv(ruta_supervisado, index=False)
        print(f"[OK] Dataset supervisado guardado: {ruta_supervisado}")
        print(f"     Filas: {len(data_supervisado)}, Columnas: {len(data_supervisado.columns)}")
        exito_supervisado = True
    except Exception as e:
        print(f"[ERROR] No se pudo guardar dataset supervisado: {e}")
    
    return exito_clustering, exito_supervisado


# ============================================================================
# FUNCION 11: VERIFICACION DE DATOS ESCALADOS
# ============================================================================

def verificar_escalado(X_escalado, nombre_dataset="Dataset"):
    """
    Verifica que los datos esten correctamente escalados.
    
    Args:
        X_escalado (pd.DataFrame): Dataset escalado
        nombre_dataset (str): Nombre del dataset para el reporte
    """
    print("\n" + "=" * 80)
    print(f"VERIFICACION DE ESCALADO: {nombre_dataset}")
    print("=" * 80)
    
    print(f"\n[RESUMEN DE ESCALADO]")
    print(f"Forma del dataset: {X_escalado.shape}")
    print(f"\nEstadisticas por columna:")
    print(f"{'Columna':<30} {'Media':<12} {'Std Dev':<12} {'Min':<12} {'Max':<12}")
    print("-" * 80)
    
    for columna in X_escalado.columns:
        media = X_escalado[columna].mean()
        std = X_escalado[columna].std()
        minimo = X_escalado[columna].min()
        maximo = X_escalado[columna].max()
        
        print(f"{columna:<30} {media:<12.4f} {std:<12.4f} {minimo:<12.4f} {maximo:<12.4f}")
    
    # Verificar que la media sea muy cercana a 0 y std cercana a 1 (para StandardScaler)
    medias = X_escalado.mean()
    stds = X_escalado.std()
    
    todas_bien = True
    for col in X_escalado.columns:
        if abs(medias[col]) > 0.1 or abs(stds[col] - 1.0) > 0.1:
            todas_bien = False
            break
    
    if todas_bien:
        print(f"\n[OK] Datos escalados correctamente usando StandardScaler")
    else:
        print(f"\n[INFO] Escalado completado")
    
    print(f"[OK] Verificacion completada")


# ============================================================================
# FUNCION PRINCIPAL
# ============================================================================

def main():
    """
    Funcion principal que ejecuta el pipeline de preprocesamiento completo.
    """
    print("\n" + "=" * 80)
    print("INICIANDO PIPELINE DE PREPROCESAMIENTO")
    print("=" * 80)
    
    # PASO 1: Cargar dataset
    print("\n[PASO 1/12] CARGANDO DATASET")
    df = cargar_dataset(DATA_RAW_PATH)
    
    if df is None:
        print("[ERROR] No se pudo cargar el dataset. Abortando.")
        return False
    
    # Crear copia para no modificar original
    df = df.copy()
    
    # PASO 2: Exploracion inicial
    print("\n[PASO 2/12] EXPLORACION INICIAL")
    exploracion_inicial(df)

    # PASO 2.5: Validacion de rangos logicos
    print("\n[PASO 2.5/12] VALIDACION DE RANGOS LOGICOS")
    validar_rangos_logicos(df)
    
    # PASO 3: Eliminar Student_ID
    print("\n[PASO 3/12] ELIMINANDO STUDENT_ID")
    df = eliminar_student_id(df)
    
    # PASO 4: Manejar valores nulos
    print("\n[PASO 4/12] MANEJANDO VALORES NULOS")
    df = manejar_valores_nulos(df)

    # PASO 4.5: Deteccion de outliers
    print("\n[PASO 4.5/12] DETECCION DE OUTLIERS")
    df_numerico = df.select_dtypes(include=[np.number])
    _ = detectar_outliers(df_numerico)
    
    # PASO 5: Separar features y label
    print("\n[PASO 5/12] SEPARANDO FEATURES Y LABEL")
    X, y = separar_features_label(df)
    
    if X is None or y is None:
        print("[ERROR] No se pudo separar features y label. Abortando.")
        return False
    
    # PASO 6: Codificar Stress_Level
    print("\n[PASO 6/12] CODIFICANDO STRESS_LEVEL")
    y_codificado, mapeo_stress = codificar_stress_level(y)

    # Guardar mapeo de codificacion para trazabilidad
    with open(MAPEO_PATH, 'w') as f:
        json.dump(mapeo_stress, f, indent=2)
    print(f"[OK] Mapeo de Stress_Level guardado en: {MAPEO_PATH}")

    # PASO 6.5: Verificacion de balance de clases
    print("\n[PASO 6.5/12] VERIFICACION DE BALANCE DE CLASES")
    _ = verificar_balance_clases(y, y_codificado, mapeo_stress)
    
    # PASO 7: Escalar datos
    print("\n[PASO 7/12] ESCALANDO DATOS NUMERICOS")
    X_escalado_array, X_escalado_df, scaler = escalar_datos(X, metodo='standard')
    
    if X_escalado_df is None:
        print("[ERROR] No se pudo escalar los datos. Abortando.")
        return False
    
    # PASO 8: Crear dataset para clustering
    print("\n[PASO 8/12] CREANDO DATASET PARA CLUSTERING")
    data_clustering = crear_dataset_clustering(X_escalado_df)
    
    # PASO 9: Crear dataset para modelos supervisados
    print("\n[PASO 9/12] CREANDO DATASET PARA MODELOS SUPERVISADOS")
    data_supervisado = crear_dataset_supervisado(X_escalado_df, y_codificado)
    
    # PASO 10: Guardar datasets
    print("\n[PASO 10/12] GUARDANDO DATASETS")
    exito_cluster, exito_super = guardar_datasets(
        data_clustering, 
        data_supervisado,
        DATA_CLUSTERING_PATH,
        DATA_SUPERVISED_PATH
    )
    
    # PASO 11: Verificacion final
    print("\n[PASO 11/12] VERIFICACION FINAL")
    print("\n" + "=" * 80)
    print("VERIFICACION FINAL")
    print("=" * 80)
    
    verificar_escalado(X_escalado_df, "Dataset Clustering")
    
    # PASO 12: Resumen final
    print("\n[PASO 12/12] RESUMEN FINAL")
    print("\n" + "=" * 80)
    print("RESUMEN DEL PREPROCESAMIENTO")
    print("=" * 80)
    
    print(f"\nDatos originales: {len(df)} registros")
    print(f"Columnas en dataset procesado: {len(df.columns)} (sin Student_ID, sin Stress_Level)")
    print(f"Datos procesados: {len(data_clustering)} registros")
    print(f"\nDataset CLUSTERING (sin Stress_Level):")
    print(f"  - Forma: {data_clustering.shape}")
    print(f"  - Archivo: {DATA_CLUSTERING_PATH}")
    print(f"  - Primera fila:")
    print(f"    {data_clustering.iloc[0].to_dict()}")
    
    print(f"\nDataset SUPERVISADO (con Stress_Level):")
    print(f"  - Forma: {data_supervisado.shape}")
    print(f"  - Archivo: {DATA_SUPERVISED_PATH}")
    print(f"  - Primera fila:")
    print(f"    {data_supervisado.iloc[0].to_dict()}")
    
    print(f"\nCodificacion de Stress_Level:")
    for orig, cod in mapeo_stress.items():
        count = (y == orig).sum()
        print(f"  - '{orig}' ({cod}): {count} registros")
    
    if exito_cluster and exito_super:
        print("\n[OK] PREPROCESAMIENTO COMPLETADO EXITOSAMENTE")
        return True
    else:
        print("\n[ADVERTENCIA] Preprocesamiento completado con algunos errores")
        return False


# ============================================================================
# EJECUTAR PROGRAMA
# ============================================================================

if __name__ == "__main__":
    exito = main()
    
    print("\n" + "=" * 80)
    if exito:
        print("Pipeline finalizado exitosamente")
    else:
        print("Pipeline finalizado con problemas")
    print("=" * 80)
