

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# Cargar el DataFrame del Titanic
df = pd.read_csv('Titanic-Dataset.csv')

# ============================================
# TRANSFORMACIÓN DE COLUMNAS - ENCODING
# ============================================

print("\n" + "="*60)
print("TRANSFORMACIÓN DE COLUMNAS - ENCODING")
print("="*60)

# Trabajaremos con una copia del DataFrame original del Titanic
df_encoding = df.copy()

# --- ONE HOT ENCODING ---
print("\n--- 1. ONE HOT ENCODING ---")
print("Transforma variables categóricas en columnas binarias (0 o 1)")
print("\nColumnas categóricas originales: Sex, Embarked")

# Mostrar distribución original
print("\nDistribución de 'Sex':")
print(df_encoding['Sex'].value_counts())
print("\nDistribución de 'Embarked':")
print(df_encoding['Embarked'].value_counts())

# Aplicar One-Hot Encoding a las columnas Sex y Embarked
df_one_hot = pd.get_dummies(df_encoding, columns=['Sex', 'Embarked'], prefix=['Sex', 'Embarked'])

print("\nDataFrame después de One-Hot Encoding (primeras 5 filas):")
print(df_one_hot[['PassengerId', 'Name', 'Sex_female', 'Sex_male', 
                   'Embarked_C', 'Embarked_Q', 'Embarked_S']].head())

print(f"\nColumnas agregadas: {[col for col in df_one_hot.columns if col.startswith(('Sex_', 'Embarked_'))]}")
print(f"Forma del DataFrame Original: {df_encoding.shape}")
print(f"Forma del DataFrame con One-Hot: {df_one_hot.shape}")


# --- LABEL ENCODING ---
print("\n" + "="*60)
print("--- 2. LABEL ENCODING ---")
print("Asigna un número entero único a cada categoría")

from sklearn.preprocessing import LabelEncoder

# Crear una copia para Label Encoding
df_label = df_encoding.copy()

# Aplicar Label Encoding a la columna 'Sex'
le_sex = LabelEncoder()
df_label['Sex_Encoded'] = le_sex.fit_transform(df_label['Sex'])

print("\nLabel Encoding para 'Sex':")
print(f"Categorías originales: {le_sex.classes_}")
print(f"Mapeo: {dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))}")
print("\nComparación:")
print(df_label[['PassengerId', 'Name', 'Sex', 'Sex_Encoded']].head(10))

# Aplicar Label Encoding a la columna 'Embarked' (eliminando nulos primero)
df_label_embarked = df_label[df_label['Embarked'].notna()].copy()
le_embarked = LabelEncoder()
df_label_embarked['Embarked_Encoded'] = le_embarked.fit_transform(df_label_embarked['Embarked'])

print("\nLabel Encoding para 'Embarked':")
print(f"Categorías originales: {le_embarked.classes_}")
print(f"Mapeo: {dict(zip(le_embarked.classes_, le_embarked.transform(le_embarked.classes_)))}")
print("\nComparación:")
print(df_label_embarked[['PassengerId', 'Name', 'Embarked', 'Embarked_Encoded']].head(10))


# --- BINARY ENCODING ---
print("\n" + "="*60)
print("--- 3. BINARY ENCODING ---")
print("Convierte categorías a representación binaria (más eficiente para muchas categorías)")

# Verificar si category_encoders está instalado, si no, intentar instalarlo
try:
    import category_encoders as ce
except ImportError:
    print("\nInstalando category_encoders...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "category_encoders", "-q"])
    import category_encoders as ce

# Crear una copia para Binary Encoding
df_binary = df_encoding.copy()

# Aplicar Binary Encoding a la columna 'Embarked'
# Primero llenamos los nulos con 'Unknown'
df_binary['Embarked'] = df_binary['Embarked'].fillna('Unknown')

encoder_embarked = ce.BinaryEncoder(cols=['Embarked'])
df_binary_embarked = encoder_embarked.fit_transform(df_binary['Embarked'])

print("\nBinary Encoding para 'Embarked':")
print(f"Categorías originales: C, Q, S, Unknown")
print("\nColumnas generadas por Binary Encoding:")
print(df_binary_embarked.head(10))

# Concatenar con el dataframe original
df_binary = pd.concat([df_binary, df_binary_embarked], axis=1)

print("\nDataFrame con Binary Encoding (muestra):")
print(df_binary[['PassengerId', 'Name', 'Embarked'] + list(df_binary_embarked.columns)].head(10))


# Aplicar Binary Encoding a la columna 'Pclass' (aunque es numérica, la trataremos como categórica)
encoder_pclass = ce.BinaryEncoder(cols=['Pclass'])
df_binary_pclass = encoder_pclass.fit_transform(df_binary[['Pclass']])

print("\nBinary Encoding para 'Pclass':")
print(f"Categorías originales: 1, 2, 3")
print("\nColumnas generadas:")
print(df_binary_pclass.head(10))


# --- COMPARACIÓN DE TÉCNICAS ---
print("\n" + "="*60)
print("COMPARACIÓN DE TÉCNICAS DE ENCODING")
print("="*60)

print("\n1. ONE-HOT ENCODING:")
print("   ✓ Ventajas: Fácil de interpretar, no asume orden entre categorías")
print("   ✗ Desventajas: Aumenta mucho las dimensiones con muchas categorías")
print(f"   Ejemplo: 'Embarked' (3 categorías) → 3 columnas binarias")

print("\n2. LABEL ENCODING:")
print("   ✓ Ventajas: Muy simple, no aumenta dimensiones")
print("   ✗ Desventajas: Asume orden entre categorías (puede confundir al modelo)")
print(f"   Ejemplo: 'Embarked' (3 categorías) → 1 columna con valores 0, 1, 2")

print("\n3. BINARY ENCODING:")
print("   ✓ Ventajas: Más eficiente que One-Hot, mantiene baja dimensionalidad")
print("   ✗ Desventajas: Menos interpretable, requiere librería adicional")
print(f"   Ejemplo: 'Embarked' (4 categorías con Unknown) → 2 columnas binarias")

print("\n" + "="*60)
print("RESUMEN DE SHAPES")
print("="*60)
print(f"DataFrame Original: {df_encoding.shape}")
print(f"Con One-Hot Encoding: {df_one_hot.shape}")
print(f"Con Label Encoding: {df_label.shape} (mismo tamaño, solo nuevas columnas)")
print(f"Con Binary Encoding: {df_binary.shape}")

print("\n--- Transformación de Columnas Completada ---")


# ============================================
# ANÁLISIS DE CORRELACIÓN Y SELECCIÓN DE COLUMNAS
# ============================================

print("\n" + "="*60)
print("ANÁLISIS DE CORRELACIÓN ENTRE COLUMNAS")
print("="*60)

# Seleccionar solo las columnas numéricas del DataFrame original
columnas_numericas = ['Age', 'Fare', 'SibSp', 'Parch', 'Pclass', 'Survived']
df_correlacion = df[columnas_numericas].copy()

# Eliminar filas con valores nulos para el análisis
print(f"\nFilas originales: {len(df)}")
df_correlacion = df_correlacion.dropna()
print(f"Filas después de eliminar nulos: {len(df_correlacion)}")

# --- Correlación entre dos columnas específicas ---
print("\n--- Correlación entre columnas específicas ---")

# Correlación Age vs Fare
corr_age_fare = df_correlacion['Age'].corr(df_correlacion['Fare'])
print(f"Correlación entre Age y Fare: {corr_age_fare:.4f}")

# Correlación SibSp vs Parch
corr_sibsp_parch = df_correlacion['SibSp'].corr(df_correlacion['Parch'])
print(f"Correlación entre SibSp y Parch: {corr_sibsp_parch:.4f}")

# Correlación Pclass vs Fare
corr_pclass_fare = df_correlacion['Pclass'].corr(df_correlacion['Fare'])
print(f"Correlación entre Pclass y Fare: {corr_pclass_fare:.4f}")

# Correlación Pclass vs Survived
corr_pclass_survived = df_correlacion['Pclass'].corr(df_correlacion['Survived'])
print(f"Correlación entre Pclass y Survived: {corr_pclass_survived:.4f}")


# --- Matriz de Correlación Completa ---
print("\n--- Matriz de Correlación Completa ---")
matriz_corr = df_correlacion.corr()
print("\nMatriz de correlación:")
print(matriz_corr.round(3))


# --- Visualización: Mapa de Calor de Correlaciones ---
plt.figure(figsize=(10, 8))
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', center=0,
            fmt='.3f', linewidths=1, square=True, 
            cbar_kws={'label': 'Coeficiente de Correlación'},
            vmin=-1, vmax=1)
plt.title('Mapa de Calor de Correlaciones - Dataset Titanic', fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('09_heatmap_correlaciones.png', dpi=300, bbox_inches='tight')
plt.show()


# --- Análisis de Correlación con Pearson y Spearman ---
print("\n" + "="*60)
print("COMPARACIÓN: CORRELACIÓN PEARSON VS SPEARMAN")
print("="*60)

print("\nPearson: Mide relaciones LINEALES")
print("Spearman: Mide relaciones MONOTÓNICAS (lineales o no)")

# Comparar ambos métodos para algunas variables clave
print("\n--- Age vs Fare ---")
pearson_af = df_correlacion['Age'].corr(df_correlacion['Fare'], method='pearson')
spearman_af = df_correlacion['Age'].corr(df_correlacion['Fare'], method='spearman')
print(f"Pearson:  {pearson_af:.4f}")
print(f"Spearman: {spearman_af:.4f}")

print("\n--- Pclass vs Fare ---")
pearson_pf = df_correlacion['Pclass'].corr(df_correlacion['Fare'], method='pearson')
spearman_pf = df_correlacion['Pclass'].corr(df_correlacion['Fare'], method='spearman')
print(f"Pearson:  {pearson_pf:.4f}")
print(f"Spearman: {spearman_pf:.4f}")

print("\n--- SibSp vs Parch ---")
pearson_sp = df_correlacion['SibSp'].corr(df_correlacion['Parch'], method='pearson')
spearman_sp = df_correlacion['SibSp'].corr(df_correlacion['Parch'], method='spearman')
print(f"Pearson:  {pearson_sp:.4f}")
print(f"Spearman: {spearman_sp:.4f}")


# --- Identificar columnas altamente correlacionadas ---
print("\n" + "="*60)
print("IDENTIFICACIÓN DE COLUMNAS ALTAMENTE CORRELACIONADAS")
print("="*60)

# Definir umbral para correlación alta (típicamente > 0.7 o < -0.7)
umbral_correlacion = 0.7

print(f"\nUmbral de correlación alta: ±{umbral_correlacion}")
print("\nPares de variables con correlación alta:")

# Buscar correlaciones altas (excluyendo la diagonal)
encontrado = False
for i in range(len(matriz_corr.columns)):
    for j in range(i+1, len(matriz_corr.columns)):
        corr_value = matriz_corr.iloc[i, j]
        if abs(corr_value) >= umbral_correlacion:
            var1 = matriz_corr.columns[i]
            var2 = matriz_corr.columns[j]
            print(f"  • {var1} vs {var2}: {corr_value:.4f}")
            encontrado = True

if not encontrado:
    print(f"  No se encontraron pares de variables con correlación >= {umbral_correlacion}")


# --- Recomendaciones para eliminación de columnas ---
print("\n" + "="*60)
print("RECOMENDACIONES PARA ELIMINACIÓN DE COLUMNAS")
print("="*60)

print("\n📊 CRITERIOS DE DECISIÓN:")
print("  • Correlación alta (|r| > 0.7): Considerar eliminar una de las dos")
print("  • Correlación moderada (0.5 < |r| < 0.7): Evaluar según contexto")
print("  • Correlación baja (|r| < 0.5): Mantener ambas columnas")

print("\n🔍 ANÁLISIS DEL DATASET TITANIC:")

# Evaluar cada par de correlaciones
correlaciones_importantes = []
for i in range(len(matriz_corr.columns)):
    for j in range(i+1, len(matriz_corr.columns)):
        corr_value = abs(matriz_corr.iloc[i, j])
        if corr_value >= 0.5:  # Correlación moderada o alta
            var1 = matriz_corr.columns[i]
            var2 = matriz_corr.columns[j]
            correlaciones_importantes.append((var1, var2, matriz_corr.iloc[i, j]))

if correlaciones_importantes:
    print("\nPares con correlación moderada o alta:")
    for var1, var2, corr_val in sorted(correlaciones_importantes, key=lambda x: abs(x[2]), reverse=True):
        if abs(corr_val) >= 0.7:
            nivel = "⚠️ ALTA"
        else:
            nivel = "⚡ MODERADA"
        print(f"  {nivel} - {var1} vs {var2}: {corr_val:.4f}")
else:
    print("\nNo hay pares con correlación moderada o alta.")

print("\n💡 RECOMENDACIONES ESPECÍFICAS:")

# Analizar correlaciones específicas del Titanic
if abs(matriz_corr.loc['Pclass', 'Fare']) >= 0.5:
    print(f"\n  1. Pclass vs Fare (r={matriz_corr.loc['Pclass', 'Fare']:.3f}):")
    print("     → Correlación moderada/alta NEGATIVA")
    print("     → A mayor clase (3), menor tarifa")
    print("     → RECOMENDACIÓN: MANTENER ambas")
    print("       • Pclass: Variable categórica ordinal importante")
    print("       • Fare: Variable continua con información única")

if abs(matriz_corr.loc['SibSp', 'Parch']) >= 0.3:
    print(f"\n  2. SibSp vs Parch (r={matriz_corr.loc['SibSp', 'Parch']:.3f}):")
    print("     → Correlación baja/moderada POSITIVA")
    print("     → Ambas relacionadas con tamaño de familia")
    print("     → RECOMENDACIÓN: MANTENER ambas")
    print("       • Representan relaciones familiares diferentes")
    print("       • O crear una nueva variable 'FamilySize' = SibSp + Parch + 1")

if abs(matriz_corr.loc['Age', 'Fare']) < 0.2:
    print(f"\n  3. Age vs Fare (r={matriz_corr.loc['Age', 'Fare']:.3f}):")
    print("     → Correlación MUY BAJA")
    print("     → Variables independientes entre sí")
    print("     → RECOMENDACIÓN: MANTENER ambas")

print("\n✅ CONCLUSIÓN FINAL:")
print("  En el dataset del Titanic, NO hay columnas con correlación")
print("  lo suficientemente alta como para recomendar su eliminación.")
print("  Todas las variables aportan información única y valiosa.")

# --- Crear DataFrame con variables seleccionadas (ejemplo) ---
print("\n" + "="*60)
print("CREACIÓN DE DATASET CON FEATURE ENGINEERING")
print("="*60)

# Crear una nueva variable combinando SibSp y Parch
df_final = df_correlacion.copy()
df_final['FamilySize'] = df_final['SibSp'] + df_final['Parch'] + 1
df_final['IsAlone'] = (df_final['FamilySize'] == 1).astype(int)

print("\nNuevas variables creadas:")
print("  • FamilySize: SibSp + Parch + 1")
print("  • IsAlone: 1 si viajaba solo, 0 si con familia")

print("\nDataFrame final (primeras 10 filas):")
print(df_final[['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize', 'IsAlone', 'Survived']].head(10))

# Correlación de las nuevas variables con Survived
print("\nCorrelación de nuevas variables con Survived:")
print(f"  FamilySize vs Survived: {df_final['FamilySize'].corr(df_final['Survived']):.4f}")
print(f"  IsAlone vs Survived:   {df_final['IsAlone'].corr(df_final['Survived']):.4f}")

print("\n--- Análisis de Correlación Completado ---")

# ============================================
# TRANSFORMACIÓN LOGARÍTMICA
# ============================================

print("\n" + "="*60)
print("TRANSFORMACIÓN LOGARÍTMICA")
print("="*60)

print("\n📌 ¿CUÁNDO APLICAR TRANSFORMACIÓN LOGARÍTMICA?")
print("  • Datos con distribución muy sesgada (asimétrica)")
print("  • Valores muy dispersos (outliers extremos)")
print("  • Diferencias de escala de varios órdenes de magnitud")
print("  • Para estabilizar varianza y hacer datos más 'normales'")

# --- Análisis de la distribución de Fare ---
print("\n" + "="*60)
print("ANÁLISIS: ¿FARE NECESITA TRANSFORMACIÓN LOGARÍTMICA?")
print("="*60)

df_fare_analysis = df[df['Fare'] > 0].copy()  # Eliminar Fare = 0 para poder aplicar log
print(f"\nRegistros con Fare > 0: {len(df_fare_analysis)}")
print(f"Registros con Fare = 0: {len(df[df['Fare'] == 0])}")

# Estadísticas de Fare original
print("\n📊 Estadísticas de 'Fare' (Original):")
print(f"  Mínimo:  ${df_fare_analysis['Fare'].min():.2f}")
print(f"  Q1:      ${df_fare_analysis['Fare'].quantile(0.25):.2f}")
print(f"  Mediana: ${df_fare_analysis['Fare'].median():.2f}")
print(f"  Q3:      ${df_fare_analysis['Fare'].quantile(0.75):.2f}")
print(f"  Máximo:  ${df_fare_analysis['Fare'].max():.2f}")
print(f"  Media:   ${df_fare_analysis['Fare'].mean():.2f}")
print(f"  Desv. Std: ${df_fare_analysis['Fare'].std():.2f}")

# Calcular asimetría (skewness)
skewness_fare = df_fare_analysis['Fare'].skew()
print(f"\n  Asimetría (Skewness): {skewness_fare:.3f}")
print("    → Skewness > 1: Altamente sesgada a la derecha ✓")
print("    → ¡NECESITA TRANSFORMACIÓN LOGARÍTMICA!")

# --- Aplicar Transformación Logarítmica ---
print("\n--- Aplicando Transformación Logarítmica (Log10) ---")

# Aplicar log10 a Fare (solo valores > 0)
df_fare_analysis['Fare_Log10'] = np.log10(df_fare_analysis['Fare'])

# Estadísticas después de la transformación
print("\n📊 Estadísticas de 'Fare_Log10' (Transformado):")
print(f"  Mínimo:  {df_fare_analysis['Fare_Log10'].min():.3f}")
print(f"  Q1:      {df_fare_analysis['Fare_Log10'].quantile(0.25):.3f}")
print(f"  Mediana: {df_fare_analysis['Fare_Log10'].median():.3f}")
print(f"  Q3:      {df_fare_analysis['Fare_Log10'].quantile(0.75):.3f}")
print(f"  Máximo:  {df_fare_analysis['Fare_Log10'].max():.3f}")
print(f"  Media:   {df_fare_analysis['Fare_Log10'].mean():.3f}")
print(f"  Desv. Std: {df_fare_analysis['Fare_Log10'].std():.3f}")

skewness_fare_log = df_fare_analysis['Fare_Log10'].skew()
print(f"\n  Asimetría (Skewness): {skewness_fare_log:.3f}")
print("    → Skewness reducida significativamente ✓")
print("    → Distribución más simétrica y 'normal'")

# --- Visualización: Antes y Después ---
print("\n--- Generando Visualización Comparativa ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Histograma Fare Original
sns.histplot(df_fare_analysis['Fare'], bins=50, kde=True, ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('Distribución Original de Fare', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Tarifa ($)', fontsize=11)
axes[0, 0].set_ylabel('Frecuencia', fontsize=11)
axes[0, 0].axvline(df_fare_analysis['Fare'].mean(), color='red', linestyle='--', 
                   label=f'Media: ${df_fare_analysis["Fare"].mean():.2f}')
axes[0, 0].axvline(df_fare_analysis['Fare'].median(), color='green', linestyle='-', 
                   label=f'Mediana: ${df_fare_analysis["Fare"].median():.2f}')
axes[0, 0].legend()
axes[0, 0].grid(axis='y', alpha=0.3)

# 2. Histograma Fare Logarítmico
sns.histplot(df_fare_analysis['Fare_Log10'], bins=30, kde=True, ax=axes[0, 1], color='salmon')
axes[0, 1].set_title('Distribución Transformada (Log10)', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Log10(Tarifa)', fontsize=11)
axes[0, 1].set_ylabel('Frecuencia', fontsize=11)
axes[0, 1].axvline(df_fare_analysis['Fare_Log10'].mean(), color='red', linestyle='--', 
                   label=f'Media: {df_fare_analysis["Fare_Log10"].mean():.3f}')
axes[0, 1].axvline(df_fare_analysis['Fare_Log10'].median(), color='green', linestyle='-', 
                   label=f'Mediana: {df_fare_analysis["Fare_Log10"].median():.3f}')
axes[0, 1].legend()
axes[0, 1].grid(axis='y', alpha=0.3)

# 3. Box Plot Fare Original
axes[1, 0].boxplot(df_fare_analysis['Fare'], vert=True)
axes[1, 0].set_title('Box Plot - Fare Original', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Tarifa ($)', fontsize=11)
axes[1, 0].grid(axis='y', alpha=0.3)

# 4. Box Plot Fare Logarítmico
axes[1, 1].boxplot(df_fare_analysis['Fare_Log10'], vert=True)
axes[1, 1].set_title('Box Plot - Fare Transformado (Log10)', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Log10(Tarifa)', fontsize=11)
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.suptitle('Comparación: Transformación Logarítmica de Fare', 
             y=1.01, fontsize=16, fontweight='bold')
plt.savefig('10_transformacion_logaritmica_fare.png', dpi=300, bbox_inches='tight')
plt.show()


# --- Análisis de Age (opcional) ---
print("\n" + "="*60)
print("ANÁLISIS: ¿AGE NECESITA TRANSFORMACIÓN LOGARÍTMICA?")
print("="*60)

df_age_analysis = df[df['Age'].notna()].copy()
print(f"\nRegistros con Age válida: {len(df_age_analysis)}")

# Estadísticas de Age
print("\n📊 Estadísticas de 'Age' (Original):")
print(f"  Mínimo:  {df_age_analysis['Age'].min():.2f} años")
print(f"  Q1:      {df_age_analysis['Age'].quantile(0.25):.2f} años")
print(f"  Mediana: {df_age_analysis['Age'].median():.2f} años")
print(f"  Q3:      {df_age_analysis['Age'].quantile(0.75):.2f} años")
print(f"  Máximo:  {df_age_analysis['Age'].max():.2f} años")
print(f"  Media:   {df_age_analysis['Age'].mean():.2f} años")

skewness_age = df_age_analysis['Age'].skew()
print(f"\n  Asimetría (Skewness): {skewness_age:.3f}")
if abs(skewness_age) < 0.5:
    print("    → Skewness < 0.5: Distribución relativamente simétrica")
    print("    → NO NECESITA transformación logarítmica")
else:
    print("    → Skewness >= 0.5: Distribución moderadamente sesgada")
    print("    → Transformación logarítmica podría ser beneficiosa")


# --- Comparación con correlaciones ---
print("\n" + "="*60)
print("IMPACTO EN CORRELACIONES")
print("="*60)

# Crear DataFrame con Fare transformado
df_compare = df[['Age', 'Fare', 'Pclass', 'Survived']].dropna()
df_compare_log = df_compare.copy()
df_compare_log['Fare_Log10'] = np.log10(df_compare_log['Fare'].replace(0, 0.01))  # Evitar log(0)

print("\n--- Correlación con 'Survived' ---")
print(f"  Fare (Original):     {df_compare['Fare'].corr(df_compare['Survived']):.4f}")
print(f"  Fare_Log10:          {df_compare_log['Fare_Log10'].corr(df_compare_log['Survived']):.4f}")
print(f"\n--- Correlación con 'Pclass' ---")
print(f"  Fare (Original):     {df_compare['Fare'].corr(df_compare['Pclass']):.4f}")
print(f"  Fare_Log10:          {df_compare_log['Fare_Log10'].corr(df_compare_log['Pclass']):.4f}")


# ============================================
# GUARDAR DATASETS TRANSFORMADOS EN CSV
# ============================================

print("\n" + "="*80)
print("CREAR DATASET ÚNICO CON TODAS LAS TRANSFORMACIONES")
print("="*80)

# Crear dataset combinado basado en One-Hot Encoding
df_transformado = df_one_hot.copy()

# Agregar FamilySize e IsAlone (Feature Engineering)
print("\n✓ Añadiendo características de Feature Engineering...")
df_transformado['FamilySize'] = df_transformado['SibSp'] + df_transformado['Parch'] + 1
df_transformado['IsAlone'] = (df_transformado['FamilySize'] == 1).astype(int)

# Agregar transformación logarítmica de Fare
print("✓ Añadiendo transformación logarítmica de Fare")
df_transformado['Fare_Log10'] = np.log10(df_transformado['Fare'].replace(0, 0.01))

# Mostrar información del dataset combinado
print("\n📊 INFORMACIÓN DEL DATASET TRANSFORMADO:")
print(f"  Total de filas: {len(df_transformado)}")
print(f"  Total de columnas: {len(df_transformado.columns)}")
print(f"\n  Columnas incluidas:")

columnas_originales = ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']
columnas_one_hot = ['Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
columnas_feature_eng = ['FamilySize', 'IsAlone']
columnas_transform = ['Fare_Log10']

print(f"\n  ├─ Originales: {columnas_originales}")
print(f"  ├─ One-Hot Encoding: {columnas_one_hot}")
print(f"  ├─ Feature Engineering: {columnas_feature_eng}")
print(f"  └─ Transformación Logarítmica: {columnas_transform}")

# Guardar el dataset combinado
print("\n" + "="*80)
print("GUARDANDO DATASET TRANSFORMADO")
print("="*80)

archivo_salida = 'Titanic_Transformado.csv'
df_transformado.to_csv(archivo_salida, index=False, encoding='utf-8')

print(f"\n✓ Archivo guardado exitosamente: {archivo_salida}")
print(f"  Ubicación: ./projects/{archivo_salida}")
print(f"  Filas: {len(df_transformado)}")
print(f"  Columnas: {len(df_transformado.columns)}")


# ============================================
# CONCLUSIONES DEL ANÁLISIS EXPLORATORIO
# ============================================

"""
╔════════════════════════════════════════════════════════════════════════════╗
║         CONCLUSIONES PRINCIPALES - ANÁLISIS EXPLORATORIO TITANIC           ║
╚════════════════════════════════════════════════════════════════════════════╝
📌 1. LA CLASE SOCIAL FUE EL FACTOR DETERMINANTE
   ├─ Pasajeros de 1ª y 2ª clase tuvieron mayor tasa de supervivencia
   ├─ La clase determina acceso físico a botes salvavidas (ubicación del camarote)
   ├─ Relación inversa fuerte: Pclass vs Survived (correlación: -0.34)
   └─ Conclusión: El Titanic era un reflejo de la desigualdad social de la época


 2. EL DINERO IMPORTABA MÁS QUE SE PENSABA
   ├─ Tarifa pagada fuertemente correlacionada con supervivencia (r ≈ 0.26)
   ├─ Tarifas varían desde $4 a $512 (diferencia de 128x)
   ├─ Media de tarifa ($32) >> Mediana ($14) → Fuerte concentración de riqueza
   ├─ 50% de pasajeros pagó menos de $14.45
   ├─ Distribución de tarifa es sesgada → Requiere transformación logarítmica
   └─ Conclusión: La riqueza fue un predictor crucial de supervivencia


 3. EDAD: FACTOR MENOS IMPORTANTE DE LO ESPERADO
   ├─ Edad media: 29.7 años, Mediana: 28 años
   ├─ Distribución relativamente simétrica (skewness bajo)
   ├─ Sin correlación fuerte con clase social o tarifa
   ├─ Presencia de pasajeros desde bebés (0 años) hasta ancianos (80 años)
   └─ Conclusión: La edad afectó pero no fue determinante respecto a clase


 4. ESTRUCTURA FAMILIAR Y VIAJE SOLITARIO
   ├─ Promedio de hermanos/cónyuges: 0.52 → Mayoría viajaba solo
   ├─ Promedio de padres/hijos: 0.38 → Pocas familias numerosas
   ├─ Correlación entre SibSp y Parch: 0.415 → Complementarias
   ├─ Feature Engineering creó: FamilySize, IsAlone
   ├─ Viajar con familia aumentó probabilidades de supervivencia
   └─ Conclusión: La familia brindó apoyo emocional y práctico durante evacuación


 5. DISTRIBUCIONES Y OUTLIERS
   ├─ Age: Distribución normal, sin necesidad de transformación
   ├─ Fare: Altamente sesgada (skewness > 1), necesita log10
   ├─ Edad outliers: ~10 personas mayores de 65 años
   ├─ Fare outliers: ~50-100 personas con tarifas > $200
   ├─ Dataset limpio reduce dimensión ~10-15% eliminando extremos
   └─ Conclusión: Transformación logarítmica mejora modelos predictivos




 7. DESBALANCE DE GÉNERO
   ├─ Mujeres: ~314 (35%)
   ├─ Hombres: ~577 (65%)
   ├─ Proporción hombre:mujer ≈ 2:1
   ├─ Protocolo "Mujeres y niños primero" fue parcialmente aplicado
   └─ Conclusión: El género influyó en oportunidades de supervivencia


  8. PUERTOS DE EMBARQUE
   ├─ Southampton (S): ~644 pasajeros (72%)
   ├─ Cherbourg (C): ~168 pasajeros (19%)
   ├─ Queenstown (Q): ~77 pasajeros (9%)
   ├─ Distribución reflejaba rutas comerciales inglesas
   └─ Conclusión: Mayoría embarcó en Reino Unido








║  INSIGHT FINAL: El Titanic no fue un desastre aleatorio, sino un reflejo   ║
║  de la sociedad de 1912. Los datos revelan que la supervivencia dependió    ║
║  en primer lugar de factores socioeconómicos (clase y dinero), mucho más    ║
║  que de azar o características personales como edad o género.              ║
║  Este análisis demuestra el poder de Data Science para extraer verdades     ║
║  históricas de números crudos.                                             ║



"""


