import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime

from scipy.stats import reciprocal

from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt

# Definamos el "random_state" para que los resultados sean reproducibles:
random_state = 42
np.random.seed(random_state)

# Cambiemos la fuente de las gráficas de matplotlib:
plt.rc('font', family='serif', size=12)

# Obtener rutas
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'Sport_car_price_clean.csv')

# Crear carpetas para resultados
graficas_dir = os.path.join(script_dir, 'graficas')
resultados_dir = os.path.join(script_dir, 'resultados')
os.makedirs(graficas_dir, exist_ok=True)
os.makedirs(resultados_dir, exist_ok=True)

# Configurar archivo de salida para resultados
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
resultados_file = os.path.join(resultados_dir, f'resultados_regresion_lineal_{timestamp}.txt')

# Redirigir salida estándar al archivo
original_stdout = sys.stdout
f_out = open(resultados_file, 'w', encoding='utf-8')
sys.stdout = f_out

# Definamos y carguemos nuestro dataset:
df = pd.read_csv(data_path)
X = df.drop('Price (in USD)', axis=1).values
y = df['Price (in USD)'].values

# Separemos nuestros datos en conjuntos de entrenamiento y prueba:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# Grafiquemos los datos junto con el modelo real:
# Usamos Horsepower (columna 2) para visualización
fig, ax = plt.subplots()
ax.scatter(X_train[:, 2], y_train, c='c', label='Training data', alpha=0.6)
ax.scatter(X_test[:, 2], y_test, c='m', label='Test data', alpha=0.6)
ax.set_xlabel('Horsepower')
ax.set_ylabel('Price (in USD)')
ax.set_title('Sport Car Dataset')
ax.legend()
fig.set_size_inches(1.6*5, 5)
plt.savefig(os.path.join(graficas_dir, 'datos_entrenamiento.png'), dpi=300, bbox_inches='tight')
plt.close()

# Definamos pipelines de scikit-learn con nuestros modelos base:

ridge_base = Pipeline([
    ('poly', PolynomialFeatures(include_bias=False)),
    ('scaler', StandardScaler()),
    ('regressor', Ridge())
])

lasso_base = Pipeline([
    ('poly', PolynomialFeatures(include_bias=False)),
    ('scaler', StandardScaler()),
    ('regressor', Lasso())
])

# Definamos las distribuciones de parámetros sobre las que haremos la búsqueda:
param_distributions = {
    'poly__degree': list(range(1, 4)),
    'regressor__alpha': reciprocal(1e-5, 1e3)
}

# Definamos nuestros modelos mediante RandomizedSearchCV:

ridge = RandomizedSearchCV(
    ridge_base,
    cv=4,
    param_distributions=param_distributions,
    n_iter=200,
    random_state=random_state
)

lasso = RandomizedSearchCV(
    lasso_base,
    cv=4,
    param_distributions=param_distributions,
    n_iter=200,
    random_state=random_state
)

# Entrenemos los modelos:
ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)

# Obtengamos los mejores hiperparámetros encontrados para el modelo ridge:
print("Mejores parámetros Ridge:")
print(ridge.best_params_)

# Obtengamos los mejores hiperparámetros encontrados para el modelo LASSO:
print("\nMejores parámetros LASSO:")
print(lasso.best_params_)

# Obtengamos el R^2 y el MAE de prueba para el modelo ridge:
print('\nModelo ridge')
print(f'R^2: {ridge.score(X_test, y_test)}')
print(f'MAE: {mean_absolute_error(y_test, ridge.predict(X_test))}')

# Obtenamos el R^2 y el MAE para el modelo LASSO:
print('\nModelo LASSO')
print(f'R^2: {lasso.score(X_test, y_test)}')
print(f'MAE: {mean_absolute_error(y_test, lasso.predict(X_test))}')

# Grafiquemos los datos junto con el modelo predicho por la regresión ridge:
# Ordenar por Horsepower para línea suave
sort_idx = X_test[:, 2].argsort()
X_test_sorted = X_test[sort_idx]
y_test_sorted = y_test[sort_idx]

fig, ax = plt.subplots()
ax.scatter(X_train[:, 2], y_train, c='c', label='Training data', alpha=0.5)
ax.scatter(X_test[:, 2], y_test, c='m', label='Test data', alpha=0.6, s=80)
ax.plot(X_test_sorted[:, 2], ridge.predict(X_test_sorted), c='r', 
        label='Predicted model (ridge)', linewidth=2)
ax.set_xlabel('Horsepower')
ax.set_ylabel('Price (in USD)')
ax.set_title('Ridge Regression - Sport Car Price Prediction')
ax.legend()
fig.set_size_inches(1.6*5, 5)
plt.savefig(os.path.join(graficas_dir, 'prediccion_ridge.png'), dpi=300, bbox_inches='tight')
plt.close()

# Grafiquemos los datos junto con el modelo predicho por la regresión LASSO:
fig, ax = plt.subplots()
ax.scatter(X_train[:, 2], y_train, c='c', label='Training data', alpha=0.5)
ax.scatter(X_test[:, 2], y_test, c='m', label='Test data', alpha=0.6, s=80)
ax.plot(X_test_sorted[:, 2], lasso.predict(X_test_sorted), c='r', 
        label='Predicted model (LASSO)', linewidth=2)
ax.set_xlabel('Horsepower')
ax.set_ylabel('Price (in USD)')
ax.set_title('LASSO Regression - Sport Car Price Prediction')
ax.legend()
fig.set_size_inches(1.6*5, 5)
plt.savefig(os.path.join(graficas_dir, 'prediccion_lasso.png'), dpi=300, bbox_inches='tight')
plt.close()

# Obtengamos los coeficientes y el intercepto de la regresión ridge:
print('\nModelo ridge')
print(f"coeficientes: {ridge.best_estimator_['regressor'].coef_}")
print(f"intercepto: {ridge.best_estimator_['regressor'].intercept_}")

# Obtengamos los coeficientes y el intercepto de la regresión LASSO:
print('\nModelo LASSO')
print(f"coeficientes: {lasso.best_estimator_['regressor'].coef_}")
print(f"intercepto: {lasso.best_estimator_['regressor'].intercept_}")

# Cerrar archivo de resultados
f_out.close()
sys.stdout = original_stdout
print(f'\nResultados guardados en: {resultados_file}')
print(f'Gráficas guardadas en: {graficas_dir}')
