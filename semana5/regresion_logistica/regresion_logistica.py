import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import reciprocal
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, roc_curve, roc_auc_score

# Definamos el "random_state" para que los resultados sean reproducibles:
random_state = 42

# Cambiemos la fuente de las gráficas de matplotlib:
plt.rc('font', family='serif', size=12)

# Obtener rutas
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'Heart_disease_statlog_processed.csv')

# Crear carpetas para resultados
graficas_dir = os.path.join(script_dir, 'graficas')
resultados_dir = os.path.join(script_dir, 'resultados')
os.makedirs(graficas_dir, exist_ok=True)
os.makedirs(resultados_dir, exist_ok=True)

# Configurar archivo de salida para resultados
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
resultados_file = os.path.join(resultados_dir, f'resultados_regresion_logistica_{timestamp}.txt')

# Redirigir salida estándar al archivo
original_stdout = sys.stdout
f_out = open(resultados_file, 'w', encoding='utf-8')
sys.stdout = f_out

# Definamos y carguemos nuestro dataset:
df = pd.read_csv(data_path)
X = df.drop('target', axis=1).values
y = df['target'].values

# Separemos nuestros datos en conjuntos de entrenamiento y prueba:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# Función auxiliar para graficar la frontera de decisión:
def plot_binary_classifcation(X, y, classifier=None, contour_alpha=0.1):
    
    def compute_predictions(X, classifier):
        return classifier.predict(X)

    cmap = ListedColormap(['#FF0000', '#0000FF'])
    
    # Como X tiene muchas dimensiones, usar PCA para reducir a 2D
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X)
    
    # plot the decision surface
    x1_min, x1_max = X_pca[:, 0].min() - 0.2, X_pca[:, 0].max() + 0.2
    x2_min, x2_max = X_pca[:, 1].min() - 0.2, X_pca[:, 1].max() + 0.2

    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, 0.02), np.arange(x2_min, x2_max, 0.02)
    )

    fig, ax = plt.subplots()

    if classifier is not None:
        # Entrenar un modelo en el espacio PCA para visualización
        X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
            X_pca, y, test_size=0.2, random_state=random_state
        )
        
        # Obtener los mejores parámetros del clasificador original
        best_params = classifier.best_params_
        model_pca = LogisticRegression(
            C=best_params['classifier__C'],
            penalty=best_params['classifier__penalty'],
            solver=best_params['classifier__solver'],
            max_iter=1000,
            random_state=random_state
        )
        model_pca.fit(X_train_pca, y_train_pca)
        
        X_ = np.array([xx1.ravel(), xx2.ravel()]).T
        z = compute_predictions(X_, model_pca)
        z = np.reshape(z, xx1.shape)
        ax.contourf(xx1, xx2, z, alpha=contour_alpha, cmap=cmap)

    ax.set_xlim(xx1.min(), xx1.max())
    ax.set_ylim(xx2.min(), xx2.max())

    ax.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap=cmap, s=20)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title('Frontera de Decisión - Heart Disease (Proyección PCA)')

    fig.set_size_inches(1.6*5, 5)

# Definamos un pipeline de scikit-learn con nuestro modelo base:
lr_base = Pipeline([
    ('classifier', LogisticRegression(max_iter=1000, random_state=random_state))
])

# Definamos las distribuciones de parámetros sobre las que haremos la búsqueda:
param_distributions = {
    'classifier__C': reciprocal(1e-5, 1e5),
    'classifier__penalty': ['l2'],
    'classifier__solver': ['lbfgs', 'liblinear']
}

# Definamos nuestro modelo mediante RandomizedSearchCV:
lr = RandomizedSearchCV(
    lr_base,
    cv=4,
    param_distributions=param_distributions,
    n_iter=50,
    random_state=random_state
)

# Entrenemos el modelo:
lr.fit(X_train, y_train)

# Obtengamos los mejores hiperparámetros encontrados para el modelo:
print("Mejores parámetros:")
print(lr.best_params_)

# Obtengamos la accuracy y el f1-score de prueba:
print(f'Accuracy: {lr.score(X_test, y_test)}')
print(f'F1 score: {f1_score(y_test, lr.predict(X_test))}')

# Grafiquemos los datos junto con la frontera de decisión del modelo:
plot_binary_classifcation(X, y, classifier=lr)
plt.savefig(os.path.join(graficas_dir, 'frontera_decision.png'), dpi=300, bbox_inches='tight')
plt.close()

# Grafiquemos la matriz de confusión de los datos de prueba:
cm = confusion_matrix(y_test, lr.predict(X_test))
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.savefig(os.path.join(graficas_dir, 'matriz_confusion.png'), dpi=300, bbox_inches='tight')
plt.close()

# Cerrar archivo de resultados
f_out.close()
sys.stdout = original_stdout
print(f'\nResultados guardados en: {resultados_file}')
print(f'Gráficas guardadas en: {graficas_dir}')
