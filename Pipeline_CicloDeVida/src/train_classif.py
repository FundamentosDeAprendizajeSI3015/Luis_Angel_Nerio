from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from .config import RANDOM_STATE


@dataclass
class ClassificationModels:
    models: Dict[str, Any]


def get_classification_models() -> ClassificationModels:
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE
        ),
        "DecisionTreeClassifier": DecisionTreeClassifier(
            random_state=RANDOM_STATE, max_depth=None
        ),
        "RandomForestClassifier": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_estimators=400,
            max_depth=None,
            n_jobs=-1
        ),
    }
    return ClassificationModels(models=models)


def train_classification_models(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Dict[str, Any]:
    clfs = get_classification_models().models
    fitted = {}
    for name, model in clfs.items():
        model.fit(X_train, y_train)
        fitted[name] = model
    return fitted


def predict_classification_models(
    fitted: Dict[str, Any], X_test: pd.DataFrame
) -> Dict[str, np.ndarray]:
    preds = {}
    for name, model in fitted.items():
        preds[name] = model.predict(X_test)
    return preds
