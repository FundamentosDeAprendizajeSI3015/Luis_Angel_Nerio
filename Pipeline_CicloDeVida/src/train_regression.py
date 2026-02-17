from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from .config import RANDOM_STATE


@dataclass
class RegressionModels:
    models: Dict[str, Any]


def get_regression_models() -> RegressionModels:
    models = {
        "LinearRegression": LinearRegression(),
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=None),
        "RandomForestRegressor": RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_estimators=300,
            max_depth=None,
            n_jobs=-1
        ),
    }
    return RegressionModels(models=models)


def train_regression_models(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Dict[str, Any]:
    regs = get_regression_models().models
    fitted = {}
    for name, model in regs.items():
        model.fit(X_train, y_train)
        fitted[name] = model
    return fitted


def predict_regression_models(
    fitted: Dict[str, Any], X_test: pd.DataFrame
) -> Dict[str, np.ndarray]:
    preds = {}
    for name, model in fitted.items():
        preds[name] = model.predict(X_test)
    return preds
