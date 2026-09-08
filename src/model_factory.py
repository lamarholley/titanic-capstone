"""Maps a config-file model name to an actual (unfitted) estimator
instance. Keeping this in one place means src/train.py never hardcodes a
hyperparameter -- everything comes from config/config.yaml.
"""
from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

_MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "xgboost": XGBClassifier,
}


def build_estimator(model_name: str, params: dict):
    """Instantiate an estimator by its config-file name with the given
    hyperparameters. Raises ValueError for an unknown name so a typo in
    config.yaml fails loudly instead of silently training the wrong model.
    """
    if model_name not in _MODEL_REGISTRY:
        known = ", ".join(sorted(_MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{model_name}'. Known models: {known}")
    params = dict(params or {})
    return _MODEL_REGISTRY[model_name](**params)
