"""Model validation tests: prediction shape/type, and a minimum
performance bar on held-out data.

These tests train a small, fast pipeline on the real dataset directly
(rather than depending on `models/production/`, which is produced by the
separate `python -m src.train` + `python -m src.select_best_model`
pipeline) so `pytest tests/ -v` is hermetic and doesn't depend on run
order or an mlflow database being present.
"""
import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from src.evaluate import compute_metrics
from src.model_factory import build_estimator
from src.preprocessing import (
    build_preprocessor,
    load_and_prepare,
    train_test_split_data,
)

RAW_DATA_PATH = "data/raw/titanic.csv"


@pytest.fixture(scope="module")
def fitted_pipeline_and_split():
    X, y = load_and_prepare(RAW_DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", build_estimator("logistic_regression", {"max_iter": 1000, "random_state": 42})),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline, X_test, y_test


def test_predictions_have_correct_type_and_shape(fitted_pipeline_and_split):
    pipeline, X_test, y_test = fitted_pipeline_and_split

    preds = pipeline.predict(X_test)
    assert preds.shape == (len(X_test),)
    assert set(np.unique(preds)).issubset({0, 1})

    proba = pipeline.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    # Every row's class probabilities must sum to 1.
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_model_beats_naive_baseline_on_held_out_data(fitted_pipeline_and_split):
    """A trained model must clear a real bar, not just 'not crash'. The
    naive baseline of always predicting the majority class ("did not
    survive") scores ~0.62 accuracy on this dataset, so 0.78 is a
    meaningful, achievable improvement -- this also guards against a
    future refactor silently breaking the training pipeline.
    """
    pipeline, X_test, y_test = fitted_pipeline_and_split
    preds = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, preds, proba)

    majority_baseline_accuracy = max(y_test.mean(), 1 - y_test.mean())
    assert metrics["accuracy"] > majority_baseline_accuracy
    assert metrics["accuracy"] >= 0.78
    assert metrics["roc_auc"] >= 0.80


def test_unknown_model_name_raises_clear_error():
    with pytest.raises(ValueError, match="Unknown model"):
        build_estimator("not_a_real_model", {})


def test_xgboost_estimator_builds_and_predicts():
    """Smoke test for the gradient-boosted config used in production
    (xgboost was the winning run in experiment tracking)."""
    X, y = load_and_prepare(RAW_DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                build_estimator(
                    "xgboost",
                    {"n_estimators": 50, "max_depth": 3, "random_state": 42, "eval_metric": "logloss"},
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    assert preds.shape == (len(X_test),)
    assert set(np.unique(preds)).issubset({0, 1})
