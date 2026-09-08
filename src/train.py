"""
Train every model configuration listed in config/config.yaml, evaluate
each on a held-out test set, and log everything (params, metrics, the
fitted pipeline, and a copy of the config) to MLflow.

Usage:
    python -m src.train
    python -m src.train --config config/config.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import yaml
from sklearn.pipeline import Pipeline

from src.evaluate import compute_metrics
from src.model_factory import build_estimator
from src.preprocessing import (
    build_preprocessor,
    load_and_prepare,
    train_test_split_data,
)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _data_fingerprint(raw_path: str) -> str:
    """A short hash of the raw data file's bytes, logged as an MLflow tag
    so every run records exactly which data version produced it."""
    data = Path(raw_path).read_bytes()
    return hashlib.sha256(data).hexdigest()[:12]


def run_one_config(run_cfg: dict, X_train, X_test, y_train, y_test, data_tag: dict):
    """Train + evaluate one model configuration inside its own MLflow run.
    Returns the run's metrics dict."""
    model_name = run_cfg["model"]
    params = run_cfg.get("params", {})

    with mlflow.start_run(run_name=run_cfg["name"]):
        mlflow.set_tags(
            {
                "model_family": model_name,
                **data_tag,
            }
        )
        mlflow.log_param("model", model_name)
        for key, value in params.items():
            mlflow.log_param(key, value)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", build_estimator(model_name, params)),
            ]
        )
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = None
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_proba)
        mlflow.log_metrics(metrics)

        # Log the fitted pipeline (preprocessing + model together) so the
        # LLM interface can load one artifact and call .predict() on raw
        # feature dicts.
        # cloudpickle (vs. the new default "skops" format) is used because
        # the pipeline embeds a ColumnTransformer/estimator combo that
        # skops' safe-deserialization allowlist doesn't fully cover yet;
        # these are artifacts we produce and load ourselves, not
        # untrusted third-party files, so pickle-based loading is fine.
        mlflow.sklearn.log_model(
            pipeline, name="model", serialization_format="cloudpickle"
        )

        print(
            f"[{run_cfg['name']:24s}] "
            + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        )
        return metrics, mlflow.active_run().info.run_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    X, y = load_and_prepare(cfg["data"]["raw_path"])
    X_train, X_test, y_train, y_test = train_test_split_data(
        X,
        y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
    )

    data_tag = {
        "data_source": cfg["data"]["raw_path"],
        "data_sha256_12": _data_fingerprint(cfg["data"]["raw_path"]),
        "n_train": str(len(X_train)),
        "n_test": str(len(X_test)),
    }

    results = []
    for run_cfg in cfg["runs"]:
        metrics, run_id = run_one_config(run_cfg, X_train, X_test, y_train, y_test, data_tag)
        results.append((run_cfg["name"], run_id, metrics))

    print("\n=== Summary (sorted by F1) ===")
    for name, run_id, metrics in sorted(
        results, key=lambda r: r[2]["f1"], reverse=True
    ):
        print(f"{name:24s} run_id={run_id}  f1={metrics['f1']:.4f}  acc={metrics['accuracy']:.4f}")

    return results


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
