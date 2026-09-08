"""
Programmatically query MLflow for the best run in the experiment and
export its model artifact for the LLM interface to load.

This satisfies the "Experiment tracking" requirement that a script use
``mlflow.search_runs()`` to compare experiments and pick a winner --
rather than a human eyeballing the MLflow UI.

Usage:
    python -m src.select_best_model
    python -m src.select_best_model --metric roc_auc
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlflow
import yaml

from src.preprocessing import reference_statistics


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def find_best_run(experiment_name: str, metric: str = "f1") -> "mlflow.entities.Run":
    """Query every run in the experiment and return the one with the
    highest value of `metric`, using mlflow.search_runs()."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(
            f"Experiment '{experiment_name}' not found. Run `python -m src.train` first."
        )

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
    )
    if runs_df.empty:
        raise RuntimeError(
            f"No runs found in experiment '{experiment_name}'. Run `python -m src.train` first."
        )

    best = runs_df.iloc[0]
    print(f"Compared {len(runs_df)} runs on metric '{metric}':")
    cols = ["tags.mlflow.runName", f"metrics.{metric}", "metrics.accuracy", "metrics.roc_auc"]
    cols = [c for c in cols if c in runs_df.columns]
    print(runs_df[cols].sort_values(f"metrics.{metric}", ascending=False).to_string(index=False))
    print(
        f"\nBest run: {best.get('tags.mlflow.runName')} "
        f"(run_id={best['run_id']}, {metric}={best[f'metrics.{metric}']:.4f})"
    )
    return best


def export_model(run_id: str, dest_dir: str = "models/production") -> str:
    """Download the logged 'model' artifact for `run_id` into `dest_dir`
    so the interface app doesn't need a live MLflow tracking connection
    at inference time."""
    dest = Path(dest_dir)
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="model", dst_path=str(dest.parent)
    )
    # download_artifacts recreates the "model" subfolder name; normalize it.
    downloaded = Path(local_path)
    if downloaded != dest:
        if dest.exists():
            shutil.rmtree(dest)
        downloaded.rename(dest)
    return str(dest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--metric",
        default="f1",
        help="Metric to rank runs by (default: f1). Any logged metric works, e.g. roc_auc.",
    )
    parser.add_argument("--dest", default="models/production")
    args = parser.parse_args()

    cfg = load_config(args.config)
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])

    best_run = find_best_run(cfg["mlflow"]["experiment_name"], metric=args.metric)
    dest = export_model(best_run["run_id"], args.dest)

    stats = reference_statistics(cfg["data"]["raw_path"])
    stats["selected_run_id"] = best_run["run_id"]
    stats["selected_run_name"] = best_run.get("tags.mlflow.runName")
    stats["selection_metric"] = args.metric
    stats["selection_metric_value"] = float(best_run[f"metrics.{args.metric}"])
    stats_path = Path(dest).parent / "reference_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))

    print(f"\nExported best model to: {dest}")
    print(f"Exported reference stats to: {stats_path}")
    print("The LLM interface (src/interface.py) loads both from these paths.")


if __name__ == "__main__":
    main()
