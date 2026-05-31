import os
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf


def configure_mlflow_s3(cfg: DictConfig) -> None:
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", cfg.mlflow.s3_endpoint_url)
    os.environ.setdefault("AWS_ACCESS_KEY_ID", cfg.mlflow.aws_access_key_id)
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", cfg.mlflow.aws_secret_access_key)


def flatten_config(cfg: DictConfig) -> dict[str, Any]:
    container = OmegaConf.to_container(cfg, resolve=True)
    flat: dict[str, Any] = {}

    def _walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                _walk(next_prefix, nested)
        else:
            flat[prefix] = value

    _walk("", container)
    return flat


def log_config_params(cfg: DictConfig) -> None:
    params = flatten_config(cfg)
    safe_params = {key: str(value) for key, value in params.items() if value is not None}
    mlflow.log_params(safe_params)


def tag_best_run_as_prd(experiment_name: str, current_run_id: str, metric_name: str = "weighted_f1") -> None:
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        order_by=[f"metrics.{metric_name} DESC"],
        max_results=50,
    )
    if not runs:
        return

    best_run = runs[0]
    if best_run.info.run_id != current_run_id:
        return

    version_number = sum(1 for run in runs if run.data.tags.get("PRD") == "Production") + 1
    for run in runs:
        if run.data.tags.get("PRD") == "Production" and run.info.run_id != current_run_id:
            client.set_tag(run.info.run_id, "PRD", "Archived")

    client.set_tag(current_run_id, "PRD", "Production")
    client.set_tag(current_run_id, "version", f"v{version_number}")


def find_prd_run_id(experiment_name: str) -> str | None:
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.PRD = 'Production'",
        order_by=["metrics.weighted_f1 DESC"],
        max_results=1,
    )
    if not runs:
        return None
    return runs[0].info.run_id
