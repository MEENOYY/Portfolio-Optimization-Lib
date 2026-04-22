from __future__ import annotations

from dataclasses import asdict
from datetime import date
from pathlib import Path
import tomllib

from .contracts import DatasetSpec, MlflowSettings


def _repo_root(repo_root: str | Path | None = None) -> Path:
    return Path("." if repo_root is None else repo_root).resolve()


def _parse_date(value: str) -> date:
    return date.fromisoformat(str(value))


def load_dataset_specs(repo_root: str | Path | None = None) -> dict[str, DatasetSpec]:
    config_path = _repo_root(repo_root) / "configs" / "datasets.toml"
    with config_path.open("rb") as handle:
        payload = tomllib.load(handle)
    specs: dict[str, DatasetSpec] = {}
    for dataset_name, values in payload.items():
        specs[dataset_name] = DatasetSpec(
            name=str(values.get("name", dataset_name)),
            tickers=[str(item).upper() for item in values.get("tickers", [])],
            benchmark_ticker=str(values.get("benchmark_ticker", "SPY")).upper(),
            start_date=_parse_date(values.get("start_date", "2014-01-02")),
            end_date=_parse_date(values.get("end_date", "2025-12-31")),
            train_start=_parse_date(values.get("train_start", "2014-01-02")),
            train_end=_parse_date(values.get("train_end", "2019-12-31")),
            val_start=_parse_date(values.get("val_start", "2020-01-02")),
            val_end=_parse_date(values.get("val_end", "2021-12-31")),
            test_start=_parse_date(values.get("test_start", "2022-01-03")),
            test_end=_parse_date(values.get("test_end", "2025-12-31")),
            cost_bps=float(values.get("cost_bps", 10.0)),
            default_benchmark=str(values.get("default_benchmark", "SPY")).upper(),
        )
    return specs


def get_dataset_spec(dataset_name: str, repo_root: str | Path | None = None) -> DatasetSpec:
    specs = load_dataset_specs(repo_root)
    try:
        return specs[dataset_name]
    except KeyError as exc:
        raise KeyError(f"unknown dataset preset '{dataset_name}'") from exc


def load_mlflow_settings(repo_root: str | Path | None = None) -> MlflowSettings:
    config_path = _repo_root(repo_root) / "configs" / "mlflow.toml"
    with config_path.open("rb") as handle:
        payload = tomllib.load(handle)
    return MlflowSettings(
        experiment_prefix=str(payload.get("experiment_prefix", "portfolio_toolkit")),
        tracking_uri=str(payload.get("tracking_uri", "sqlite:///mlflow/mlflow.db")),
        backend_store_uri=str(payload.get("backend_store_uri", "sqlite:///mlflow/mlflow.db")),
        artifact_root=str(payload.get("artifact_root", "mlflow/artifacts")),
        host=str(payload.get("host", "127.0.0.1")),
        port=int(payload.get("port", 5000)),
    )


def dataset_spec_dict(dataset_name: str, repo_root: str | Path | None = None) -> dict[str, object]:
    return asdict(get_dataset_spec(dataset_name, repo_root=repo_root))
