from __future__ import annotations

from dataclasses import asdict
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import re
import tomllib
from typing import Iterable

from .contracts import DatasetSpec, MlflowSettings


def _repo_root(repo_root: str | Path | None = None) -> Path:
    return Path("." if repo_root is None else repo_root).resolve()


def _parse_date(value: str) -> date:
    return date.fromisoformat(str(value))


def _parse_runtime_date(value: str | date) -> date:
    return value if isinstance(value, date) else _parse_date(str(value))


def _normalize_identifier(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_").lower()
    if not cleaned:
        raise ValueError("dataset name must contain at least one alphanumeric character")
    return cleaned


def _normalize_tickers(tickers: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for item in tickers:
        ticker = str(item).strip().upper()
        if not ticker:
            raise ValueError("tickers cannot contain empty values")
        if ticker not in seen:
            normalized.append(ticker)
            seen.add(ticker)
    if not normalized:
        raise ValueError("tickers cannot be empty")
    return normalized


def _allocate_split_days(total_days: int) -> tuple[int, int, int]:
    if total_days < 3:
        raise ValueError("custom datasets must span at least 3 calendar days")
    train_days = max(1, int(total_days * 0.6))
    val_days = max(1, int(total_days * 0.2))
    test_days = total_days - train_days - val_days
    while test_days < 1:
        if train_days >= val_days and train_days > 1:
            train_days -= 1
        elif val_days > 1:
            val_days -= 1
        else:
            raise ValueError("unable to allocate non-empty train/val/test splits")
        test_days = total_days - train_days - val_days
    return train_days, val_days, test_days


def _runtime_split_boundaries(start_date: date, end_date: date) -> dict[str, date]:
    total_days = (end_date - start_date).days + 1
    train_days, val_days, _ = _allocate_split_days(total_days)
    train_start = start_date
    train_end = date.fromordinal(train_start.toordinal() + train_days - 1)
    val_start = date.fromordinal(train_end.toordinal() + 1)
    val_end = date.fromordinal(val_start.toordinal() + val_days - 1)
    test_start = date.fromordinal(val_end.toordinal() + 1)
    return {
        "train_start": train_start,
        "train_end": train_end,
        "val_start": val_start,
        "val_end": val_end,
        "test_start": test_start,
        "test_end": end_date,
    }


def _custom_dataset_identifier(tickers: Iterable[str], start_date: date, end_date: date, benchmark: str) -> str:
    payload = json.dumps(
        {
            "benchmark": str(benchmark).upper(),
            "end": end_date.isoformat(),
            "start": start_date.isoformat(),
            "tickers": sorted(_normalize_tickers(tickers)),
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"custom_{digest}"


def custom_dataset(
    tickers: Iterable[str],
    start: str | date,
    end: str | date,
    benchmark: str = "SPY",
    name: str | None = None,
    cost_bps: float = 10.0,
) -> DatasetSpec:
    normalized_tickers = _normalize_tickers(tickers)
    benchmark_ticker = str(benchmark).strip().upper()
    if not benchmark_ticker:
        raise ValueError("benchmark cannot be empty")
    start_date = _parse_runtime_date(start)
    end_date = _parse_runtime_date(end)
    if end_date < start_date:
        raise ValueError("end date must be on or after start date")
    identifier = _normalize_identifier(name) if name is not None else _custom_dataset_identifier(
        normalized_tickers, start_date, end_date, benchmark_ticker
    )
    splits = _runtime_split_boundaries(start_date, end_date)
    return DatasetSpec(
        name=identifier,
        tickers=normalized_tickers,
        benchmark_ticker=benchmark_ticker,
        start_date=start_date,
        end_date=end_date,
        train_start=splits["train_start"],
        train_end=splits["train_end"],
        val_start=splits["val_start"],
        val_end=splits["val_end"],
        test_start=splits["test_start"],
        test_end=splits["test_end"],
        cost_bps=float(cost_bps),
        default_benchmark=benchmark_ticker,
        kind="custom",
        dataset_id=identifier,
    )


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
            kind="preset",
            dataset_id=dataset_name,
        )
    return specs


def get_dataset_spec(dataset_name: str, repo_root: str | Path | None = None) -> DatasetSpec:
    specs = load_dataset_specs(repo_root)
    try:
        return specs[dataset_name]
    except KeyError as exc:
        raise KeyError(f"unknown dataset preset '{dataset_name}'") from exc


def resolve_dataset_spec(dataset_name: str | DatasetSpec, repo_root: str | Path | None = None) -> DatasetSpec:
    if isinstance(dataset_name, DatasetSpec):
        return dataset_name
    return get_dataset_spec(dataset_name, repo_root=repo_root)


def dataset_identifier(dataset_name: str | DatasetSpec, repo_root: str | Path | None = None) -> str:
    if isinstance(dataset_name, str):
        return dataset_name
    return resolve_dataset_spec(dataset_name, repo_root=repo_root).identifier


def dataset_kind(dataset_name: str | DatasetSpec, repo_root: str | Path | None = None) -> str:
    return resolve_dataset_spec(dataset_name, repo_root=repo_root).kind


def load_mlflow_settings(repo_root: str | Path | None = None) -> MlflowSettings:
    config_path = _repo_root(repo_root) / "configs" / "mlflow.toml"
    with config_path.open("rb") as handle:
        payload = tomllib.load(handle)
    tracking_uri = str(payload.get("tracking_uri", "https://adams-macbook-pro.tail5ddc35.ts.net"))
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", tracking_uri)
    return MlflowSettings(
        experiment_prefix=str(payload.get("experiment_prefix", "portfolio_toolkit")),
        tracking_uri=tracking_uri,
        backend_store_uri=str(payload.get("backend_store_uri", "sqlite:///mlflow/mlflow.db")),
        artifact_root=str(payload.get("artifact_root", "mlflow/artifacts")),
        host=str(payload.get("host", "127.0.0.1")),
        port=int(payload.get("port", 5000)),
    )


def dataset_spec_dict(dataset_name: str | DatasetSpec, repo_root: str | Path | None = None) -> dict[str, object]:
    payload = asdict(resolve_dataset_spec(dataset_name, repo_root=repo_root))
    for key, value in payload.items():
        if isinstance(value, date):
            payload[key] = value.isoformat()
    return payload
