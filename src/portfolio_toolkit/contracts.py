from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(slots=True)
class DatasetSpec:
    name: str
    tickers: list[str]
    benchmark_ticker: str = "SPY"
    start_date: date = date(2014, 1, 2)
    end_date: date = date(2025, 12, 31)
    train_start: date = date(2014, 1, 2)
    train_end: date = date(2019, 12, 31)
    val_start: date = date(2020, 1, 2)
    val_end: date = date(2021, 12, 31)
    test_start: date = date(2022, 1, 3)
    test_end: date = date(2025, 12, 31)
    cost_bps: float = 10.0
    default_benchmark: str = "SPY"
    kind: str = "preset"
    dataset_id: str = ""

    @property
    def identifier(self) -> str:
        return self.dataset_id or self.name

    @property
    def all_tickers(self) -> list[str]:
        tickers = list(dict.fromkeys(ticker.upper() for ticker in self.tickers))
        benchmark = self.benchmark_ticker.upper()
        return tickers if benchmark in tickers else [*tickers, benchmark]


@dataclass(slots=True)
class MlflowSettings:
    experiment_prefix: str = "portfolio_toolkit"
    tracking_uri: str = "https://adams-macbook-pro.tail5ddc35.ts.net"
    backend_store_uri: str = "sqlite:///mlflow/mlflow.db"
    artifact_root: str = "mlflow/artifacts"
    host: str = "127.0.0.1"
    port: int = 5000

    def artifact_root_path(self, repo_root: str | Path = ".") -> Path:
        return Path(repo_root).resolve() / self.artifact_root


@dataclass(slots=True)
class PortfolioWeights:
    weights: pd.DataFrame
    dataset_name: str
    strategy_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BacktestResult:
    strategy_name: str
    dataset_name: str
    weights: pd.DataFrame
    nav: pd.Series
    returns: pd.Series
    turnover: pd.Series
    benchmark_returns: pd.DataFrame
    metrics: dict[str, float]
    artifact_paths: dict[str, str] = field(default_factory=dict)
