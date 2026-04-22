from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import get_dataset_spec


PRICE_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
PREDICTION_REQUIRED_COLUMNS = ["date", "ticker", "horizon", "expected_return"]
OPTIONAL_PREDICTION_COLUMNS = ["expected_alpha", "expected_volatility", "uncertainty"]


def _normalize_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True).dt.tz_localize(None)


def _normalize_tickers(series: pd.Series) -> pd.Series:
    return series.astype(str).str.upper()


def _ensure_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def validate_prices_frame(
    df: pd.DataFrame,
    dataset_name: str | None = None,
    *,
    repo_root: str | Path | None = None,
) -> pd.DataFrame:
    _ensure_columns(df, PRICE_COLUMNS, "prices frame")
    validated = df.copy()
    validated.columns.name = None
    validated["date"] = _normalize_dates(validated["date"])
    validated["ticker"] = _normalize_tickers(validated["ticker"])
    for column in ["open", "high", "low", "close", "adj_close", "volume"]:
        validated[column] = pd.to_numeric(validated[column], errors="raise")
    if validated.duplicated(["date", "ticker"]).any():
        raise ValueError("prices frame contains duplicate date/ticker rows")
    if (validated[["open", "high", "low", "close", "adj_close", "volume"]] < 0).any().any():
        raise ValueError("prices frame contains negative OHLCV values")
    if (validated["high"] < validated["low"]).any():
        raise ValueError("prices frame contains rows where high < low")
    if dataset_name is not None:
        spec = get_dataset_spec(dataset_name, repo_root=repo_root)
        allowed = set(spec.all_tickers)
        unexpected = sorted(set(validated["ticker"]) - allowed)
        if unexpected:
            raise ValueError(f"prices frame contains unexpected tickers for {dataset_name}: {unexpected}")
        if spec.benchmark_ticker not in set(validated["ticker"]):
            raise ValueError(f"prices frame must include benchmark ticker {spec.benchmark_ticker}")
    validated = validated.sort_values(["ticker", "date"]).reset_index(drop=True)
    validated.columns.name = None
    return validated


def validate_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    _ensure_columns(df, ["date", "ticker"], "feature frame")
    validated = df.copy()
    validated.columns.name = None
    validated["date"] = _normalize_dates(validated["date"])
    validated["ticker"] = _normalize_tickers(validated["ticker"])
    if validated.duplicated(["date", "ticker"]).any():
        raise ValueError("feature frame contains duplicate date/ticker rows")
    feature_columns = [column for column in validated.columns if column not in {"date", "ticker"}]
    if not feature_columns:
        raise ValueError("feature frame must contain at least one feature column")
    validated = validated.sort_values(["ticker", "date"]).reset_index(drop=True)
    validated.columns.name = None
    return validated


def validate_prediction_frame(
    df: pd.DataFrame,
    dataset_name: str | None = None,
    horizon: int | None = None,
    *,
    repo_root: str | Path | None = None,
) -> pd.DataFrame:
    _ensure_columns(df, PREDICTION_REQUIRED_COLUMNS, "prediction frame")
    validated = df.copy()
    validated["date"] = _normalize_dates(validated["date"])
    validated["ticker"] = _normalize_tickers(validated["ticker"])
    validated["horizon"] = pd.to_numeric(validated["horizon"], errors="raise").astype(int)
    validated["expected_return"] = pd.to_numeric(validated["expected_return"], errors="raise")
    for column in OPTIONAL_PREDICTION_COLUMNS:
        if column in validated.columns:
            validated[column] = pd.to_numeric(validated[column], errors="coerce")
    if (validated["horizon"] <= 0).any():
        raise ValueError("prediction horizon must be positive")
    if validated["expected_return"].isna().any():
        raise ValueError("prediction frame contains null expected_return values")
    if validated.duplicated(["date", "ticker", "horizon"]).any():
        raise ValueError("prediction frame contains duplicate date/ticker/horizon rows")
    if horizon is not None and set(validated["horizon"].unique()) != {int(horizon)}:
        raise ValueError(f"prediction frame must contain exactly horizon={horizon}")
    if dataset_name is not None:
        spec = get_dataset_spec(dataset_name, repo_root=repo_root)
        allowed = set(spec.tickers)
        unexpected = sorted(set(validated["ticker"]) - allowed)
        if unexpected:
            raise ValueError(f"prediction frame contains unexpected tickers for {dataset_name}: {unexpected}")
    return validated.sort_values(["date", "ticker", "horizon"]).reset_index(drop=True)


def validate_weights_frame(
    df: pd.DataFrame,
    dataset_name: str | None = None,
    *,
    repo_root: str | Path | None = None,
) -> pd.DataFrame:
    if df.empty:
        raise ValueError("weights frame cannot be empty")
    validated = df.copy()
    validated.index = pd.to_datetime(validated.index, utc=True).tz_localize(None)
    validated.index.name = "date"
    validated.columns = [str(column).upper() for column in validated.columns]
    validated = validated.sort_index()
    numeric = validated.apply(pd.to_numeric, errors="raise")
    if numeric.isna().any().any():
        raise ValueError("weights frame cannot contain null values")
    if (numeric < 0).any().any() or (numeric > 1).any().any():
        raise ValueError("weights frame must be long-only with values in [0, 1]")
    row_sums = numeric.sum(axis=1)
    if not np.allclose(row_sums.to_numpy(dtype=float), np.ones(len(row_sums)), atol=1e-6):
        raise ValueError("each weights row must sum to 1.0")
    if dataset_name is not None:
        spec = get_dataset_spec(dataset_name, repo_root=repo_root)
        unexpected = sorted(set(numeric.columns) - set(spec.tickers))
        if unexpected:
            raise ValueError(f"weights frame contains unexpected tickers for {dataset_name}: {unexpected}")
    return numeric
