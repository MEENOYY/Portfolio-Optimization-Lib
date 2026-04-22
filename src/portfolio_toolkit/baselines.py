from __future__ import annotations

import numpy as np
import pandas as pd

from .config import get_dataset_spec
from .data import load_prices
from .features import build_features
from .portfolio import weights_from_predictions_rank_long_only
from .contracts import PortfolioWeights
from .splits import slice_split
from .validation import validate_weights_frame


def _daily_split_dates(
    prices: pd.DataFrame,
    dataset_name: str,
    split: str,
    *,
    repo_root: str | None = None,
) -> pd.DatetimeIndex:
    split_frame = slice_split(prices, dataset_name, split, repo_root=repo_root)
    return pd.DatetimeIndex(pd.to_datetime(split_frame["date"].sort_values().unique()))


def _equal_weight_frame(dates: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    value = 1.0 / len(tickers)
    frame = pd.DataFrame(value, index=dates, columns=tickers, dtype=float)
    frame.index.name = "date"
    return frame


def _inverse_volatility_frame(prices: pd.DataFrame, dates: pd.DatetimeIndex, tickers: list[str], lookback: int = 20) -> pd.DataFrame:
    returns = prices.loc[prices["ticker"].isin(tickers)].pivot(index="date", columns="ticker", values="adj_close").sort_index().pct_change()
    rows: list[pd.Series] = []
    for date_value in dates:
        history = returns.loc[returns.index <= date_value].tail(lookback)
        vols = history.std(ddof=0).replace(0.0, np.nan)
        usable = vols.dropna()
        if usable.empty:
            row = pd.Series(1.0 / len(tickers), index=tickers, dtype=float)
        else:
            inv = 1.0 / usable
            inv = inv / inv.sum()
            row = pd.Series(0.0, index=tickers, dtype=float)
            row.loc[inv.index] = inv.to_numpy(dtype=float)
        row.name = pd.Timestamp(date_value)
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.index = dates
    frame.index.name = "date"
    return frame


def baseline_weights(
    dataset_name: str,
    strategy_name: str,
    split: str = "test",
    *,
    repo_root: str | None = None,
) -> PortfolioWeights:
    spec = get_dataset_spec(dataset_name, repo_root=repo_root)
    if not spec.tickers:
        raise ValueError(f"dataset preset '{dataset_name}' has no tickers configured")
    prices = load_prices(dataset_name, repo_root=repo_root)
    dates = _daily_split_dates(prices, dataset_name, split, repo_root=repo_root)
    tickers = list(spec.tickers)

    if strategy_name == "equal_weight":
        weights = _equal_weight_frame(dates, tickers)
    elif strategy_name == "inverse_volatility":
        weights = _inverse_volatility_frame(prices, dates, tickers, lookback=20)
    elif strategy_name == "momentum_20d":
        feature_frame = build_features(prices, feature_names=["momentum_20d"])
        signal_frame = slice_split(feature_frame, dataset_name, split, repo_root=repo_root)
        signal_frame = signal_frame.loc[signal_frame["ticker"].isin(spec.tickers)].reset_index(drop=True)
        signal_frame = signal_frame.rename(columns={"momentum_20d": "expected_return"})
        signal_frame["horizon"] = 20
        weights_obj = weights_from_predictions_rank_long_only(
            signal_frame.loc[:, ["date", "ticker", "horizon", "expected_return"]],
            dataset_name=dataset_name,
            strategy_name="momentum_20d",
        )
        weights = weights_obj.weights
    else:
        raise KeyError(f"unknown baseline strategy '{strategy_name}'")

    weights = validate_weights_frame(weights, dataset_name=dataset_name, repo_root=repo_root)
    return PortfolioWeights(weights=weights, dataset_name=dataset_name, strategy_name=strategy_name, metadata={"split": split})
