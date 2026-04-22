from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import PortfolioWeights
from .validation import validate_prediction_frame, validate_weights_frame


def _empty_weights(dates: pd.Index, tickers: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(0.0, index=pd.DatetimeIndex(pd.to_datetime(dates)), columns=[ticker.upper() for ticker in tickers])
    frame.index.name = "date"
    return frame


def weights_from_predictions_top_k_equal(
    predictions: pd.DataFrame,
    k: int,
    score_column: str = "expected_return",
    *,
    dataset_name: str = "unknown",
    strategy_name: str | None = None,
) -> PortfolioWeights:
    if k <= 0:
        raise ValueError("k must be positive")
    validated = validate_prediction_frame(predictions)
    if score_column not in validated.columns:
        raise ValueError(f"score column '{score_column}' is missing from predictions")
    tickers = sorted(validated["ticker"].unique().tolist())
    weights = _empty_weights(validated["date"].drop_duplicates(), tickers)
    for date_value, frame in validated.groupby("date", sort=True):
        chosen = frame.sort_values(score_column, ascending=False).head(k)
        if chosen.empty:
            continue
        value = 1.0 / len(chosen)
        weights.loc[pd.Timestamp(date_value), chosen["ticker"]] = value
    weights = validate_weights_frame(weights)
    return PortfolioWeights(weights=weights, dataset_name=dataset_name, strategy_name=strategy_name or f"top_{k}_equal")


def weights_from_predictions_rank_long_only(
    predictions: pd.DataFrame,
    score_column: str = "expected_return",
    *,
    dataset_name: str = "unknown",
    strategy_name: str = "rank_long_only",
) -> PortfolioWeights:
    validated = validate_prediction_frame(predictions)
    if score_column not in validated.columns:
        raise ValueError(f"score column '{score_column}' is missing from predictions")
    tickers = sorted(validated["ticker"].unique().tolist())
    weights = _empty_weights(validated["date"].drop_duplicates(), tickers)
    for date_value, frame in validated.groupby("date", sort=True):
        ranked = frame.sort_values(score_column, ascending=False).reset_index(drop=True)
        raw_scores = pd.Series(np.arange(len(ranked), 0, -1, dtype=float), index=ranked["ticker"])
        raw_scores = raw_scores / raw_scores.sum()
        weights.loc[pd.Timestamp(date_value), raw_scores.index] = raw_scores.to_numpy(dtype=float)
    weights = validate_weights_frame(weights)
    return PortfolioWeights(weights=weights, dataset_name=dataset_name, strategy_name=strategy_name)


def weights_from_predictions_risk_adjusted(
    predictions: pd.DataFrame,
    return_col: str = "expected_return",
    vol_col: str = "expected_volatility",
    *,
    dataset_name: str = "unknown",
    strategy_name: str = "risk_adjusted",
) -> PortfolioWeights:
    validated = validate_prediction_frame(predictions)
    if return_col not in validated.columns:
        raise ValueError(f"return column '{return_col}' is missing from predictions")
    if vol_col not in validated.columns:
        raise ValueError(f"volatility column '{vol_col}' is missing from predictions")
    tickers = sorted(validated["ticker"].unique().tolist())
    weights = _empty_weights(validated["date"].drop_duplicates(), tickers)
    for date_value, frame in validated.groupby("date", sort=True):
        working = frame.copy()
        working[vol_col] = working[vol_col].replace(0.0, np.nan)
        working["risk_adjusted_score"] = (working[return_col] / working[vol_col]).replace([np.inf, -np.inf], np.nan)
        positive = working.loc[working["risk_adjusted_score"] > 0.0, ["ticker", "risk_adjusted_score"]].dropna()
        if positive.empty:
            equal_value = 1.0 / len(working)
            weights.loc[pd.Timestamp(date_value), working["ticker"]] = equal_value
            continue
        scores = positive.set_index("ticker")["risk_adjusted_score"]
        scores = scores / scores.sum()
        weights.loc[pd.Timestamp(date_value), scores.index] = scores.to_numpy(dtype=float)
    weights = validate_weights_frame(weights)
    return PortfolioWeights(weights=weights, dataset_name=dataset_name, strategy_name=strategy_name)
