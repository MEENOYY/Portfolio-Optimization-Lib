from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd

from .validation import validate_prices_frame, validate_feature_frame


FEATURE_NAMES = [
    "return_1d",
    "return_2d",
    "return_5d",
    "return_10d",
    "return_20d",
    "return_60d",
    "log_return_1d",
    "log_return_5d",
    "log_return_20d",
    "vol_5d",
    "vol_10d",
    "vol_20d",
    "vol_60d",
    "downside_vol_20d",
    "upside_vol_20d",
    "atr_14",
    "beta_20d_spy",
    "beta_60d_spy",
    "momentum_5d",
    "momentum_10d",
    "momentum_20d",
    "momentum_60d",
    "momentum_120d",
    "price_to_sma_10d",
    "price_to_sma_20d",
    "price_to_sma_50d",
    "price_to_sma_200d",
    "price_to_ema_12d",
    "price_to_ema_26d",
    "macd",
    "macd_signal",
    "macd_hist",
    "rsi_7",
    "rsi_14",
    "rsi_28",
    "bollinger_z_20d",
    "stoch_k_14",
    "stoch_d_14",
    "volume_change_1d",
    "volume_change_5d",
    "volume_change_20d",
    "volume_zscore_20d",
    "volume_zscore_60d",
    "dollar_volume",
    "dollar_volume_ratio_20d",
    "intraday_range",
    "close_open_gap",
    "close_location_in_range",
    "distance_to_20d_high",
    "distance_to_20d_low",
    "distance_to_60d_high",
    "distance_to_60d_low",
    "excess_return_5d_vs_spy",
    "excess_return_20d_vs_spy",
    "excess_return_60d_vs_spy",
    "relative_momentum_20d_vs_spy",
    "skew_20d",
    "kurtosis_20d",
]


def list_features() -> list[str]:
    return list(FEATURE_NAMES)


def _rolling_apply(values: pd.Series, window: int, func) -> pd.Series:
    return values.rolling(window, min_periods=window).apply(func, raw=False)


def _rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.rolling(window, min_periods=window).mean()
    avg_loss = losses.rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _prepare_benchmark_returns(prices: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    benchmark_prices = (
        prices.loc[prices["ticker"] == benchmark.upper(), ["date", "adj_close"]]
        .rename(columns={"adj_close": "benchmark_adj_close"})
        .sort_values("date")
    )
    benchmark_prices["benchmark_return_1d"] = benchmark_prices["benchmark_adj_close"].pct_change()
    for horizon in [5, 20, 60]:
        benchmark_prices[f"benchmark_return_{horizon}d"] = benchmark_prices["benchmark_adj_close"].pct_change(horizon)
    return benchmark_prices


def build_features(prices: pd.DataFrame, feature_names: list[str] | None = None) -> pd.DataFrame:
    validated = validate_prices_frame(prices)
    selected = list_features() if feature_names is None else list(feature_names)
    unknown = sorted(set(selected) - set(FEATURE_NAMES))
    if unknown:
        raise ValueError(f"unknown feature names requested: {unknown}")

    panel = validated.sort_values(["ticker", "date"]).copy()
    benchmark_data = _prepare_benchmark_returns(panel, benchmark="SPY")
    panel = panel.merge(benchmark_data, on="date", how="left")
    panel["dollar_volume"] = panel["adj_close"] * panel["volume"]

    results: OrderedDict[str, pd.Series] = OrderedDict()
    grouped = panel.groupby("ticker", sort=False)

    daily_return = grouped["adj_close"].pct_change()
    results["return_1d"] = daily_return
    results["return_2d"] = grouped["adj_close"].pct_change(2)
    results["return_5d"] = grouped["adj_close"].pct_change(5)
    results["return_10d"] = grouped["adj_close"].pct_change(10)
    results["return_20d"] = grouped["adj_close"].pct_change(20)
    results["return_60d"] = grouped["adj_close"].pct_change(60)
    results["log_return_1d"] = np.log1p(results["return_1d"])
    results["log_return_5d"] = np.log1p(results["return_5d"])
    results["log_return_20d"] = np.log1p(results["return_20d"])
    results["vol_5d"] = daily_return.groupby(panel["ticker"]).transform(lambda s: s.rolling(5, min_periods=5).std(ddof=0))
    results["vol_10d"] = daily_return.groupby(panel["ticker"]).transform(lambda s: s.rolling(10, min_periods=10).std(ddof=0))
    results["vol_20d"] = daily_return.groupby(panel["ticker"]).transform(lambda s: s.rolling(20, min_periods=20).std(ddof=0))
    results["vol_60d"] = daily_return.groupby(panel["ticker"]).transform(lambda s: s.rolling(60, min_periods=60).std(ddof=0))
    results["downside_vol_20d"] = (
        daily_return.clip(upper=0.0).groupby(panel["ticker"]).transform(lambda s: s.rolling(20, min_periods=20).std(ddof=0))
    )
    results["upside_vol_20d"] = (
        daily_return.clip(lower=0.0).groupby(panel["ticker"]).transform(lambda s: s.rolling(20, min_periods=20).std(ddof=0))
    )

    previous_close = grouped["close"].shift(1)
    true_range = pd.concat(
        [
            panel["high"] - panel["low"],
            (panel["high"] - previous_close).abs(),
            (panel["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    results["atr_14"] = true_range.groupby(panel["ticker"]).transform(lambda s: s.rolling(14, min_periods=14).mean())

    benchmark_return_1d = panel["benchmark_return_1d"]
    benchmark_var_20 = benchmark_return_1d.rolling(20, min_periods=20).var(ddof=0)
    benchmark_var_60 = benchmark_return_1d.rolling(60, min_periods=60).var(ddof=0)
    cov_20 = daily_return.groupby(panel["ticker"]).transform(
        lambda s: s.rolling(20, min_periods=20).cov(benchmark_return_1d.loc[s.index])
    )
    cov_60 = daily_return.groupby(panel["ticker"]).transform(
        lambda s: s.rolling(60, min_periods=60).cov(benchmark_return_1d.loc[s.index])
    )
    results["beta_20d_spy"] = cov_20 / benchmark_var_20.replace(0.0, np.nan)
    results["beta_60d_spy"] = cov_60 / benchmark_var_60.replace(0.0, np.nan)

    results["momentum_5d"] = results["return_5d"]
    results["momentum_10d"] = results["return_10d"]
    results["momentum_20d"] = results["return_20d"]
    results["momentum_60d"] = results["return_60d"]
    results["momentum_120d"] = grouped["adj_close"].pct_change(120)

    for window in [10, 20, 50, 200]:
        sma = grouped["adj_close"].transform(lambda s: s.rolling(window, min_periods=window).mean())
        results[f"price_to_sma_{window}d"] = panel["adj_close"] / sma - 1.0

    ema_12 = grouped["adj_close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    ema_26 = grouped["adj_close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    results["price_to_ema_12d"] = panel["adj_close"] / ema_12 - 1.0
    results["price_to_ema_26d"] = panel["adj_close"] / ema_26 - 1.0
    results["macd"] = ema_12 - ema_26
    results["macd_signal"] = results["macd"].groupby(panel["ticker"]).transform(lambda s: s.ewm(span=9, adjust=False).mean())
    results["macd_hist"] = results["macd"] - results["macd_signal"]

    results["rsi_7"] = grouped["adj_close"].transform(lambda s: _rsi(s, 7))
    results["rsi_14"] = grouped["adj_close"].transform(lambda s: _rsi(s, 14))
    results["rsi_28"] = grouped["adj_close"].transform(lambda s: _rsi(s, 28))
    sma_20 = grouped["adj_close"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    std_20 = grouped["adj_close"].transform(lambda s: s.rolling(20, min_periods=20).std(ddof=0))
    results["bollinger_z_20d"] = (panel["adj_close"] - sma_20) / std_20.replace(0.0, np.nan)
    rolling_high_14 = grouped["high"].transform(lambda s: s.rolling(14, min_periods=14).max())
    rolling_low_14 = grouped["low"].transform(lambda s: s.rolling(14, min_periods=14).min())
    results["stoch_k_14"] = 100.0 * (panel["close"] - rolling_low_14) / (rolling_high_14 - rolling_low_14).replace(0.0, np.nan)
    results["stoch_d_14"] = results["stoch_k_14"].groupby(panel["ticker"]).transform(lambda s: s.rolling(3, min_periods=3).mean())

    results["volume_change_1d"] = grouped["volume"].pct_change()
    results["volume_change_5d"] = grouped["volume"].pct_change(5)
    results["volume_change_20d"] = grouped["volume"].pct_change(20)
    vol_mean_20 = grouped["volume"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    vol_std_20 = grouped["volume"].transform(lambda s: s.rolling(20, min_periods=20).std(ddof=0))
    vol_mean_60 = grouped["volume"].transform(lambda s: s.rolling(60, min_periods=60).mean())
    vol_std_60 = grouped["volume"].transform(lambda s: s.rolling(60, min_periods=60).std(ddof=0))
    results["volume_zscore_20d"] = (panel["volume"] - vol_mean_20) / vol_std_20.replace(0.0, np.nan)
    results["volume_zscore_60d"] = (panel["volume"] - vol_mean_60) / vol_std_60.replace(0.0, np.nan)
    results["dollar_volume"] = panel["dollar_volume"]
    dollar_mean_20 = panel["dollar_volume"].groupby(panel["ticker"]).transform(lambda s: s.rolling(20, min_periods=20).mean())
    results["dollar_volume_ratio_20d"] = panel["dollar_volume"] / dollar_mean_20.replace(0.0, np.nan)

    results["intraday_range"] = (panel["high"] - panel["low"]) / panel["close"].replace(0.0, np.nan)
    results["close_open_gap"] = (panel["open"] - previous_close) / previous_close.replace(0.0, np.nan)
    results["close_location_in_range"] = (panel["close"] - panel["low"]) / (panel["high"] - panel["low"]).replace(0.0, np.nan)
    rolling_high_20 = grouped["adj_close"].transform(lambda s: s.rolling(20, min_periods=20).max())
    rolling_low_20 = grouped["adj_close"].transform(lambda s: s.rolling(20, min_periods=20).min())
    rolling_high_60 = grouped["adj_close"].transform(lambda s: s.rolling(60, min_periods=60).max())
    rolling_low_60 = grouped["adj_close"].transform(lambda s: s.rolling(60, min_periods=60).min())
    results["distance_to_20d_high"] = panel["adj_close"] / rolling_high_20 - 1.0
    results["distance_to_20d_low"] = panel["adj_close"] / rolling_low_20 - 1.0
    results["distance_to_60d_high"] = panel["adj_close"] / rolling_high_60 - 1.0
    results["distance_to_60d_low"] = panel["adj_close"] / rolling_low_60 - 1.0

    results["excess_return_5d_vs_spy"] = results["return_5d"] - panel["benchmark_return_5d"]
    results["excess_return_20d_vs_spy"] = results["return_20d"] - panel["benchmark_return_20d"]
    results["excess_return_60d_vs_spy"] = results["return_60d"] - panel["benchmark_return_60d"]
    results["relative_momentum_20d_vs_spy"] = ((1.0 + results["return_20d"]) / (1.0 + panel["benchmark_return_20d"])) - 1.0

    results["skew_20d"] = daily_return.groupby(panel["ticker"]).transform(lambda s: _rolling_apply(s, 20, pd.Series.skew))
    results["kurtosis_20d"] = daily_return.groupby(panel["ticker"]).transform(lambda s: _rolling_apply(s, 20, pd.Series.kurt))

    feature_frame = panel.loc[:, ["date", "ticker"]].copy()
    for name in selected:
        feature_frame[name] = results[name]
    feature_frame.columns.name = None
    return validate_feature_frame(feature_frame)


def make_forward_return_target(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    validated = validate_prices_frame(prices)
    target_name = f"forward_return_{int(horizon)}d"
    result = validated.loc[:, ["date", "ticker"]].copy()
    result[target_name] = (
        validated.groupby("ticker", sort=False)["adj_close"].shift(-int(horizon)) / validated["adj_close"] - 1.0
    )
    return result


def make_forward_alpha_target(
    prices: pd.DataFrame,
    horizon: int,
    benchmark: str = "SPY",
) -> pd.DataFrame:
    validated = validate_prices_frame(prices)
    target_name = f"forward_alpha_{int(horizon)}d_vs_{benchmark.lower()}"
    result = make_forward_return_target(validated, horizon)
    benchmark_frame = (
        result.loc[result["ticker"] == benchmark.upper(), ["date", f"forward_return_{int(horizon)}d"]]
        .rename(columns={f"forward_return_{int(horizon)}d": "benchmark_forward_return"})
    )
    merged = result.merge(benchmark_frame, on="date", how="left")
    merged[target_name] = merged[f"forward_return_{int(horizon)}d"] - merged["benchmark_forward_return"]
    return merged.loc[:, ["date", "ticker", target_name]]


def make_forward_realized_vol_target(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    validated = validate_prices_frame(prices)
    target_name = f"forward_realized_vol_{int(window)}d"
    daily_return = validated.groupby("ticker", sort=False)["adj_close"].pct_change()

    def _future_realized_vol(series: pd.Series) -> pd.Series:
        values = series.to_numpy(dtype=float)
        result = np.full(len(values), np.nan, dtype=float)
        for idx in range(len(values)):
            future = values[idx + 1 : idx + 1 + int(window)]
            if len(future) == int(window) and np.isfinite(future).all():
                result[idx] = float(np.std(future, ddof=0) * np.sqrt(252.0))
        return pd.Series(result, index=series.index)

    target_values = daily_return.groupby(validated["ticker"]).transform(_future_realized_vol)
    result = validated.loc[:, ["date", "ticker"]].copy()
    result[target_name] = target_values
    return result
