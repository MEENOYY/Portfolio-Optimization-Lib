from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest


def make_fixture_prices() -> pd.DataFrame:
    dates = pd.bdate_range("2018-01-02", "2025-12-31")
    tickers = ["AAPL", "MSFT", "NVDA", "SPY"]
    rows: list[dict[str, object]] = []
    for ticker_index, ticker in enumerate(tickers):
        base = 80.0 + 20.0 * ticker_index
        trend = np.linspace(0.0, 120.0, len(dates))
        seasonal = 4.0 * np.sin(np.linspace(0.0, 12.0 * np.pi, len(dates)) + ticker_index)
        prices = base + trend + seasonal
        for idx, date_value in enumerate(dates):
            close = float(prices[idx])
            open_price = close * (0.998 + 0.004 * ((idx + ticker_index) % 5) / 4.0)
            high = max(open_price, close) * 1.01
            low = min(open_price, close) * 0.99
            adj_close = close
            volume = 1_000_000 + 10_000 * ticker_index + (idx % 30) * 1_500
            rows.append(
                {
                    "date": date_value,
                    "ticker": ticker,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "adj_close": adj_close,
                    "volume": volume,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def repo_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir(parents=True)
    (root / "configs").mkdir()
    (root / "data_cache").mkdir()
    (root / "runs").mkdir()
    (root / "mlflow").mkdir()
    shutil.copyfile(Path.cwd() / "configs" / "mlflow.toml", root / "configs" / "mlflow.toml")
    (root / "configs" / "datasets.toml").write_text(
        """
[shared_set_1]
name = "shared_set_1"
tickers = ["AAPL", "MSFT", "NVDA"]
benchmark_ticker = "SPY"
start_date = "2014-01-02"
end_date = "2025-12-31"
train_start = "2014-01-02"
train_end = "2019-12-31"
val_start = "2020-01-02"
val_end = "2021-12-31"
test_start = "2022-01-03"
test_end = "2025-12-31"
cost_bps = 10.0
default_benchmark = "SPY"

[shared_set_2]
name = "shared_set_2"
tickers = []
benchmark_ticker = "SPY"
start_date = "2014-01-02"
end_date = "2025-12-31"
train_start = "2014-01-02"
train_end = "2019-12-31"
val_start = "2020-01-02"
val_end = "2021-12-31"
test_start = "2022-01-03"
test_end = "2025-12-31"
cost_bps = 10.0
default_benchmark = "SPY"

[shared_set_3]
name = "shared_set_3"
tickers = []
benchmark_ticker = "SPY"
start_date = "2014-01-02"
end_date = "2025-12-31"
train_start = "2014-01-02"
train_end = "2019-12-31"
val_start = "2020-01-02"
val_end = "2021-12-31"
test_start = "2022-01-03"
test_end = "2025-12-31"
cost_bps = 10.0
default_benchmark = "SPY"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    prices = make_fixture_prices()
    prices.to_parquet(root / "data_cache" / "shared_set_1.parquet", index=False)
    return root
