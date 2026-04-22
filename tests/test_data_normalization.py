from __future__ import annotations

import pandas as pd

from portfolio_toolkit.data import _normalize_downloaded_frame


def test_normalize_downloaded_frame_flattens_single_ticker_multiindex_columns() -> None:
    index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    columns = pd.MultiIndex.from_tuples(
        [
            ("Open", "AAPL"),
            ("High", "AAPL"),
            ("Low", "AAPL"),
            ("Close", "AAPL"),
            ("Adj Close", "AAPL"),
            ("Volume", "AAPL"),
        ],
        names=["Price", "Ticker"],
    )
    raw = pd.DataFrame(
        [
            [100.0, 101.0, 99.0, 100.5, 100.4, 1_000_000],
            [101.0, 102.0, 100.0, 101.5, 101.4, 1_100_000],
        ],
        index=index,
        columns=columns,
    )
    raw.index.name = "Date"

    normalized = _normalize_downloaded_frame(raw, "AAPL")

    assert list(normalized.columns) == ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
    assert normalized.columns.name is None
    assert normalized["ticker"].tolist() == ["AAPL", "AAPL"]
    assert normalized["close"].tolist() == [100.5, 101.5]
