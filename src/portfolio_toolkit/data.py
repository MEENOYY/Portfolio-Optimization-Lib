from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from .config import get_dataset_spec
from .validation import validate_prices_frame


def _repo_root(repo_root: str | Path | None = None) -> Path:
    return Path("." if repo_root is None else repo_root).resolve()


def _cache_path(dataset_name: str, repo_root: str | Path | None = None) -> Path:
    return _repo_root(repo_root) / "data_cache" / f"{dataset_name}.parquet"


def _normalize_downloaded_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    normalized = frame.copy()
    if isinstance(normalized.columns, pd.MultiIndex):
        normalized.columns = normalized.columns.get_level_values(0)
    normalized.columns.name = None

    normalized = normalized.reset_index().rename(
        columns={
            "Date": "date",
            "Datetime": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    normalized["ticker"] = ticker.upper()
    normalized = normalized.loc[:, ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]]
    normalized.columns.name = None
    return normalized


def _download_prices_for_dataset(dataset_name: str, repo_root: str | Path | None = None) -> pd.DataFrame:
    spec = get_dataset_spec(dataset_name, repo_root=repo_root)
    if not spec.tickers:
        raise ValueError(
            f"dataset preset '{dataset_name}' has no tickers yet; fill the ticker list in configs/datasets.toml first"
        )
    frames: list[pd.DataFrame] = []
    start = spec.start_date.isoformat()
    end = (spec.end_date + timedelta(days=1)).isoformat()
    for ticker in spec.all_tickers:
        downloaded = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if downloaded.empty:
            continue
        frames.append(_normalize_downloaded_frame(downloaded, ticker))
    if not frames:
        raise ValueError(f"no data could be downloaded for dataset '{dataset_name}'")
    combined = pd.concat(frames, ignore_index=True)
    return validate_prices_frame(combined, dataset_name=dataset_name, repo_root=repo_root)


def load_prices(
    dataset_name: str,
    refresh: bool = False,
    *,
    repo_root: str | Path | None = None,
) -> pd.DataFrame:
    cache_path = _cache_path(dataset_name, repo_root=repo_root)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not refresh:
        return validate_prices_frame(pd.read_parquet(cache_path), dataset_name=dataset_name, repo_root=repo_root)
    prices = _download_prices_for_dataset(dataset_name, repo_root=repo_root)
    prices.to_parquet(cache_path, index=False)
    return prices
