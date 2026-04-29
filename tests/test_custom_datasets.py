from __future__ import annotations

from pathlib import Path

from mlflow.tracking import MlflowClient
import pandas as pd
import pytest

from portfolio_toolkit import (
    baseline_weights,
    backtest_weights,
    custom_dataset,
    init_mlflow,
    load_prices,
    split_dates,
    start_run,
    validate_prediction_frame,
    validate_weights_frame,
    write_backtest_artifacts,
)
from portfolio_toolkit.data import _cache_path


def _to_download_frame(prices: pd.DataFrame, ticker: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    frame = (
        prices.loc[prices["ticker"] == ticker, ["date", "open", "high", "low", "close", "adj_close", "volume"]]
        .rename(
            columns={
                "date": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "adj_close": "Adj Close",
                "volume": "Volume",
            }
        )
        .set_index("Date")
        .sort_index()
    )
    if start is not None:
        frame = frame.loc[frame.index >= pd.Timestamp(start)]
    if end is not None:
        frame = frame.loc[frame.index < pd.Timestamp(end)]
    return frame


def test_custom_dataset_generates_stable_identifier_and_contiguous_splits() -> None:
    left = custom_dataset(["msft", "aapl"], "2024-01-01", "2024-01-10", benchmark="spy")
    right = custom_dataset(["AAPL", "MSFT"], "2024-01-01", "2024-01-10", benchmark="SPY")
    named = custom_dataset(["AAPL", "MSFT"], "2024-01-01", "2024-01-10", name="Secret Semis")

    assert left.name == right.name
    assert left.kind == "custom"
    assert left.benchmark_ticker == "SPY"
    assert left.tickers == ["MSFT", "AAPL"]
    assert left.train_start <= left.train_end < left.val_start <= left.val_end < left.test_start <= left.test_end
    assert left.train_start == pd.Timestamp("2024-01-01").date()
    assert left.test_end == pd.Timestamp("2024-01-10").date()
    assert named.name == "secret_semis"


def test_custom_dataset_rejects_invalid_date_ranges() -> None:
    with pytest.raises(ValueError):
        custom_dataset(["AAPL"], "2024-01-03", "2024-01-01")

    with pytest.raises(ValueError):
        custom_dataset(["AAPL"], "2024-01-01", "2024-01-02")


def test_custom_dataset_cache_and_validation(repo_root, monkeypatch) -> None:
    dataset = custom_dataset(["AAPL", "MSFT"], "2018-01-02", "2025-12-31")
    fixture_prices = pd.read_parquet(repo_root / "data_cache" / "shared_set_1.parquet")
    calls: list[str] = []

    def fake_download(
        ticker: str,
        start: str | None = None,
        end: str | None = None,
        auto_adjust: bool = False,
        progress: bool = False,
        threads: bool = False,
    ) -> pd.DataFrame:
        calls.append(ticker)
        return _to_download_frame(fixture_prices, ticker, start=start, end=end)

    monkeypatch.setattr("portfolio_toolkit.data.yf.download", fake_download)

    loaded = load_prices(dataset, repo_root=repo_root)
    cache_path = _cache_path(dataset, repo_root=repo_root)
    assert cache_path.exists()
    assert calls == ["AAPL", "MSFT", "SPY"]

    dataset_same = custom_dataset(["MSFT", "AAPL"], "2018-01-02", "2025-12-31")
    assert _cache_path(dataset_same, repo_root=repo_root) == cache_path

    def fail_download(*args, **kwargs):
        raise AssertionError("cached custom dataset should not redownload prices")

    monkeypatch.setattr("portfolio_toolkit.data.yf.download", fail_download)
    cached = load_prices(dataset_same, repo_root=repo_root)
    pd.testing.assert_frame_equal(loaded, cached)

    predictions = loaded.loc[loaded["ticker"].isin(["AAPL", "MSFT"]), ["date", "ticker"]].head(6).copy()
    predictions["horizon"] = 5
    predictions["expected_return"] = 0.01
    validated_predictions = validate_prediction_frame(predictions, dataset_name=dataset, horizon=5, repo_root=repo_root)
    assert len(validated_predictions) == len(predictions)

    weights = pd.DataFrame({"AAPL": [0.5], "MSFT": [0.5]}, index=pd.DatetimeIndex(["2024-01-02"]))
    validated_weights = validate_weights_frame(weights, dataset_name=dataset, repo_root=repo_root)
    assert validated_weights.columns.tolist() == ["AAPL", "MSFT"]


def test_custom_dataset_baseline_backtest_and_quantstats(repo_root, monkeypatch, tmp_path) -> None:
    dataset = custom_dataset(["AAPL", "MSFT", "NVDA"], "2018-01-02", "2025-12-31")
    fixture_prices = pd.read_parquet(repo_root / "data_cache" / "shared_set_1.parquet")

    def fake_download(
        ticker: str,
        start: str | None = None,
        end: str | None = None,
        auto_adjust: bool = False,
        progress: bool = False,
        threads: bool = False,
    ) -> pd.DataFrame:
        return _to_download_frame(fixture_prices, ticker, start=start, end=end)

    monkeypatch.setattr("portfolio_toolkit.data.yf.download", fake_download)

    splits = split_dates(dataset, repo_root=repo_root)
    assert splits["train"][1] < splits["val"][0]
    assert splits["val"][1] < splits["test"][0]

    weights = baseline_weights(dataset, "equal_weight", repo_root=repo_root)
    result = backtest_weights(dataset, weights, repo_root=repo_root)
    artifact_paths = write_backtest_artifacts(result, tmp_path / "custom_dataset_backtest")

    assert weights.dataset_name == dataset.name
    assert result.dataset_name == dataset.name
    assert "total_return" in result.metrics
    assert Path(artifact_paths["quantstats_report"]).exists()


def test_start_run_logs_custom_dataset_metadata(repo_root) -> None:
    dataset = custom_dataset(["AAPL", "MSFT", "NVDA"], "2018-01-02", "2025-12-31")
    layout = init_mlflow(repo_root)

    with start_run("custom_dataset_run", dataset, repo_root=repo_root) as run:
        run_id = run.info.run_id

    client = MlflowClient(tracking_uri=layout["tracking_uri"])
    stored_run = client.get_run(run_id)
    assert stored_run.data.tags["dataset_name"] == dataset.name
    assert stored_run.data.tags["dataset_kind"] == "custom"
    assert stored_run.data.params["dataset_benchmark_ticker"] == "SPY"
    assert stored_run.data.params["dataset_start_date"] == "2018-01-02"
    assert stored_run.data.params["dataset_end_date"] == "2025-12-31"
    assert stored_run.data.params["dataset_ticker_count"] == "3"

    manifest_path = client.download_artifacts(run_id, "dataset_spec.json", str(repo_root / "runs" / "dataset_spec_download"))
    payload = Path(manifest_path).read_text(encoding="utf-8")
    assert dataset.name in payload
