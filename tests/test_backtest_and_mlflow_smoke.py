from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_toolkit import baseline_weights, backtest_weights, init_mlflow, start_run, write_backtest_artifacts, log_backtest
from portfolio_toolkit.backtest import _mask_unavailable_weights


def test_backtest_and_mlflow_smoke(repo_root):
    layout = init_mlflow(repo_root)
    assert Path(layout["db_path"]).exists()

    weights = baseline_weights("shared_set_1", "equal_weight", repo_root=repo_root)
    result = backtest_weights("shared_set_1", weights, repo_root=repo_root)
    artifact_paths = write_backtest_artifacts(result, repo_root / "runs" / "equal_weight_smoke")
    assert "total_return" in result.metrics
    assert Path(artifact_paths["quantstats_report"]).exists()

    with start_run("equal_weight_smoke", "shared_set_1", repo_root=repo_root):
        log_backtest(result)


def test_baseline_weights_respects_repo_root_from_nested_working_directory(repo_root, monkeypatch):
    nested_dir = repo_root / "notebooks" / "templates"
    nested_dir.mkdir(parents=True)
    monkeypatch.chdir(nested_dir)

    weights = baseline_weights("shared_set_1", "momentum_20d", repo_root=repo_root)

    assert not weights.weights.empty
    assert weights.dataset_name == "shared_set_1"


def test_mask_unavailable_weights_renormalizes_rows() -> None:
    weights = pd.DataFrame(
        {
            "AAPL": [0.5, 0.4],
            "MSFT": [0.3, 0.4],
            "CEG": [0.2, 0.2],
        },
        index=pd.to_datetime(["2022-01-03", "2022-01-04"]),
    )
    weights.index.name = "date"

    prices = pd.DataFrame(
        {
            "AAPL": [100.0, 101.0],
            "MSFT": [200.0, 201.0],
            "CEG": [np.nan, 50.0],
        },
        index=pd.to_datetime(["2022-01-03", "2022-01-04"]),
    )
    prices.index.name = "date"

    adjusted = _mask_unavailable_weights(weights, prices)

    assert np.isclose(adjusted.loc[pd.Timestamp("2022-01-03")].sum(), 1.0)
    assert adjusted.loc[pd.Timestamp("2022-01-03"), "CEG"] == 0.0
    assert np.isclose(adjusted.loc[pd.Timestamp("2022-01-03"), "AAPL"], 0.625)
    assert np.isclose(adjusted.loc[pd.Timestamp("2022-01-03"), "MSFT"], 0.375)
