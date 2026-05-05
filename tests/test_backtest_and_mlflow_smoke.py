from __future__ import annotations

import json
from pathlib import Path

from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import pytest

from portfolio_toolkit import (
    baseline_weights,
    backtest_weights,
    build_metrics,
    init_mlflow,
    log_backtest,
    log_model_submission,
    start_run,
    write_backtest_artifacts,
)
from portfolio_toolkit.backtest import _mask_unavailable_weights
from portfolio_toolkit.contracts import BacktestResult


def test_backtest_and_mlflow_smoke(repo_root):
    layout = init_mlflow(repo_root)
    assert Path(layout["db_path"]).exists()

    weights = baseline_weights("shared_set_1", "equal_weight", repo_root=repo_root)
    result = backtest_weights("shared_set_1", weights, repo_root=repo_root)
    artifact_paths = write_backtest_artifacts(result, repo_root / "runs" / "equal_weight_smoke")
    assert "total_return" in result.metrics
    assert Path(artifact_paths["quantstats_report"]).exists()
    report_html = Path(artifact_paths["quantstats_report"]).read_text(encoding="utf-8")
    assert "QuantStats Alpha vs SPY" in report_html
    assert "not the model's <code>expected_alpha</code>" in report_html

    with start_run("equal_weight_smoke", "shared_set_1", repo_root=repo_root):
        log_backtest(result)


def test_metrics_annualize_over_active_weight_window():
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2022-01-03", "2023-01-03"])
    nav = pd.Series([100.0, 100.0, 100.0, 121.0], index=dates, name="nav")
    returns = nav.pct_change().fillna(0.0).rename("returns")
    weights = pd.DataFrame({"AAA": [1.0]}, index=pd.to_datetime(["2022-01-03"]))
    benchmark_returns = pd.DataFrame({"SPY": [0.0, 0.0, 0.0, 0.10]}, index=dates)
    result = BacktestResult(
        strategy_name="active_window_test",
        dataset_name="dummy",
        weights=weights,
        nav=nav,
        returns=returns,
        turnover=pd.Series([1.0], index=weights.index, name="turnover"),
        benchmark_returns=benchmark_returns,
        metrics={},
    )

    metrics = build_metrics(result)
    expected_years = (pd.Timestamp("2023-01-03") - pd.Timestamp("2022-01-03")).days / 365.25
    expected_annual_return = (1.21 ** (1.0 / expected_years)) - 1.0
    expected_benchmark_annual_return = (1.10 ** (1.0 / expected_years)) - 1.0

    assert metrics["evaluation_years"] == pytest.approx(expected_years)
    assert metrics["evaluation_trading_days"] == 2.0
    assert metrics["total_return"] == pytest.approx(0.21)
    assert metrics["annual_return"] == pytest.approx(expected_annual_return)
    assert metrics["benchmark_total_return"] == pytest.approx(0.10)
    assert metrics["benchmark_annual_return"] == pytest.approx(expected_benchmark_annual_return)
    assert metrics["benchmark_sharpe"] > 0.0
    assert metrics["benchmark_max_drawdown"] == pytest.approx(0.0)
    assert metrics["excess_return_vs_benchmark"] == pytest.approx(0.11)
    assert metrics["sharpe_vs_benchmark"] == pytest.approx(metrics["sharpe"] - metrics["benchmark_sharpe"])


def test_log_model_submission_writes_manifest_and_artifacts(repo_root, tmp_path):
    layout = init_mlflow(repo_root)
    model_path = tmp_path / "dummy_model.pt"
    model_path.write_text("model-bytes", encoding="utf-8")
    source_path = tmp_path / "notebook.ipynb"
    source_path.write_text("{}", encoding="utf-8")

    with start_run("model_submission_smoke", "shared_set_1", repo_root=repo_root) as run:
        manifest = log_model_submission(
            {"model": model_path},
            model_name="dummy_torch_model",
            model_family="torch",
            feature_names=["momentum_20d", "vol_20d"],
            target="forward_alpha_5d_vs_spy",
            horizon=5,
            rebalance_frequency="weekly",
            preprocessing={
                "scaler": "train_mean_std",
                "train_means": {"momentum_20d": 0.1, "vol_20d": 0.2},
                "train_stds": {"momentum_20d": 1.0, "vol_20d": 2.0},
            },
            model_config={"latent_dim": 8},
            source_files=[source_path],
            notes="test submission",
        )
        run_id = run.info.run_id

    assert manifest["feature_names"] == ["momentum_20d", "vol_20d"]
    assert manifest["artifact_map"] == {"model": "artifacts/dummy_model.pt"}
    assert manifest["source_files"] == ["source/notebook.ipynb"]

    client = MlflowClient(tracking_uri=layout["tracking_uri"])
    downloaded = client.download_artifacts(run_id, "model_submission/manifest.json", str(tmp_path / "downloaded"))
    stored_manifest = json.loads(Path(downloaded).read_text(encoding="utf-8"))

    assert stored_manifest["model_name"] == "dummy_torch_model"
    assert stored_manifest["model_family"] == "torch"
    assert stored_manifest["target"] == "forward_alpha_5d_vs_spy"
    assert stored_manifest["horizon"] == 5
    assert stored_manifest["rebalance_frequency"] == "weekly"
    assert stored_manifest["feature_names"] == ["momentum_20d", "vol_20d"]

    mlflow_run = client.get_run(run_id)
    assert mlflow_run.data.params["submission_model_name"] == "dummy_torch_model"
    assert mlflow_run.data.params["submission_feature_count"] == "2"
    assert mlflow_run.data.params["submission_rebalance_frequency"] == "weekly"
    assert mlflow_run.data.tags["has_model_submission"] == "true"


def test_log_model_submission_rejects_missing_artifact(tmp_path):
    with pytest.raises(FileNotFoundError):
        log_model_submission(
            {"model": tmp_path / "missing.pt"},
            model_name="missing",
            model_family="torch",
            feature_names=["momentum_20d"],
            target="forward_return_5d",
            horizon=5,
        )


def test_log_model_submission_rejects_empty_features(tmp_path):
    model_path = tmp_path / "model.txt"
    model_path.write_text("model", encoding="utf-8")

    with pytest.raises(ValueError):
        log_model_submission(
            [model_path],
            model_name="empty_features",
            model_family="other",
            feature_names=[],
            target="forward_return_5d",
            horizon=5,
        )


def test_log_model_submission_rejects_empty_rebalance_frequency(tmp_path):
    model_path = tmp_path / "model.txt"
    model_path.write_text("model", encoding="utf-8")

    with pytest.raises(ValueError):
        log_model_submission(
            [model_path],
            model_name="bad_rebalance",
            model_family="other",
            feature_names=["momentum_20d"],
            target="forward_return_5d",
            horizon=5,
            rebalance_frequency=" ",
        )


def test_log_model_submission_rejects_non_json_metadata(tmp_path):
    model_path = tmp_path / "model.txt"
    model_path.write_text("model", encoding="utf-8")

    with pytest.raises(TypeError):
        log_model_submission(
            [model_path],
            model_name="bad_metadata",
            model_family="other",
            feature_names=["momentum_20d"],
            target="forward_return_5d",
            horizon=5,
            model_config={"bad": {object()}},
        )


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
