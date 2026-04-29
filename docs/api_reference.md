# API Reference

This is a compact reference for the public surface most notebook users should rely on.

## Configuration

### `load_dataset_specs(repo_root=None) -> dict[str, DatasetSpec]`

Load all dataset presets from `configs/datasets.toml`.

### `custom_dataset(tickers, start, end, benchmark="SPY", name=None, cost_bps=10.0) -> DatasetSpec`

Construct a runtime dataset without editing `configs/datasets.toml`.

- `tickers`
  Tradable universe for the dataset.
- `start`, `end`
  Inclusive calendar-date bounds.
- `benchmark`
  Benchmark ticker used in downloads and backtests.
- `name`
  Optional dataset identifier. If omitted, the toolkit generates a stable `custom_<hash>` identifier.
- `cost_bps`
  Flat transaction cost used by the backtest layer.

Runtime datasets use contiguous calendar-date `60/20/20` train/validation/test splits.

### `get_dataset_spec(dataset_name, repo_root=None) -> DatasetSpec`

Load one dataset preset.

### `load_mlflow_settings(repo_root=None) -> MlflowSettings`

Load the shared MLflow configuration from `configs/mlflow.toml`. The tracking URI can also be overridden with the `MLFLOW_TRACKING_URI` environment variable.

## Data And Splits

### `load_prices(dataset_name, refresh=False, repo_root=None) -> pd.DataFrame`

Download or load cached daily OHLCV prices for either a preset dataset name or a runtime `DatasetSpec`.

### `split_dates(dataset_name, repo_root=None) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]`

Return train, validation, and test boundaries for either a preset dataset name or a runtime `DatasetSpec`.

### `slice_split(frame, dataset_name, split_name, repo_root=None) -> pd.DataFrame`

Slice any dataframe with a `date` column to the requested split window for either a preset dataset name or a runtime `DatasetSpec`.

## Features And Targets

### `list_features() -> list[str]`

Return the names of all built-in features.

### `build_features(prices, feature_names=None) -> pd.DataFrame`

Build the long-form feature frame from the shared prices dataframe.

### `make_forward_return_target(prices, horizon) -> pd.DataFrame`

Create a forward return target column such as `forward_return_5d`.

### `make_forward_alpha_target(prices, horizon, benchmark="SPY") -> pd.DataFrame`

Create a forward alpha target such as `forward_alpha_5d_vs_spy`.

### `make_forward_realized_vol_target(prices, window) -> pd.DataFrame`

Create a forward realized volatility target such as `forward_realized_vol_5d`.

## Validation

### `validate_prices_frame(df, dataset_name=None, repo_root=None) -> pd.DataFrame`

Validate and normalize a price dataframe. `dataset_name` may be a preset dataset name or a runtime `DatasetSpec`.

### `validate_feature_frame(df) -> pd.DataFrame`

Validate and normalize a feature dataframe.

### `validate_prediction_frame(df, dataset_name=None, horizon=None, repo_root=None) -> pd.DataFrame`

Validate and normalize a standardized prediction dataframe. `dataset_name` may be a preset dataset name or a runtime `DatasetSpec`.

### `validate_weights_frame(df, dataset_name=None, repo_root=None) -> pd.DataFrame`

Validate and normalize a weights dataframe. `dataset_name` may be a preset dataset name or a runtime `DatasetSpec`.

## Contracts

### `DatasetSpec`

Fields:

- `name`
- `dataset_id`
- `kind`
- `tickers`
- `benchmark_ticker`
- `start_date`
- `end_date`
- `train_start`
- `train_end`
- `val_start`
- `val_end`
- `test_start`
- `test_end`
- `cost_bps`
- `default_benchmark`

### `MlflowSettings`

Fields:

- `experiment_prefix`
- `tracking_uri`
- `backend_store_uri`
- `artifact_root`
- `host`
- `port`

### `PortfolioWeights`

Fields:

- `weights`
- `dataset_name`
- `strategy_name`
- `metadata`

### `BacktestResult`

Fields:

- `strategy_name`
- `dataset_name`
- `weights`
- `nav`
- `returns`
- `turnover`
- `benchmark_returns`
- `metrics`
- `artifact_paths`

## Portfolio Builders

### `weights_from_predictions_top_k_equal(predictions, k, score_column="expected_return", dataset_name="unknown", strategy_name=None) -> PortfolioWeights`

Create equal weights across the top-`k` predictions at each rebalance date.

### `weights_from_predictions_rank_long_only(predictions, score_column="expected_return", dataset_name="unknown", strategy_name="rank_long_only") -> PortfolioWeights`

Create descending long-only weights across the ranked universe.

### `weights_from_predictions_risk_adjusted(predictions, return_col="expected_return", vol_col="expected_volatility", dataset_name="unknown", strategy_name="risk_adjusted") -> PortfolioWeights`

Create long-only weights using return divided by expected volatility.

## Baselines

### `baseline_weights(dataset_name, strategy_name, split="test", repo_root=None) -> PortfolioWeights`

Built-in strategies:

- `equal_weight`
- `inverse_volatility`
- `momentum_20d`

`dataset_name` may be a preset dataset name or a runtime `DatasetSpec`.

## Backtesting

### `backtest_weights(dataset_name, portfolio_weights, benchmark="SPY", repo_root=None) -> BacktestResult`

Backtest a direct `PortfolioWeights` object against either a preset dataset name or a runtime `DatasetSpec`.

### `backtest_predictions(dataset_name, predictions, builder="top_k_equal", repo_root=None, **builder_kwargs) -> BacktestResult`

Convert predictions into weights with a shared portfolio builder and then backtest them against either a preset dataset name or a runtime `DatasetSpec`.

Supported builders:

- `top_k_equal`
- `rank_long_only`
- `risk_adjusted`

## Reporting

### `build_metrics(result) -> dict[str, float]`

Compute the standard shared metrics from a `BacktestResult`.

Annualized metrics use the active evaluation window: first submitted weight date through final NAV date. The metric bundle includes `evaluation_years` and `evaluation_trading_days` so the annualization period is explicit.

### `write_quantstats_report(result, output_dir) -> Path`

Write the QuantStats HTML tear sheet for a strategy.

When a benchmark is present, the QuantStats `Alpha` row is relabeled in the generated HTML as `QuantStats Alpha vs <benchmark>`. This is annualized realized-return regression alpha, not the model prediction column `expected_alpha`, not a `forward_alpha_*` target, and not the simple total-return difference in `excess_return_vs_benchmark`.

### `write_backtest_artifacts(result, output_dir) -> dict[str, str]`

Write the shared output bundle for a backtest.

## Tracking

### `init_mlflow(repo_root=".") -> dict[str, str]`

Initialize the local MLflow SQLite database and artifact directory.

### `start_run(run_name, dataset_name, tags=None, repo_root=".")`

Start an MLflow run as a context manager.

`dataset_name` may be a preset dataset name or a runtime `DatasetSpec`. Runtime datasets automatically log dataset metadata and a `dataset_spec.json` artifact so runs remain reconstructable.

### `log_predictions(df) -> None`

Log a prediction dataframe as an MLflow artifact.

### `log_portfolio(weights) -> None`

Log a portfolio weights artifact.

### `log_backtest(result) -> None`

Log metrics and any existing report artifacts from a backtest result.

### `log_model_submission(model_artifacts, *, model_name, model_family, feature_names, target, horizon, rebalance_frequency=None, preprocessing=None, model_config=None, source_files=None, notes=None, artifact_dir="model_submission") -> dict`

Log a reconstructable model bundle to the active MLflow run. The caller saves framework-specific model files first, then this helper packages them with ordered feature names, target/horizon details, optional rebalance frequency, optional preprocessing metadata, optional model config, and optional source notebooks/code.

Expected notebook-side submission functions:

- `build_model_features(prices)`: required. Rebuilds the exact feature dataframe, including custom features, from standard prices.
- `predict_from_prices(model, prices, dates=None, tickers=None)`: required. Runs inference on arbitrary dates/ticker subsets and returns the standard prediction frame.

`load_submission_model(...)` is optional. Model loading can be reconstructed from the model artifact, notebook source, and manifest.

### `log_report_artifacts(paths) -> None`

Log explicit artifact paths to MLflow.
