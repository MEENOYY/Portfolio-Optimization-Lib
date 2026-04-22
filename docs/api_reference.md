# API Reference

This is a compact reference for the public surface most notebook users should rely on.

## Configuration

### `load_dataset_specs(repo_root=None) -> dict[str, DatasetSpec]`

Load all dataset presets from `configs/datasets.toml`.

### `get_dataset_spec(dataset_name, repo_root=None) -> DatasetSpec`

Load one dataset preset.

### `load_mlflow_settings(repo_root=None) -> MlflowSettings`

Load local MLflow configuration from `configs/mlflow.toml`.

## Data And Splits

### `load_prices(dataset_name, refresh=False, repo_root=None) -> pd.DataFrame`

Download or load cached daily OHLCV prices for the dataset preset.

### `split_dates(dataset_name, repo_root=None) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]`

Return train, validation, and test boundaries for a dataset preset.

### `slice_split(frame, dataset_name, split_name, repo_root=None) -> pd.DataFrame`

Slice any dataframe with a `date` column to the requested split window.

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

Validate and normalize a price dataframe.

### `validate_feature_frame(df) -> pd.DataFrame`

Validate and normalize a feature dataframe.

### `validate_prediction_frame(df, dataset_name=None, horizon=None, repo_root=None) -> pd.DataFrame`

Validate and normalize a standardized prediction dataframe.

### `validate_weights_frame(df, dataset_name=None, repo_root=None) -> pd.DataFrame`

Validate and normalize a weights dataframe.

## Contracts

### `DatasetSpec`

Fields:

- `name`
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

## Backtesting

### `backtest_weights(dataset_name, portfolio_weights, benchmark="SPY", repo_root=None) -> BacktestResult`

Backtest a direct `PortfolioWeights` object.

### `backtest_predictions(dataset_name, predictions, builder="top_k_equal", repo_root=None, **builder_kwargs) -> BacktestResult`

Convert predictions into weights with a shared portfolio builder and then backtest them.

Supported builders:

- `top_k_equal`
- `rank_long_only`
- `risk_adjusted`

## Reporting

### `build_metrics(result) -> dict[str, float]`

Compute the standard shared metrics from a `BacktestResult`.

### `write_quantstats_report(result, output_dir) -> Path`

Write the QuantStats HTML tear sheet for a strategy.

### `write_backtest_artifacts(result, output_dir) -> dict[str, str]`

Write the shared output bundle for a backtest.

## Tracking

### `init_mlflow(repo_root=".") -> dict[str, str]`

Initialize the local MLflow SQLite database and artifact directory.

### `start_run(run_name, dataset_name, tags=None, repo_root=".")`

Start an MLflow run as a context manager.

### `log_predictions(df) -> None`

Log a prediction dataframe as an MLflow artifact.

### `log_portfolio(weights) -> None`

Log a portfolio weights artifact.

### `log_backtest(result) -> None`

Log metrics and any existing report artifacts from a backtest result.

### `log_report_artifacts(paths) -> None`

Log explicit artifact paths to MLflow.
