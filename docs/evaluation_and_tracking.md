# Evaluation And Tracking

This guide explains the shared evaluation layer: portfolio objects, baselines, backtesting, metrics, QuantStats, and MLflow.

## 1. PortfolioWeights

The core container for any allocation is `PortfolioWeights`.

It holds:

- `weights`
- `dataset_name`
- `strategy_name`
- `metadata`

The underlying weights dataframe must satisfy these rules:

- index is rebalance dates
- columns are tickers
- all weights are in `[0, 1]`
- long-only only
- each row sums to `1.0`
- tickers must belong to the dataset universe

Use:

```python
from portfolio_toolkit import validate_weights_frame, PortfolioWeights
```

## 2. Built-In Portfolio Builders

These are deterministic bridges from predictions to allocations.

### `weights_from_predictions_top_k_equal`

- ranks stocks by a score column
- keeps the top `k`
- assigns equal weight across the selected names

Best for:

- simple stock-picking models
- high-conviction top-bucket research

### `weights_from_predictions_rank_long_only`

- ranks all names
- assigns descending long-only weights across the entire universe

Best for:

- broad-exposure ranking models
- signal ranking notebooks

### `weights_from_predictions_risk_adjusted`

- uses `expected_return / expected_volatility`
- normalizes positive scores into long-only weights
- falls back to equal-weight within the date if no positive scores exist

Best for:

- models that predict both return and volatility

## 3. Built-In Baselines

The toolkit ships three baselines:

- `equal_weight`
- `inverse_volatility`
- `momentum_20d`

Use them with:

```python
from portfolio_toolkit import baseline_weights, backtest_weights
```

Why they matter:

- `equal_weight` tells you whether your model beats a naive diversified allocation
- `inverse_volatility` tells you whether risk scaling alone gets similar performance
- `momentum_20d` tells you whether a simple technical signal rivals your model

If a fancy notebook cannot beat those on the test split, that is useful information.

## 4. BacktestResult

The shared backtest output is `BacktestResult`.

It contains:

- `strategy_name`
- `dataset_name`
- `weights`
- `nav`
- `returns`
- `turnover`
- `benchmark_returns`
- `metrics`
- `artifact_paths`

This object is the main result container after a formal run.

## 5. What The Backtest Layer Does

The shared backtest wrapper:

- loads the shared dataset prices
- pivots `adj_close` into a wide price matrix
- validates incoming weights
- aligns weight dates to the next available trading day
- runs a `bt` strategy using:
  - `RunDaily`
  - `SelectAll`
  - `WeighTarget`
  - `Rebalance`
- applies flat commissions from `cost_bps`
- computes turnover from changes in target weights
- produces benchmark return series
- computes summary metrics

This is intentionally a lightweight evaluation wrapper, not a brokerage simulation engine.

## 6. Benchmarks

The shared backtest compares a strategy against:

- `SPY` when available
- an equal-weight benchmark across the strategy universe

That gives every strategy a broad market baseline and a same-universe baseline.

## 7. Metrics

Use:

```python
from portfolio_toolkit import build_metrics
```

Metrics computed:

- `evaluation_years`
- `evaluation_trading_days`
- `total_return`
- `annual_return`
- `annual_volatility`
- `sharpe`
- `sortino`
- `max_drawdown`
- `calmar`
- `average_turnover`
- `benchmark_total_return`
- `benchmark_annual_return`
- `benchmark_annual_volatility`
- `benchmark_sharpe`
- `benchmark_max_drawdown`
- `excess_return_vs_benchmark`
- `annual_excess_return_vs_benchmark`
- `sharpe_vs_benchmark`

Interpretation:

- `evaluation_years`
  Calendar years from the first active weight date through the final NAV date.
- `evaluation_trading_days`
  Number of return observations in that active evaluation window.
- `annual_return`
  CAGR over the active evaluation window, using actual elapsed calendar years.
- `annual_volatility`
  Realized daily return volatility annualized with the active window's observed trading-days-per-year rate.
- `sharpe`
  Return per unit of realized volatility.
- `sortino`
  Return per unit of downside volatility.
- `max_drawdown`
  Worst peak-to-trough decline.
- `calmar`
  Return relative to worst drawdown.
- `average_turnover`
  How aggressively the portfolio changes through time.
- `benchmark_total_return`
  Total return for the preferred benchmark. The toolkit uses `SPY` when it is available, otherwise the first benchmark series in the backtest result.
- `benchmark_annual_return`
  Benchmark CAGR over the same active evaluation window.
- `benchmark_annual_volatility`
  Benchmark realized daily return volatility annualized over the same active evaluation window.
- `benchmark_sharpe`
  Benchmark annual return per unit of realized benchmark volatility.
- `benchmark_max_drawdown`
  Benchmark worst peak-to-trough decline over the same active evaluation window.
- `excess_return_vs_benchmark`
  Strategy total return minus benchmark total return. This is the simple benchmark-relative performance number.
- `annual_excess_return_vs_benchmark`
  Strategy CAGR minus benchmark CAGR over the same active evaluation window.
- `sharpe_vs_benchmark`
  Strategy Sharpe minus benchmark Sharpe. Use this to check whether a strategy beat the benchmark on risk-adjusted return, not just total return.

The active evaluation window starts at the first date in the submitted `weights` frame. This prevents years of inactive cash or zero returns before the first model position from diluting CAGR, volatility, Sharpe, Sortino, Calmar, and benchmark-relative metrics.

## 8. Writing Reports

Use:

```python
from portfolio_toolkit import write_quantstats_report, write_backtest_artifacts
```

`write_backtest_artifacts` writes:

- `weights.parquet`
- `nav.parquet`
- `returns.parquet`
- `turnover.parquet`
- `benchmarks.parquet`
- `metrics.json`
- `metrics_table.parquet`
- `quantstats.html`

This gives you a clean artifact folder per notebook run.

### QuantStats Alpha

The QuantStats HTML report has its own `Alpha` metric. In toolkit-generated reports this row is relabeled as `QuantStats Alpha vs <benchmark>` and a note is inserted at the top of the HTML.

That number is annualized regression alpha from realized daily returns:

```text
annualized mean(strategy_return - beta * benchmark_return)
```

It is not:

- the model's `expected_alpha` prediction column
- a `forward_alpha_*` supervised-learning target
- the same as `excess_return_vs_benchmark`

Use `excess_return_vs_benchmark` in `metrics.json` when you want the simple total-return difference between the strategy and the benchmark.

## 9. MLflow

The toolkit keeps MLflow intentionally simple.

It is team-shared by default:

- shared tracking server at `https://adams-macbook-pro.tail5ddc35.ts.net`
- no model registry
- no orchestration layer
- one consistent place to compare runs across the team

Initialize once:

```python
from portfolio_toolkit import init_mlflow

init_mlflow()
```

If you need to override the shared tracking server for a one-off local experiment, set:

```bash
export MLFLOW_TRACKING_URI=<another-uri>
```

Start a run:

```python
from portfolio_toolkit import start_run

with start_run("ridge_run_01", "shared_set_1", tags={"model_type": "ridge"}):
    ...
```

Log notebook outputs:

```python
from portfolio_toolkit import (
    log_predictions,
    log_portfolio,
    log_backtest,
    log_model_submission,
)

log_predictions(predictions)
log_portfolio(portfolio)
log_backtest(result)
```

`log_portfolio(...)` writes `weights.parquet`, which is the backtested portfolio allocation by date and ticker. It is not a trained model artifact.

Use `log_model_submission(...)` when you want the MLflow run to carry enough files and metadata to recreate model inference later:

```python
log_model_submission(
    {"model": "hannah_model.pt"},
    model_name="autoencoder_mlp_downstream",
    model_family="torch",
    feature_names=all_feature_names,
    target="forward_alpha_5d_vs_spy",
    horizon=5,
    rebalance_frequency="weekly",
    preprocessing={
        "scaler": "train_mean_std",
        "train_means": train_means.to_dict(),
        "train_stds": train_stds.to_dict(),
    },
    model_config={"latent_dim": LATENT_DIM},
    source_files=["MODELS/Hannah/baseline_autoencoder.ipynb"],
)
```

This creates a standardized MLflow artifact directory:

- `model_submission/manifest.json`
- `model_submission/artifacts/<model files>`
- `model_submission/source/<optional notebooks or code files>`

The manifest also records `rebalance_frequency`, such as `daily`, `weekly`, or `monthly`. The backtest still uses the dates in `weights.parquet`; this field makes the intended cadence visible in MLflow.

## 10. Model Submission Requirements

The submission bundle is meant to answer one practical question:

> Can another person recreate this model's predictions on a different dataset by reading the notebook and downloading the MLflow artifacts?

For that to work, every submission notebook must include two reusable functions.

### Required Function 1: `build_model_features(prices)`

This function rebuilds the exact model input table from a raw price frame.

Required behavior:

- accepts one long-form price dataframe with the standard price columns
- calls `build_features(...)` for shared toolkit features
- adds every custom notebook-local feature used by the model
- returns a dataframe with `date`, `ticker`, and all model feature columns
- does not use forward target columns such as `forward_return_*`, `forward_alpha_*`, or `forward_realized_vol_*`
- does not assume a fixed ticker universe beyond the rows provided in `prices`

Example:

```python
base_feature_names = [
    "momentum_20d",
    "vol_20d",
    "rsi_14",
    "price_to_sma_20d",
]
custom_feature_names = ["mom_vol_ratio"]
feature_names = base_feature_names + custom_feature_names

def build_model_features(prices):
    features = build_features(prices, feature_names=base_feature_names)
    features["mom_vol_ratio"] = (
        features["momentum_20d"] / features["vol_20d"].replace(0.0, float("nan"))
    )
    return features
```

### Required Function 2: `predict_from_prices(model, prices, dates=None, tickers=None)`

This function is the inference entrypoint. It should rebuild features, filter to requested dates/tickers when provided, run the model, and return the standard prediction contract.

Required output columns:

- `date`
- `ticker`
- `horizon`
- `expected_return`

Optional output columns:

- `expected_alpha`
- `expected_volatility`
- `uncertainty`

Required behavior:

- uses `build_model_features(prices)` rather than duplicating feature logic
- preserves the exact `feature_names` order logged in the manifest
- applies the same preprocessing used during training
- supports arbitrary ticker subsets through the optional `tickers` argument
- supports arbitrary evaluation dates through the optional `dates` argument
- returns only rows that the model actually scored
- validates with `validate_prediction_frame(...)` before backtesting

Example:

```python
def predict_from_prices(model, prices, dates=None, tickers=None):
    features = build_model_features(prices)

    if dates is not None:
        features = features.loc[features["date"].isin(pd.to_datetime(dates))].copy()

    if tickers is not None:
        tickers = [ticker.upper() for ticker in tickers]
        features = features.loc[features["ticker"].isin(tickers)].copy()

    scoring_frame = features.dropna(subset=feature_names).reset_index(drop=True)
    scores = model.predict(scoring_frame[feature_names])

    predictions = scoring_frame[["date", "ticker"]].copy()
    predictions["horizon"] = horizon
    predictions["expected_return"] = scores
    return predictions
```

### Model Loading Is Not A Required Function

Participants do not have to define a required `load_submission_model(...)` function. It is useful, but optional.

For official evaluation, model loading can be reconstructed from:

- the saved model artifact in `model_submission/artifacts/`
- the source notebook in `model_submission/source/`
- `manifest.json`
- the framework-specific model format

Common formats:

- Torch: `.pt` or `.pth`
- sklearn: `.pkl` or `.joblib`
- XGBoost: `.json`
- LightGBM: `.txt` or `.pkl`
- CatBoost: `.cbm`

If a participant uses a non-obvious save/load pattern, they should document it in the notebook or include an optional helper function.

### What `manifest.json` Must Explain

The manifest should make inference reconstruction unambiguous:

- `model_name`: submission/model name
- `model_family`: framework family, such as `torch`, `sklearn`, `xgboost`, `lightgbm`, `catboost`, or `other`
- `target`: training target, such as `forward_return_5d`
- `horizon`: prediction horizon
- `rebalance_frequency`: intended portfolio rebalance cadence, such as `daily`, `weekly`, or `monthly`
- `feature_names`: exact ordered model input columns
- `preprocessing`: scaler or normalization information
- `model_config`: architecture and hyperparameter details needed to recreate the model
- `artifact_files`: model files copied into MLflow
- `source_files`: notebooks or code files copied into MLflow

For Torch models, save enough config to instantiate the model class before loading the `state_dict`. For tree/sklearn models, prefer native portable formats and record the feature order.

### Minimum MLflow Submission Block

At the bottom of a notebook, after training and backtesting:

```python
with start_run(...):
    log_predictions(predictions)
    log_portfolio(portfolio)
    log_backtest(result)

    log_model_submission(
        {"model": model_path},
        model_name=model_name,
        model_family="xgboost",
        feature_names=feature_names,
        target=target_col,
        horizon=horizon,
        rebalance_frequency=rebalance_frequency,
        preprocessing={"scaler": "none"},
        model_config={
            "portfolio_builder": "weights_from_predictions_rank_long_only",
            "required_functions": ["build_model_features", "predict_from_prices"],
        },
        source_files=["MODELS/<Name>/<notebook>.ipynb"],
    )
```

### What Not To Do

- Do not treat `weights.parquet` as model weights; it is only portfolio allocations.
- Do not use future target columns inside `build_model_features(...)` or `predict_from_prices(...)`.
- Do not hide custom feature logic in an unreusable notebook cell.
- Do not rely on a fixed ticker list or fixed ticker order.
- Do not log a model artifact without the feature order and preprocessing metadata.

## 11. What To Log From Notebooks

Recommended parameters:

- dataset name
- split used for evaluation
- model family
- horizon
- feature names
- training window
- validation window
- portfolio builder
- top-k or rank settings
- cost basis points

Recommended artifacts:

- predictions parquet
- weights parquet for portfolio allocations
- `model_submission/` for reconstructable model files and inference metadata
- metrics json
- QuantStats report
- any notebook-local config dump or parameter summary

Recommended tags:

- project name
- owner or researcher name
- model family
- strategy type
- target horizon

## 12. Comparing Multiple Strategies

A good habit is to compare at least:

- your strategy
- equal weight
- inverse volatility
- momentum baseline

You can do that in one notebook and assemble a comparison dataframe from each `BacktestResult.metrics`.

## 13. Suggested Folder Pattern For Notebook Runs

Use one output folder per notebook experiment:

- `runs/ridge_h5_v1/`
- `runs/lstm_seq_v2/`
- `runs/direct_policy_ablation_01/`

Inside, keep:

- backtest artifacts
- optional prediction parquet
- optional notebook-local notes or config dump

This makes MLflow and local artifacts easier to reconcile later.
