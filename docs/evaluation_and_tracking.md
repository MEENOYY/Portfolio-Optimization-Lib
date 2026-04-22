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

- `total_return`
- `annual_return`
- `annual_volatility`
- `sharpe`
- `sortino`
- `max_drawdown`
- `calmar`
- `average_turnover`
- `benchmark_total_return`
- `excess_return_vs_benchmark`

Interpretation:

- `annual_return`
  How fast capital compounded over the evaluation window.
- `annual_volatility`
  Realized daily return volatility annualized.
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

## 9. MLflow

The toolkit keeps MLflow intentionally simple.

It is local-first:

- local SQLite database
- local artifact folder
- no model registry
- no orchestration layer

Initialize once:

```python
from portfolio_toolkit import init_mlflow

init_mlflow()
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
)

log_predictions(predictions)
log_portfolio(portfolio)
log_backtest(result)
```

## 10. What To Log From Notebooks

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
- weights parquet
- metrics json
- QuantStats report
- any notebook-local config dump or parameter summary

Recommended tags:

- project name
- owner or researcher name
- model family
- strategy type
- target horizon

## 11. Comparing Multiple Strategies

A good habit is to compare at least:

- your strategy
- equal weight
- inverse volatility
- momentum baseline

You can do that in one notebook and assemble a comparison dataframe from each `BacktestResult.metrics`.

## 12. Suggested Folder Pattern For Notebook Runs

Use one output folder per notebook experiment:

- `runs/ridge_h5_v1/`
- `runs/lstm_seq_v2/`
- `runs/direct_policy_ablation_01/`

Inside, keep:

- backtest artifacts
- optional prediction parquet
- optional notebook-local notes or config dump

This makes MLflow and local artifacts easier to reconcile later.
