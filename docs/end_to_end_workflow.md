# End-To-End Workflow

This guide shows the full notebook-first workflow from raw shared prices to a comparable backtest result.

## Overview

There are three layers in a normal research notebook:

1. shared input layer
2. model-specific research layer
3. shared evaluation layer

The shared layers make runs comparable. The middle layer is where developers keep their freedom.

## Step 1. Load The Shared Dataset

```python
from portfolio_toolkit import load_prices, get_dataset_spec, split_dates

dataset_name = "shared_set_1"
spec = get_dataset_spec(dataset_name)
prices = load_prices(dataset_name)
splits = split_dates(dataset_name)
```

What this gives you:

- a long-form OHLCV dataframe
- the exact ticker universe for that dataset preset
- the canonical train, validation, and test windows

Price columns:

- `date`
- `ticker`
- `open`
- `high`
- `low`
- `close`
- `adj_close`
- `volume`

## Step 2. Build Shared Features And Targets

```python
from portfolio_toolkit import (
    build_features,
    make_forward_return_target,
    make_forward_alpha_target,
    make_forward_realized_vol_target,
)

features = build_features(prices)
target_return = make_forward_return_target(prices, horizon=5)
target_alpha = make_forward_alpha_target(prices, horizon=5)
target_vol = make_forward_realized_vol_target(prices, window=5)
```

At this point you usually merge them:

```python
dataset = features.merge(target_return, on=["date", "ticker"], how="left")
dataset = dataset.merge(target_alpha, on=["date", "ticker"], how="left")
dataset = dataset.merge(target_vol, on=["date", "ticker"], how="left")
```

If you want custom features, add them here before splitting.

## Step 3. Slice Train, Validation, And Test

```python
from portfolio_toolkit import slice_split

train = slice_split(dataset, dataset_name, "train")
val = slice_split(dataset, dataset_name, "val")
test = slice_split(dataset, dataset_name, "test")
```

The usual pattern is:

- train on `train`
- tune on `val`
- report final comparisons on `test`

## Step 4. Train Your Model In The Notebook

This is intentionally up to the researcher.

Examples:

- regression on the panel dataframe
- boosted trees on the panel dataframe
- notebook-built rolling windows for an LSTM
- an autoencoder that learns compressed representations and then a downstream predictor
- a model that directly outputs portfolio weights

The toolkit does not force one training style.

## Step 5A. Forecast Model Path

If your model predicts stock-level quantities, the standard output is a prediction table.

Required columns:

- `date`
- `ticker`
- `horizon`
- `expected_return`

Optional standardized columns:

- `expected_alpha`
- `expected_volatility`
- `uncertainty`

Example:

```python
predictions = test.loc[:, ["date", "ticker"]].copy()
predictions["horizon"] = 5
predictions["expected_return"] = model_predictions
predictions["expected_alpha"] = alpha_predictions
predictions["expected_volatility"] = vol_predictions
predictions["uncertainty"] = uncertainty_estimates
```

Validate before doing anything else:

```python
from portfolio_toolkit import validate_prediction_frame

predictions = validate_prediction_frame(
    predictions,
    dataset_name=dataset_name,
    horizon=5,
)
```

## Step 5B. Direct Weights Path

If your notebook directly outputs allocations, skip the prediction contract and move straight to a weights table.

Shape:

- index: rebalance dates
- columns: tickers
- values: long-only weights that sum to `1.0`

Example:

```python
import pandas as pd
from portfolio_toolkit import validate_weights_frame

weights = pd.DataFrame(
    {
        "AAPL": [0.40, 0.35],
        "MSFT": [0.35, 0.40],
        "NVDA": [0.25, 0.25],
    },
    index=pd.to_datetime(["2022-01-03", "2022-02-01"]),
)

weights = validate_weights_frame(weights, dataset_name=dataset_name)
```

## Step 6A. Convert Predictions Into Weights

If you took the forecast path, use one of the portfolio builders.

```python
from portfolio_toolkit import (
    weights_from_predictions_top_k_equal,
    weights_from_predictions_rank_long_only,
    weights_from_predictions_risk_adjusted,
)

top_k_portfolio = weights_from_predictions_top_k_equal(
    predictions,
    k=5,
    dataset_name=dataset_name,
)

rank_portfolio = weights_from_predictions_rank_long_only(
    predictions,
    dataset_name=dataset_name,
)

risk_adjusted_portfolio = weights_from_predictions_risk_adjusted(
    predictions,
    dataset_name=dataset_name,
)
```

General guidance:

- `top_k_equal`
  Good first baseline when you want only the highest-scoring names.
- `rank_long_only`
  Good when you want every name in the universe to get some long-only weight.
- `risk_adjusted`
  Good when your model also produces `expected_volatility`.

## Step 6B. Wrap Direct Weights

If you already have weights, create a `PortfolioWeights` object:

```python
from portfolio_toolkit import PortfolioWeights

portfolio = PortfolioWeights(
    weights=weights,
    dataset_name=dataset_name,
    strategy_name="my_direct_policy",
    metadata={"notes": "custom direct allocation notebook"},
)
```

## Step 7. Run The Shared Backtest

Forecast path:

```python
from portfolio_toolkit import backtest_predictions

result = backtest_predictions(
    dataset_name,
    predictions,
    builder="top_k_equal",
    k=5,
)
```

Direct weights path:

```python
from portfolio_toolkit import backtest_weights

result = backtest_weights(dataset_name, portfolio)
```

What happens inside the backtest layer:

- prices are pivoted to wide format from `adj_close`
- weights are aligned to the next available trading day
- `bt` runs on daily prices and rebalances only on dates present in your weights index
- flat commissions are applied from the dataset preset
- benchmark return series are produced
- turnover is computed
- summary metrics are computed

Your rebalance frequency is therefore controlled by the dates you score and convert into weights. Daily prediction dates create daily target weights; weekly prediction dates create weekly target weights.

## Step 8. Write Reports And Artifacts

```python
from portfolio_toolkit import write_backtest_artifacts, build_metrics

metrics = build_metrics(result)
artifact_paths = write_backtest_artifacts(result, "runs/my_first_strategy")
```

Artifacts written:

- `weights.parquet`
- `nav.parquet`
- `returns.parquet`
- `turnover.parquet`
- `benchmarks.parquet`
- `metrics.json`
- `metrics_table.parquet`
- `quantstats.html`

Metrics and the QuantStats report are evaluated from the first submitted weight date through the final NAV date. This keeps inactive pre-test years from diluting annual return, volatility, Sharpe, Sortino, Calmar, and benchmark-relative annual metrics.

The QuantStats HTML report includes a benchmark-regression alpha metric. In generated reports this is labeled `QuantStats Alpha vs <benchmark>` because it is not the same as `expected_alpha`, `forward_alpha_*`, or `excess_return_vs_benchmark`. Use `metrics.json` for the toolkit's shared evaluation metrics.

## Step 9. Log To MLflow

```python
from portfolio_toolkit import (
    init_mlflow,
    start_run,
    log_predictions,
    log_portfolio,
    log_backtest,
)

init_mlflow()

with start_run(
    run_name="ridge_shared_set_1_h5",
    dataset_name=dataset_name,
    tags={"model_type": "ridge", "horizon": "5"},
):
    log_predictions(predictions)
    log_portfolio(top_k_portfolio)
    log_backtest(result)
```

Recommended params and tags to log from notebooks:

- dataset name
- split used for evaluation
- model family
- horizon
- feature names
- portfolio builder name
- rebalance frequency, such as `daily`, `weekly`, or `monthly`
- top-k or allocation assumptions

## Step 10. Compare Against Baselines

Use the built-in baselines to make sure your model is actually beating something simple.

```python
from portfolio_toolkit import baseline_weights, backtest_weights

equal_weight = baseline_weights(dataset_name, "equal_weight")
equal_result = backtest_weights(dataset_name, equal_weight)
```

Available baseline strategies:

- `equal_weight`
- `inverse_volatility`
- `momentum_20d`

## A Minimal Forecast Notebook Skeleton

```python
from portfolio_toolkit import (
    load_prices,
    build_features,
    make_forward_return_target,
    slice_split,
    validate_prediction_frame,
    weights_from_predictions_top_k_equal,
    backtest_weights,
    write_backtest_artifacts,
)

dataset_name = "shared_set_1"
prices = load_prices(dataset_name)
features = build_features(prices, feature_names=["momentum_20d", "vol_20d", "price_to_sma_20d"])
target = make_forward_return_target(prices, horizon=5)
frame = features.merge(target, on=["date", "ticker"], how="left").dropna()

train = slice_split(frame, dataset_name, "train")
test = slice_split(frame, dataset_name, "test")

feature_cols = ["momentum_20d", "vol_20d", "price_to_sma_20d"]
coefs = train[feature_cols].fillna(0.0).to_numpy()
y = train["forward_return_5d"].to_numpy()
beta, *_ = __import__("numpy").linalg.lstsq(coefs, y, rcond=None)

predictions = test.loc[:, ["date", "ticker"]].copy()
predictions["horizon"] = 5
predictions["expected_return"] = test[feature_cols].fillna(0.0).to_numpy() @ beta
predictions = validate_prediction_frame(predictions, dataset_name=dataset_name, horizon=5)

portfolio = weights_from_predictions_top_k_equal(predictions, k=5, dataset_name=dataset_name)
result = backtest_weights(dataset_name, portfolio)
artifacts = write_backtest_artifacts(result, "runs/minimal_example")
```

That is the core pattern the rest of the docs elaborate on.
