# Troubleshooting

This guide covers the most common issues a notebook user will hit.

## 1. `dataset preset 'shared_set_1' has no tickers yet`

Cause:

- the team has not filled in `configs/datasets.toml`

Fix:

- add real ticker symbols to the chosen dataset preset
- rerun `load_prices`

## 2. `prices frame must include benchmark ticker SPY`

Cause:

- the dataframe you passed into validation does not contain `SPY`
- or you hand-built a partial price frame that excludes the benchmark

Fix:

- use `load_prices(dataset_name)` instead of building the frame manually
- if you build a custom frame, make sure benchmark rows are included

## 3. `prediction frame contains unexpected tickers`

Cause:

- notebook predictions include tickers outside the shared dataset universe

Fix:

- confirm `dataset_name`
- confirm your model only predicts over `DatasetSpec.tickers`
- inspect `get_dataset_spec(dataset_name).tickers`

## 4. `prediction horizon must be positive`

Cause:

- horizon is zero, negative, or non-numeric

Fix:

- use a positive integer horizon such as `1`, `5`, `10`, or `20`

## 5. `prediction frame must contain exactly horizon=...`

Cause:

- you passed mixed horizons into a validation or backtest call that expects one horizon

Fix:

- filter the prediction dataframe to one horizon before calling validation or backtesting

## 6. `weights frame cannot be empty`

Cause:

- your notebook created no allocations
- filtering removed all weight rows

Fix:

- inspect the prediction stage
- inspect the date alignment between your predictions and the test split
- inspect whether all signals were null or filtered out

## 7. `each weights row must sum to 1.0`

Cause:

- portfolio logic created rows that do not fully invest capital

Fix:

- renormalize the row sums in the notebook
- rerun `validate_weights_frame`

## 8. `weights frame contains unexpected tickers`

Cause:

- your direct allocation notebook produced names outside the configured universe

Fix:

- make sure the weight columns match `get_dataset_spec(dataset_name).tickers`

## 9. `no weight rows align to the available trading calendar`

Cause:

- your rebalance dates are after the dataset ends
- your index is malformed
- your weights are for dates with no forward market data left

Fix:

- confirm the weights index is a valid datetime index
- confirm the dates lie inside the dataset range
- confirm you are not generating rebalance dates after `test_end`

## 10. Too Many Missing Feature Values

Cause:

- long-window features such as `price_to_sma_200d` need a large warmup period
- many rolling features naturally create missing values at the start of each ticker history

Fix:

- begin with shorter-window features
- drop missing rows only after you build the final modeling table
- inspect row counts before and after dropping nulls

## 11. My Sequence Model Wants Tensors, Not Long-Form Data

That is expected.

The toolkit intentionally stops at long-form features in v1.

Recommended fix:

- build shared features first
- sort by `ticker,date`
- create rolling windows in the notebook
- convert those windows into tensors for your framework

This is not a bug. It is part of the notebook-first design.

## 12. QuantStats Warning Noise

You may see deprecation warnings from seaborn through QuantStats.

If the report still writes successfully, that is usually non-blocking.

Fix:

- treat it as warning noise unless report generation fails
- upgrade the dependency set later if it becomes disruptive

## 13. MLflow Runs Are Not Showing Up

Checklist:

1. call `init_mlflow()` before logging
2. wrap logging in `with start_run(...):`
3. make sure the run block actually executes
4. make sure the `mlflow/` folder is writable

Optional local UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db
```

## 14. Downloaded Data Looks Old

Cause:

- `load_prices` reuses cached parquet files by default

Fix:

```python
prices = load_prices("shared_set_1", refresh=True)
```

## 15. Which Validation Test Should I Run?

Use:

```bash
python3 -m pytest -q tests/test_prediction_contract.py
```

When:

- your notebook emits predictions and you want to verify the prediction schema

Use:

```bash
python3 -m pytest -q tests/test_portfolio_validation.py
```

When:

- your notebook emits or builds weights and you want to verify long-only fully-invested rules

Use:

```bash
python3 -m pytest -q tests/test_backtest_and_mlflow_smoke.py
```

When:

- you want to verify that the shared backtest, QuantStats, and MLflow flow still works

## 16. Best Debugging Pattern Inside A Notebook

When a notebook is failing, debug in this order:

1. validate prices
2. validate features
3. inspect split row counts
4. validate predictions
5. validate weights
6. backtest
7. write artifacts
8. log to MLflow

That narrows failures to the exact layer that broke.
