# Data, Features, And Targets

This guide explains the shared input layer of the toolkit.

## 1. Shared Dataset Presets

The repo defines exactly three dataset presets in `configs/datasets.toml`:

- `shared_set_1`
- `shared_set_2`
- `shared_set_3`

The starter intent of each preset is:

- `shared_set_1`
  Full S&P 500 universe
- `shared_set_2`
  Growth/tech/innovation names
- `shared_set_3`
  Defensive/quality/value names

Each preset represents:

- a shared ticker universe
- a benchmark ticker
- a date range for data download
- fixed train, validation, and test windows
- shared transaction-cost assumptions

The intended usage is simple:

- the team agrees on the ticker lists
- everyone uses the same preset names in notebooks
- everyone inherits the same split boundaries

## 2. Price Loading

The main entrypoint is:

```python
from portfolio_toolkit import load_prices

prices = load_prices("shared_set_1")
```

Behavior:

- downloads daily data from `yfinance` the first time
- caches the result to `data_cache/<dataset_name>.parquet`
- always includes `SPY` in the downloaded universe
- validates the result before returning it

Use `refresh=True` when you need a fresh pull:

```python
prices = load_prices("shared_set_1", refresh=True)
```

## 3. Price Frame Contract

The price dataframe is long-form and must contain:

- `date`
- `ticker`
- `open`
- `high`
- `low`
- `close`
- `adj_close`
- `volume`

Validation rules:

- no duplicate `date,ticker` rows
- no negative OHLCV values
- `high >= low`
- tickers must belong to the dataset universe when a dataset name is provided
- the benchmark ticker must be present

## 4. Shared Split Helpers

Use:

```python
from portfolio_toolkit import split_dates, slice_split
```

Examples:

```python
split_dates("shared_set_1")
train = slice_split(prices, "shared_set_1", "train")
val = slice_split(prices, "shared_set_1", "val")
test = slice_split(prices, "shared_set_1", "test")
```

This is the only split logic the team should use for formal comparisons.

## 5. Built-In Feature Catalog

Use:

```python
from portfolio_toolkit import list_features, build_features

all_feature_names = list_features()
feature_frame = build_features(prices)
```

Or select a subset:

```python
feature_frame = build_features(
    prices,
    feature_names=[
        "momentum_20d",
        "vol_20d",
        "rsi_14",
        "price_to_sma_20d",
    ],
)
```

### Returns And Log Returns

- `return_1d`
- `return_2d`
- `return_5d`
- `return_10d`
- `return_20d`
- `return_60d`
- `log_return_1d`
- `log_return_5d`
- `log_return_20d`

### Volatility And Risk

- `vol_5d`
- `vol_10d`
- `vol_20d`
- `vol_60d`
- `downside_vol_20d`
- `upside_vol_20d`
- `atr_14`
- `beta_20d_spy`
- `beta_60d_spy`

### Momentum And Trend

- `momentum_5d`
- `momentum_10d`
- `momentum_20d`
- `momentum_60d`
- `momentum_120d`
- `price_to_sma_10d`
- `price_to_sma_20d`
- `price_to_sma_50d`
- `price_to_sma_200d`
- `price_to_ema_12d`
- `price_to_ema_26d`
- `macd`
- `macd_signal`
- `macd_hist`

### Oscillators

- `rsi_7`
- `rsi_14`
- `rsi_28`
- `bollinger_z_20d`
- `stoch_k_14`
- `stoch_d_14`

### Volume And Liquidity

- `volume_change_1d`
- `volume_change_5d`
- `volume_change_20d`
- `volume_zscore_20d`
- `volume_zscore_60d`
- `dollar_volume`
- `dollar_volume_ratio_20d`

### Range And Positioning

- `intraday_range`
- `close_open_gap`
- `close_location_in_range`
- `distance_to_20d_high`
- `distance_to_20d_low`
- `distance_to_60d_high`
- `distance_to_60d_low`

### Benchmark-Relative

- `excess_return_5d_vs_spy`
- `excess_return_20d_vs_spy`
- `excess_return_60d_vs_spy`
- `relative_momentum_20d_vs_spy`

### Distribution Shape

- `skew_20d`
- `kurtosis_20d`

## 6. Feature Output Contract

The shared feature frame is also long-form:

- `date`
- `ticker`
- one or more feature columns

Validation rules:

- `date,ticker` must be unique
- at least one feature column must be present

## 7. Built-In Target Helpers

These helpers make it easier to train richer stock-level forecast models.

### Forward Return

```python
target_return = make_forward_return_target(prices, horizon=5)
```

Output column:

- `forward_return_5d`

Supported practice:

- use `1`, `5`, `10`, or `20` day horizons

### Forward Alpha

```python
target_alpha = make_forward_alpha_target(prices, horizon=5)
```

Output column:

- `forward_alpha_5d_vs_spy`

This measures forward return minus benchmark forward return.

### Forward Realized Volatility

```python
target_vol = make_forward_realized_vol_target(prices, window=5)
```

Output column:

- `forward_realized_vol_5d`

This is annualized realized volatility over the next `window` trading days.

## 8. How To Use Features For Different Model Families

### Linear And Tree Models

Use the long-form feature frame directly. Typical workflow:

- build features
- merge targets
- drop rows with missing target
- split into train, validation, and test
- fit on panel rows

### Sequence Models

The toolkit does not ship a tensor pipeline in v1. That is intentional.

For sequence models:

- build the long-form feature frame first
- sort by `ticker,date`
- create rolling windows in the notebook
- convert them into tensors for your framework

This keeps the shared layer simple while still letting deep-learning users work normally.

### Autoencoders And Representation Learning

Use the same shared feature frame as the input table for:

- reconstruction targets
- compressed latent factors
- downstream fine-tuning

Again, the tensor or latent-data mechanics live in the notebook.

## 9. Recommended Feature Engineering Practice

Start simple:

- one momentum feature
- one volatility feature
- one price-to-trend feature
- one benchmark-relative feature
- one volume/liquidity feature

Then add complexity only if validation and test results justify it.

Good first subsets:

- `momentum_20d`
- `vol_20d`
- `price_to_sma_20d`
- `rsi_14`
- `volume_zscore_20d`
- `excess_return_20d_vs_spy`

## 10. Adding Notebook-Local Features

The easiest pattern is:

1. build the shared feature frame
2. add your custom columns in the notebook
3. validate your final modeling frame if needed

Example:

```python
features = build_features(prices, feature_names=["momentum_20d", "vol_20d"])
features["mom_vol_ratio"] = features["momentum_20d"] / features["vol_20d"].replace(0.0, float("nan"))
```

That keeps the shared layer useful without blocking experimentation.

## 11. Common Pitfalls

- Using features from one dataset preset and predictions from another
- Forgetting that `SPY` is included as the benchmark
- Creating sequence tensors before sorting by `ticker,date`
- Dropping too many rows after adding long-window features like `price_to_sma_200d`
- Comparing models with different split boundaries instead of the shared dataset splits
