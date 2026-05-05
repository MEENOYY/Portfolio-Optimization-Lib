# Model Workflows

This guide explains how to use the toolkit with any model type.

The core design principle is simple:

- the toolkit standardizes inputs and evaluation
- developers still train their models however they normally would

## 1. Two Supported Research Styles

### Forecast Models

These models output stock-level forecasts such as:

- expected return
- expected alpha
- expected volatility
- uncertainty

Forecast models are usually the cleanest fit for this toolkit because they separate:

- stock forecasting
- portfolio construction

### Direct-Weights Models

These models or procedures output:

- one weight vector per rebalance date

Use this path when the notebook directly decides the portfolio rather than first predicting stock-level quantities.

## 2. Standard Prediction Contract

If your model follows the forecast path, the prediction table must contain:

Required:

- `date`
- `ticker`
- `horizon`
- `expected_return`

Optional:

- `expected_alpha`
- `expected_volatility`
- `uncertainty`

Why these fields:

- `expected_return`
  Gives the shared portfolio builders a ranking or allocation signal.
- `expected_alpha`
  Lets researchers reason about benchmark-relative edge.
- `expected_volatility`
  Supports risk-aware ranking or risk-adjusted portfolio builders.
- `uncertainty`
  Gives room for model-confidence experiments without forcing one interpretation.

What is not required:

- predicted Sharpe

If you have both expected return and expected volatility, compute Sharpe downstream in the notebook rather than making it part of the required contract.

## 3. Linear And Classical Regression Models

Examples:

- linear regression
- ridge
- lasso
- elastic net
- partial least squares

These models usually work best directly on the long-form feature panel.

Typical workflow:

1. build the shared feature frame
2. merge the return or alpha target
3. select feature columns
4. fit the model on train rows
5. generate predictions for validation or test rows
6. assemble the standardized prediction table
7. validate predictions
8. convert predictions into weights
9. backtest

Use cases:

- strong baseline models
- interpretable feature importance studies
- feature selection experiments
- fast iteration when data is limited

## 4. Tree And Boosted Tree Models

Examples:

- random forest
- XGBoost
- LightGBM
- CatBoost

The shared toolkit is still useful here because the input format is the same.

Typical workflow:

1. start from the same panel features and targets as linear models
2. handle missing values and categorical choices in the notebook as needed
3. train the model with your preferred library
4. emit the standard prediction table
5. evaluate with the shared backtest layer

Good reasons to use this path:

- nonlinear interactions
- robust performance on tabular features
- quick experimentation without writing sequence code

## 5. Neural Forecasting Models

Examples:

- MLPs on panel features
- sequence MLPs
- LSTM models
- GRU models
- TCNs
- transformers over time

The key point is that the toolkit does not try to hide your deep-learning workflow from you.

You still:

- build tensors in the notebook
- define your own network class
- choose your optimizer, scheduler, batching, and loss
- handle masking, padding, normalization, or scaling however you normally do

The shared layer only gives you:

- shared raw price data
- shared starter features
- shared targets
- shared split boundaries
- shared output validation
- shared evaluation

### Recommended Neural Workflow

1. load prices
2. build shared features
3. optionally add custom features
4. sort by `ticker,date`
5. build rolling windows per ticker in the notebook
6. train your network
7. map outputs back to `date,ticker,horizon`
8. create the standardized prediction table
9. validate predictions
10. backtest

That pattern works for shallow or deep LSTMs, TCNs with many blocks, transformers, or custom attention models.

## 6. Autoencoders And Representation Learning

Autoencoder-style models fit naturally too, but usually in two stages:

1. learn a latent representation from the shared feature frame or price-derived tensors
2. train a downstream head or portfolio rule on top of the learned representation

Possible outputs:

- expected return from a predictor head
- expected alpha from a predictor head
- direct scores used to build weights

The toolkit does not care how the representation is learned as long as the final artifact becomes either:

- a valid prediction dataframe
- or valid portfolio weights

## 7. Direct Portfolio Management Models

Some researchers may want their notebook to output weights directly.

That is allowed.

In that case the model can produce a dataframe like:

- index: rebalance dates
- columns: tickers
- values: long-only weights

Then validate it:

```python
from portfolio_toolkit import validate_weights_frame

weights = validate_weights_frame(weights, dataset_name=dataset_name)
```

And wrap it:

```python
from portfolio_toolkit import PortfolioWeights

portfolio = PortfolioWeights(
    weights=weights,
    dataset_name=dataset_name,
    strategy_name="my_direct_allocator",
)
```

Then backtest with the shared wrapper.

## 8. Choosing Between Forecast Models And Direct Weights

Use forecast models when:

- you want a cleaner research decomposition
- you want to compare stock-picking quality separately from allocation rules
- you want to reuse the same portfolio builders across many models

Use direct weights when:

- allocation is the research problem itself
- the notebook already solves a constrained portfolio problem
- you are testing an end-to-end policy model

For team comparability, forecast models are usually easier to compare. For advanced research, direct weights are fine as long as the weights pass validation.

## 9. Recommended Output Fields By Model Family

### For most tabular regressors

- required: `expected_return`
- optional: `expected_alpha`

### For uncertainty-aware models

- required: `expected_return`
- optional: `uncertainty`

### For risk-aware models

- required: `expected_return`
- optional: `expected_volatility`

### For benchmark-relative models

- required: `expected_return`
- optional: `expected_alpha`

### For direct portfolio models

- no prediction table required
- emit validated weights directly

## 10. Portfolio Builders For Forecast Models

If your model emits predictions, you can convert them into weights with:

- `weights_from_predictions_top_k_equal`
- `weights_from_predictions_rank_long_only`
- `weights_from_predictions_risk_adjusted`

How to choose:

- use `top_k_equal` when you only trust your top names
- use `rank_long_only` when you want broad exposure across the full universe
- use `risk_adjusted` when you predict both return and volatility

## 11. Notebook Discipline That Helps Every Model Type

- keep one notebook focused on one modeling idea
- save the exact feature list used in that notebook
- choose and log the rebalance frequency used to turn predictions into weights
- validate predictions or weights before backtesting
- compare against at least one shared baseline
- log the run to MLflow with enough tags to understand it later

## 12. Minimal Forecast Output Example

```python
predictions = test.loc[:, ["date", "ticker"]].copy()
predictions["horizon"] = 5
predictions["expected_return"] = test_scores
predictions["expected_alpha"] = test_alpha
predictions["expected_volatility"] = test_vol
predictions["uncertainty"] = test_uncertainty
```

For formal runs, filter the prediction dates to the intended rebalance cadence before converting predictions into weights. For example, weekly scoring should produce one prediction row per ticker per weekly rebalance date, not one row per ticker per trading day.

## 13. Minimal Direct Weights Example

```python
weights = pd.DataFrame(
    {
        "AAPL": [0.5, 0.4],
        "MSFT": [0.3, 0.4],
        "NVDA": [0.2, 0.2],
    },
    index=pd.to_datetime(["2022-01-03", "2022-02-01"]),
)
```

## 14. What The Toolkit Expects From Researchers

It expects only a few things:

- use a shared dataset preset
- respect the shared split boundaries for formal comparisons
- output a valid prediction table or a valid weights table
- use the shared backtest/statistics layer when reporting results

Everything else is intentionally left to the notebook author.
