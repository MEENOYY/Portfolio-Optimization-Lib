# Documentation

This toolkit is meant to be used as a shared research foundation for a portfolio project.

It is not a training platform and it is not a production trading system. The intended use is:

1. agree on shared datasets and split boundaries
2. load shared prices in a notebook
3. use shared feature helpers when they are useful
4. add custom features or custom tensors in the notebook if needed
5. train any model type you want
6. emit a standardized prediction table or a direct weights table
7. run the shared backtest, statistics, QuantStats, and MLflow logging flow

## Start Here

- [Getting Started](getting_started.md)
- [End-to-End Workflow](end_to_end_workflow.md)

## Core Topics

- [Data, Features, and Targets](data_features_and_targets.md)
- [Model Workflows](model_workflows.md)
- [Evaluation and Tracking](evaluation_and_tracking.md)
- [API Reference](api_reference.md)
- [Troubleshooting](troubleshooting.md)

## Mental Model

There are two normal workflows in this repo.

### 1. Forecast-first workflow

Use this when your model predicts stock-level quantities such as:

- `expected_return`
- `expected_alpha`
- `expected_volatility`
- `uncertainty`

You then convert those predictions into portfolio weights with one of the shared portfolio builders and run the shared backtest.

This is the default path for:

- linear models
- ridge/lasso/elastic-net style models
- tree models
- boosted tree models
- neural forecasting models
- LSTM, TCN, transformer, or autoencoder-based predictors

### 2. Direct-weights workflow

Use this when your model or rule directly outputs portfolio weights.

You skip the prediction-to-portfolio conversion step and validate the weights directly before backtesting.

This is the right path for:

- direct allocation heuristics
- optimization notebooks
- end-to-end learned portfolio policies
- any custom direct portfolio management model

## What The Toolkit Standardizes

- dataset presets
- date splits
- cached price loading
- built-in starter features
- target helpers
- prediction schema
- weight validation rules
- backtest implementation
- statistics and QuantStats output
- MLflow logging

## What The Toolkit Does Not Standardize

- the model class
- the training framework
- the notebook workflow
- the feature set a researcher must use
- the loss function
- the hyperparameter search approach
- the internal training loop

That split is intentional. The shared layer exists to make comparisons fair without forcing everyone into the same research style.
