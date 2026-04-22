# Getting Started

This guide gets a new teammate from a blank checkout to a working notebook and a first shared backtest.

## 1. What You Need

- Python 3.12 or newer
- a virtual environment or pyenv environment
- internet access when downloading prices for the first time
- the shared dataset presets from `configs/datasets.toml`

Install the repo:

```bash
python3 -m pip install -e ".[dev]"
```

## 2. Review The Shared Datasets

The repo ships three starter shared dataset presets in `configs/datasets.toml`:

- `shared_set_1`
- `shared_set_2`
- `shared_set_3`

They are intended as three different research universes:

- `shared_set_1`
  Full S&P 500 universe
- `shared_set_2`
  Growth, tech, internet, software, and semiconductor names
- `shared_set_3`
  More defensive, quality-oriented names across staples, healthcare, utilities, energy, and financials

You can keep them as-is or replace them with your team's preferred universes later.

All three presets already share the same fixed windows:

- train: `2014-01-02` to `2019-12-31`
- val: `2020-01-02` to `2021-12-31`
- test: `2022-01-03` to `2025-12-31`

Those shared windows are one of the main ways the repo keeps comparisons fair.

Important details:

- `SPY` is always used as the default benchmark
- `SPY` is always included in downloads even if it is not in `tickers`
- `cost_bps` defaults to `10.0`
- the first `shared_set_1` download is much larger than the other sets because it pulls the full S&P 500 universe

## 3. Understand The Repo Layout

- `src/portfolio_toolkit/`
  The library code you import from notebooks.
- `configs/datasets.toml`
  Shared dataset presets and split values.
- `configs/mlflow.toml`
  Local MLflow settings.
- `notebooks/templates/`
  Starting points for forecast models, features, direct weights, and baseline comparisons.
- `tests/`
  Lightweight validation tests you can run when debugging a notebook workflow.
- `data_cache/`
  Created automatically when price data is downloaded and cached.
- `mlflow/`
  Created automatically when MLflow is initialized and runs are logged.

## 4. Pick The Right First Notebook

Use one of the committed notebook templates:

- `starter_forecast_model.ipynb`
  Start here if your model predicts returns, alpha, volatility, or uncertainty.
- `mlp_end_to_end_workflow.ipynb`
  Start here if you want a heavily commented end-to-end example of a neural-style forecasting workflow using the shared toolkit.
- `feature_playground.ipynb`
  Start here if you want to explore the built-in feature set or create your own features.
- `direct_weights_workflow.ipynb`
  Start here if your research directly outputs portfolio weights.
- `baseline_compare.ipynb`
  Start here if you want to understand the default backtest/statistics layer before building a model.

## 5. Your First End-To-End Run

The fastest way to sanity-check the repo is:

1. use `shared_set_1` if you want the broadest shared comparison universe, or start with `shared_set_2` / `shared_set_3` if you want faster first downloads
2. open `baseline_compare.ipynb`
3. run the baseline comparison
4. confirm that:
   - data loads
   - splits are respected
   - weights validate
   - backtest metrics are produced
   - QuantStats report is written

After that, move to `starter_forecast_model.ipynb`.

## 6. Typical Team Workflow

For most researchers, the normal flow is:

1. choose one shared dataset preset
2. load the shared prices
3. build shared features and optional targets
4. create notebook-local custom features if needed
5. slice train, validation, and test windows
6. train a model in the notebook
7. produce a standardized prediction table
8. validate the prediction table
9. convert predictions into weights
10. validate the weights
11. run the shared backtest
12. generate QuantStats output
13. log params, artifacts, and metrics to MLflow

## 7. When To Use Shared Features vs Custom Features

Use shared features when:

- you want a fast baseline
- you are building a first version of a model
- you want easier comparability across notebooks
- you do not have strong finance-specific feature ideas yet

Use custom notebook-local features when:

- you have a new idea you want to test quickly
- you need architecture-specific preprocessing
- you want to build tensors or windows for deep sequence models

The team-friendly pattern is:

- start with shared features
- add notebook-local experiments
- move generally useful ideas back into the shared feature layer later

## 8. When To Use Predictions vs Direct Weights

Use predictions when:

- your model forecasts expected return or alpha
- you want to compare multiple models through the same portfolio construction rules
- you want cleaner separation between forecasting and allocation

Use direct weights when:

- your notebook directly solves for allocations
- you are testing an optimization method
- you have a model that already produces weights

Both are supported. Forecast models use the prediction contract. Direct-weight workflows use `PortfolioWeights`.

## 9. Recommended First Week For A New Developer

Day 1:

- review the starter dataset presets with the team
- run `baseline_compare.ipynb`
- read [Data, Features, and Targets](data_features_and_targets.md)

Day 2:

- train a simple linear baseline in `starter_forecast_model.ipynb`
- produce a prediction table
- backtest it

Day 3 and beyond:

- branch into your own model family
- add custom features or tensors
- keep using the same shared validation and backtest layer

## 10. Minimum Commands To Know

Install:

```bash
python3 -m pip install -e ".[dev]"
```

Validation checks:

```bash
python3 -m pytest -q tests/test_prediction_contract.py
python3 -m pytest -q tests/test_portfolio_validation.py
python3 -m pytest -q tests/test_backtest_and_mlflow_smoke.py
```

Optional MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db
```
