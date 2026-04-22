# Portfolio Toolkit

Minimal notebook-first research toolkit for shared datasets, feature engineering, portfolio construction, backtesting, QuantStats reporting, and MLflow tracking.

This repo is intentionally small. It does not train models for your team. It gives everyone:

- the same datasets and split boundaries
- the same starter feature helpers and targets
- the same prediction and weight validation rules
- the same backtesting and statistics layer
- the same MLflow logging pattern

Your developers should still work in notebooks, choose their own model classes, write their own training loops, and experiment however they normally would.

## Quick Start

1. Review the starter ticker universes in `configs/datasets.toml` and change them if your team wants a different shared universe
2. Install with:

```bash
python3 -m pip install -e ".[dev]"
```

3. Start with one of the committed notebook templates in `notebooks/templates/`
4. Load shared prices, build features, train your own model, emit predictions or weights, backtest, and log to MLflow

Starter datasets included:

- `shared_set_1`
  Full S&P 500 universe
- `shared_set_2`
  Growth/tech/innovation universe
- `shared_set_3`
  Defensive/quality/value universe

## Documentation

- [Docs Index](docs/README.md)
- [Getting Started](docs/getting_started.md)
- [End-to-End Workflow](docs/end_to_end_workflow.md)
- [Data, Features, and Targets](docs/data_features_and_targets.md)
- [Model Workflows](docs/model_workflows.md)
- [Evaluation and Tracking](docs/evaluation_and_tracking.md)
- [API Reference](docs/api_reference.md)
- [Troubleshooting](docs/troubleshooting.md)

## Notebook Templates

- `notebooks/templates/starter_forecast_model.ipynb`
- `notebooks/templates/mlp_end_to_end_workflow.ipynb`
- `notebooks/templates/feature_playground.ipynb`
- `notebooks/templates/direct_weights_workflow.ipynb`
- `notebooks/templates/baseline_compare.ipynb`

## Validation Checks

Run targeted validation checks when you want to sanity-check a notebook output before sharing results:

```bash
python3 -m pytest -q tests/test_prediction_contract.py
python3 -m pytest -q tests/test_portfolio_validation.py
python3 -m pytest -q tests/test_backtest_and_mlflow_smoke.py
```
