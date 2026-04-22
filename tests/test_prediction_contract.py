from __future__ import annotations

import pandas as pd
import pytest

from portfolio_toolkit import load_prices, validate_prediction_frame


def test_prediction_contract_accepts_valid_frame(repo_root):
    prices = load_prices("shared_set_1", repo_root=repo_root)
    sample = prices.loc[prices["ticker"] != "SPY", ["date", "ticker"]].head(10).copy()
    sample["horizon"] = 5
    sample["expected_return"] = 0.01
    validated = validate_prediction_frame(sample, dataset_name="shared_set_1", horizon=5, repo_root=repo_root)
    assert len(validated) == len(sample)


def test_prediction_contract_rejects_missing_columns_and_bad_horizon(repo_root):
    with pytest.raises(ValueError):
        validate_prediction_frame(pd.DataFrame({"date": [], "ticker": []}), dataset_name="shared_set_1", repo_root=repo_root)

    bad = pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "ticker": ["AAPL"],
            "horizon": [0],
            "expected_return": [0.01],
        }
    )
    with pytest.raises(ValueError):
        validate_prediction_frame(bad, dataset_name="shared_set_1", repo_root=repo_root)


def test_prediction_contract_rejects_unknown_ticker(repo_root):
    bad = pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "ticker": ["ZZZZ"],
            "horizon": [5],
            "expected_return": [0.01],
        }
    )
    with pytest.raises(ValueError):
        validate_prediction_frame(bad, dataset_name="shared_set_1", horizon=5, repo_root=repo_root)
