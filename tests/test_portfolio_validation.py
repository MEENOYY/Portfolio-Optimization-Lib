from __future__ import annotations

import pandas as pd
import pytest

from portfolio_toolkit import validate_weights_frame


def test_portfolio_validation_accepts_good_weights(repo_root):
    weights = pd.DataFrame({"AAPL": [0.5], "MSFT": [0.5]}, index=pd.DatetimeIndex(["2024-01-02"]))
    validated = validate_weights_frame(weights, dataset_name="shared_set_1", repo_root=repo_root)
    assert abs(validated.sum(axis=1).iloc[0] - 1.0) < 1e-9


def test_portfolio_validation_catches_bad_rows(repo_root):
    bad_sum = pd.DataFrame({"AAPL": [0.7], "MSFT": [0.1]}, index=pd.DatetimeIndex(["2024-01-02"]))
    with pytest.raises(ValueError):
        validate_weights_frame(bad_sum, dataset_name="shared_set_1", repo_root=repo_root)

    negative = pd.DataFrame({"AAPL": [-0.2], "MSFT": [1.2]}, index=pd.DatetimeIndex(["2024-01-02"]))
    with pytest.raises(ValueError):
        validate_weights_frame(negative, dataset_name="shared_set_1", repo_root=repo_root)

    unknown = pd.DataFrame({"AAPL": [0.5], "ZZZZ": [0.5]}, index=pd.DatetimeIndex(["2024-01-02"]))
    with pytest.raises(ValueError):
        validate_weights_frame(unknown, dataset_name="shared_set_1", repo_root=repo_root)
