from __future__ import annotations

from portfolio_toolkit import get_dataset_spec, split_dates


def test_dataset_presets_load_and_have_non_overlapping_splits(repo_root):
    spec = get_dataset_spec("shared_set_1", repo_root=repo_root)
    splits = split_dates("shared_set_1", repo_root=repo_root)
    assert spec.benchmark_ticker == "SPY"
    assert splits["train"][1] < splits["val"][0]
    assert splits["val"][1] < splits["test"][0]
