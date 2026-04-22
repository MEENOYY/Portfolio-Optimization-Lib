from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import get_dataset_spec


def split_dates(
    dataset_name: str,
    *,
    repo_root: str | Path | None = None,
) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    spec = get_dataset_spec(dataset_name, repo_root=repo_root)
    return {
        "train": (pd.Timestamp(spec.train_start), pd.Timestamp(spec.train_end)),
        "val": (pd.Timestamp(spec.val_start), pd.Timestamp(spec.val_end)),
        "test": (pd.Timestamp(spec.test_start), pd.Timestamp(spec.test_end)),
    }


def slice_split(
    frame: pd.DataFrame,
    dataset_name: str,
    split_name: str,
    *,
    repo_root: str | Path | None = None,
) -> pd.DataFrame:
    if "date" not in frame.columns:
        raise ValueError("slice_split expects a dataframe with a date column")
    splits = split_dates(dataset_name, repo_root=repo_root)
    try:
        start, end = splits[split_name]
    except KeyError as exc:
        raise KeyError(f"unknown split '{split_name}'") from exc
    dated = frame.copy()
    dated["date"] = pd.to_datetime(dated["date"], utc=True).dt.tz_localize(None)
    return dated.loc[(dated["date"] >= start) & (dated["date"] <= end)].reset_index(drop=True)
