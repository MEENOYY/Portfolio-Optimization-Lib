from __future__ import annotations

from contextlib import contextmanager, nullcontext
import json
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import mlflow
import pandas as pd

from .config import dataset_identifier, dataset_spec_dict, load_mlflow_settings, resolve_dataset_spec
from .contracts import BacktestResult, DatasetSpec, PortfolioWeights


def _repo_root(repo_root: str | Path | None = None) -> Path:
    return Path("." if repo_root is None else repo_root).resolve()


def _resolve_sqlite_uri(uri: str, repo_root: Path) -> str:
    prefix = "sqlite:///"
    if not uri.startswith(prefix):
        return uri
    target = Path(uri.removeprefix(prefix))
    if target.is_absolute():
        return uri
    return f"sqlite:///{(repo_root / target).resolve()}"


def _is_remote_tracking_uri(uri: str) -> bool:
    return uri.startswith("http://") or uri.startswith("https://")


def _tracking_paths(repo_root: Path) -> tuple[Path, Path]:
    mlflow_dir = repo_root / "mlflow"
    lock_path = mlflow_dir / ".mlflow.lock"
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    return mlflow_dir, lock_path


@contextmanager
def _mlflow_lock(repo_root: Path):
    if not HAS_FCNTL:
        yield
        return
    _, lock_path = _tracking_paths(repo_root)
    with lock_path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def init_mlflow(repo_root: str | Path = ".") -> dict[str, str]:
    root = _repo_root(repo_root)
    settings = load_mlflow_settings(root)
    tracking_uri = _resolve_sqlite_uri(settings.tracking_uri, root)
    mlflow.set_tracking_uri(tracking_uri)
    if _is_remote_tracking_uri(tracking_uri):
        return {
            "tracking_uri": tracking_uri,
            "backend_store_uri": tracking_uri,
            "artifact_root": "",
            "db_path": "",
        }

    mlflow_dir, _ = _tracking_paths(root)
    artifact_root = settings.artifact_root_path(root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    db_path = mlflow_dir / "mlflow.db"
    db_path.touch(exist_ok=True)
    return {
        "tracking_uri": tracking_uri,
        "backend_store_uri": _resolve_sqlite_uri(settings.backend_store_uri, root),
        "artifact_root": str(artifact_root.resolve()),
        "db_path": str(db_path.resolve()),
    }


def _get_or_create_experiment(experiment_name: str, artifact_root: Path | None) -> str:
    existing = mlflow.get_experiment_by_name(experiment_name)
    if existing is not None:
        return existing.experiment_id
    if artifact_root is None:
        return mlflow.create_experiment(experiment_name)
    return mlflow.create_experiment(experiment_name, artifact_location=artifact_root.as_uri())


@contextmanager
def start_run(
    run_name: str,
    dataset_name: str | DatasetSpec,
    tags: dict[str, str] | None = None,
    *,
    repo_root: str | Path = ".",
):
    root = _repo_root(repo_root)
    spec = resolve_dataset_spec(dataset_name, repo_root=root)
    dataset_id = dataset_identifier(spec, repo_root=root)
    settings = load_mlflow_settings(root)
    layout = init_mlflow(root)
    artifact_root = Path(layout["artifact_root"]) if layout["artifact_root"] else None
    lock = _mlflow_lock(root) if not _is_remote_tracking_uri(layout["tracking_uri"]) else nullcontext()
    with lock:
        mlflow.set_tracking_uri(layout["tracking_uri"])
        experiment_name = f"{settings.experiment_prefix}_{dataset_id}"
        experiment_id = _get_or_create_experiment(experiment_name, artifact_root)
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
            merged_tags = {"dataset_name": dataset_id, **(tags or {})}
            if spec.kind == "custom":
                merged_tags["dataset_kind"] = "custom"
            mlflow.set_tags(merged_tags)
            if spec.kind == "custom":
                mlflow.log_params(
                    {
                        "dataset_kind": spec.kind,
                        "dataset_benchmark_ticker": spec.benchmark_ticker,
                        "dataset_start_date": spec.start_date.isoformat(),
                        "dataset_end_date": spec.end_date.isoformat(),
                        "dataset_ticker_count": len(spec.tickers),
                    }
                )
                _log_json_artifact(dataset_spec_dict(spec), "dataset_spec.json")
            yield run


def _log_dataframe(df: pd.DataFrame, artifact_name: str) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / artifact_name
        df.to_parquet(path, index=False)
        mlflow.log_artifact(str(path))


def _log_json_artifact(payload: dict[str, Any], artifact_name: str) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / artifact_name
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        mlflow.log_artifact(str(path))


def log_predictions(df: pd.DataFrame) -> None:
    _log_dataframe(df, "predictions.parquet")


def log_portfolio(weights: PortfolioWeights | pd.DataFrame) -> None:
    frame = weights.weights if isinstance(weights, PortfolioWeights) else weights
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "weights.parquet"
        frame.to_parquet(path)
        mlflow.log_artifact(str(path))


def log_backtest(result: BacktestResult) -> None:
    if result.metrics:
        mlflow.log_metrics({key: float(value) for key, value in result.metrics.items()})
    if result.artifact_paths:
        log_report_artifacts(result.artifact_paths)


def log_report_artifacts(paths: dict[str, str] | list[str]) -> None:
    if isinstance(paths, dict):
        iterable = paths.values()
    else:
        iterable = paths
    for path in iterable:
        artifact_path = Path(path)
        if artifact_path.exists():
            mlflow.log_artifact(str(artifact_path))


def _validate_json_serializable(value: Any, label: str) -> None:
    try:
        json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be JSON-serializable") from exc


def _normalize_model_artifacts(model_artifacts: Mapping[str, str | Path] | Sequence[str | Path]) -> dict[str, Path]:
    if isinstance(model_artifacts, Mapping):
        if not model_artifacts:
            raise ValueError("model_artifacts cannot be empty")
        normalized = {str(name): Path(path) for name, path in model_artifacts.items()}
    else:
        if isinstance(model_artifacts, (str, Path)):
            raise TypeError("model_artifacts must be a mapping or sequence of file paths")
        normalized = {Path(path).name: Path(path) for path in model_artifacts}
        if not normalized:
            raise ValueError("model_artifacts cannot be empty")

    for logical_name, path in normalized.items():
        if not logical_name:
            raise ValueError("model_artifacts cannot contain empty logical names")
        if not path.exists():
            raise FileNotFoundError(f"model artifact does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"model artifact must be a file: {path}")
    return normalized


def _normalize_source_files(source_files: Sequence[str | Path] | None) -> list[Path]:
    if source_files is None:
        return []
    if isinstance(source_files, (str, Path)):
        raise TypeError("source_files must be a sequence of file paths")
    normalized = [Path(path) for path in source_files]
    for path in normalized:
        if not path.exists():
            raise FileNotFoundError(f"source file does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"source file must be a file: {path}")
    return normalized


def _validate_artifact_dir(artifact_dir: str) -> str:
    cleaned = artifact_dir.strip("/")
    if not cleaned:
        raise ValueError("artifact_dir cannot be empty")
    path = Path(cleaned)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("artifact_dir must be a relative MLflow artifact path")
    return cleaned


def _normalize_rebalance_frequency(rebalance_frequency: str | None) -> str | None:
    if rebalance_frequency is None:
        return None
    cleaned = str(rebalance_frequency).strip()
    if not cleaned:
        raise ValueError("rebalance_frequency cannot be empty")
    return cleaned


def _copy_unique(source: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    candidate = destination_dir / source.name
    if candidate.exists():
        stem = source.stem
        suffix = source.suffix
        counter = 1
        while candidate.exists():
            candidate = destination_dir / f"{stem}_{counter}{suffix}"
            counter += 1
    shutil.copy2(source, candidate)
    return candidate


def log_model_submission(
    model_artifacts: Mapping[str, str | Path] | Sequence[str | Path],
    *,
    model_name: str,
    model_family: str,
    feature_names: Sequence[str],
    target: str,
    horizon: int,
    rebalance_frequency: str | None = None,
    preprocessing: Mapping[str, Any] | None = None,
    model_config: Mapping[str, Any] | None = None,
    source_files: Sequence[str | Path] | None = None,
    notes: str | None = None,
    artifact_dir: str = "model_submission",
) -> dict[str, Any]:
    """Log a reconstructable model bundle to the active MLflow run.

    The caller is responsible for saving framework-specific model files before
    calling this helper. This function packages those files with ordered feature
    metadata and optional preprocessing/config details needed for later inference.
    """

    artifacts = _normalize_model_artifacts(model_artifacts)
    sources = _normalize_source_files(source_files)
    ordered_features = [str(feature) for feature in feature_names]
    if not ordered_features:
        raise ValueError("feature_names cannot be empty")
    if int(horizon) <= 0:
        raise ValueError("horizon must be positive")
    cleaned_rebalance_frequency = _normalize_rebalance_frequency(rebalance_frequency)
    cleaned_artifact_dir = _validate_artifact_dir(artifact_dir)

    manifest: dict[str, Any] = {
        "model_name": str(model_name),
        "model_family": str(model_family),
        "target": str(target),
        "horizon": int(horizon),
        "rebalance_frequency": cleaned_rebalance_frequency,
        "feature_names": ordered_features,
        "preprocessing": dict(preprocessing or {}),
        "model_config": dict(model_config or {}),
        "notes": notes,
        "artifact_files": [],
        "artifact_map": {},
        "source_files": [],
    }
    _validate_json_serializable(manifest, "model submission metadata")

    with tempfile.TemporaryDirectory() as tmp_dir:
        bundle_dir = Path(tmp_dir) / "bundle"
        artifacts_dir = bundle_dir / "artifacts"
        source_dir = bundle_dir / "source"
        bundle_dir.mkdir(parents=True, exist_ok=True)

        for logical_name, path in artifacts.items():
            copied = _copy_unique(path, artifacts_dir)
            relative = copied.relative_to(bundle_dir).as_posix()
            manifest["artifact_files"].append(relative)
            manifest["artifact_map"][logical_name] = relative

        for path in sources:
            copied = _copy_unique(path, source_dir)
            manifest["source_files"].append(copied.relative_to(bundle_dir).as_posix())

        _validate_json_serializable(manifest, "model submission manifest")
        manifest_path = bundle_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        mlflow.log_params(
            {
                "submission_model_name": manifest["model_name"],
                "submission_model_family": manifest["model_family"],
                "submission_target": manifest["target"],
                "submission_horizon": manifest["horizon"],
                "submission_feature_count": len(ordered_features),
            }
        )
        if cleaned_rebalance_frequency is not None:
            mlflow.log_param("submission_rebalance_frequency", cleaned_rebalance_frequency)
        mlflow.set_tag("has_model_submission", "true")
        mlflow.log_artifacts(str(bundle_dir), artifact_path=cleaned_artifact_dir)

    return manifest
