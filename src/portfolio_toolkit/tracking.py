from __future__ import annotations

from contextlib import contextmanager, nullcontext
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False
from pathlib import Path
import tempfile

import mlflow
import pandas as pd

from .config import load_mlflow_settings
from .contracts import BacktestResult, PortfolioWeights


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
    dataset_name: str,
    tags: dict[str, str] | None = None,
    *,
    repo_root: str | Path = ".",
):
    root = _repo_root(repo_root)
    settings = load_mlflow_settings(root)
    layout = init_mlflow(root)
    artifact_root = Path(layout["artifact_root"]) if layout["artifact_root"] else None
    lock = _mlflow_lock(root) if not _is_remote_tracking_uri(layout["tracking_uri"]) else nullcontext()
    with lock:
        mlflow.set_tracking_uri(layout["tracking_uri"])
        experiment_name = f"{settings.experiment_prefix}_{dataset_name}"
        experiment_id = _get_or_create_experiment(experiment_name, artifact_root)
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
            mlflow.set_tags({"dataset_name": dataset_name, **(tags or {})})
            yield run


def _log_dataframe(df: pd.DataFrame, artifact_name: str) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / artifact_name
        df.to_parquet(path, index=False)
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
