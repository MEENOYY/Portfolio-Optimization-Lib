from __future__ import annotations

import json
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import quantstats as qs

from .contracts import BacktestResult


def build_metrics(result: BacktestResult) -> dict[str, float]:
    nav = result.nav.astype(float)
    returns = result.returns.astype(float)
    turnover = result.turnover.astype(float)
    total_return = float(nav.iloc[-1] / nav.iloc[0] - 1.0) if len(nav) > 1 else 0.0
    annual_return = float((1.0 + total_return) ** (252.0 / max(len(returns), 1)) - 1.0) if len(returns) else 0.0
    annual_volatility = float(returns.std(ddof=0) * sqrt(252.0)) if len(returns) else 0.0
    downside = returns[returns < 0.0]
    downside_vol = float(downside.std(ddof=0) * sqrt(252.0)) if len(downside) else 0.0
    sharpe = annual_return / annual_volatility if annual_volatility > 0 else 0.0
    sortino = annual_return / downside_vol if downside_vol > 0 else 0.0
    drawdown = nav / nav.cummax() - 1.0
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
    metrics = {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_drawdown,
        "calmar": float(calmar),
        "average_turnover": float(turnover.mean()) if len(turnover) else 0.0,
    }
    benchmark_total_return = 0.0
    excess_return = total_return
    if not result.benchmark_returns.empty:
        preferred = "SPY" if "SPY" in result.benchmark_returns.columns else result.benchmark_returns.columns[0]
        benchmark_nav = (1.0 + result.benchmark_returns[preferred]).cumprod()
        benchmark_total_return = float(benchmark_nav.iloc[-1] - 1.0) if len(benchmark_nav) else 0.0
        excess_return = total_return - benchmark_total_return
    metrics["benchmark_total_return"] = benchmark_total_return
    metrics["excess_return_vs_benchmark"] = excess_return
    return metrics


def write_quantstats_report(result: BacktestResult, output_dir: str | Path) -> Path:
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "quantstats.html"
    benchmark = None
    if not result.benchmark_returns.empty:
        benchmark = result.benchmark_returns["SPY"] if "SPY" in result.benchmark_returns.columns else result.benchmark_returns.iloc[:, 0]
    qs.reports.html(result.returns, benchmark=benchmark, output=str(report_path), title=f"{result.strategy_name} tear sheet")
    return report_path


def write_backtest_artifacts(result: BacktestResult, output_dir: str | Path) -> dict[str, str]:
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    weights_path = output_path / "weights.parquet"
    nav_path = output_path / "nav.parquet"
    returns_path = output_path / "returns.parquet"
    turnover_path = output_path / "turnover.parquet"
    benchmarks_path = output_path / "benchmarks.parquet"
    metrics_path = output_path / "metrics.json"
    metrics_table_path = output_path / "metrics_table.parquet"
    result.weights.to_parquet(weights_path)
    result.nav.to_frame(name="nav").to_parquet(nav_path)
    result.returns.to_frame(name="returns").to_parquet(returns_path)
    result.turnover.to_frame(name="turnover").to_parquet(turnover_path)
    result.benchmark_returns.to_parquet(benchmarks_path)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(result.metrics, handle, indent=2, sort_keys=True)
    pd.DataFrame([{"metric": key, "value": value} for key, value in sorted(result.metrics.items())]).to_parquet(metrics_table_path, index=False)
    report_path = write_quantstats_report(result, output_path)
    artifact_paths = {
        "weights": str(weights_path),
        "nav": str(nav_path),
        "returns": str(returns_path),
        "turnover": str(turnover_path),
        "benchmarks": str(benchmarks_path),
        "metrics": str(metrics_path),
        "metrics_table": str(metrics_table_path),
        "quantstats_report": str(report_path),
    }
    result.artifact_paths.update(artifact_paths)
    return artifact_paths
