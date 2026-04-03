"""
Interactive reporting helpers for simulation and evaluation outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px


SUMMARY_METRICS: Dict[str, str] = {
    "mean_collisions": "Mean Collisions",
    "mean_fuel": "Mean Fuel Used (kg)",
    "mean_maneuvers": "Mean Maneuvers Executed",
    "mean_secondary_conjunctions": "Mean Secondary Conjunctions",
    "mean_near_misses": "Mean Near Misses",
    "mean_min_separation_km": "Mean Minimum Separation (km)",
}

RUN_METRICS: Dict[str, str] = {
    "total_collisions": "Episode Collisions",
    "total_fuel_used": "Episode Fuel Used (kg)",
    "total_maneuvers_executed": "Episode Maneuvers Executed",
    "total_secondary_conjunctions": "Episode Secondary Conjunctions",
    "total_near_misses": "Episode Near Misses",
}


def _group_col(df: pd.DataFrame) -> Optional[str]:
    if "test_case" in df.columns:
        return "test_case"
    return None


def build_summary_bar_figure(summary_df: pd.DataFrame, metric: str):
    """Build a grouped summary bar chart for one metric."""
    if summary_df.empty or metric not in summary_df.columns or "policy" not in summary_df.columns:
        return None

    label = SUMMARY_METRICS.get(metric, metric.replace("_", " ").title())
    group_col = _group_col(summary_df)

    if group_col and summary_df[group_col].nunique() > 1:
        fig = px.bar(
            summary_df,
            x=group_col,
            y=metric,
            color="policy",
            barmode="group",
            title=f"{label} by Policy and Test Case",
            hover_data=[c for c in ["policy_label", "std_collisions", "std_fuel"] if c in summary_df.columns],
        )
    else:
        fig = px.bar(
            summary_df,
            x="policy",
            y=metric,
            color="policy",
            text_auto=".3g",
            title=f"{label} by Policy",
            hover_data=[c for c in ["policy_label", "test_case"] if c in summary_df.columns],
        )

    fig.update_layout(xaxis_title=None, yaxis_title=label, legend_title="Policy")
    return fig


def build_pareto_figure(summary_df: pd.DataFrame):
    """Build a fuel-vs-collisions scatter chart."""
    if summary_df.empty or "mean_fuel" not in summary_df.columns or "mean_collisions" not in summary_df.columns:
        return None

    size_col = "mean_maneuvers" if "mean_maneuvers" in summary_df.columns else None
    group_col = _group_col(summary_df)

    if group_col and summary_df[group_col].nunique() > 1:
        fig = px.scatter(
            summary_df,
            x="mean_fuel",
            y="mean_collisions",
            color="policy",
            facet_col=group_col,
            facet_col_wrap=2,
            hover_name="policy",
            size=size_col,
            title="Fuel vs Collisions Pareto View",
        )
    else:
        fig = px.scatter(
            summary_df,
            x="mean_fuel",
            y="mean_collisions",
            color="policy",
            hover_name="policy",
            size=size_col,
            title="Fuel vs Collisions Pareto View",
        )

    fig.update_layout(xaxis_title="Mean Fuel Used (kg)", yaxis_title="Mean Collisions")
    return fig


def build_run_distribution_figure(runs_df: pd.DataFrame, metric: str):
    """Build a policy-level box plot for raw episode runs."""
    if runs_df.empty or metric not in runs_df.columns or "policy" not in runs_df.columns:
        return None

    label = RUN_METRICS.get(metric, metric.replace("_", " ").title())
    group_col = _group_col(runs_df)

    if group_col and runs_df[group_col].nunique() > 1:
        fig = px.box(
            runs_df,
            x="policy",
            y=metric,
            color="policy",
            points="all",
            facet_col=group_col,
            facet_col_wrap=2,
            title=f"{label} Distribution by Policy",
        )
    else:
        fig = px.box(
            runs_df,
            x="policy",
            y=metric,
            color="policy",
            points="all",
            title=f"{label} Distribution by Policy",
        )

    fig.update_layout(xaxis_title=None, yaxis_title=label, showlegend=False)
    return fig


def build_training_progress_figure(train_df: pd.DataFrame, metric: str):
    """Build a simple training-progress line chart."""
    if train_df.empty or metric not in train_df.columns:
        return None

    plot_df = train_df.reset_index(drop=True).copy()
    plot_df["train_iteration"] = plot_df.index + 1
    color_col = "scenario" if "scenario" in plot_df.columns else None

    fig = px.line(
        plot_df,
        x="train_iteration",
        y=metric,
        color=color_col,
        markers=True,
        title=f"Training Progress: {metric.replace('_', ' ').title()}",
    )
    fig.update_layout(xaxis_title="Training Iteration", yaxis_title=metric.replace("_", " ").title())
    return fig


def _write_figures(figures: List[tuple[str, object]], output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for filename, fig in figures:
        if fig is None:
            continue
        path = output_dir / filename
        fig.write_html(str(path), include_plotlyjs="cdn")
        written.append(path)
    return written


def save_summary_charts(summary_df: pd.DataFrame, output_dir: str | Path, prefix: str) -> List[Path]:
    """Save interactive summary charts as HTML files."""
    figures: List[tuple[str, object]] = []
    for metric in SUMMARY_METRICS:
        figures.append((f"{prefix}_{metric}.html", build_summary_bar_figure(summary_df, metric)))
    figures.append((f"{prefix}_pareto.html", build_pareto_figure(summary_df)))
    return _write_figures(figures, Path(output_dir))


def save_run_distribution_charts(runs_df: pd.DataFrame, output_dir: str | Path, prefix: str) -> List[Path]:
    """Save interactive raw-run distribution charts as HTML files."""
    figures: List[tuple[str, object]] = []
    for metric in RUN_METRICS:
        figures.append((f"{prefix}_{metric}.html", build_run_distribution_figure(runs_df, metric)))
    return _write_figures(figures, Path(output_dir))


def save_training_progress_charts(train_df: pd.DataFrame, output_dir: str | Path, prefix: str) -> List[Path]:
    """Save interactive training-progress charts as HTML files."""
    figures: List[tuple[str, object]] = []
    for metric in ["final_collisions", "final_fuel_used", "final_steps", "actor_loss", "critic_loss"]:
        figures.append((f"{prefix}_{metric}.html", build_training_progress_figure(train_df, metric)))
    return _write_figures(figures, Path(output_dir))
