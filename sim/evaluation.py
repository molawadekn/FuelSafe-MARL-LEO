"""
Evaluation helpers shared by training and experiment scripts.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def compute_efficiency_score(collisions: float, fuel_used: float, maneuvers: float) -> float:
    """Shared scalar objective used for quick policy comparisons."""
    return float(-10.0 * float(collisions) - float(fuel_used) - 0.3 * float(maneuvers))


def compute_tc8_success_rate(tc8_collisions: float, tc8_runs: int) -> float:
    """Convert TC8 collision counts into a bounded success rate."""
    if int(tc8_runs) <= 0:
        return float("nan")
    success = 1.0 - (float(tc8_collisions) / float(tc8_runs))
    return float(np.clip(success, 0.0, 1.0))


def save_pareto_artifacts(eval_history: List[Dict[str, float]], output_dir: str | Path, prefix: str) -> None:
    """
    Save both CSV metrics and publication-friendly Pareto plots.

    The plots keep the three objectives visible at once:
    collisions, fuel, and maneuver count.
    """
    if not eval_history:
        return

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{prefix}_pareto.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["episode"] + list(eval_history[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, metrics in enumerate(eval_history):
            row = {"episode": idx}
            row.update(metrics)
            writer.writerow(row)

    collisions = np.asarray([m.get("mean_collisions", 0.0) for m in eval_history], dtype=np.float64)
    fuel = np.asarray([m.get("mean_fuel", 0.0) for m in eval_history], dtype=np.float64)
    maneuvers = np.asarray([m.get("mean_maneuvers", 0.0) for m in eval_history], dtype=np.float64)
    color_metric = np.asarray([m.get("mean_score", 0.0) for m in eval_history], dtype=np.float64)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        collisions,
        fuel,
        maneuvers,
        c=color_metric,
        cmap="viridis",
        s=60,
    )
    ax.set_xlabel("Mean Collisions")
    ax.set_ylabel("Mean Fuel Used")
    ax.set_zlabel("Mean Maneuvers")
    ax.set_title(f"Pareto Progress: {prefix}")
    plt.colorbar(scatter, label="Efficiency Score")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_pareto_plot.png")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    bubble_sizes = 80.0 + 20.0 * np.maximum(maneuvers, 0.0)
    bubble = ax2.scatter(
        fuel,
        collisions,
        s=bubble_sizes,
        c=color_metric,
        cmap="viridis",
        alpha=0.85,
        edgecolors="black",
        linewidths=0.4,
    )
    ax2.set_xlabel("Mean Fuel Used")
    ax2.set_ylabel("Mean Collisions")
    ax2.set_title(f"Fuel vs Collisions vs Maneuvers: {prefix}")
    plt.colorbar(bubble, label="Efficiency Score")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_pareto_bubble.png")
    plt.close(fig2)


def compute_detailed_metrics(episode_stats_list: List[Dict]) -> Dict[str, float]:
    """Compute publication-quality aggregate metrics from a list of episode stats.

    Designed to work with the enriched info dict produced by the realism-aware
    environment (fields: total_collisions, total_fuel_used,
    total_maneuvers_executed, total_near_misses, min_separation_distance_km,
    episode_min_separation_distance_km, etc.).

    Returns a flat dict of scalar metrics suitable for CSV export or logging.
    """
    if not episode_stats_list:
        return {}

    collisions = np.asarray(
        [float(s.get("total_collisions", 0)) for s in episode_stats_list], dtype=np.float64
    )
    fuel = np.asarray(
        [float(s.get("total_fuel_used", 0.0)) for s in episode_stats_list], dtype=np.float64
    )
    maneuvers = np.asarray(
        [float(s.get("total_maneuvers_executed", 0)) for s in episode_stats_list], dtype=np.float64
    )
    near_misses = np.asarray(
        [float(s.get("total_near_misses", 0)) for s in episode_stats_list], dtype=np.float64
    )
    min_sep_m = np.asarray(
        [
            float(s.get("min_separation_distance_km", s.get("episode_min_separation_distance_km", float("inf")))) * 1000.0
            for s in episode_stats_list
        ],
        dtype=np.float64,
    )

    collision_rate = float(np.mean(collisions > 0))
    near_miss_rate = float(np.mean(near_misses > 0))

    # Maneuver efficiency: collisions avoided per kg fuel.
    # Defined as (1 - collision_rate) / max(mean_fuel, 1e-6) so higher is better.
    mean_fuel = float(np.mean(fuel))
    maneuver_efficiency = (1.0 - collision_rate) / max(mean_fuel, 1e-6)

    # TC8 difficulty score: fraction of episodes that are TC8-like AND have collisions.
    tc8_episodes = [s for s in episode_stats_list if bool(s.get("tc8_active", False))]
    tc8_runs = len(tc8_episodes)
    tc8_collisions = sum(1 for s in tc8_episodes if float(s.get("total_collisions", 0)) > 0)
    tc8_difficulty_score = float(tc8_collisions / max(tc8_runs, 1))

    return {
        "num_episodes": len(episode_stats_list),
        "mean_collisions": float(np.mean(collisions)),
        "std_collisions": float(np.std(collisions)),
        "collision_rate": collision_rate,
        "mean_near_misses": float(np.mean(near_misses)),
        "std_near_misses": float(np.std(near_misses)),
        "near_miss_rate": near_miss_rate,
        "mean_fuel_used": mean_fuel,
        "std_fuel_used": float(np.std(fuel)),
        "mean_maneuvers": float(np.mean(maneuvers)),
        "std_maneuvers": float(np.std(maneuvers)),
        "maneuver_efficiency": maneuver_efficiency,
        "mean_min_separation_m": float(np.mean(min_sep_m)),
        "min_min_separation_m": float(np.min(min_sep_m)),
        "tc8_difficulty_score": tc8_difficulty_score,
        "tc8_runs": tc8_runs,
    }
