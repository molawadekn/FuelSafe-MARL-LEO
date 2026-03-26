"""
Collision Avoidance Test-Case Framework
-----------------------------------------
Runs reproducible Monte Carlo evaluations for:
  - worst-case no-maneuver (no_op)
  - deterministic threshold rules (threshold_rule)
  - fuel-aware threshold rules (fuel_aware_threshold_rule)
  - optional MARL evaluation (if enabled)

Outputs:
  - per-test-case CSV (one row per Monte Carlo run per policy)
  - aggregated summary CSV
  - a few PNG plots for paper/demo usage
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sim.simulator import SimulationRunner

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    scipy_stats = None


def _orbital_period_seconds(orbit_altitude_km: float) -> float:
    # mu in km^3/s^2
    mu = 398600.4418
    r_earth_km = 6378.0
    a_km = r_earth_km + orbit_altitude_km
    return float(2.0 * np.pi * np.sqrt((a_km ** 3) / mu))


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
        if np.isinf(v):
            return float("nan")
        return v
    except Exception:
        return float("nan")


@dataclass(frozen=True)
class ScenarioSpec:
    test_case: str
    num_satellites: int
    num_debris: int
    orbit_altitude_band_km: tuple[float, float]
    use_high_risk_mode: bool
    policy_params: Dict[str, Any]
    # Simulation / metrics
    distance_threshold_km: float = 250.0
    collision_threshold_km: float = 5.0
    safety_threshold_km: float = 0.5
    dt_sec: float = 60.0


def build_test_cases(
    max_debris: Optional[int],
    orbit_altitude_band_km: tuple[float, float] = (500.0, 800.0),
) -> Dict[str, ScenarioSpec]:
    low, high = orbit_altitude_band_km

    def cap(x: int) -> int:
        if max_debris is None:
            return x
        return min(int(x), int(max_debris))

    # Default "as specified" N ranges, but with caps controlled by CLI to avoid timeouts.
    return {
        # TC1: baseline worst-case (no maneuvers)
        "TC1_no_maneuver": ScenarioSpec(
            test_case="TC1_no_maneuver",
            num_satellites=3,
            num_debris=cap(1000),
            orbit_altitude_band_km=(low, high),
            use_high_risk_mode=True,
            policy_params={"fuel_kg": 1000.0},
            collision_threshold_km=5.0,
        ),
        "TC2_threshold_rule": ScenarioSpec(
            test_case="TC2_threshold_rule",
            num_satellites=3,
            num_debris=cap(100),
            orbit_altitude_band_km=(low, high),
            use_high_risk_mode=True,
            policy_params={"threshold_km": 5.0, "dv_action": 1, "fuel_kg": 1000.0},
            collision_threshold_km=5.0,
        ),
        "TC3_fuel_aware_rule": ScenarioSpec(
            test_case="TC3_fuel_aware_rule",
            num_satellites=3,
            num_debris=cap(100),
            orbit_altitude_band_km=(low, high),
            use_high_risk_mode=True,
            policy_params={"threshold_km": 5.0, "dv_action": 1, "min_fuel_ratio": 0.2, "fuel_kg": 1000.0},
            collision_threshold_km=5.0,
        ),
        # TC4: MARL (optional)
        "TC4_marl": ScenarioSpec(
            test_case="TC4_marl",
            num_satellites=3,
            num_debris=cap(100),
            orbit_altitude_band_km=(low, high),
            use_high_risk_mode=True,
            policy_params={"fuel_kg": 1000.0},
            collision_threshold_km=5.0,
        ),
        # TC5: high-density stress test
        "TC5_high_density_stress": ScenarioSpec(
            test_case="TC5_high_density_stress",
            num_satellites=50,
            num_debris=cap(1000),
            orbit_altitude_band_km=(low, high),
            use_high_risk_mode=True,
            policy_params={"fuel_kg": 1000.0},
            collision_threshold_km=5.0,
        ),
        # TC6: fuel-constrained scenario
        "TC6_fuel_constrained": ScenarioSpec(
            test_case="TC6_fuel_constrained",
            num_satellites=10,
            num_debris=cap(100),
            orbit_altitude_band_km=(low, high),
            use_high_risk_mode=True,
            policy_params={
                "fuel_kg": 0.5,
                "threshold_km": 5.0,
                "dv_action": 1,
                "min_fuel_ratio": 0.2,
            },
            collision_threshold_km=5.0,
        ),
        # TC7: secondary conjunction risk test
        "TC7_secondary_conjunctions": ScenarioSpec(
            test_case="TC7_secondary_conjunctions",
            num_satellites=3,
            num_debris=cap(50),
            orbit_altitude_band_km=(low, high),
            # Less extreme orbit closeness so we can observe non-collision conjunctions
            # and measure maneuver-induced secondary risk.
            use_high_risk_mode=False,
            policy_params={"threshold_km": 5.0, "dv_action": 1, "min_fuel_ratio": 0.2, "fuel_kg": 1000.0},
            collision_threshold_km=5.0,
            distance_threshold_km=100.0,
        ),
    }


def pick_orbit_altitude(rs: np.random.RandomState, band_km: tuple[float, float]) -> float:
    lo, hi = band_km
    return float(rs.uniform(lo, hi))


def run_policy_on_scenario(
    *,
    scenario: ScenarioSpec,
    policy_type: str,
    mc_idx: int,
    base_epoch: datetime,
    run_seed: int,
    marl_trainer: Optional[object] = None,
    include_marl: bool = False,
) -> Dict[str, Any]:
    # Fairness: deterministic epoch+altitude for all policies within a run index.
    rs = np.random.RandomState(run_seed + mc_idx)
    orbit_altitude_km = pick_orbit_altitude(rs, scenario.orbit_altitude_band_km)
    epoch_datetime = base_epoch + timedelta(days=mc_idx)

    fuel_kg = float(scenario.policy_params.get("fuel_kg", 1000.0))

    runner_kwargs = dict(
        num_satellites=scenario.num_satellites,
        num_debris=scenario.num_debris,
        use_safety_filter=True,
        safety_threshold_km=scenario.safety_threshold_km,
        distance_threshold_km=scenario.distance_threshold_km,
        collision_threshold_km=scenario.collision_threshold_km,
        high_risk_mode=scenario.use_high_risk_mode,
        policy_type=policy_type,
        enable_logging=False,
        dt_sec=scenario.dt_sec,
        orbit_altitude_km=orbit_altitude_km,
        epoch_datetime=epoch_datetime,
        initial_fuel_kg=fuel_kg,
        max_fuel_kg=fuel_kg,
        secondary_conjunction_risk_threshold=0.3,
        policy_kwargs={
            # Defaults for policies that accept/ignore these.
            "threshold_km": scenario.policy_params.get("threshold_km", scenario.collision_threshold_km),
            "dv_action": scenario.policy_params.get("dv_action", 1),
            "min_fuel_ratio": scenario.policy_params.get("min_fuel_ratio", 0.1),
            "baseline_risk_threshold": 0.5,
            "rule_based_aggression": 0.5,
        },
    )

    # MARL policy is optional.
    if include_marl:
        runner_kwargs["marl_trainer"] = marl_trainer

    runner = SimulationRunner(**runner_kwargs)

    # Simulate ~1 orbit.
    period_s = _orbital_period_seconds(orbit_altitude_km)
    max_steps = int(np.ceil(period_s / scenario.dt_sec))

    stats = runner.run_episode(max_steps=max_steps, verbose=False)
    return {
        "mc_idx": mc_idx,
        "orbit_altitude_km": orbit_altitude_km,
        "epoch_datetime": epoch_datetime.isoformat(),
        **stats,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=str, default="outputs/test_framework")
    ap.add_argument("--mc-runs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max-debris", type=int, default=None, help="Caps debris count for runtime safety.")
    ap.add_argument("--test-cases", type=str, default=None, help="Comma-separated test case keys to run.")
    ap.add_argument("--quick", action="store_true", help="Reduce mc-runs and cap debris automatically.")
    ap.add_argument("--include-marl", action="store_true", help="Include MARL policy evaluation (untrained unless model is loaded elsewhere).")
    ap.add_argument("--marl-untrained", action="store_true", help="Allow running untrained MARL weights when no model is provided.")
    ap.add_argument("--marl-model-path", type=str, default=None, help="Optional path to MARL weights (not provided by repo by default).")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        args.mc_runs = min(args.mc_runs, 3)
        if args.max_debris is None:
            args.max_debris = 200

    scenario_specs = build_test_cases(max_debris=args.max_debris)

    # Policy set. TC4 will add MARL only if enabled.
    base_policies = [
        ("no_op", "No maneuver (no_op)"),
        ("rule_based", "Rule-based (existing TCA logic)"),
        ("threshold_rule", "Threshold rule"),
        ("fuel_aware_threshold_rule", "Fuel-aware threshold rule"),
    ]

    all_rows: List[Dict[str, Any]] = []

    base_epoch = datetime(2020, 1, 1, 0, 0, 0)

    selected_keys: Optional[List[str]] = None
    if args.test_cases:
        selected_keys = [k.strip() for k in args.test_cases.split(",") if k.strip()]

    scenario_iter = scenario_specs.items()
    if selected_keys is not None:
        scenario_iter = [(k, scenario_specs[k]) for k in selected_keys if k in scenario_specs]

    for tc_key, scenario in scenario_iter:
        print(f"Running {tc_key}: N_sats={scenario.num_satellites}, N_debris={scenario.num_debris}")

        # Optional MARL trainer: create one per satellite count.
        marl_trainer = None
        include_marl_here = args.include_marl and tc_key == "TC4_marl" or (args.include_marl and tc_key in ("TC5_high_density_stress", "TC7_secondary_conjunctions"))
        if include_marl_here:
            from marl.marl_trainer import MARLTrainer

            # Create a trainer matching number of satellites in this scenario.
            marl_trainer = MARLTrainer(num_agents=scenario.num_satellites)

            if args.marl_model_path:
                model_path = Path(args.marl_model_path)
                if model_path.exists():
                    marl_trainer.load(str(model_path))
                else:
                    raise FileNotFoundError(f"MARL model path not found: {model_path}")
            else:
                if not args.marl_untrained:
                    print("Skipping MARL (no model path provided and --marl-untrained not set).")
                    include_marl_here = False
                    marl_trainer = None

        policies_to_run = list(base_policies)
        if include_marl_here:
            policies_to_run.append(("marl", "MARL policy"))

        for policy_type, policy_label in policies_to_run:
            for mc_idx in range(args.mc_runs):
                row = run_policy_on_scenario(
                    scenario=scenario,
                    policy_type=policy_type,
                    mc_idx=mc_idx,
                    base_epoch=base_epoch,
                    run_seed=args.seed,
                    marl_trainer=marl_trainer,
                    include_marl=(policy_type == "marl"),
                )
                row["test_case"] = tc_key
                row["policy"] = policy_type
                row["policy_label"] = policy_label
                all_rows.append(row)

    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "test_runs_per_policy.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote: {csv_path}")

    # Aggregated summary table.
    agg_cols = [
        "total_collisions",
        "total_fuel_used",
        "total_maneuvers_executed",
        "total_secondary_conjunctions",
        "total_near_misses",
    ]
    # Mean aggregation
    summary = (
        df.groupby(["test_case", "policy"], dropna=False)
        .agg(
            mean_collisions=("total_collisions", "mean"),
            std_collisions=("total_collisions", "std"),
            mean_fuel=("total_fuel_used", "mean"),
            std_fuel=("total_fuel_used", "std"),
            mean_maneuvers=("total_maneuvers_executed", "mean"),
            mean_secondary_conjunctions=("total_secondary_conjunctions", "mean"),
            mean_near_misses=("total_near_misses", "mean"),
        )
        .reset_index()
    )
    summary_path = out_dir / "aggregated_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote: {summary_path}")

    # BONUS 1: Statistical significance (t-test on collisions)
    if args.mc_runs >= 2 and scipy_stats is not None:
        ttest_rows: List[Dict[str, Any]] = []
        policy_pairs = [
            ("no_op", "rule_based"),
            ("no_op", "threshold_rule"),
            ("threshold_rule", "fuel_aware_threshold_rule"),
        ]

        for tc_key in df["test_case"].unique():
            df_tc = df[df["test_case"] == tc_key]
            for p0, p1 in policy_pairs:
                if p0 not in set(df_tc["policy"]) or p1 not in set(df_tc["policy"]):
                    continue

                x0 = df_tc[df_tc["policy"] == p0]["total_collisions"].astype(float).values
                x1 = df_tc[df_tc["policy"] == p1]["total_collisions"].astype(float).values

                if len(x0) < 2 or len(x1) < 2:
                    continue

                t_stat, p_val = scipy_stats.ttest_ind(
                    x0, x1, equal_var=False, nan_policy="omit"
                )
                ttest_rows.append(
                    {
                        "test_case": tc_key,
                        "policy_a": p0,
                        "policy_b": p1,
                        "mean_a": float(np.mean(x0)),
                        "mean_b": float(np.mean(x1)),
                        "t_stat": float(t_stat),
                        "p_value": float(p_val),
                    }
                )

        if ttest_rows:
            ttest_df = pd.DataFrame(ttest_rows)
            ttest_path = out_dir / "ttest_collisions.csv"
            ttest_df.to_csv(ttest_path, index=False)
            print(f"Wrote: {ttest_path}")

    # BONUS 2: Pareto frontier (fuel vs collisions)
    pareto_rows: List[Dict[str, Any]] = []

    def dominates(a_f: float, a_c: float, b_f: float, b_c: float) -> bool:
        # a dominates b if it's no worse in both and strictly better in at least one.
        return (a_f <= b_f and a_c <= b_c) and (a_f < b_f or a_c < b_c)

    for tc_key in summary["test_case"].unique():
        sub = summary[summary["test_case"] == tc_key].copy()
        sub = sub[np.isfinite(sub["mean_fuel"]) & np.isfinite(sub["mean_collisions"])]
        if sub.empty:
            continue

        pts = sub[["policy", "mean_fuel", "mean_collisions"]].values.tolist()
        frontier = []

        for i, (p_i, f_i, c_i) in enumerate(pts):
            is_dom = False
            for j, (p_j, f_j, c_j) in enumerate(pts):
                if i == j:
                    continue
                if dominates(f_j, c_j, f_i, c_i):
                    is_dom = True
                    break
            if not is_dom:
                frontier.append((p_i, f_i, c_i))

        for p_i, f_i, c_i in frontier:
            pareto_rows.append(
                {
                    "test_case": tc_key,
                    "policy": p_i,
                    "mean_fuel": f_i,
                    "mean_collisions": c_i,
                }
            )

    if pareto_rows:
        pareto_df = pd.DataFrame(pareto_rows)
        pareto_path = out_dir / "pareto_frontier_fuel_vs_collisions.csv"
        pareto_df.to_csv(pareto_path, index=False)
        print(f"Wrote: {pareto_path}")

    # Quick plots (mean collisions + fuel).
    # Keep plotting robust even if some policies were skipped.
    for metric, ylabel in [
        ("mean_collisions", "Mean Collisions"),
        ("mean_fuel", "Mean Fuel Used (kg)"),
        ("mean_maneuvers", "Mean Maneuvers Executed"),
        ("mean_secondary_conjunctions", "Mean Secondary Conjunctions"),
    ]:
        plt.figure(figsize=(10, 5))
        for tc_key in df["test_case"].unique():
            sub = summary[summary["test_case"] == tc_key]
            # Plot in stable order
            order = ["no_op", "rule_based", "threshold_rule", "fuel_aware_threshold_rule", "marl"]
            sub = sub.set_index("policy").reindex(order).reset_index()
            plt.plot(sub["policy_label"] if "policy_label" in sub.columns else sub["policy"], sub[metric], marker="o")
        plt.title(f"{metric} by policy (per test case)")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plot_path = out_dir / f"plot_{metric}.png"
        plt.savefig(plot_path)
        plt.close()

    print("Finished test framework run.")


if __name__ == "__main__":
    main()

