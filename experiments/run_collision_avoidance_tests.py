"""
Collision Avoidance Test-Case Framework
-----------------------------------------
Runs reproducible Monte Carlo evaluations for:
  - worst-case no-maneuver (no_op)
  - baseline heuristic policy (baseline)
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
import logging
import os
import platform
import sys
import time
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
from sim.evaluation import compute_tc8_success_rate
from sim.realism import RealismConfig
from sim.reporting import save_run_distribution_charts, save_summary_charts

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    scipy_stats = None


LOGGER = logging.getLogger(__name__)


def _configure_logging(*, level: int, log_file: Optional[Path] = None) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s.%(msecs)03d %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )

    # Keep verbose runs readable by suppressing noisy third-party DEBUG logs.
    for noisy in [
        "matplotlib",
        "matplotlib.font_manager",
        "PIL",
        "PIL.PngImagePlugin",
    ]:
        logging.getLogger(noisy).setLevel(max(logging.WARNING, level))


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


def _predictive_collision_probability(dist_m: float, tca_sec: float) -> float:
    """Mirror the environment's physics-based Pc approximation for test-case setup."""
    return float(
        np.exp(-max(0.0, float(dist_m)) / 100.0)
        * np.exp(-max(0.0, float(tca_sec)) / 600.0)
    )


def _build_close_approach_entry(
    *,
    debris_index: int,
    miss_distance_m: float,
    tca_sec: float,
    relative_speed_m_s: float,
    phase_rad: float,
) -> Dict[str, Any]:
    """Create a deterministic physically plausible relative state for a future conjunction."""
    approach_dir = np.array(
        [
            np.cos(phase_rad),
            -0.65 + 0.2 * np.sin(phase_rad),
            0.25 * np.cos(0.5 * phase_rad + 0.3),
        ],
        dtype=np.float64,
    )
    approach_dir /= max(float(np.linalg.norm(approach_dir)), 1e-9)

    reference_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(approach_dir, reference_axis))) > 0.95:
        reference_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    miss_dir = np.cross(approach_dir, reference_axis)
    miss_dir /= max(float(np.linalg.norm(miss_dir)), 1e-9)

    rel_velocity_m_s = approach_dir * float(relative_speed_m_s)
    rel_position_m = (-rel_velocity_m_s * float(tca_sec)) + (miss_dir * float(miss_distance_m))
    pc = _predictive_collision_probability(miss_distance_m, tca_sec)

    return {
        "debris_index": int(debris_index),
        "raw_features": {
            "relative_position_r": float(rel_position_m[0]),
            "relative_position_t": float(rel_position_m[1]),
            "relative_position_n": float(rel_position_m[2]),
            "relative_velocity_r": float(rel_velocity_m_s[0]),
            "relative_velocity_t": float(rel_velocity_m_s[1]),
            "relative_velocity_n": float(rel_velocity_m_s[2]),
        },
        "conjunction_info": {
            "miss_distance": float(miss_distance_m),
            "relative_speed": float(relative_speed_m_s),
            "time_to_tca": float(tca_sec),
            "risk_score": float(pc),
        },
    }


def _build_cluster_scenario(
    *,
    name: str,
    profiles: List[tuple[float, float, float, float]],
    distance_threshold_km: float,
    collision_threshold_km: float,
    risk_level: str,
    high_risk_mode: bool = False,
    duration_hours: float = 1.0,
) -> Dict[str, Any]:
    """Build a deterministic multi-object close-approach cluster for experiment cases."""
    cluster_offsets = [
        _build_close_approach_entry(
            debris_index=idx,
            miss_distance_m=miss_distance_m,
            tca_sec=tca_sec,
            relative_speed_m_s=relative_speed_m_s,
            phase_rad=phase_rad,
        )
        for idx, (miss_distance_m, tca_sec, relative_speed_m_s, phase_rad) in enumerate(profiles)
    ]

    primary_entry = cluster_offsets[0]
    return {
        "name": name,
        "duration_hours": float(duration_hours),
        "risk_level": risk_level,
        "multi_object": len(cluster_offsets) > 1,
        "high_risk_mode": bool(high_risk_mode),
        "distance_threshold_km": float(distance_threshold_km),
        "collision_threshold_km": float(collision_threshold_km),
        "target_features": {"sma": 6978.0, "ecc": 0.0001, "inc": 51.6},
        "chaser_features": {"sma": 6978.001, "ecc": 0.0001, "inc": 51.6},
        "conjunction_info": primary_entry["conjunction_info"],
        "raw_features": primary_entry["raw_features"],
        "cluster_offsets": cluster_offsets,
    }


@dataclass(frozen=True)
class ScenarioSpec:
    test_case: str
    description: str
    scenario_family: str
    num_satellites: int
    num_debris: int
    orbit_altitude_band_km: tuple[float, float]
    use_high_risk_mode: bool
    policy_params: Dict[str, Any]
    scenario_config: Optional[Dict[str, Any]] = None
    # Simulation / metrics
    distance_threshold_km: float = 250.0
    collision_threshold_km: float = 5.0
    safety_threshold_km: float = 0.5
    secondary_conjunction_risk_threshold: float = 0.01
    dt_sec: float = 60.0


def _cluster_summary(scenario: ScenarioSpec) -> Dict[str, Any]:
    scenario_config = scenario.scenario_config or {}
    primary = scenario_config.get("conjunction_info", {})
    cluster = scenario_config.get("cluster_offsets", [])
    return {
        "family": scenario.scenario_family,
        "cluster_size": len(cluster),
        "risk_level": scenario_config.get("risk_level", "UNKNOWN"),
        "primary_miss_m": _safe_float(primary.get("miss_distance")),
        "primary_tca_s": _safe_float(primary.get("time_to_tca")),
        "primary_pc": _safe_float(primary.get("risk_score")),
    }


def _log_scenario_catalog(scenarios: Dict[str, ScenarioSpec]) -> None:
    for tc_key, scenario in scenarios.items():
        summary = _cluster_summary(scenario)
        LOGGER.info(
            "Configured %s | family=%s | sats=%d | debris=%d | cluster=%d | primary_miss_m=%.1f | primary_tca_s=%.1f | primary_pc=%.4f | %s",
            tc_key,
            summary["family"],
            scenario.num_satellites,
            scenario.num_debris,
            summary["cluster_size"],
            summary["primary_miss_m"],
            summary["primary_tca_s"],
            summary["primary_pc"],
            scenario.description,
        )


def build_test_cases(
    max_debris: Optional[int],
    orbit_altitude_band_km: tuple[float, float] = (500.0, 800.0),
) -> Dict[str, ScenarioSpec]:
    low, high = orbit_altitude_band_km

    def cap(x: int) -> int:
        if max_debris is None:
            return x
        return min(int(x), int(max_debris))

    def debris_count(target: int, minimum: int = 1) -> int:
        capped = int(cap(target))
        if max_debris is not None:
            return max(1, capped)
        return max(int(minimum), capped)

    tc1_num_debris = debris_count(150, minimum=4)
    tc2_num_debris = debris_count(120, minimum=3)
    tc3_num_debris = debris_count(120, minimum=4)
    tc4_num_debris = debris_count(180, minimum=4)
    tc5_num_debris = debris_count(1000, minimum=6)
    tc6_num_debris = debris_count(160, minimum=4)
    tc7_num_debris = debris_count(60, minimum=4)
    tc8_num_debris = debris_count(80, minimum=6)

    tc1_cluster = _build_cluster_scenario(
        name="Predictive_Normal_NoManeuver",
        profiles=[
            (650.0, 1500.0, 2200.0, 0.10),
            (980.0, 2100.0, 1800.0, 1.05),
            (1400.0, 2700.0, 1600.0, 2.10),
            (1800.0, 3200.0, 1400.0, 3.05),
        ][:tc1_num_debris],
        distance_threshold_km=250.0,
        collision_threshold_km=0.25,
        risk_level="MODERATE",
        high_risk_mode=True,
        duration_hours=1.25,
    )
    tc2_cluster = _build_cluster_scenario(
        name="Predictive_Normal_Threshold",
        profiles=[
            (320.0, 900.0, 3400.0, 0.30),
            (480.0, 1200.0, 3000.0, 1.40),
            (900.0, 1800.0, 2400.0, 2.50),
        ][:tc2_num_debris],
        distance_threshold_km=250.0,
        collision_threshold_km=0.20,
        risk_level="ELEVATED",
        high_risk_mode=False,
        duration_hours=1.10,
    )
    tc3_cluster = _build_cluster_scenario(
        name="Predictive_Normal_FuelAware",
        profiles=[
            (280.0, 840.0, 3600.0, 0.45),
            (430.0, 1140.0, 3200.0, 1.35),
            (760.0, 1680.0, 2600.0, 2.20),
            (1100.0, 2460.0, 2100.0, 3.05),
        ][:tc3_num_debris],
        distance_threshold_km=250.0,
        collision_threshold_km=0.20,
        risk_level="ELEVATED",
        high_risk_mode=False,
        duration_hours=1.20,
    )
    tc4_cluster = _build_cluster_scenario(
        name="Predictive_MARL_Benchmark",
        profiles=[
            (260.0, 780.0, 3800.0, 0.20),
            (390.0, 1080.0, 3300.0, 1.00),
            (620.0, 1440.0, 2900.0, 1.85),
            (980.0, 2040.0, 2400.0, 2.75),
        ][:tc4_num_debris],
        distance_threshold_km=300.0,
        collision_threshold_km=0.20,
        risk_level="ELEVATED",
        high_risk_mode=True,
        duration_hours=1.40,
    )
    tc5_cluster = _build_cluster_scenario(
        name="Predictive_HighDensity_Stress",
        profiles=[
            (240.0, 480.0, 5200.0, 0.15),
            (320.0, 660.0, 4700.0, 1.00),
            (480.0, 900.0, 4300.0, 1.80),
            (700.0, 1320.0, 3900.0, 2.60),
            (950.0, 1860.0, 3400.0, 3.40),
            (1300.0, 2400.0, 2800.0, 4.20),
        ][:tc5_num_debris],
        distance_threshold_km=400.0,
        collision_threshold_km=0.20,
        risk_level="HIGH",
        high_risk_mode=True,
        duration_hours=1.50,
    )
    tc6_cluster = _build_cluster_scenario(
        name="Predictive_Fuel_Constrained",
        profiles=[
            (190.0, 360.0, 5400.0, 0.20),
            (270.0, 600.0, 4800.0, 1.20),
            (450.0, 960.0, 4200.0, 2.10),
            (800.0, 1500.0, 3600.0, 3.00),
        ][:tc6_num_debris],
        distance_threshold_km=350.0,
        collision_threshold_km=0.10,
        risk_level="HIGH",
        high_risk_mode=True,
        duration_hours=1.10,
    )
    tc7_cluster = _build_cluster_scenario(
        name="Synthetic_Secondary_Cluster",
        profiles=[
            (85.0, 420.0, 3200.0, 0.15),
            (130.0, 540.0, 2800.0, 1.25),
            (260.0, 900.0, 2400.0, 2.35),
            (420.0, 1200.0, 2200.0, 3.1),
        ][:tc7_num_debris],
        distance_threshold_km=250.0,
        collision_threshold_km=0.05,
        risk_level="ELEVATED",
        high_risk_mode=False,
        duration_hours=1.00,
    )
    tc8_cluster = _build_cluster_scenario(
        name="Synthetic_Close_Cluster",
        profiles=[
            (5.0, 45.0, 6200.0, 0.2),
            (8.0, 90.0, 5400.0, 1.1),
            (11.0, 135.0, 4800.0, 2.0),
            (14.0, 180.0, 4200.0, 2.8),
            (17.0, 240.0, 3600.0, 3.5),
            (20.0, 295.0, 3000.0, 4.2),
        ][:tc8_num_debris],
        distance_threshold_km=500.0,
        collision_threshold_km=0.025,
        risk_level="HIGH",
        high_risk_mode=False,
        duration_hours=1.00,
    )

    scenarios = {
        # TC1: baseline worst-case (no maneuvers) under predictive multi-object screening.
        "TC1_no_maneuver": ScenarioSpec(
            test_case="TC1_no_maneuver",
            description="No-op worst case in a predictive normal cluster with long warning and dense background traffic.",
            scenario_family="normal",
            num_satellites=3,
            num_debris=tc1_num_debris,
            orbit_altitude_band_km=(low, high),
            use_high_risk_mode=True,
            policy_params={
                "fuel_kg": 1000.0,
                "baseline_risk_threshold": 0.30,
                "rule_based_aggression": 0.60,
            },
            scenario_config=tc1_cluster,
            collision_threshold_km=0.25,
            distance_threshold_km=250.0,
        ),
        "TC2_threshold_rule": ScenarioSpec(
            test_case="TC2_threshold_rule",
            description="Predictive normal conjunction set tuned to exercise threshold and emergency escalation logic.",
            scenario_family="normal",
            num_satellites=3,
            num_debris=tc2_num_debris,
            orbit_altitude_band_km=(low, high),
            use_high_risk_mode=False,
            policy_params={
                "threshold_km": 3.0,
                "dv_action": 1,
                "fuel_kg": 1000.0,
                "baseline_risk_threshold": 0.22,
                "rule_based_aggression": 0.72,
            },
            scenario_config=tc2_cluster,
            collision_threshold_km=0.20,
            distance_threshold_km=250.0,
        ),
        "TC3_fuel_aware_rule": ScenarioSpec(
            test_case="TC3_fuel_aware_rule",
            description="Normal predictive cluster with sequential threats to reward fuel-aware pacing rather than reactive jitter.",
            scenario_family="normal",
            num_satellites=3,
            num_debris=tc3_num_debris,
            orbit_altitude_band_km=(low, high),
            use_high_risk_mode=False,
            policy_params={
                "threshold_km": 2.5,
                "dv_action": 1,
                "min_fuel_ratio": 0.30,
                "fuel_kg": 2.5,
                "baseline_risk_threshold": 0.20,
                "rule_based_aggression": 0.68,
            },
            scenario_config=tc3_cluster,
            collision_threshold_km=0.20,
            distance_threshold_km=250.0,
        ),
        # TC4: MARL benchmark under predictive multi-object threat ordering.
        "TC4_marl": ScenarioSpec(
            test_case="TC4_marl",
            description="Mixed-warning predictive benchmark for MARL evaluation against the updated Pc-aware baselines.",
            scenario_family="normal",
            num_satellites=5,
            num_debris=tc4_num_debris,
            orbit_altitude_band_km=(low, high),
            use_high_risk_mode=True,
            policy_params={
                "fuel_kg": 250.0,
                "baseline_risk_threshold": 0.20,
                "rule_based_aggression": 0.80,
            },
            scenario_config=tc4_cluster,
            collision_threshold_km=0.20,
            distance_threshold_km=300.0,
        ),
        # TC5: dense stress test with many active objects and multi-object foreground risk.
        "TC5_high_density_stress": ScenarioSpec(
            test_case="TC5_high_density_stress",
            description="Large-fleet stress case combining dense background traffic with a prioritized multi-object risk cluster.",
            scenario_family="normal",
            num_satellites=50,
            num_debris=tc5_num_debris,
            orbit_altitude_band_km=(low, high),
            use_high_risk_mode=True,
            policy_params={
                "fuel_kg": 250.0,
                "threshold_km": 4.0,
                "dv_action": 1,
                "min_fuel_ratio": 0.25,
                "baseline_risk_threshold": 0.18,
                "rule_based_aggression": 0.82,
            },
            scenario_config=tc5_cluster,
            collision_threshold_km=0.20,
            distance_threshold_km=400.0,
        ),
        # TC6: limited-fuel short-warning predictive case.
        "TC6_fuel_constrained": ScenarioSpec(
            test_case="TC6_fuel_constrained",
            description="Fuel-limited predictive case with short warning so the policy must pick one or two efficient maneuvers.",
            scenario_family="hybrid",
            num_satellites=10,
            num_debris=tc6_num_debris,
            orbit_altitude_band_km=(low, high),
            use_high_risk_mode=True,
            policy_params={
                "fuel_kg": 0.75,
                "threshold_km": 4.0,
                "dv_action": 1,
                "min_fuel_ratio": 0.35,
                "baseline_risk_threshold": 0.16,
                "rule_based_aggression": 0.88,
            },
            scenario_config=tc6_cluster,
            collision_threshold_km=0.10,
            distance_threshold_km=350.0,
        ),
        # TC7: secondary conjunction risk test
        "TC7_secondary_conjunctions": ScenarioSpec(
            test_case="TC7_secondary_conjunctions",
            description="Staggered near-miss cluster that should expose maneuver-linked secondary conjunctions without direct primary collisions.",
            scenario_family="normal",
            num_satellites=3,
            num_debris=tc7_num_debris,
            orbit_altitude_band_km=(low, high),
            # Deterministic multi-object cluster so maneuver-linked secondary risk
            # is repeatable under Pc-based evaluation.
            use_high_risk_mode=False,
            policy_params={
                "threshold_km": 2.0,
                "dv_action": 1,
                "min_fuel_ratio": 0.2,
                "fuel_kg": 1000.0,
                "baseline_risk_threshold": 0.10,
                "rule_based_aggression": 0.70,
            },
            scenario_config=tc7_cluster,
            collision_threshold_km=0.05,
            distance_threshold_km=250.0,
            secondary_conjunction_risk_threshold=0.01,
        ),
        # TC8: predictive short-warning multi-object cluster aligned with the new Pc reward.
        "TC8_hypothetical_collision_cluster": ScenarioSpec(
            test_case="TC8_hypothetical_collision_cluster",
            description="TC8-style short-warning hard cluster with sub-100 m miss distances and aggressive approach rates.",
            scenario_family="hard",
            num_satellites=3,
            num_debris=tc8_num_debris,
            orbit_altitude_band_km=(low, high),
            use_high_risk_mode=False,
            policy_params={
                "fuel_kg": 0.75,
                "threshold_km": 2.5,
                "dv_action": 1,
                "min_fuel_ratio": 0.2,
                "baseline_risk_threshold": 0.10,
                "rule_based_aggression": 0.9,
            },
            scenario_config=tc8_cluster,
            collision_threshold_km=0.025,
            distance_threshold_km=500.0,
            secondary_conjunction_risk_threshold=0.01,
        ),
    }
    _log_scenario_catalog(scenarios)
    return scenarios


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
    realism_config: Optional[RealismConfig] = None,
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
        secondary_conjunction_risk_threshold=scenario.secondary_conjunction_risk_threshold,
        scenario_config=scenario.scenario_config,
    )

    import copy
    effective_realism_config = realism_config
    if policy_type == "no_op" and realism_config is not None:
        effective_realism_config = copy.deepcopy(realism_config)
        effective_realism_config.decision_noise = False

    runner_kwargs["realism_config"] = effective_realism_config
    runner_kwargs["policy_kwargs"] = {
        # Defaults for policies that accept/ignore these.
        "threshold_km": scenario.policy_params.get("threshold_km", scenario.collision_threshold_km),
        "dv_action": scenario.policy_params.get("dv_action", 1),
        "min_fuel_ratio": scenario.policy_params.get("min_fuel_ratio", 0.1),
        "baseline_risk_threshold": scenario.policy_params.get("baseline_risk_threshold", 0.5),
        "rule_based_aggression": scenario.policy_params.get("rule_based_aggression", 0.5),
    }

    # MARL policy is optional.
    if include_marl:
        runner_kwargs["marl_trainer"] = marl_trainer

    LOGGER.debug(
        "Executing %s | policy=%s | family=%s | mc_idx=%d | orbit_altitude_km=%.2f | fuel_kg=%.2f | cluster_size=%d",
        scenario.test_case,
        policy_type,
        scenario.scenario_family,
        mc_idx,
        orbit_altitude_km,
        fuel_kg,
        len((scenario.scenario_config or {}).get("cluster_offsets", [])),
    )

    runner = SimulationRunner(**runner_kwargs)

    # Simulate ~1 orbit.
    period_s = _orbital_period_seconds(orbit_altitude_km)
    max_steps = int(np.ceil(period_s / scenario.dt_sec))

    stats = runner.run_episode(max_steps=max_steps, verbose=False)
    LOGGER.debug(
        "Completed %s | policy=%s | mc_idx=%d | collisions=%s | fuel=%.4f | maneuvers=%s | secondary=%s | near_misses=%s",
        scenario.test_case,
        policy_type,
        mc_idx,
        stats.get("total_collisions"),
        float(stats.get("total_fuel_used", 0.0)),
        stats.get("total_maneuvers_executed"),
        stats.get("total_secondary_conjunctions"),
        stats.get("total_near_misses"),
    )
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
    ap.add_argument("--realism", type=str, default="true", help="Enable realism uncertainty injection (true/false).")
    ap.add_argument("--log-level", type=str, default=None, help="Python logging level (DEBUG, INFO, WARNING, ERROR).")
    ap.add_argument("--verbose", action="store_true", help="Alias for --log-level DEBUG.")
    ap.add_argument("--debug", action="store_true", help="Alias for --log-level DEBUG.")
    ap.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional path to write logs (in addition to console). If omitted, logs only to console.",
    )
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    level_name = (args.log_level or "").upper().strip() if args.log_level else ""
    if args.verbose or args.debug:
        level = logging.DEBUG
    elif level_name:
        level = getattr(logging, level_name, logging.INFO)
    else:
        level = logging.INFO

    log_file = Path(args.log_file) if args.log_file else None
    if log_file is not None and not log_file.is_absolute():
        log_file = (out_dir / log_file).resolve()

    _configure_logging(level=level, log_file=log_file)

    LOGGER.info("Collision avoidance test framework starting.")
    LOGGER.info("Python=%s | platform=%s", sys.version.split()[0], platform.platform())
    LOGGER.info("Output dir: %s", out_dir.resolve())
    LOGGER.info(
        "Args: mc_runs=%d seed=%d max_debris=%s quick=%s include_marl=%s marl_model_path=%s",
        int(args.mc_runs),
        int(args.seed),
        str(args.max_debris),
        bool(args.quick),
        bool(args.include_marl),
        str(args.marl_model_path),
    )
    if log_file is not None:
        LOGGER.info("Writing logs to %s", log_file)

    if args.quick:
        args.mc_runs = min(args.mc_runs, 3)
        if args.max_debris is None:
            args.max_debris = 200

    _use_realism = str(args.realism).strip().lower() in {"1", "true", "yes", "y", "on"}
    realism_config = RealismConfig(enabled=_use_realism)
    LOGGER.info("Realism layer: enabled=%s", _use_realism)

    scenario_specs = build_test_cases(max_debris=args.max_debris)
    LOGGER.info("Prepared %d experiment scenarios.", len(scenario_specs))

    # Policy set. TC4 will add MARL only if enabled.
    base_policies = [
        ("no_op", "No maneuver (no_op)"),
        ("baseline", "Baseline Pc-threshold heuristic"),
        ("rule_based", "Rule-based predictive heuristic"),
        ("threshold_rule", "Threshold + emergency rule"),
        ("fuel_aware_threshold_rule", "Fuel-aware threshold + emergency rule"),
    ]

    all_rows: List[Dict[str, Any]] = []

    base_epoch = datetime(2020, 1, 1, 0, 0, 0)

    selected_keys: Optional[List[str]] = None
    if args.test_cases:
        selected_keys = [k.strip() for k in args.test_cases.split(",") if k.strip()]

    scenario_iter = scenario_specs.items()
    if selected_keys is not None:
        scenario_iter = [(k, scenario_specs[k]) for k in selected_keys if k in scenario_specs]

    if selected_keys is not None:
        missing = [k for k in selected_keys if k not in scenario_specs]
        if missing:
            LOGGER.warning("Ignoring unknown test cases: %s", ",".join(missing))

    for tc_key, scenario in scenario_iter:
        LOGGER.info(
            "Running %s | family=%s | sats=%d | debris=%d | %s",
            tc_key,
            scenario.scenario_family,
            scenario.num_satellites,
            scenario.num_debris,
            scenario.description,
        )

        # Optional MARL trainer: create one per satellite count.
        marl_trainer = None
        marl_enabled_cases = {
            "TC4_marl",
            "TC5_high_density_stress",
            "TC6_fuel_constrained",
            "TC7_secondary_conjunctions",
            "TC8_hypothetical_collision_cluster",
        }
        include_marl_here = bool(args.include_marl and tc_key in marl_enabled_cases)
        if include_marl_here:
            from marl.marl_trainer import MARLTrainer

            # Create a trainer matching number of satellites in this scenario.
            marl_trainer = MARLTrainer(num_agents=scenario.num_satellites)

            if args.marl_model_path:
                model_path = Path(args.marl_model_path)
                if model_path.exists():
                    marl_trainer.load(str(model_path))
                    LOGGER.info("Loaded MARL weights from %s for %s.", model_path, tc_key)
                else:
                    raise FileNotFoundError(f"MARL model path not found: {model_path}")
            else:
                if not args.marl_untrained:
                    LOGGER.warning(
                        "Skipping MARL for %s because no model path was provided and --marl-untrained is not set.",
                        tc_key,
                    )
                    include_marl_here = False
                    marl_trainer = None

        policies_to_run = list(base_policies)
        if include_marl_here:
            policies_to_run.append(("marl", "MARL policy"))

        for policy_type, policy_label in policies_to_run:
            LOGGER.info("Evaluating %s with policy=%s across %d Monte Carlo runs.", tc_key, policy_type, args.mc_runs)
            t0 = time.perf_counter()
            policy_rows: List[Dict[str, Any]] = []
            for mc_idx in range(args.mc_runs):
                LOGGER.debug("Starting %s | policy=%s | mc_idx=%d/%d", tc_key, policy_type, mc_idx + 1, args.mc_runs)
                row = run_policy_on_scenario(
                    scenario=scenario,
                    policy_type=policy_type,
                    mc_idx=mc_idx,
                    base_epoch=base_epoch,
                    run_seed=args.seed,
                    marl_trainer=marl_trainer,
                    include_marl=(policy_type == "marl"),
                    realism_config=realism_config,
                )
                row["test_case"] = tc_key
                row["policy"] = policy_type
                row["policy_label"] = policy_label
                all_rows.append(row)
                policy_rows.append(row)

            dt = time.perf_counter() - t0
            if policy_rows:
                collisions = [_safe_float(r.get("total_collisions")) for r in policy_rows]
                fuel = [_safe_float(r.get("total_fuel_used")) for r in policy_rows]
                maneuvers = [_safe_float(r.get("total_maneuvers_executed")) for r in policy_rows]
                LOGGER.info(
                    "Completed %s | policy=%s | mean_collisions=%.3f | mean_fuel=%.3f | mean_maneuvers=%.3f | elapsed=%.2fs",
                    tc_key,
                    policy_type,
                    float(np.nanmean(collisions)) if collisions else float("nan"),
                    float(np.nanmean(fuel)) if fuel else float("nan"),
                    float(np.nanmean(maneuvers)) if maneuvers else float("nan"),
                    float(dt),
                )

    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "test_runs_per_policy.csv"
    df.to_csv(csv_path, index=False)
    LOGGER.info("Wrote per-run results to %s", csv_path)

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
            policy_label=("policy_label", "first"),
            mean_collisions=("total_collisions", "mean"),
            std_collisions=("total_collisions", "std"),
            collision_rate=("total_collisions", lambda s: float((pd.Series(s).astype(float) > 0.0).mean())),
            mean_fuel=("total_fuel_used", "mean"),
            std_fuel=("total_fuel_used", "std"),
            mean_maneuvers=("total_maneuvers_executed", "mean"),
            mean_secondary_conjunctions=("total_secondary_conjunctions", "mean"),
            mean_near_misses=("total_near_misses", "mean"),
            mean_min_separation_km=("min_separation_distance_km", "mean"),
            mean_efficiency_score=("efficiency_score", "mean"),
        )
        .reset_index()
    )
    def _summary_tc8_success_rate(row: pd.Series) -> float:
        if row["test_case"] != "TC8_hypothetical_collision_cluster":
            return float("nan")
        subset = df[(df["test_case"] == row["test_case"]) & (df["policy"] == row["policy"])]
        return compute_tc8_success_rate(
            float(subset["total_collisions"].sum()),
            int(len(subset)),
        )

    summary["tc8_success_rate"] = summary.apply(_summary_tc8_success_rate, axis=1)
    summary_path = out_dir / "aggregated_summary.csv"
    summary.to_csv(summary_path, index=False)
    LOGGER.info("Wrote aggregated summary to %s", summary_path)

    # BONUS 1: Statistical significance (t-test on collisions)
    if args.mc_runs >= 2 and scipy_stats is not None:
        ttest_rows: List[Dict[str, Any]] = []
        policy_pairs = [
            ("no_op", "rule_based"),
            ("no_op", "baseline"),
            ("baseline", "rule_based"),
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
            LOGGER.info("Wrote collision t-test summary to %s", ttest_path)

    # BONUS 2: Pareto frontier (fuel vs collisions)
    pareto_rows: List[Dict[str, Any]] = []

    def dominates(a_f: float, a_c: float, a_m: float, b_f: float, b_c: float, b_m: float) -> bool:
        # a dominates b if it is no worse in all objectives and strictly
        # better in at least one. All three metrics are minimized.
        return (
            a_f <= b_f
            and a_c <= b_c
            and a_m <= b_m
            and (a_f < b_f or a_c < b_c or a_m < b_m)
        )

    for tc_key in summary["test_case"].unique():
        sub = summary[summary["test_case"] == tc_key].copy()
        sub = sub[np.isfinite(sub["mean_fuel"]) & np.isfinite(sub["mean_collisions"])]
        if sub.empty:
            continue

        pts = sub[["policy", "mean_fuel", "mean_collisions", "mean_maneuvers"]].values.tolist()
        frontier = []

        for i, (p_i, f_i, c_i, m_i) in enumerate(pts):
            is_dom = False
            for j, (p_j, f_j, c_j, m_j) in enumerate(pts):
                if i == j:
                    continue
                if dominates(f_j, c_j, m_j, f_i, c_i, m_i):
                    is_dom = True
                    break
            if not is_dom:
                frontier.append((p_i, f_i, c_i, m_i))

        for p_i, f_i, c_i, m_i in frontier:
            pareto_rows.append(
                {
                    "test_case": tc_key,
                    "policy": p_i,
                    "mean_fuel": f_i,
                    "mean_collisions": c_i,
                    "mean_maneuvers": m_i,
                }
            )

    if pareto_rows:
        pareto_df = pd.DataFrame(pareto_rows)
        pareto_path = out_dir / "pareto_frontier_fuel_vs_collisions_vs_maneuvers.csv"
        pareto_df.to_csv(pareto_path, index=False)
        LOGGER.info("Wrote Pareto frontier data to %s", pareto_path)

    # Quick plots (mean collisions + fuel).
    # Keep plotting robust even if some policies were skipped.
    for metric, ylabel in [
        ("mean_collisions", "Mean Collisions"),
        ("collision_rate", "Collision Rate"),
        ("mean_fuel", "Mean Fuel Used (kg)"),
        ("mean_maneuvers", "Mean Maneuvers Executed"),
        ("mean_secondary_conjunctions", "Mean Secondary Conjunctions"),
        ("mean_efficiency_score", "Mean Efficiency Score"),
        ("tc8_success_rate", "TC8 Success Rate"),
    ]:
        plt.figure(figsize=(10, 5))
        for tc_key in df["test_case"].unique():
            sub = summary[summary["test_case"] == tc_key]
            # Plot in stable order
            order = ["no_op", "baseline", "rule_based", "threshold_rule", "fuel_aware_threshold_rule", "marl"]
            sub = sub.set_index("policy").reindex(order).reset_index()
            sub = sub[sub[metric].notna()].copy()
            if "policy_label" in sub.columns:
                sub["policy_label"] = sub["policy_label"].fillna(sub["policy"])
                x_values = sub["policy_label"]
            else:
                x_values = sub["policy"]
            plt.plot(x_values, sub[metric], marker="o")
        plt.title(f"{metric} by policy (per test case)")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plot_path = out_dir / f"plot_{metric}.png"
        plt.savefig(plot_path)
        plt.close()

    save_summary_charts(summary, out_dir, prefix="interactive_summary")
    save_run_distribution_charts(df, out_dir, prefix="interactive_runs")

    LOGGER.info("Finished collision avoidance test framework run.")


if __name__ == "__main__":
    main()

