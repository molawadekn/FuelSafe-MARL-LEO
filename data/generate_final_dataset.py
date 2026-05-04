from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "final_dataset.jsonl"
SEED = 20260421
DEFAULT_RECORDS = 100000

NORMAL_RATIO = 0.70
HARD_RATIO = 0.30

EXTREME_HARD_RATIO = 0.40
CLUSTER_MEMBER_RATIO_HARD = 1.0 / 3.0
SENSOR_GAP_RATIO = 0.25

MU_EARTH_M3_S2 = 3.986004418e14
EARTH_RADIUS_M = 6378.0e3
LEO_MIN_RADIUS_M = 6678.0e3
MEO_MAX_RADIUS_M = 26378.0e3


@dataclass(frozen=True)
class RecordSpec:
    index: int
    scenario_class: str
    hard_band: Optional[str]
    outcome: str
    has_sensor_gap: bool
    cluster_id: Optional[str]
    cluster_size: int
    cluster_member_index: int


def isoformat_z(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def clip(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


def rotation_matrix_1(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float64,
    )


def rotation_matrix_3(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def orbital_elements_to_state(
    *,
    semi_major_axis_km: float,
    eccentricity: float,
    inclination_deg: float,
    raan_deg: float,
    argp_deg: float,
    true_anomaly_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    a_m = float(semi_major_axis_km) * 1000.0
    e = clip(float(eccentricity), 0.0, 0.2)
    inc = math.radians(float(inclination_deg))
    raan = math.radians(float(raan_deg))
    argp = math.radians(float(argp_deg))
    ta = math.radians(float(true_anomaly_deg))

    p_m = a_m * (1.0 - e * e)
    radius_m = p_m / max(1.0 + e * math.cos(ta), 1e-9)

    r_pf = np.array(
        [
            radius_m * math.cos(ta),
            radius_m * math.sin(ta),
            0.0,
        ],
        dtype=np.float64,
    )
    v_scale = math.sqrt(MU_EARTH_M3_S2 / max(p_m, 1.0))
    v_pf = np.array(
        [
            -v_scale * math.sin(ta),
            v_scale * (e + math.cos(ta)),
            0.0,
        ],
        dtype=np.float64,
    )

    transform = rotation_matrix_3(raan) @ rotation_matrix_1(inc) @ rotation_matrix_3(argp)
    position = transform @ r_pf
    velocity = transform @ v_pf
    return position, velocity


def local_orbital_frame(position_m: np.ndarray, velocity_m_s: np.ndarray) -> np.ndarray:
    r_hat = position_m / max(np.linalg.norm(position_m), 1e-9)
    h_vec = np.cross(position_m, velocity_m_s)
    h_hat = h_vec / max(np.linalg.norm(h_vec), 1e-9)
    t_hat = np.cross(h_hat, r_hat)
    t_hat = t_hat / max(np.linalg.norm(t_hat), 1e-9)
    return np.column_stack([r_hat, t_hat, h_hat]).astype(np.float64)


def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-9)


def random_orthogonal_unit(reference: np.ndarray, rng: random.Random) -> np.ndarray:
    candidate = np.array(
        [
            rng.uniform(-1.0, 1.0),
            rng.uniform(-1.0, 1.0),
            rng.uniform(-1.0, 1.0),
        ],
        dtype=np.float64,
    )
    candidate = normalize(candidate)
    projection = np.dot(candidate, reference) * reference
    orthogonal = candidate - projection
    if np.linalg.norm(orthogonal) < 1e-6:
        fallback = np.cross(reference, np.array([0.0, 0.0, 1.0], dtype=np.float64))
        if np.linalg.norm(fallback) < 1e-6:
            fallback = np.cross(reference, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        orthogonal = fallback
    return normalize(orthogonal)


def round_list(values: np.ndarray, digits: int = 6) -> List[float]:
    return [round(float(v), digits) for v in values.tolist()]


def flatten_matrix(matrix: np.ndarray, digits: int = 6) -> List[float]:
    return [round(float(v), digits) for v in matrix.reshape(-1).tolist()]


def build_covariance_matrix(
    *,
    rng: random.Random,
    position_sigma_range_m: Tuple[float, float],
    velocity_sigma_range_m_s: Tuple[float, float],
    correlation_scale: float,
) -> np.ndarray:
    pos_sigmas = np.array(
        [rng.uniform(*position_sigma_range_m) for _ in range(3)],
        dtype=np.float64,
    )
    vel_sigmas = np.array(
        [rng.uniform(*velocity_sigma_range_m_s) for _ in range(3)],
        dtype=np.float64,
    )
    sigmas = np.concatenate([pos_sigmas, vel_sigmas])

    lower = np.zeros((6, 6), dtype=np.float64)
    for row in range(6):
        lower[row, row] = sigmas[row]
        for col in range(row):
            base_scale = min(sigmas[row], sigmas[col])
            lower[row, col] = rng.uniform(-correlation_scale, correlation_scale) * base_scale

    covariance = lower @ lower.T
    covariance += np.eye(6, dtype=np.float64) * 1e-9
    return covariance


def label_from_risk(collision_probability: float, miss_distance_m: float) -> str:
    if collision_probability >= 0.1 or miss_distance_m < 5.0:
        return "collision_imminent"
    if collision_probability >= 1e-3 and miss_distance_m < 500.0:
        return "avoidance_maneuver"
    return "no_action"


def recommended_action(
    *,
    label: str,
    scenario_class: str,
    is_cluster: bool,
    severity: float,
    rng: random.Random,
) -> str:
    if label == "no_action":
        return "no_action"

    if label == "collision_imminent":
        delta_v = 2.0 + 3.0 * severity + rng.uniform(0.1, 0.8)
        if is_cluster:
            return f"perform_cluster_emergency_burn_{delta_v:.2f}_mps"
        return f"perform_emergency_burn_{delta_v:.2f}_mps"

    base = 0.15 if scenario_class == "normal" else 0.55
    delta_v = base + 1.8 * severity + rng.uniform(0.05, 0.4)
    if is_cluster:
        return f"perform_coordinated_burn_{delta_v:.2f}_mps"
    return f"perform_minimum_burn_{delta_v:.2f}_mps"


def risk_category(collision_probability: float, miss_distance_m: float) -> str:
    if collision_probability >= 0.1 or miss_distance_m < 5.0:
        return "CRITICAL"
    if collision_probability >= 1e-2 or miss_distance_m < 100.0:
        return "HIGH"
    if collision_probability >= 1e-3 or miss_distance_m < 500.0:
        return "MEDIUM"
    return "LOW"


def severity_score(
    *,
    miss_distance_m: float,
    tca_s: float,
    miss_bounds: Tuple[float, float],
    tca_bounds: Tuple[float, float],
) -> float:
    miss_low, miss_high = miss_bounds
    tca_low, tca_high = tca_bounds
    miss_norm = 1.0 - clip(
        (math.log10(max(miss_distance_m, miss_low)) - math.log10(miss_low))
        / max(math.log10(miss_high) - math.log10(miss_low), 1e-9),
        0.0,
        1.0,
    )
    tca_norm = 1.0 - clip(
        (tca_s - tca_low) / max(tca_high - tca_low, 1e-9),
        0.0,
        1.0,
    )
    return clip(0.68 * miss_norm + 0.32 * tca_norm, 0.0, 1.0)


def log_collision_probability(
    *,
    spec: RecordSpec,
    severity: float,
    uncertainty_boost: float,
    density_boost: float,
    rng: random.Random,
) -> Tuple[float, float]:
    if spec.scenario_class == "normal":
        log_before = -5.4 + 3.2 * severity + 0.15 * uncertainty_boost + 0.12 * density_boost
        log_before += rng.uniform(-0.10, 0.10)
        log_before = clip(log_before, -5.9, -3.0)

        if spec.outcome == "success":
            log_after = log_before - rng.uniform(0.20, 0.55)
        elif spec.outcome == "delayed":
            log_after = log_before - rng.uniform(0.03, 0.12)
        else:
            log_after = log_before + rng.uniform(0.02, 0.16)

        log_after = clip(log_after, -6.0, -3.0)
        return log_before, log_after

    if spec.hard_band == "extreme":
        log_before = -1.55 + 0.85 * severity + 0.14 * uncertainty_boost + 0.10 * density_boost
        log_before += rng.uniform(-0.08, 0.08)
        log_before = clip(log_before, -1.95, -0.45)

        if spec.outcome == "success":
            log_after = log_before - rng.uniform(0.08, 0.26)
        elif spec.outcome == "delayed":
            log_after = log_before - rng.uniform(0.01, 0.08)
        else:
            log_after = log_before + rng.uniform(0.00, 0.10)

        log_after = max(log_after, -1.995)
        log_after = clip(log_after, -1.995, -0.40)
        return log_before, log_after

    log_before = -2.55 + 0.95 * severity + 0.18 * uncertainty_boost + 0.08 * density_boost
    log_before += rng.uniform(-0.08, 0.08)
    log_before = clip(log_before, -2.95, -1.85)

    if spec.outcome == "success":
        log_after = log_before - rng.uniform(0.12, 0.35)
    elif spec.outcome == "delayed":
        log_after = log_before - rng.uniform(0.02, 0.12)
    else:
        log_after = log_before + rng.uniform(0.00, 0.10)

    log_after = clip(log_after, -3.0, -2.0)
    return log_before, log_after


def sample_epoch(rng: random.Random) -> datetime:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    delta_seconds = rng.randint(0, 365 * 24 * 3600 - 1)
    return start + timedelta(seconds=delta_seconds)


def sample_primary_orbit(rng: random.Random, regime: Optional[str] = None) -> Dict[str, float]:
    orbit_regime = regime or ("LEO" if rng.random() < 0.78 else "MEO")
    if orbit_regime == "LEO":
        semi_major_axis_km = rng.uniform(6778.0, 8378.0)
        eccentricity = rng.uniform(0.0001, 0.01)
        inclination_deg = rng.uniform(0.0, 98.7)
    else:
        semi_major_axis_km = rng.uniform(9000.0, 26378.0)
        eccentricity = rng.uniform(0.0001, 0.02)
        inclination_deg = rng.uniform(0.0, 70.0)

    return {
        "regime": orbit_regime,
        "a_km": semi_major_axis_km,
        "e": eccentricity,
        "i_deg": inclination_deg,
        "raan_deg": rng.uniform(0.0, 360.0),
        "argp_deg": rng.uniform(0.0, 360.0),
        "ta_deg": rng.uniform(0.0, 360.0),
    }


def sample_miss_distance_and_tca(spec: RecordSpec, rng: random.Random) -> Tuple[float, float]:
    if spec.scenario_class == "normal":
        return rng.uniform(100.0, 2000.0), rng.uniform(600.0, 3600.0)

    if spec.hard_band == "extreme":
        miss_distance = 10 ** rng.uniform(math.log10(1.0), math.log10(35.0))
        tca_seconds = rng.uniform(60.0, 300.0)
        return miss_distance, tca_seconds

    miss_distance = 10 ** rng.uniform(math.log10(20.0), math.log10(99.5))
    tca_seconds = rng.uniform(120.0, 600.0)
    return miss_distance, tca_seconds


def sample_relative_speed(regime: str, spec: RecordSpec, rng: random.Random) -> float:
    if spec.scenario_class == "normal":
        if regime == "LEO":
            return rng.uniform(120.0, 1400.0)
        return rng.uniform(80.0, 900.0)

    if regime == "LEO":
        if spec.hard_band == "extreme":
            return rng.uniform(5000.0, 14500.0)
        return rng.uniform(2500.0, 9000.0)

    if spec.hard_band == "extreme":
        return rng.uniform(2500.0, 8500.0)
    return rng.uniform(1500.0, 5500.0)


def sample_density(spec: RecordSpec, rng: random.Random) -> Tuple[int, str]:
    if spec.cluster_id is not None:
        return spec.cluster_size, "high_density"

    if spec.scenario_class == "normal":
        nearby_objects = rng.randint(1, 4)
        return nearby_objects, "low_density" if nearby_objects <= 2 else "moderate_density"

    nearby_objects = rng.randint(2, 8)
    return nearby_objects, "moderate_density" if nearby_objects <= 4 else "high_density"


def sample_sensor_coverage(spec: RecordSpec, rng: random.Random) -> Dict[str, object]:
    if spec.has_sensor_gap:
        mode = rng.choice(["sensor_gap", "radar_dropout"])
        coverage_fraction = rng.uniform(0.45, 0.82)
        position_sigma = rng.uniform(180.0, 750.0)
        velocity_sigma = rng.uniform(0.18, 1.10)
        tracking_sources = ["ground_radar", "space_fence"] if mode == "sensor_gap" else ["ground_radar"]
    elif spec.scenario_class == "hard":
        mode = rng.choice(["degraded", "partial"])
        coverage_fraction = rng.uniform(0.82, 0.97)
        position_sigma = rng.uniform(110.0, 320.0)
        velocity_sigma = rng.uniform(0.08, 0.60)
        tracking_sources = ["ground_radar", "optical"]
    else:
        mode = "full"
        coverage_fraction = rng.uniform(0.95, 1.00)
        position_sigma = rng.uniform(30.0, 120.0)
        velocity_sigma = rng.uniform(0.02, 0.20)
        tracking_sources = ["ground_radar", "optical", "space_fence"]

    return {
        "mode": mode,
        "has_gap": spec.has_sensor_gap,
        "coverage_fraction": round(coverage_fraction, 6),
        "tracking_sources": tracking_sources,
        "position_sigma_m": round(position_sigma, 6),
        "velocity_sigma_m_s": round(velocity_sigma, 6),
    }


def secondary_type(spec: RecordSpec, rng: random.Random) -> str:
    if spec.cluster_id is not None:
        return "fragment"
    if spec.scenario_class == "hard":
        return rng.choices(["fragment", "rocket_body", "payload"], weights=[0.70, 0.20, 0.10], k=1)[0]
    return rng.choices(["fragment", "rocket_body", "payload"], weights=[0.55, 0.30, 0.15], k=1)[0]


def primary_satellite_properties(spec: RecordSpec, rng: random.Random) -> Tuple[float, float]:
    if spec.scenario_class == "hard":
        return rng.uniform(250.0, 1800.0), rng.uniform(4.0, 22.0)
    return rng.uniform(300.0, 2200.0), rng.uniform(5.0, 24.0)


def secondary_object_properties(obj_type: str, spec: RecordSpec, rng: random.Random) -> Tuple[float, float]:
    if obj_type == "fragment":
        mass = rng.uniform(0.2, 80.0 if spec.scenario_class == "normal" else 120.0)
        area = rng.uniform(0.01, 3.5 if spec.scenario_class == "normal" else 5.0)
    elif obj_type == "rocket_body":
        mass = rng.uniform(20.0, 300.0)
        area = rng.uniform(1.0, 12.0)
    else:
        mass = rng.uniform(50.0, 900.0)
        area = rng.uniform(1.0, 18.0)
    return mass, area


def build_record_specs(record_count: int, rng: random.Random) -> List[RecordSpec]:
    normal_count = int(round(record_count * NORMAL_RATIO))
    hard_count = record_count - normal_count

    extreme_hard_count = int(round(hard_count * EXTREME_HARD_RATIO))
    cluster_member_count = int(round(hard_count * CLUSTER_MEMBER_RATIO_HARD))
    cluster_member_count -= cluster_member_count % 4
    sensor_gap_count = int(round(record_count * SENSOR_GAP_RATIO))

    normal_success = int(round(normal_count * 0.80))
    normal_delayed = 10000 if record_count == DEFAULT_RECORDS else int(round(normal_count * 0.14))
    normal_failed = normal_count - normal_success - normal_delayed

    hard_success = 70000 - normal_success if record_count == DEFAULT_RECORDS else int(round(hard_count * 0.47))
    hard_delayed = 20000 - normal_delayed if record_count == DEFAULT_RECORDS else int(round(hard_count * 0.33))
    hard_failed = hard_count - hard_success - hard_delayed

    if record_count == DEFAULT_RECORDS:
        normal_success = 56000
        normal_delayed = 10000
        normal_failed = 4000
        hard_success = 14000
        hard_delayed = 10000
        hard_failed = 6000

    normal_gap_count = max(0, sensor_gap_count - min(hard_count, 18000 if record_count == DEFAULT_RECORDS else int(round(hard_count * 0.60))))
    hard_gap_count = sensor_gap_count - normal_gap_count

    normal_outcomes = (["success"] * normal_success) + (["delayed"] * normal_delayed) + (["failed"] * normal_failed)
    hard_outcomes = (["success"] * hard_success) + (["delayed"] * hard_delayed) + (["failed"] * hard_failed)
    rng.shuffle(normal_outcomes)
    rng.shuffle(hard_outcomes)

    normal_gaps = ([True] * normal_gap_count) + ([False] * (normal_count - normal_gap_count))
    hard_gaps = ([True] * hard_gap_count) + ([False] * (hard_count - hard_gap_count))
    rng.shuffle(normal_gaps)
    rng.shuffle(hard_gaps)

    hard_bands = (["extreme"] * extreme_hard_count) + (["moderate"] * (hard_count - extreme_hard_count))
    rng.shuffle(hard_bands)

    hard_cluster_map: Dict[int, Tuple[str, int, int]] = {}
    cluster_candidates = list(range(hard_count))
    rng.shuffle(cluster_candidates)
    for cluster_number in range(cluster_member_count // 4):
        member_indices = cluster_candidates[cluster_number * 4 : (cluster_number + 1) * 4]
        cluster_id = f"TC8-CL-{cluster_number:05d}"
        for member_index, hard_local_index in enumerate(member_indices):
            hard_cluster_map[hard_local_index] = (cluster_id, 4, member_index)

    specs: List[RecordSpec] = []
    running_index = 0

    for normal_idx in range(normal_count):
        specs.append(
            RecordSpec(
                index=running_index,
                scenario_class="normal",
                hard_band=None,
                outcome=normal_outcomes[normal_idx],
                has_sensor_gap=normal_gaps[normal_idx],
                cluster_id=None,
                cluster_size=1,
                cluster_member_index=0,
            )
        )
        running_index += 1

    for hard_idx in range(hard_count):
        cluster_info = hard_cluster_map.get(hard_idx)
        specs.append(
            RecordSpec(
                index=running_index,
                scenario_class="tc8_hard",
                hard_band=hard_bands[hard_idx],
                outcome=hard_outcomes[hard_idx],
                has_sensor_gap=hard_gaps[hard_idx],
                cluster_id=cluster_info[0] if cluster_info else None,
                cluster_size=cluster_info[1] if cluster_info else 1,
                cluster_member_index=cluster_info[2] if cluster_info else 0,
            )
        )
        running_index += 1

    rng.shuffle(specs)
    return specs


def cluster_context(
    *,
    spec: RecordSpec,
    cache: Dict[str, Dict[str, object]],
    rng: random.Random,
) -> Optional[Dict[str, object]]:
    if spec.cluster_id is None:
        return None

    if spec.cluster_id in cache:
        return cache[spec.cluster_id]

    orbit = sample_primary_orbit(rng, regime="LEO" if rng.random() < 0.82 else "MEO")
    epoch = sample_epoch(rng)
    primary_mass_kg, primary_area_m2 = primary_satellite_properties(spec, rng)
    position_m, velocity_m_s = orbital_elements_to_state(
        semi_major_axis_km=orbit["a_km"],
        eccentricity=orbit["e"],
        inclination_deg=orbit["i_deg"],
        raan_deg=orbit["raan_deg"],
        argp_deg=orbit["argp_deg"],
        true_anomaly_deg=orbit["ta_deg"],
    )
    cache[spec.cluster_id] = {
        "epoch": epoch,
        "orbit": orbit,
        "primary_mass_kg": primary_mass_kg,
        "primary_area_m2": primary_area_m2,
        "position_m": position_m,
        "velocity_m_s": velocity_m_s,
        "sat_id": f"SAT-{rng.randint(100, 999)}",
    }
    return cache[spec.cluster_id]


def generate_relative_state(
    *,
    primary_position_m: np.ndarray,
    primary_velocity_m_s: np.ndarray,
    miss_distance_m: float,
    tca_s: float,
    relative_speed_m_s: float,
    spec: RecordSpec,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = local_orbital_frame(primary_position_m, primary_velocity_m_s)

    if spec.scenario_class == "normal":
        velocity_local = np.array(
            [
                rng.uniform(-0.15, 0.15),
                rng.choice([-1.0, 1.0]) * rng.uniform(0.72, 1.00),
                rng.uniform(-0.20, 0.20),
            ],
            dtype=np.float64,
        )
    else:
        velocity_local = np.array(
            [
                rng.uniform(-0.35, 0.35),
                rng.choice([-1.0, 1.0]) * rng.uniform(0.48, 0.98),
                rng.uniform(-0.55, 0.55),
            ],
            dtype=np.float64,
        )

    velocity_local = normalize(velocity_local)
    miss_local = random_orthogonal_unit(velocity_local, rng)
    miss_local *= miss_distance_m
    rel_vel_local = velocity_local * relative_speed_m_s
    rel_pos_local = miss_local - rel_vel_local * tca_s

    rel_pos_eci = frame @ rel_pos_local
    rel_vel_eci = frame @ rel_vel_local
    secondary_position_m = primary_position_m + rel_pos_eci
    secondary_velocity_m_s = primary_velocity_m_s + rel_vel_eci
    return secondary_position_m, secondary_velocity_m_s, rel_pos_local, rel_vel_local


def chaser_features_from_primary(
    primary_orbit: Dict[str, float],
    rel_pos_local_m: np.ndarray,
    rel_vel_local_m_s: np.ndarray,
) -> Dict[str, float]:
    sma_delta_km = clip(rel_pos_local_m[0] / 1000.0, -20.0, 20.0)
    inc_delta_deg = clip(rel_pos_local_m[2] / 50000.0, -2.0, 2.0)
    ecc_delta = clip(np.linalg.norm(rel_vel_local_m_s) / 300000.0, 0.0, 0.01)
    return {
        "a_km": round(primary_orbit["a_km"] + sma_delta_km, 6),
        "e": round(clip(primary_orbit["e"] + ecc_delta, 0.0, 0.03), 9),
        "i_deg": round(clip(primary_orbit["i_deg"] + inc_delta_deg, 0.0, 98.7), 6),
    }


def generate_record(
    *,
    spec: RecordSpec,
    cdm_id: str,
    rng: random.Random,
    cluster_cache: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    shared = cluster_context(spec=spec, cache=cluster_cache, rng=rng)
    if shared is None:
        orbit = sample_primary_orbit(rng)
        epoch = sample_epoch(rng)
        primary_mass_kg, primary_area_m2 = primary_satellite_properties(spec, rng)
        primary_position_m, primary_velocity_m_s = orbital_elements_to_state(
            semi_major_axis_km=orbit["a_km"],
            eccentricity=orbit["e"],
            inclination_deg=orbit["i_deg"],
            raan_deg=orbit["raan_deg"],
            argp_deg=orbit["argp_deg"],
            true_anomaly_deg=orbit["ta_deg"],
        )
        sat_id = f"SAT-{rng.randint(100, 999)}"
    else:
        orbit = dict(shared["orbit"])
        epoch = shared["epoch"]
        primary_mass_kg = float(shared["primary_mass_kg"])
        primary_area_m2 = float(shared["primary_area_m2"])
        primary_position_m = np.asarray(shared["position_m"], dtype=np.float64)
        primary_velocity_m_s = np.asarray(shared["velocity_m_s"], dtype=np.float64)
        sat_id = str(shared["sat_id"])

    miss_distance_m, tca_s = sample_miss_distance_and_tca(spec, rng)
    if spec.cluster_id is not None:
        miss_distance_m *= 1.0 + 0.05 * spec.cluster_member_index
        tca_s *= 1.0 + 0.04 * spec.cluster_member_index
        miss_distance_m = clip(miss_distance_m, 1.0, 99.5)
        tca_s = clip(tca_s, 60.0, 600.0)

    regime = orbit["regime"]
    relative_speed_m_s = sample_relative_speed(regime, spec, rng)
    nearby_objects, density_tag = sample_density(spec, rng)

    sensor = sample_sensor_coverage(spec, rng)
    uncertainty_boost = (float(sensor["position_sigma_m"]) / 120.0) + (float(sensor["velocity_sigma_m_s"]) / 0.2)
    uncertainty_boost = clip(uncertainty_boost, 0.8, 8.0)
    density_boost = 0.12 * max(nearby_objects - 1, 0)

    severity = severity_score(
        miss_distance_m=miss_distance_m,
        tca_s=tca_s,
        miss_bounds=(100.0, 2000.0) if spec.scenario_class == "normal" else (1.0, 100.0),
        tca_bounds=(600.0, 3600.0) if spec.scenario_class == "normal" else (60.0, 600.0),
    )
    log_pc_before, log_pc_after = log_collision_probability(
        spec=spec,
        severity=severity,
        uncertainty_boost=uncertainty_boost,
        density_boost=density_boost,
        rng=rng,
    )
    collision_probability_before = 10.0 ** log_pc_before
    collision_probability = 10.0 ** log_pc_after

    secondary_position_m, secondary_velocity_m_s, rel_pos_local_m, rel_vel_local_m_s = generate_relative_state(
        primary_position_m=primary_position_m,
        primary_velocity_m_s=primary_velocity_m_s,
        miss_distance_m=miss_distance_m,
        tca_s=tca_s,
        relative_speed_m_s=relative_speed_m_s,
        spec=spec,
        rng=rng,
    )

    secondary_radius_m = float(np.linalg.norm(secondary_position_m))
    if secondary_radius_m < LEO_MIN_RADIUS_M or secondary_radius_m > MEO_MAX_RADIUS_M:
        scale = clip((secondary_radius_m - np.linalg.norm(primary_position_m)) / max(np.linalg.norm(rel_pos_local_m), 1.0), -0.4, 0.4)
        rel_pos_local_m[0] -= scale * 500000.0
        secondary_position_m, secondary_velocity_m_s, rel_pos_local_m, rel_vel_local_m_s = generate_relative_state(
            primary_position_m=primary_position_m,
            primary_velocity_m_s=primary_velocity_m_s,
            miss_distance_m=miss_distance_m,
            tca_s=tca_s,
            relative_speed_m_s=relative_speed_m_s * 0.92,
            spec=spec,
            rng=rng,
        )

    secondary_obj_type = secondary_type(spec, rng)
    secondary_mass_kg, secondary_area_m2 = secondary_object_properties(secondary_obj_type, spec, rng)

    primary_cov = build_covariance_matrix(
        rng=rng,
        position_sigma_range_m=(25.0, 80.0) if spec.scenario_class == "normal" else (35.0, 140.0),
        velocity_sigma_range_m_s=(0.01, 0.08) if spec.scenario_class == "normal" else (0.02, 0.20),
        correlation_scale=0.12,
    )
    secondary_cov = build_covariance_matrix(
        rng=rng,
        position_sigma_range_m=(
            max(float(sensor["position_sigma_m"]) * 0.8, 25.0),
            float(sensor["position_sigma_m"]) * 1.35,
        ),
        velocity_sigma_range_m_s=(
            max(float(sensor["velocity_sigma_m_s"]) * 0.8, 0.01),
            float(sensor["velocity_sigma_m_s"]) * 1.40,
        ),
        correlation_scale=0.18,
    )

    label = label_from_risk(collision_probability, miss_distance_m)
    action = recommended_action(
        label=label,
        scenario_class=spec.scenario_class,
        is_cluster=spec.cluster_id is not None,
        severity=severity,
        rng=rng,
    )
    category = risk_category(collision_probability, miss_distance_m)

    maneuver_time = epoch + timedelta(seconds=clip(0.28 * tca_s, 15.0, max(tca_s - 5.0, 15.0)))
    execution_delay_s = 0.0 if spec.outcome == "success" else rng.uniform(20.0, 180.0) if spec.outcome == "delayed" else rng.uniform(60.0, 240.0)
    planned_delta_v = 0.0 if label == "no_action" else clip(0.2 + 4.2 * severity + rng.uniform(-0.1, 0.3), 0.05, 5.5)
    executed_delta_v = planned_delta_v
    if spec.outcome == "delayed":
        executed_delta_v = planned_delta_v * rng.uniform(0.65, 0.95)
    elif spec.outcome == "failed":
        executed_delta_v = planned_delta_v * rng.uniform(0.0, 0.25)

    fuel_used_kg = max(0.0, executed_delta_v * rng.uniform(0.05, 0.18))
    cluster_size = spec.cluster_size if spec.cluster_id is not None else nearby_objects

    scenario_tags = [
        orbit["regime"].lower(),
        "normal" if spec.scenario_class == "normal" else "tc8_hard",
        density_tag,
        str(sensor["mode"]),
        spec.outcome,
    ]
    if spec.cluster_id is not None:
        scenario_tags.extend(
            [
                "multi_object_cluster",
                f"cluster_size_{cluster_size}",
                spec.cluster_id,
            ]
        )
    if spec.hard_band is not None:
        scenario_tags.append(spec.hard_band)
    if spec.has_sensor_gap:
        scenario_tags.append("sensor_gap_present")

    raw_features = {
        "relative_position_r": round(float(rel_pos_local_m[0]), 6),
        "relative_position_t": round(float(rel_pos_local_m[1]), 6),
        "relative_position_n": round(float(rel_pos_local_m[2]), 6),
        "relative_velocity_r": round(float(rel_vel_local_m_s[0]), 6),
        "relative_velocity_t": round(float(rel_vel_local_m_s[1]), 6),
        "relative_velocity_n": round(float(rel_vel_local_m_s[2]), 6),
    }

    record = {
        "cdm_id": cdm_id,
        "epoch_utc": isoformat_z(epoch),
        "primary": {
            "sat_id": sat_id,
            "mass_kg": round(primary_mass_kg, 6),
            "area_m2": round(primary_area_m2, 6),
            "orbit": {
                "regime": orbit["regime"],
                "a_km": round(orbit["a_km"], 6),
                "e": round(orbit["e"], 9),
                "i_deg": round(orbit["i_deg"], 6),
                "raan_deg": round(orbit["raan_deg"], 6),
                "argp_deg": round(orbit["argp_deg"], 6),
                "ta_deg": round(orbit["ta_deg"], 6),
            },
        },
        "secondary": {
            "obj_id": f"OBJ-{rng.randint(1000, 9999)}",
            "type": secondary_obj_type,
            "mass_kg": round(secondary_mass_kg, 6),
            "area_m2": round(secondary_area_m2, 6),
        },
        "state_vector_primary": round_list(np.concatenate([primary_position_m, primary_velocity_m_s]), digits=6),
        "state_vector_secondary": round_list(np.concatenate([secondary_position_m, secondary_velocity_m_s]), digits=6),
        "covariance_primary": flatten_matrix(primary_cov, digits=6),
        "covariance_secondary": flatten_matrix(secondary_cov, digits=6),
        "TCA_utc": isoformat_z(epoch + timedelta(seconds=tca_s)),
        "miss_distance_m": round(miss_distance_m, 6),
        "collision_probability": round(collision_probability, 12),
        "risk_category": category,
        "maneuver_history": [
            {
                "time_utc": isoformat_z(maneuver_time),
                "planned_delta_v_m_s": round(planned_delta_v, 6),
                "executed_delta_v_m_s": round(executed_delta_v, 6),
                "fuel_used_kg": round(fuel_used_kg, 6),
                "status": spec.outcome,
                "execution_delay_s": round(execution_delay_s, 6),
                "risk_before": round(collision_probability_before, 12),
                "risk_after": round(collision_probability, 12),
            }
        ],
        "sensor_coverage": sensor,
        "recommended_action": action,
        "label_marl": label,
        "scenario_tags": scenario_tags,
        "num_debris": cluster_size,
        "target_features": {
            "sma": round(orbit["a_km"], 6),
            "ecc": round(orbit["e"], 9),
            "inc": round(orbit["i_deg"], 6),
        },
        "chaser_features": chaser_features_from_primary(orbit, rel_pos_local_m, rel_vel_local_m_s),
        "conjunction_info": {
            "miss_distance": round(miss_distance_m, 6),
            "relative_speed": round(relative_speed_m_s, 6),
            "time_to_tca_s": round(tca_s, 6),
            "time_to_tca_min": round(tca_s / 60.0, 6),
            "risk_score": round(-math.log10(max(collision_probability, 1e-12)), 6),
        },
        "raw_features": raw_features,
    }
    return record


def write_dataset(output_path: Path, record_count: int, seed: int) -> Dict[str, float]:
    rng = random.Random(seed)
    np.random.seed(seed)

    specs = build_record_specs(record_count, rng)
    cluster_cache: Dict[str, Dict[str, object]] = {}

    outcome_counts = {"success": 0, "delayed": 0, "failed": 0}
    scenario_counts = {"normal": 0, "tc8_hard": 0}
    hard_cluster_members = 0
    gap_count = 0
    close_count = 0
    high_pc_count = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for line_index, spec in enumerate(specs):
            cdm_id = f"CDM-FINAL-{line_index:06d}"
            record = generate_record(
                spec=spec,
                cdm_id=cdm_id,
                rng=rng,
                cluster_cache=cluster_cache,
            )
            handle.write(json.dumps(record, separators=(",", ":")))
            handle.write("\n")

            scenario_counts[spec.scenario_class] += 1
            outcome_counts[spec.outcome] += 1
            if spec.cluster_id is not None:
                hard_cluster_members += 1
            if spec.has_sensor_gap:
                gap_count += 1
            if float(record["miss_distance_m"]) < 100.0:
                close_count += 1
            if float(record["collision_probability"]) > 1e-2:
                high_pc_count += 1

    return {
        "records": float(record_count),
        "normal": float(scenario_counts["normal"]),
        "tc8_hard": float(scenario_counts["tc8_hard"]),
        "hard_cluster_members": float(hard_cluster_members),
        "sensor_gaps": float(gap_count),
        "miss_lt_100": float(close_count),
        "pc_gt_1e_2": float(high_pc_count),
        "success": float(outcome_counts["success"]),
        "delayed": float(outcome_counts["delayed"]),
        "failed": float(outcome_counts["failed"]),
    }


def validate_dataset(path: Path) -> Dict[str, float]:
    totals = {
        "records": 0,
        "normal": 0,
        "tc8_hard": 0,
        "sensor_gaps": 0,
        "miss_lt_100": 0,
        "pc_gt_1e_2": 0,
        "success": 0,
        "delayed": 0,
        "failed": 0,
        "hard_cluster_members": 0,
    }

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            totals["records"] += 1

            scenario_tags = [str(tag) for tag in record.get("scenario_tags", [])]
            if "tc8_hard" in scenario_tags:
                totals["tc8_hard"] += 1
            else:
                totals["normal"] += 1

            if any(tag == "multi_object_cluster" for tag in scenario_tags):
                totals["hard_cluster_members"] += 1

            sensor = record.get("sensor_coverage", {})
            if isinstance(sensor, dict) and bool(sensor.get("has_gap", False)):
                totals["sensor_gaps"] += 1

            if float(record.get("miss_distance_m", 0.0)) < 100.0:
                totals["miss_lt_100"] += 1
            if float(record.get("collision_probability", 0.0)) > 1e-2:
                totals["pc_gt_1e_2"] += 1

            history = record.get("maneuver_history", [])
            if history:
                status = str(history[0].get("status", ""))
                if status in {"success", "delayed", "failed"}:
                    totals[status] += 1

    return {key: float(value) for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the final MARL collision-avoidance JSONL dataset.")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--count", type=int, default=DEFAULT_RECORDS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    summary = write_dataset(args.output, args.count, args.seed)
    validation = validate_dataset(args.output)

    print(json.dumps({"write_summary": summary, "validation": validation}, indent=2))


if __name__ == "__main__":
    main()
