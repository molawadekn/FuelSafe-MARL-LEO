"""
Observation schema helpers shared across the simulator, policies, and MARL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


OBS_SIZE = 96
OWN_FEATURE_COUNT = 12
MAX_NEARBY_OBJECTS = 7
THREAT_FEATURE_COUNT = 12
TCA_NORMALIZATION_S = 3600.0
MISS_DISTANCE_NORMALIZATION_KM = 100.0
DISTANCE_NORMALIZATION_KM = 250.0
RELATIVE_SPEED_NORMALIZATION_KMS = 15.0

OWN_POS_SLICE = slice(0, 3)
OWN_VEL_SLICE = slice(3, 6)
FUEL_RATIO_INDEX = 6
STEP_NORM_INDEX = 7
MIN_MISS_DISTANCE_INDEX = 8
MAX_RISK_INDEX = 9
COMBINED_RISK_TOP3_INDEX = 10
MIN_TCA_TOP3_INDEX = 11
THREATS_START_INDEX = OWN_FEATURE_COUNT


@dataclass
class ThreatObservation:
    """Threat descriptor embedded into the per-agent observation vector."""

    rel_pos: np.ndarray
    rel_vel: np.ndarray
    distance_km: float
    miss_distance_estimate_km: float
    time_to_closest_approach_s: float
    risk_score: float
    collision_probability: float
    relative_speed_kms: float


def _normalize_miss_distance(distance_km: float) -> float:
    return float(np.clip(distance_km / MISS_DISTANCE_NORMALIZATION_KM, 0.0, 1.0))


def _normalize_distance(distance_km: float) -> float:
    return float(np.clip(distance_km / DISTANCE_NORMALIZATION_KM, 0.0, 1.0))


def _normalize_tca(time_to_closest_approach_s: float) -> float:
    return float(np.clip(time_to_closest_approach_s, 0.0, TCA_NORMALIZATION_S) / TCA_NORMALIZATION_S)


def _normalize_relative_speed(relative_speed_kms: float) -> float:
    return float(
        np.clip(relative_speed_kms / RELATIVE_SPEED_NORMALIZATION_KMS, 0.0, 1.0)
    )


def encode_observation(
    *,
    own_state: np.ndarray,
    fuel_ratio: float,
    step_normalized: float,
    min_miss_distance_km: Optional[float],
    max_risk: float,
    combined_risk_top3: float,
    min_tca_top3_s: float,
    threats: List[ThreatObservation],
) -> np.ndarray:
    """Encode one satellite observation into the shared fixed-size layout."""
    own_state = np.asarray(own_state, dtype=np.float32)
    base = np.concatenate(
        [
            own_state[:6],
            np.asarray(
                [
                    float(np.clip(fuel_ratio, 0.0, 1.0)),
                    float(np.clip(step_normalized, 0.0, 1.0)),
                    _normalize_miss_distance(
                        MISS_DISTANCE_NORMALIZATION_KM
                        if min_miss_distance_km is None
                        else float(min_miss_distance_km)
                    ),
                    float(np.clip(max_risk, 0.0, 1.0)),
                    float(np.clip(combined_risk_top3, 0.0, 1.0)),
                    _normalize_tca(min_tca_top3_s),
                ],
                dtype=np.float32,
            ),
        ]
    )

    blocks: List[np.ndarray] = []
    for idx in range(MAX_NEARBY_OBJECTS):
        if idx < len(threats):
            threat = threats[idx]
            block = np.concatenate(
                [
                    np.asarray(threat.rel_pos, dtype=np.float32)[:3],
                    np.asarray(threat.rel_vel, dtype=np.float32)[:3],
                    np.asarray(
                        [
                            _normalize_distance(threat.distance_km),
                            _normalize_relative_speed(threat.relative_speed_kms),
                            _normalize_miss_distance(threat.miss_distance_estimate_km),
                            _normalize_tca(threat.time_to_closest_approach_s),
                            float(np.clip(threat.risk_score, 0.0, 1.0)),
                            float(np.clip(threat.collision_probability, 0.0, 1.0)),
                        ],
                        dtype=np.float32,
                    ),
                ]
            )
        else:
            block = np.zeros(THREAT_FEATURE_COUNT, dtype=np.float32)
        blocks.append(block)

    obs = np.concatenate([base] + blocks).astype(np.float32)
    if obs.shape[0] != OBS_SIZE:
        raise ValueError(f"Observation schema mismatch: expected {OBS_SIZE}, got {obs.shape[0]}")
    return obs


def decode_observation(observation: np.ndarray) -> Dict[str, object]:
    """Decode a shared observation vector into structured fields."""
    obs = np.asarray(observation, dtype=np.float32).reshape(-1)
    if obs.shape[0] < OBS_SIZE:
        padded = np.zeros(OBS_SIZE, dtype=np.float32)
        padded[: obs.shape[0]] = obs
        obs = padded
    elif obs.shape[0] > OBS_SIZE:
        obs = obs[:OBS_SIZE]

    threats: List[ThreatObservation] = []
    for idx in range(MAX_NEARBY_OBJECTS):
        start = THREATS_START_INDEX + idx * THREAT_FEATURE_COUNT
        end = start + THREAT_FEATURE_COUNT
        block = obs[start:end]
        if np.allclose(block, 0.0):
            continue

        rel_pos = block[0:3].astype(np.float64)
        rel_vel = block[3:6].astype(np.float64)
        distance_km = float(block[6]) * DISTANCE_NORMALIZATION_KM
        relative_speed_kms = float(block[7]) * RELATIVE_SPEED_NORMALIZATION_KMS
        miss_distance_estimate_km = float(block[8]) * MISS_DISTANCE_NORMALIZATION_KM
        time_to_closest_approach_s = float(block[9]) * TCA_NORMALIZATION_S
        risk_score = float(np.clip(block[10], 0.0, 1.0))
        collision_probability = float(np.clip(block[11], 0.0, 1.0))
        threats.append(
            ThreatObservation(
                rel_pos=rel_pos,
                rel_vel=rel_vel,
                distance_km=distance_km,
                miss_distance_estimate_km=miss_distance_estimate_km,
                time_to_closest_approach_s=time_to_closest_approach_s,
                risk_score=risk_score,
                collision_probability=collision_probability,
                relative_speed_kms=relative_speed_kms,
            )
        )

    return {
        "own_pos": obs[OWN_POS_SLICE].astype(np.float64),
        "own_vel": obs[OWN_VEL_SLICE].astype(np.float64),
        "fuel_ratio": float(np.clip(obs[FUEL_RATIO_INDEX], 0.0, 1.0)),
        "step_normalized": float(np.clip(obs[STEP_NORM_INDEX], 0.0, 1.0)),
        "min_miss_distance_km": float(obs[MIN_MISS_DISTANCE_INDEX]) * MISS_DISTANCE_NORMALIZATION_KM,
        "max_risk": float(np.clip(obs[MAX_RISK_INDEX], 0.0, 1.0)),
        "combined_risk_top3": float(np.clip(obs[COMBINED_RISK_TOP3_INDEX], 0.0, 1.0)),
        "min_tca_top3_s": float(np.clip(obs[MIN_TCA_TOP3_INDEX], 0.0, 1.0)) * TCA_NORMALIZATION_S,
        "threats": threats,
    }


def rank_threats(threats: List[ThreatObservation]) -> List[ThreatObservation]:
    """Rank threats by Pc first, then urgency, then current distance."""
    return sorted(
        threats,
        key=lambda threat: (
            -float(threat.collision_probability),
            float(threat.time_to_closest_approach_s),
            float(threat.distance_km),
        ),
    )


def primary_threat(observation: np.ndarray) -> Optional[ThreatObservation]:
    """Return the highest-priority threat from an encoded observation."""
    threats = decode_observation(observation)["threats"]
    ranked = rank_threats(threats)
    return ranked[0] if ranked else None
