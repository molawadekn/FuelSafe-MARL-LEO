"""
MODULE 8: Plugin Policy Interface
Enables pluggable policy comparison (baseline vs MARL).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from sim.maneuver_engine import ACTION_COUNT, EMERGENCY_ACTION_INDEX
from sim.observation_utils import ThreatObservation, decode_observation, rank_threats


class PolicyType(Enum):
    """Types of available policies."""

    BASELINE = "baseline"
    MARL = "marl"
    RANDOM = "random"
    RULE_BASED = "rule_based"


def _decoded_float(decoded: Dict[str, object], key: str, default: float = 0.0) -> float:
    value = decoded.get(key, default)
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    return float(default)


def _decoded_threats(decoded: Dict[str, object]) -> List[ThreatObservation]:
    value = decoded.get("threats", [])
    if not isinstance(value, list):
        return []
    return [threat for threat in value if isinstance(threat, ThreatObservation)]


def _decoded_safe_risk_threshold(decoded: Dict[str, object]) -> float:
    min_tca_top3_s = _decoded_float(decoded, "min_tca_top3_s", 3600.0)
    return float(max(0.001, 0.02 * np.exp(-max(0.0, min_tca_top3_s) / 300.0)))


def _is_emergency_threat(
    threat: ThreatObservation,
    *,
    risk_threshold: float,
    tca_threshold_s: float,
    miss_distance_threshold_km: float,
) -> bool:
    return bool(
        threat.risk_score >= risk_threshold
        or threat.time_to_closest_approach_s <= tca_threshold_s
        or threat.miss_distance_estimate_km <= miss_distance_threshold_km
    )


def _directional_action_for_threat(threat: ThreatObservation) -> int:
    rel_pos = np.asarray(threat.rel_pos, dtype=np.float64)
    dominant_axis = int(np.argmax(np.abs(rel_pos))) if rel_pos.size else 0

    if dominant_axis == 0:
        return 3 if rel_pos[0] >= 0.0 else 4
    if dominant_axis == 1:
        return 2 if rel_pos[1] >= 0.0 else 1
    return 5


def _select_avoidance_action(
    threat: ThreatObservation,
    *,
    emergency_risk_threshold: float,
    emergency_tca_threshold_s: float,
    emergency_miss_distance_km: float,
) -> int:
    if _is_emergency_threat(
        threat,
        risk_threshold=emergency_risk_threshold,
        tca_threshold_s=emergency_tca_threshold_s,
        miss_distance_threshold_km=emergency_miss_distance_km,
    ):
        return EMERGENCY_ACTION_INDEX
    return _directional_action_for_threat(threat)


class BasePolicy(ABC):
    """Abstract base class for policies."""

    @abstractmethod
    def select_action(self, state: np.ndarray, agent_id: str) -> Union[int, Tuple[int, float]]:
        """Select action given state."""

    @abstractmethod
    def reset(self):
        """Reset policy state."""

    def select_actions(self, observations: Dict[str, np.ndarray]) -> Dict[str, Union[int, Tuple[int, float]]]:
        """Select actions for a full observation dict."""
        return {
            agent_id: self.select_action(state, agent_id)
            for agent_id, state in observations.items()
        }

    def name(self) -> str:
        return self.__class__.__name__


class BaselinePolicy(BasePolicy):
    """Simple risk-threshold heuristic using the shared threat features."""

    def __init__(self, risk_threshold: float = 0.5):
        self.risk_threshold = float(risk_threshold)
        self.last_risks: Dict[str, float] = {}

    def select_action(self, state: np.ndarray, agent_id: str) -> int:
        decoded = decode_observation(state)
        fuel_ratio = _decoded_float(decoded, "fuel_ratio", 0.0)
        max_risk = _decoded_float(decoded, "max_risk", 0.0)
        combined_risk_top3 = _decoded_float(decoded, "combined_risk_top3", 0.0)
        threats = rank_threats(_decoded_threats(decoded))
        safe_risk = _decoded_safe_risk_threshold(decoded)

        self.last_risks[agent_id] = max_risk
        if fuel_ratio < 0.05 or not threats or (combined_risk_top3 <= safe_risk and max_risk < self.risk_threshold):
            return 0

        return _select_avoidance_action(
            threats[0],
            emergency_risk_threshold=0.92,
            emergency_tca_threshold_s=120.0,
            emergency_miss_distance_km=1.25,
        )

    def reset(self):
        self.last_risks = {}


class RuleBasedPolicy(BasePolicy):
    """More aggressive heuristic that prioritizes high-risk / short-TCA threats."""

    def __init__(self, aggression: float = 0.5):
        self.aggression = float(np.clip(aggression, 0.0, 1.0))
        self.maneuver_history: Dict[str, int] = {}

    def select_action(self, state: np.ndarray, agent_id: str) -> int:
        decoded = decode_observation(state)
        fuel_ratio = _decoded_float(decoded, "fuel_ratio", 0.0)
        combined_risk_top3 = _decoded_float(decoded, "combined_risk_top3", 0.0)
        threats = rank_threats(_decoded_threats(decoded))
        safe_risk = _decoded_safe_risk_threshold(decoded)

        if fuel_ratio < 0.05 or not threats:
            return 0

        primary = threats[0]
        risk_gate = 0.35 - 0.15 * self.aggression
        if combined_risk_top3 <= safe_risk and primary.risk_score < risk_gate and primary.distance_km > 10.0:
            return 0

        emergency_risk = 0.8 - 0.1 * self.aggression
        emergency_tca = 240.0 - 120.0 * self.aggression
        emergency_miss_distance = 1.5 - 0.5 * self.aggression

        if fuel_ratio < (0.35 if self.aggression > 0.7 else 0.1) and primary.risk_score < 0.9:
            return 0

        action = _select_avoidance_action(
            primary,
            emergency_risk_threshold=emergency_risk,
            emergency_tca_threshold_s=max(60.0, emergency_tca),
            emergency_miss_distance_km=max(0.75, emergency_miss_distance),
        )
        self.maneuver_history[agent_id] = action
        return action

    def reset(self):
        self.maneuver_history = {}


class NoOpPolicy(BasePolicy):
    """Worst-case policy: always choose NO_OP."""

    def select_action(self, state: np.ndarray, agent_id: str) -> int:
        return 0

    def reset(self):
        pass


class ThresholdRulePolicy(BasePolicy):
    """Distance-threshold rule with emergency escalation for imminent threats."""

    def __init__(self, threshold_km: float = 5.0, dv_action: int = 1):
        self.threshold_km = float(threshold_km)
        self.dv_action = int(dv_action)

    def select_action(self, state: np.ndarray, agent_id: str) -> int:
        decoded = decode_observation(state)
        threats = rank_threats(_decoded_threats(decoded))
        if not threats:
            return 0

        primary = threats[0]
        if (
            primary.distance_km < self.threshold_km
            or primary.miss_distance_estimate_km < self.threshold_km
        ):
            if _is_emergency_threat(
                primary,
                risk_threshold=0.9,
                tca_threshold_s=180.0,
                miss_distance_threshold_km=max(1.0, 0.5 * self.threshold_km),
            ):
                return EMERGENCY_ACTION_INDEX
            return self.dv_action
        return 0

    def reset(self):
        pass


class FuelAwareThresholdRulePolicy(BasePolicy):
    """Distance-threshold rule gated by minimum remaining fuel."""

    def __init__(
        self,
        threshold_km: float = 5.0,
        dv_action: int = 1,
        min_fuel_ratio: float = 0.1,
    ):
        self.threshold_km = float(threshold_km)
        self.dv_action = int(dv_action)
        self.min_fuel_ratio = float(min_fuel_ratio)

    def select_action(self, state: np.ndarray, agent_id: str) -> int:
        decoded = decode_observation(state)
        fuel_ratio = _decoded_float(decoded, "fuel_ratio", 0.0)
        if fuel_ratio <= self.min_fuel_ratio:
            return 0

        threats = rank_threats(_decoded_threats(decoded))
        if not threats:
            return 0

        primary = threats[0]
        if (
            primary.distance_km < self.threshold_km
            or primary.miss_distance_estimate_km < self.threshold_km
        ):
            if _is_emergency_threat(
                primary,
                risk_threshold=0.92,
                tca_threshold_s=150.0,
                miss_distance_threshold_km=max(0.8, 0.5 * self.threshold_km),
            ):
                return EMERGENCY_ACTION_INDEX
            return self.dv_action
        return 0

    def reset(self):
        pass


class MARLPolicy(BasePolicy):
    """MARL policy wrapper (uses trained MARL model)."""

    def __init__(self, marl_trainer, deterministic: bool = True):
        self.marl_trainer = marl_trainer
        self.deterministic = bool(deterministic)

    def select_action(self, state: np.ndarray, agent_id: str) -> Union[int, Tuple[int, float]]:
        observations = {agent_id: state}
        actions = self.select_actions(observations)
        return actions.get(agent_id, 0)

    def select_actions(self, observations: Dict[str, np.ndarray]) -> Dict[str, Union[int, Tuple[int, float]]]:
        return self.marl_trainer.get_actions(
            observations,
            deterministic=self.deterministic,
        )

    def reset(self):
        pass


class RandomPolicy(BasePolicy):
    """Random policy for baseline comparison."""

    def select_action(self, state: np.ndarray, agent_id: str) -> int:
        return int(np.random.randint(0, ACTION_COUNT))

    def reset(self):
        pass


class PolicyManager:
    """Manages multiple policies and enables easy switching."""

    def __init__(self):
        self.policies: Dict[str, BasePolicy] = {}
        self.active_policy: Optional[str] = None

    def register_policy(self, name: str, policy: BasePolicy) -> None:
        self.policies[name] = policy

    def use_policy(self, name: str) -> None:
        if name not in self.policies:
            raise ValueError(f"Policy '{name}' not registered")
        self.active_policy = name

    def select_action(self, state: np.ndarray, agent_id: str) -> Union[int, Tuple[int, float]]:
        if self.active_policy is None:
            raise ValueError("No active policy selected")
        return self.policies[self.active_policy].select_action(state, agent_id)

    def select_actions(self, observations: Dict[str, np.ndarray]) -> Dict[str, Union[int, Tuple[int, float]]]:
        if self.active_policy is None:
            raise ValueError("No active policy selected")
        return self.policies[self.active_policy].select_actions(observations)

    def get_available_policies(self) -> list:
        return list(self.policies.keys())

    def get_active_policy_name(self) -> str:
        if self.active_policy is None:
            raise ValueError("No active policy selected")
        return self.active_policy
