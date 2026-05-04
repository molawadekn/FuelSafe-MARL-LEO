"""
Realism Layer for Orbital Collision Avoidance Simulation
---------------------------------------------------------
Encapsulates all uncertainty injection components to transform the
simulation from an artificially safe environment into a physically
plausible evaluation testbed.

Each component is individually toggleable via RealismConfig.

Reference noise magnitudes:
  - Position σ = 20 m  (Space Surveillance Network tracking accuracy)
  - Velocity σ = 0.02 m/s
  - Maneuver ΔV noise ±10% (chemical thruster accuracy)
  - Partial observability drop rate 15% (catalog gap estimate)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

import numpy as np


@dataclass
class RealismConfig:
    """Master toggle and parameter set for all realism components."""

    enabled: bool = True

    # --- State estimation noise ---
    observation_noise: bool = True
    position_noise_sigma_km: float = 0.020       # 20 m in km
    velocity_noise_sigma_kms: float = 0.00002     # 0.02 m/s in km/s

    # --- Covariance growth ---
    covariance_growth: bool = True
    covariance_growth_rate: float = 0.01          # 1% per second

    # --- Partial observability ---
    partial_observability: bool = True
    debris_drop_probability: float = 0.15         # 15% chance per debris per step

    # --- Maneuver execution noise ---
    maneuver_noise: bool = True
    maneuver_noise_scale: float = 0.10            # ±10% ΔV magnitude

    # --- Maneuver delay ---
    maneuver_delay: bool = True
    maneuver_delay_steps: int = 1                 # 1-step buffer

    # --- Reaction delay ---
    reaction_delay: bool = True
    reaction_delay_tca_threshold_s: float = 120.0  # Imminent TCA threshold
    reaction_delay_probability: float = 0.20       # 20% chance of delayed decision

    # --- Suboptimal decision noise ---
    decision_noise: bool = True
    random_action_probability: float = 0.05        # 5% random action replacement


# ---------------------------------------------------------------------------
# Component implementations
# ---------------------------------------------------------------------------

class StateEstimationNoise:
    """Adds Gaussian noise to observed positions and velocities."""

    def __init__(self, config: RealismConfig):
        self.config = config

    def apply(self, state_array: np.ndarray) -> np.ndarray:
        """Add noise to a [pos(3), vel(3)] state vector (km, km/s units).

        Returns a noisy copy; the original is not modified.
        """
        if not self.config.enabled or not self.config.observation_noise:
            return state_array.copy()

        noisy = state_array.copy()
        noisy[:3] += np.random.normal(0.0, self.config.position_noise_sigma_km, size=3)
        noisy[3:6] += np.random.normal(0.0, self.config.velocity_noise_sigma_kms, size=3)
        return noisy


class CovarianceTracker:
    """Models covariance growth over time between tracking updates."""

    def __init__(self, config: RealismConfig):
        self.config = config
        self._scales: Dict[str, float] = {}

    def reset(self) -> None:
        self._scales.clear()

    def update(self, agent_id: str, dt: float) -> float:
        """Grow the covariance scale for *agent_id* by one timestep.

        Returns the current covariance scale factor (≥ 1.0).
        """
        if not self.config.enabled or not self.config.covariance_growth:
            return 1.0

        current = self._scales.get(agent_id, 1.0)
        current *= 1.0 + self.config.covariance_growth_rate * max(0.0, float(dt))
        self._scales[agent_id] = current
        return current

    def get_scale(self, agent_id: str) -> float:
        return self._scales.get(agent_id, 1.0)

    def reset_agent(self, agent_id: str) -> None:
        """Reset covariance after a tracking update (e.g. post-maneuver)."""
        self._scales[agent_id] = 1.0


class PartialObservability:
    """Randomly drops debris from observations to simulate catalog gaps."""

    def __init__(self, config: RealismConfig):
        self.config = config

    def filter_visible_debris(self, debris_ids: list[str]) -> list[str]:
        """Return a subset of *debris_ids* that are 'visible' this step."""
        if not self.config.enabled or not self.config.partial_observability:
            return list(debris_ids)

        return [
            did for did in debris_ids
            if np.random.rand() > self.config.debris_drop_probability
        ]


class ManeuverNoise:
    """Adds execution noise to planned ΔV vectors."""

    def __init__(self, config: RealismConfig):
        self.config = config

    def perturb_delta_v(self, delta_v: np.ndarray) -> np.ndarray:
        """Return a noisy copy of *delta_v* (km/s).

        Magnitude is scaled by uniform(1 - scale, 1 + scale).
        """
        if not self.config.enabled or not self.config.maneuver_noise:
            return delta_v.copy()

        scale = self.config.maneuver_noise_scale
        factor = float(np.random.uniform(1.0 - scale, 1.0 + scale))
        return delta_v * factor


class ManeuverDelay:
    """Buffers maneuver commands by N steps, simulating slew/pointing time."""

    def __init__(self, config: RealismConfig):
        self.config = config
        # Per-agent FIFO queues storing buffered actions.
        self._buffers: Dict[str, deque] = {}

    def reset(self) -> None:
        self._buffers.clear()

    def submit(self, agent_id: str, action: Any) -> Any:
        """Submit *action* and return the action that should execute this step.

        If the delay is disabled or 0, the action passes through immediately.
        Otherwise the oldest buffered action is popped and the new one queued.
        """
        if not self.config.enabled or not self.config.maneuver_delay:
            return action

        delay = max(0, self.config.maneuver_delay_steps)
        if delay == 0:
            return action

        buf = self._buffers.setdefault(agent_id, deque())
        buf.append(action)

        if len(buf) > delay:
            return buf.popleft()
        # Buffer not yet full — execute NO_OP (action index 0).
        return 0


class ReactionDelay:
    """Stochastic decision delay for imminent TCAs."""

    def __init__(self, config: RealismConfig):
        self.config = config

    def should_delay(self, tca_s: float) -> bool:
        """Return True if the agent should be forced to NO_OP this step."""
        if not self.config.enabled or not self.config.reaction_delay:
            return False

        if tca_s > self.config.reaction_delay_tca_threshold_s:
            return False

        return float(np.random.rand()) < self.config.reaction_delay_probability


class DecisionNoise:
    """Occasionally replaces the chosen action with a random one."""

    def __init__(self, config: RealismConfig, action_space_size: int = 7):
        self.config = config
        self.action_space_size = action_space_size

    def maybe_randomize(self, action: int) -> int:
        """With small probability, replace *action* with a random one."""
        if not self.config.enabled or not self.config.decision_noise:
            return action

        if float(np.random.rand()) < self.config.random_action_probability:
            return int(np.random.randint(0, self.action_space_size))
        return action


# ---------------------------------------------------------------------------
# Convenience aggregator
# ---------------------------------------------------------------------------

class RealismLayer:
    """Aggregates all realism components under one interface.

    Usage in the environment:
        self.realism = RealismLayer(realism_config)
        ...
        noisy_state = self.realism.noise.apply(state_array)
        visible = self.realism.observability.filter_visible_debris(ids)
        action = self.realism.delay.submit(agent_id, action)
        ...
    """

    def __init__(self, config: Optional[RealismConfig] = None):
        self.config = config or RealismConfig()
        self.noise = StateEstimationNoise(self.config)
        self.covariance = CovarianceTracker(self.config)
        self.observability = PartialObservability(self.config)
        self.maneuver_noise = ManeuverNoise(self.config)
        self.delay = ManeuverDelay(self.config)
        self.reaction_delay = ReactionDelay(self.config)
        self.decision_noise = DecisionNoise(self.config)

    def reset(self) -> None:
        """Reset all stateful components for a new episode."""
        self.covariance.reset()
        self.delay.reset()
