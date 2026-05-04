"""
MODULE 4: Maneuver Engine
Applies discrete Delta-V maneuvers and manages fuel consumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict

import numpy as np


MPS_TO_KMS = 1e-3
SMALL_BURN_RANGE_MPS = (0.1, 0.5)
MEDIUM_BURN_RANGE_MPS = (0.5, 1.5)
EMERGENCY_BURN_RANGE_MPS = (2.0, 5.0)
MAX_DELTA_V_PER_STEP_KMS = EMERGENCY_BURN_RANGE_MPS[1] * MPS_TO_KMS


class ManeuverType(Enum):
    """Types of discrete maneuvers."""

    NO_OP = 0
    PROGRADE = 1
    RETROGRADE = 2
    RADIAL_OUT = 3
    RADIAL_IN = 4
    NORMAL = 5
    EMERGENCY_RADIAL_OUT = 6


EMERGENCY_ACTION_INDEX = ManeuverType.EMERGENCY_RADIAL_OUT.value
ACTION_COUNT = len(ManeuverType)


@dataclass
class ManeuverResult:
    """Result of applying a maneuver."""

    new_position: np.ndarray
    new_velocity: np.ndarray
    delta_v_magnitude: float
    fuel_consumed: float
    maneuver_type: ManeuverType | None
    success: bool
    reason: str


class ManeuverEngine:
    """
    Applies trajectory maneuvers and tracks fuel consumption.
    Supports discrete and continuous action spaces.
    """

    def __init__(
        self,
        max_delta_v_per_step: float = MAX_DELTA_V_PER_STEP_KMS,
        fuel_consumption_factor: float = 1000.0,
        discrete_delta_v: float = 1.0e-3,
        emergency_delta_v: float = 3.5e-3,
    ):
        # The simulator propagates velocity in km/s, but the control design is
        # specified in m/s. Internally we keep magnitudes in km/s and convert
        # fuel-equivalent accounting with a 1000x scale.
        self.max_delta_v = float(max_delta_v_per_step)
        self.fuel_factor = float(fuel_consumption_factor)
        self.small_burn_range_kms = tuple(self._mps_to_kms(v) for v in SMALL_BURN_RANGE_MPS)
        self.medium_burn_range_kms = tuple(self._mps_to_kms(v) for v in MEDIUM_BURN_RANGE_MPS)
        self.emergency_burn_range_kms = tuple(self._mps_to_kms(v) for v in EMERGENCY_BURN_RANGE_MPS)
        self.discrete_dv = float(np.clip(discrete_delta_v, *self.medium_burn_range_kms))
        self.emergency_dv = min(
            float(np.clip(emergency_delta_v, *self.emergency_burn_range_kms)),
            self.max_delta_v,
        )

    @staticmethod
    def _mps_to_kms(value_m_s: float) -> float:
        return float(value_m_s) * MPS_TO_KMS

    @staticmethod
    def kms_to_mps(value_km_s: float) -> float:
        return float(value_km_s) / MPS_TO_KMS

    def select_magnitude_kms(
        self,
        *,
        action_idx: int,
        risk_level: float,
        tca_s: float,
        requested_kms: float | None = None,
    ) -> float:
        """Map an action into the requested burn bands while staying bounded."""
        if int(action_idx) == int(ManeuverType.NO_OP.value):
            return 0.0

        if int(action_idx) == int(ManeuverType.EMERGENCY_RADIAL_OUT.value) or tca_s < 180.0 or risk_level >= 0.02:
            band = self.emergency_burn_range_kms
            default = self.emergency_dv
        elif risk_level >= 0.01 or tca_s < 300.0:
            band = self.medium_burn_range_kms
            default = self.discrete_dv
        else:
            band = self.small_burn_range_kms
            default = float(np.mean(self.small_burn_range_kms))

        magnitude = default if requested_kms is None else float(requested_kms)
        return float(np.clip(magnitude, band[0], min(band[1], self.max_delta_v)))

    def apply_discrete_maneuver(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        maneuver_type: ManeuverType,
        fuel_available: float,
        dt: float = 1.0,
        propagate_position: bool = True,
        magnitude: float | None = None,
        noise_scale: float = 0.0,
    ) -> ManeuverResult:
        """Apply a discrete maneuver with optional continuous magnitude."""
        if maneuver_type == ManeuverType.NO_OP:
            return ManeuverResult(
                new_position=np.asarray(position, dtype=np.float64).copy(),
                new_velocity=np.asarray(velocity, dtype=np.float64).copy(),
                delta_v_magnitude=0.0,
                fuel_consumed=0.0,
                maneuver_type=maneuver_type,
                success=True,
                reason="No operation",
            )

        if magnitude is not None:
            direction = self._get_maneuver_direction(velocity, maneuver_type)
            delta_v = direction * magnitude
        else:
            delta_v = self.action_index_to_delta_v(int(maneuver_type.value), velocity)
            
        return self.apply_continuous_maneuver(
            position,
            velocity,
            delta_v,
            fuel_available,
            dt=dt,
            maneuver_type=maneuver_type,
            propagate_position=propagate_position,
            noise_scale=noise_scale,
        )

    def apply_continuous_maneuver(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        delta_v: np.ndarray,
        fuel_available: float,
        dt: float = 1.0,
        maneuver_type: ManeuverType | None = None,
        propagate_position: bool = True,
        noise_scale: float = 0.0,
    ) -> ManeuverResult:
        """Apply a continuous maneuver (3D Delta-V vector)."""
        position = np.asarray(position, dtype=np.float64)
        velocity = np.asarray(velocity, dtype=np.float64)
        delta_v = np.asarray(delta_v, dtype=np.float64)

        # Apply thruster execution noise when requested.
        if noise_scale > 0.0 and np.linalg.norm(delta_v) > 1e-15:
            factor = float(np.random.uniform(1.0 - noise_scale, 1.0 + noise_scale))
            delta_v = delta_v * factor

        dv_magnitude = float(np.linalg.norm(delta_v))

        if dv_magnitude > self.max_delta_v + 1e-12:
            return ManeuverResult(
                new_position=position.copy(),
                new_velocity=velocity.copy(),
                delta_v_magnitude=0.0,
                fuel_consumed=0.0,
                maneuver_type=maneuver_type,
                success=False,
                reason=f"Delta-V {dv_magnitude:.4f} exceeds max {self.max_delta_v:.4f}",
            )

        fuel_needed = self._compute_fuel_required(dv_magnitude)
        if fuel_needed > float(fuel_available) + 1e-12:
            return ManeuverResult(
                new_position=position.copy(),
                new_velocity=velocity.copy(),
                delta_v_magnitude=0.0,
                fuel_consumed=0.0,
                maneuver_type=maneuver_type,
                success=False,
                reason=(
                    f"Insufficient fuel: need {fuel_needed:.2f} kg, "
                    f"have {float(fuel_available):.2f} kg"
                ),
            )

        new_velocity = velocity + delta_v
        if propagate_position:
            new_position = position + velocity * float(dt) + 0.5 * delta_v * (float(dt) ** 2)
        else:
            new_position = position.copy()

        return ManeuverResult(
            new_position=new_position,
            new_velocity=new_velocity,
            delta_v_magnitude=dv_magnitude,
            fuel_consumed=fuel_needed,
            maneuver_type=maneuver_type,
            success=True,
            reason="Maneuver executed",
        )

    def _get_maneuver_direction(
        self, velocity: np.ndarray, maneuver_type: ManeuverType
    ) -> np.ndarray:
        """Get a unit vector in the maneuver direction."""
        velocity = np.asarray(velocity, dtype=np.float64)
        vel_mag = float(np.linalg.norm(velocity))

        if maneuver_type == ManeuverType.PROGRADE:
            return velocity / (vel_mag + 1e-10)
        if maneuver_type == ManeuverType.RETROGRADE:
            return -velocity / (vel_mag + 1e-10)
        if maneuver_type in (
            ManeuverType.RADIAL_OUT,
            ManeuverType.RADIAL_IN,
            ManeuverType.EMERGENCY_RADIAL_OUT,
        ):
            radial = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if maneuver_type == ManeuverType.RADIAL_IN:
                radial = -radial
            return radial
        if maneuver_type == ManeuverType.NORMAL:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return np.zeros(3, dtype=np.float64)

    def _compute_fuel_required(self, delta_v_magnitude: float) -> float:
        """Compute fuel required for a maneuver."""
        return float(self.fuel_factor * float(delta_v_magnitude))

    def get_discrete_action_space(self) -> Dict[int, ManeuverType]:
        """Get the mapping of action indices to maneuver types."""
        return {
            0: ManeuverType.NO_OP,
            1: ManeuverType.PROGRADE,
            2: ManeuverType.RETROGRADE,
            3: ManeuverType.RADIAL_OUT,
            4: ManeuverType.RADIAL_IN,
            5: ManeuverType.NORMAL,
            6: ManeuverType.EMERGENCY_RADIAL_OUT,
        }

    def action_index_to_delta_v(self, action_idx: int, velocity: np.ndarray) -> np.ndarray:
        """Convert a discrete action index to a Delta-V vector."""
        maneuver_type = self.get_discrete_action_space()[int(action_idx)]
        direction = self._get_maneuver_direction(velocity, maneuver_type)
        magnitude = self.emergency_dv if maneuver_type == ManeuverType.EMERGENCY_RADIAL_OUT else self.discrete_dv
        return direction * magnitude
