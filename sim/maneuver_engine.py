"""
MODULE 4: Maneuver Engine
Applies ΔV maneuvers and manages fuel consumption.
"""

import numpy as np
from enum import Enum
from typing import Tuple, Dict
from dataclasses import dataclass


class ManeuverType(Enum):
    """Types of discrete maneuvers."""
    NO_OP = 0
    PROGRADE = 1  # Along velocity direction
    RETROGRADE = 2  # Against velocity direction
    RADIAL_OUT = 3  # Outward from Earth
    RADIAL_IN = 4  # Inward toward Earth
    NORMAL = 5  # Perpendicular to orbital plane


@dataclass
class ManeuverResult:
    """Result of applying a maneuver."""
    new_position: np.ndarray
    new_velocity: np.ndarray
    delta_v_magnitude: float
    fuel_consumed: float
    maneuver_type: ManeuverType
    success: bool
    reason: str


class ManeuverEngine:
    """
    Applies trajectory maneuvers and tracks fuel consumption.
    Supports discrete and continuous action spaces.
    """
    
    def __init__(self,
                 max_delta_v_per_step: float = 0.1,  # km/s
                 fuel_consumption_factor: float = 1.0,  # kg/km·s
                 discrete_delta_v: float = 0.05):  # km/s for discrete actions
        """
        Initialize maneuver engine.
        
        Args:
            max_delta_v_per_step: Maximum ΔV per timestep
            fuel_consumption_factor: Fuel consumption coefficient
            discrete_delta_v: ΔV magnitude for discrete maneuvers
        """
        self.max_delta_v = max_delta_v_per_step
        self.fuel_factor = fuel_consumption_factor
        self.discrete_dv = discrete_delta_v
        
    def apply_discrete_maneuver(self,
                               position: np.ndarray,
                               velocity: np.ndarray,
                               maneuver_type: ManeuverType,
                               fuel_available: float,
                               dt: float = 1.0) -> ManeuverResult:
        """
        Apply a discrete maneuver.
        
        Args:
            position: Current position [km]
            velocity: Current velocity [km/s]
            maneuver_type: Type of maneuver
            fuel_available: Available fuel [kg]
            dt: Time step [s]
            
        Returns:
            ManeuverResult
        """
        if maneuver_type == ManeuverType.NO_OP:
            return ManeuverResult(
                new_position=position.copy(),
                new_velocity=velocity.copy(),
                delta_v_magnitude=0.0,
                fuel_consumed=0.0,
                maneuver_type=maneuver_type,
                success=True,
                reason="No operation"
            )
        
        # Compute maneuver direction
        delta_v = self._get_maneuver_direction(velocity, maneuver_type) * self.discrete_dv
        
        return self.apply_continuous_maneuver(
            position, velocity, delta_v, fuel_available, dt
        )
    
    def apply_continuous_maneuver(self,
                                 position: np.ndarray,
                                 velocity: np.ndarray,
                                 delta_v: np.ndarray,
                                 fuel_available: float,
                                 dt: float = 1.0) -> ManeuverResult:
        """
        Apply a continuous maneuver (3D ΔV vector).
        
        Args:
            position: Current position [km]
            velocity: Current velocity [km/s]
            delta_v: ΔV vector [km/s]
            fuel_available: Available fuel [kg]
            dt: Time step [s]
            
        Returns:
            ManeuverResult
        """
        delta_v = np.asarray(delta_v)
        dv_magnitude = np.linalg.norm(delta_v)
        
        # Check constraints
        if dv_magnitude > self.max_delta_v:
            return ManeuverResult(
                new_position=position.copy(),
                new_velocity=velocity.copy(),
                delta_v_magnitude=0.0,
                fuel_consumed=0.0,
                maneuver_type=None,
                success=False,
                reason=f"ΔV {dv_magnitude:.4f} exceeds max {self.max_delta_v:.4f}"
            )
        
        # Compute fuel consumption
        fuel_needed = self._compute_fuel_required(dv_magnitude)
        
        if fuel_needed > fuel_available:
            return ManeuverResult(
                new_position=position.copy(),
                new_velocity=velocity.copy(),
                delta_v_magnitude=0.0,
                fuel_consumed=0.0,
                maneuver_type=None,
                success=False,
                reason=f"Insufficient fuel: need {fuel_needed:.2f} kg, have {fuel_available:.2f} kg"
            )
        
        # Apply maneuver (impulse approximation)
        new_velocity = velocity + delta_v
        
        # Position update (simplified: assumes impulse at current position)
        # For more accuracy, should integrate over burn time
        new_position = position + velocity * dt + 0.5 * delta_v * (dt ** 2)
        
        return ManeuverResult(
            new_position=new_position,
            new_velocity=new_velocity,
            delta_v_magnitude=dv_magnitude,
            fuel_consumed=fuel_needed,
            maneuver_type=None,
            success=True,
            reason="Maneuver executed"
        )
    
    def _get_maneuver_direction(self, velocity: np.ndarray,
                               maneuver_type: ManeuverType) -> np.ndarray:
        """
        Get unit vector in direction of maneuver.
        
        Args:
            velocity: Current velocity vector
            maneuver_type: Type of maneuver
            
        Returns:
            Unit vector in maneuver direction
        """
        vel_mag = np.linalg.norm(velocity)
        
        if maneuver_type == ManeuverType.PROGRADE:
            return velocity / (vel_mag + 1e-10)
        
        elif maneuver_type == ManeuverType.RETROGRADE:
            return -velocity / (vel_mag + 1e-10)
        
        elif maneuver_type in [ManeuverType.RADIAL_OUT, ManeuverType.RADIAL_IN]:
            # Radial direction (outward positive)
            radial = np.array([1.0, 0.0, 0.0])  # Simplified: assume Earth at origin
            if maneuver_type == ManeuverType.RADIAL_IN:
                radial = -radial
            return radial
        
        elif maneuver_type == ManeuverType.NORMAL:
            # Normal to orbital plane
            return np.array([0.0, 0.0, 1.0])
        
        else:
            return np.array([0.0, 0.0, 0.0])
    
    def _compute_fuel_required(self, delta_v_magnitude: float) -> float:
        """
        Compute fuel required for a maneuver.
        
        Args:
            delta_v_magnitude: ΔV magnitude [km/s]
            
        Returns:
            Fuel required [kg]
        """
        # Simple model: fuel ∝ ΔV magnitude
        # More sophisticated: Tsiolkovsky rocket equation
        return self.fuel_factor * delta_v_magnitude
    
    def get_discrete_action_space(self) -> Dict[int, ManeuverType]:
        """
        Get mapping of action indices to maneuver types.
        
        Returns:
            Dict mapping action index to ManeuverType
        """
        return {
            0: ManeuverType.NO_OP,
            1: ManeuverType.PROGRADE,
            2: ManeuverType.RETROGRADE,
            3: ManeuverType.RADIAL_OUT,
            4: ManeuverType.RADIAL_IN,
            5: ManeuverType.NORMAL,
        }
    
    def action_index_to_delta_v(self, action_idx: int,
                               velocity: np.ndarray) -> np.ndarray:
        """
        Convert discrete action index to ΔV vector.
        
        Args:
            action_idx: Action index
            velocity: Current velocity
            
        Returns:
            ΔV vector [km/s]
        """
        maneuver_type = self.get_discrete_action_space()[action_idx]
        direction = self._get_maneuver_direction(velocity, maneuver_type)
        return direction * self.discrete_dv
