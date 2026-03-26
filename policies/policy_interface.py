"""
MODULE 8: Plugin Policy Interface
Enables pluggable policy comparison (baseline vs MARL).
"""

import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Tuple


class PolicyType(Enum):
    """Types of available policies."""
    BASELINE = "baseline"
    MARL = "marl"
    RANDOM = "random"
    RULE_BASED = "rule_based"


class BasePolicy(ABC):
    """Abstract base class for policies."""
    
    @abstractmethod
    def select_action(self, state: np.ndarray, agent_id: str) -> int:
        """
        Select action given state.
        
        Args:
            state: Observation/state
            agent_id: Agent identifier
            
        Returns:
            Action index (0-5)
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset policy state."""
        pass
    
    def name(self) -> str:
        """Return policy name."""
        return self.__class__.__name__


class BaselinePolicy(BasePolicy):
    """
    Baseline heuristic policy.
    
    Rule:
    - If high-risk conjunction ahead: execute avoidance maneuver
    - Otherwise: no-op
    """
    
    def __init__(self, risk_threshold: float = 0.5):
        """
        Initialize baseline policy.
        
        Args:
            risk_threshold: Risk score threshold for action
        """
        self.risk_threshold = risk_threshold
        self.last_risks = {}
    
    def select_action(self, state: np.ndarray, agent_id: str) -> int:
        """
        Select action using simple heuristic.
        
        State format: [pos(3), vel(3), fuel_ratio, steps_norm, nearby_objects(42)]
        
        Args:
            state: Observation
            agent_id: Agent identifier
            
        Returns:
            Action (0: no-op, 1-5: maneuvers)
        """
        if len(state) < 10:
            return 0  # No-op
        
        # Extract own state
        own_pos = state[:3]
        own_vel = state[3:6]
        fuel_ratio = state[6]
        
        # If out of fuel, no-op
        if fuel_ratio < 0.05:
            return 0
        
        # Check nearby objects (starting at index 8)
        nearby_start = 8
        max_risk = 0.0
        closest_distance = float('inf')
        closest_relative_pos = np.zeros(3)
        
        # Scan nearby objects
        for i in range(7):  # Max 7 nearby objects
            obj_start = nearby_start + i * 6
            if obj_start + 6 > len(state):
                break
            
            obj_pos = state[obj_start:obj_start+3]
            obj_vel = state[obj_start+3:obj_start+6]
            
            # Skip zero objects (empty slots)
            if np.allclose(obj_pos, 0) and np.allclose(obj_vel, 0):
                continue
            
            # Compute distance and relative velocity
            rel_pos = obj_pos - own_pos
            rel_vel = obj_vel - own_vel
            distance = np.linalg.norm(rel_pos)
            rel_speed = np.linalg.norm(rel_vel)
            
            # Estimate risk (simplified)
            risk = 1.0 - np.clip(distance / 100.0, 0, 1)  # Higher risk if closer
            
            if risk > max_risk:
                max_risk = risk
                closest_distance = distance
                closest_relative_pos = rel_pos
        
        self.last_risks[agent_id] = max_risk
        
        # Decision logic
        if max_risk < self.risk_threshold:
            return 0  # No-op
        
        # If collision risk, choose maneuver based on relative geometry
        # Simplified: use radial out maneuver (away from Earth)
        if closest_distance < 1.0:
            return 3  # RADIAL_OUT
        elif closest_relative_pos[0] > 0:
            return 1  # PROGRADE (move away)
        else:
            return 2  # RETROGRADE
    
    def reset(self):
        """Reset policy."""
        self.last_risks = {}


class RuleBasedPolicy(BasePolicy):
    """
    Advanced rule-based policy with multiple strategies.
    Uses more sophisticated decision logic than baseline.
    """
    
    def __init__(self, aggression: float = 0.5):
        """
        Initialize rule-based policy.
        
        Args:
            aggression: How aggressively to maneuver (0-1)
        """
        self.aggression = np.clip(aggression, 0, 1)
        self.maneuver_history = {}
    
    def select_action(self, state: np.ndarray, agent_id: str) -> int:
        """
        Select action using rule-based strategy.
        
        Args:
            state: Observation
            agent_id: Agent identifier
            
        Returns:
            Action index
        """
        if len(state) < 10:
            return 0
        
        own_pos = state[:3]
        own_vel = state[3:6]
        fuel_ratio = state[6]
        
        if fuel_ratio < 0.05:
            return 0
        
        # Analyze nearby objects
        nearby_start = 8
        threats = []
        
        for i in range(7):
            obj_start = nearby_start + i * 6
            if obj_start + 6 > len(state):
                break
            
            obj_pos = state[obj_start:obj_start+3]
            obj_vel = state[obj_start+3:obj_start+6]
            
            if np.allclose(obj_pos, 0) and np.allclose(obj_vel, 0):
                continue
            
            rel_pos = obj_pos - own_pos
            rel_vel = obj_vel - own_vel
            distance = np.linalg.norm(rel_pos)
            
            # Estimate TCA (time to closest approach)
            dot_rv = np.dot(rel_pos, rel_vel)
            dot_vv = np.dot(rel_vel, rel_vel)
            
            if dot_vv > 1e-8:
                tca = -dot_rv / dot_vv
                if 0 < tca < 3600:  # Next hour
                    threats.append({
                        'distance': distance,
                        'tca': tca,
                        'rel_pos': rel_pos,
                        'rel_vel': rel_vel
                    })
        
        if not threats:
            return 0
        
        # Sort by urgency (TCA / distance)
        threats.sort(key=lambda t: t['tca'] / (t['distance'] + 0.1))
        primary_threat = threats[0]
        
        # Choose maneuver based on threat geometry and aggression
        distance = primary_threat['distance']
        tca = primary_threat['tca']
        rel_pos = primary_threat['rel_pos']
        
        # More aggressive with lower fuel threshold
        fuel_limit = 0.2 if self.aggression > 0.7 else 0.05
        
        if fuel_ratio < fuel_limit:
            return 0  # Conserve fuel
        
        # Select maneuver
        if distance < 0.5 and tca < 300:
            # Imminent threat - aggressive maneuver
            return 3  # RADIAL_OUT
        elif rel_pos[0] < 0:
            # Object ahead - go prograde
            return 1  # PROGRADE
        else:
            # Object behind - go retrograde
            return 2  # RETROGRADE
    
    def reset(self):
        """Reset policy."""
        self.maneuver_history = {}


class NoOpPolicy(BasePolicy):
    """Worst-case policy: always choose NO_OP."""

    def select_action(self, state: np.ndarray, agent_id: str) -> int:
        return 0

    def reset(self):
        pass


class ThresholdRulePolicy(BasePolicy):
    """
    Simple threshold-based deterministic rule.

    If any nearby object is within `threshold_km`, execute `dv_action`
    (e.g., PROGRADE) else NO_OP.
    """

    def __init__(self, threshold_km: float = 5.0, dv_action: int = 1):
        self.threshold_km = float(threshold_km)
        self.dv_action = int(dv_action)

    def select_action(self, state: np.ndarray, agent_id: str) -> int:
        if len(state) < 10:
            return 0

        own_pos = state[:3]

        nearby_start = 8
        min_distance = float("inf")

        for i in range(7):
            obj_start = nearby_start + i * 6
            if obj_start + 6 > len(state):
                break

            obj_pos = state[obj_start : obj_start + 3]
            obj_vel = state[obj_start + 3 : obj_start + 6]

            # Empty slots are encoded as zeros.
            if np.allclose(obj_pos, 0) and np.allclose(obj_vel, 0):
                continue

            dist = np.linalg.norm(obj_pos - own_pos)
            if dist < min_distance:
                min_distance = dist

        if min_distance < self.threshold_km:
            return self.dv_action
        return 0

    def reset(self):
        pass


class FuelAwareThresholdRulePolicy(BasePolicy):
    """
    Distance-threshold rule gated by minimum remaining fuel.
    """

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
        if len(state) < 10:
            return 0

        own_pos = state[:3]
        fuel_ratio = float(state[6])

        # Conserve fuel if below threshold.
        if fuel_ratio <= self.min_fuel_ratio:
            return 0

        nearby_start = 8
        min_distance = float("inf")

        for i in range(7):
            obj_start = nearby_start + i * 6
            if obj_start + 6 > len(state):
                break

            obj_pos = state[obj_start : obj_start + 3]
            obj_vel = state[obj_start + 3 : obj_start + 6]

            if np.allclose(obj_pos, 0) and np.allclose(obj_vel, 0):
                continue

            dist = np.linalg.norm(obj_pos - own_pos)
            if dist < min_distance:
                min_distance = dist

        if min_distance < self.threshold_km:
            return self.dv_action
        return 0

    def reset(self):
        pass


class MARLPolicy(BasePolicy):
    """
    MARL policy wrapper (uses trained MARL model).
    """
    
    def __init__(self, marl_trainer):
        """
        Initialize MARL policy.
        
        Args:
            marl_trainer: Trained MARL trainer instance
        """
        self.marl_trainer = marl_trainer
    
    def select_action(self, state: np.ndarray, agent_id: str) -> int:
        """
        Select action using MARL policy.
        
        Args:
            state: Observation
            agent_id: Agent identifier
            
        Returns:
            Action index
        """
        observations = {agent_id: state}
        actions = self.marl_trainer.get_actions(observations)
        return actions.get(agent_id, 0)
    
    def reset(self):
        """Reset policy."""
        pass


class RandomPolicy(BasePolicy):
    """Random policy for baseline comparison."""
    
    def select_action(self, state: np.ndarray, agent_id: str) -> int:
        """Select random action."""
        return np.random.randint(0, 6)
    
    def reset(self):
        """Reset policy."""
        pass


class PolicyManager:
    """
    Manages multiple policies and enables easy switching.
    """
    
    def __init__(self):
        """Initialize policy manager."""
        self.policies: Dict[str, BasePolicy] = {}
        self.active_policy = None
    
    def register_policy(self, name: str, policy: BasePolicy) -> None:
        """Register a policy."""
        self.policies[name] = policy
    
    def use_policy(self, name: str) -> None:
        """Switch to a policy."""
        if name not in self.policies:
            raise ValueError(f"Policy '{name}' not registered")
        self.active_policy = name
    
    def select_action(self, state: np.ndarray, agent_id: str) -> int:
        """Select action using active policy."""
        if self.active_policy is None:
            raise ValueError("No active policy selected")
        
        return self.policies[self.active_policy].select_action(state, agent_id)
    
    def get_available_policies(self) -> list:
        """Get list of registered policies."""
        return list(self.policies.keys())
    
    def get_active_policy_name(self) -> str:
        """Get name of active policy."""
        return self.active_policy
