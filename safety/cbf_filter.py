"""
MODULE 7: Control Barrier Function (CBF) Safety Filter
Prevents unsafe maneuvers by projecting actions to safe set.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy.optimize import minimize


class CBFSafetyFilter:
    """
    Control Barrier Function safety filter.
    Projects actions to ensure safety constraints are maintained.
    """
    
    def __init__(self,
                 min_safe_distance_km: float = 0.1,
                 decay_rate: float = 1.0,
                 alpha: float = 1.0):
        """
        Initialize CBF safety filter.
        
        Args:
            min_safe_distance_km: Minimum safe separation distance
            decay_rate: CBF decay rate parameter
            alpha: CBF alpha parameter (larger = more aggressive safety)
        """
        self.min_safe_distance = min_safe_distance_km
        self.decay_rate = decay_rate
        self.alpha = alpha
    
    def filter_action(self, state: np.ndarray,
                     action_dv: np.ndarray,
                     nearby_objects: Dict[str, np.ndarray],
                     max_dv: float = 0.1) -> np.ndarray:
        """
        Filter action through CBF safety constraint.
        
        Args:
            state: Current state [pos, vel] (6d)
            action_dv: Proposed ΔV from policy (3d)
            nearby_objects: Dict of nearby object states
            max_dv: Maximum allowed ΔV magnitude
            
        Returns:
            Safe ΔV vector
        """
        # If no nearby objects, return original action (clipped)
        if not nearby_objects:
            return self._clip_action(action_dv, max_dv)
        
        # Check safety constraints
        constraints = []
        for obj_id, obj_state in nearby_objects.items():
            constraint = self._compute_cbf_constraint(
                state, obj_state, action_dv
            )
            if constraint is not None:
                constraints.append(constraint)
        
        if not constraints:
            return self._clip_action(action_dv, max_dv)
        
        # If any constraint violated, solve for safe action
        has_violation = any(c['value'] < 0 for c in constraints)
        
        if not has_violation:
            return self._clip_action(action_dv, max_dv)
        
        # Solve QP to find closest safe action
        safe_dv = self._solve_safe_action_qp(
            action_dv, constraints, max_dv
        )
        
        return safe_dv
    
    def _compute_cbf_constraint(self, own_state: np.ndarray,
                               other_state: np.ndarray,
                               action_dv: np.ndarray) -> Optional[Dict]:
        """
        Compute CBF constraint for a pair of objects.
        
        The barrier function is: h = ||r||^2 - d_min^2
        where r is relative position and d_min is minimum safe distance.
        
        We require: dh/dt + alpha * h ≥ 0 (ensures h remains positive)
        
        Args:
            own_state: Own state [pos, vel]
            other_state: Other object state [pos, vel]
            action_dv: Proposed action
            
        Returns:
            Constraint dict or None if not violated
        """
        own_pos = own_state[:3]
        own_vel = own_state[3:]
        other_pos = other_state[:3]
        other_vel = other_state[3:]
        
        # Relative state
        rel_pos = own_pos - other_pos
        rel_vel = own_vel - other_vel
        
        # Barrier function: h = ||r||^2 - d_min^2
        distance = np.linalg.norm(rel_pos)
        h = distance ** 2 - self.min_safe_distance ** 2
        
        # Skip if already safely away
        if h > 0.1:  # Buffer zone
            return None
        
        # Compute h_dot
        # h_dot = 2 * r · v
        h_dot = 2 * np.dot(rel_pos, rel_vel)
        
        # Compute h_double_dot (only from own action)
        # action changes own_vel, so dv_own appears in rel_vel
        # h_double_dot ≈ 2 * r · (action_dv) + 2 * v · v
        h_double_dot = 2 * np.dot(rel_pos, action_dv) + 2 * np.dot(rel_vel, rel_vel)
        
        # Constraint: h_double_dot + 2*alpha*h_dot + alpha^2*h ≥ 0
        constraint_value = h_double_dot + 2 * self.alpha * h_dot + self.alpha ** 2 * h
        
        return {
            'distance': distance,
            'h': h,
            'h_dot': h_dot,
            'h_double_dot': h_double_dot,
            'value': constraint_value,
            'rel_pos': rel_pos,
            'rel_vel': rel_vel
        }
    
    def _solve_safe_action_qp(self, desired_action: np.ndarray,
                             constraints: list,
                             max_dv: float) -> np.ndarray:
        """
        Solve quadratic program to find closest safe action.
        
        Minimize: ||u - u_desired||^2
        Subject to: constraint_value(u) ≥ 0 for all constraints
                    ||u|| ≤ max_dv
        
        Args:
            desired_action: Desired action
            constraints: List of constraint dicts
            max_dv: Maximum ΔV magnitude
            
        Returns:
            Safe ΔV vector
        """
        def objective(u):
            return np.sum((u - desired_action) ** 2)
        
        def constraint_func(u, idx):
            # Recompute constraint value for action u
            c = constraints[idx]
            rel_pos = c['rel_pos']
            rel_vel = c['rel_vel']
            h = c['h']
            
            h_2 = 2 * np.dot(rel_pos, u) + 2 * np.dot(rel_vel, rel_vel)
            h_1 = c['h_dot']
            
            return h_2 + 2 * self.alpha * h_1 + self.alpha ** 2 * h
        
        # Linear constraints for each CBF
        cons = []
        for i in range(len(constraints)):
            cons.append({
                'type': 'ineq',
                'fun': lambda u, idx=i: constraint_func(u, idx)
            })
        
        # Norm constraint
        cons.append({
            'type': 'ineq',
            'fun': lambda u: max_dv - np.linalg.norm(u)
        })
        
# Ensure numeric stability by using float64 in optimization
        desired_action = np.asarray(desired_action, dtype=np.float64)

        # Initial guess
        x0 = desired_action.copy()
        if np.linalg.norm(x0) > max_dv:
            x0 = x0 / np.linalg.norm(x0) * max_dv
        
        # Solve
        result = minimize(objective, x0, method='SLSQP', constraints=cons,
                         options={'ftol': 1e-6, 'maxiter': 20})
        
        if result.success:
            return result.x
        else:
            # If optimization fails, return clipped desired action
            return self._clip_action(desired_action, max_dv)
    
    def _clip_action(self, action: np.ndarray, max_magnitude: float) -> np.ndarray:
        """Clip action to maximum magnitude."""
        magnitude = np.linalg.norm(action)
        if magnitude > max_magnitude:
            return action * (max_magnitude / magnitude)
        return action
