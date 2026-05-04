"""
MODULE 7: Control Barrier Function (CBF) Safety Filter
Projects proposed Delta-V actions into a threat-aware safe set.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.optimize import minimize


class CBFSafetyFilter:
    """
    Linearized CBF safety filter over relative position / velocity threats.

    The filter enforces a one-step barrier condition on the radial closing rate:
        d/dt(||r|| - d_safe) + alpha * (||r|| - d_safe) >= 0

    with r defined from the controlled satellite to the threatening object.
    """

    def __init__(
        self,
        min_safe_distance_km: float = 0.1,
        alpha: float = 0.8,
        time_horizon_s: float = 1200.0,
        risk_threshold: float = 0.35,
    ):
        self.min_safe_distance = float(min_safe_distance_km)
        self.alpha = float(alpha)
        self.time_horizon_s = float(time_horizon_s)
        self.risk_threshold = float(risk_threshold)

    def filter_action(
        self,
        state: np.ndarray,
        action_dv: np.ndarray,
        threats: List[Dict[str, np.ndarray | float]],
        max_dv: float = 0.1,
    ) -> np.ndarray:
        """Project a proposed Delta-V to the closest barrier-safe action."""
        desired_action = self._clip_action(np.asarray(action_dv, dtype=np.float64), max_dv)
        active_constraints = self._build_constraints(threats)

        if not active_constraints:
            return desired_action

        if all(self._constraint_value(desired_action, constraint) >= -1e-9 for constraint in active_constraints):
            return desired_action

        def objective(u: np.ndarray) -> float:
            return float(np.sum((u - desired_action) ** 2))

        cons = [
            {"type": "ineq", "fun": lambda u, c=constraint: self._constraint_value(u, c)}
            for constraint in active_constraints
        ]
        cons.append({"type": "ineq", "fun": lambda u: float(max_dv) - np.linalg.norm(u)})

        result = minimize(
            objective,
            x0=desired_action.copy(),
            method="SLSQP",
            constraints=cons,
            options={"ftol": 1e-6, "maxiter": 40},
        )

        if result.success:
            return self._clip_action(np.asarray(result.x, dtype=np.float64), max_dv)

        # If optimization fails, favor a conservative action.
        return np.zeros(3, dtype=np.float64)

    def _build_constraints(self, threats: List[Dict[str, np.ndarray | float]]) -> List[Dict[str, np.ndarray | float]]:
        constraints: List[Dict[str, np.ndarray | float]] = []
        for threat in threats:
            rel_pos = np.asarray(threat.get("rel_pos", np.zeros(3)), dtype=np.float64)
            rel_vel = np.asarray(threat.get("rel_vel", np.zeros(3)), dtype=np.float64)
            distance = float(threat.get("distance_km", np.linalg.norm(rel_pos)))
            if distance <= 1e-9:
                continue

            risk_score = float(threat.get("risk_score", 0.0))
            time_to_ca = float(threat.get("time_to_closest_approach_s", 0.0))
            closing_rate = float(np.dot(rel_pos / distance, rel_vel))
            is_closing = closing_rate < 0.0
            is_active = (
                distance <= 2.0 * self.min_safe_distance
                or risk_score >= self.risk_threshold
                or (0.0 <= time_to_ca <= self.time_horizon_s and is_closing)
            )
            if not is_active:
                continue

            effective_safe_distance = self.min_safe_distance * (1.0 + 0.5 * max(risk_score - 0.5, 0.0))
            constraints.append(
                {
                    "radial_unit": rel_pos / distance,
                    "closing_rate": closing_rate,
                    "distance_margin": distance - effective_safe_distance,
                }
            )
        return constraints

    def _constraint_value(self, action_dv: np.ndarray, constraint: Dict[str, np.ndarray | float]) -> float:
        radial_unit = np.asarray(constraint["radial_unit"], dtype=np.float64)
        closing_rate = float(constraint["closing_rate"])
        distance_margin = float(constraint["distance_margin"])
        # rel_pos points from the controlled object to the threat, so own Delta-V
        # subtracts from the relative velocity along that direction.
        return closing_rate - float(np.dot(radial_unit, action_dv)) + self.alpha * distance_margin

    def _clip_action(self, action: np.ndarray, max_magnitude: float) -> np.ndarray:
        magnitude = float(np.linalg.norm(action))
        if magnitude > float(max_magnitude) + 1e-12:
            return action * (float(max_magnitude) / magnitude)
        return action
