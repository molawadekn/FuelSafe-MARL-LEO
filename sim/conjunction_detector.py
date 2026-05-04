"""
MODULE 3: Conjunction Detection
Detects conjunction events in real-time during simulation.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from datetime import datetime


@dataclass
class ConjunctionAlert:
    """Represents a detected conjunction event."""
    object1_id: str
    object2_id: str
    distance_km: float
    miss_distance_estimate_km: float  # Minimum distance along trajectory
    time_to_closest_approach_s: float
    relative_velocity_kms: float
    risk_score: float  # 0-1, normalized risk metric
    collision_probability: float  # 0-1, physics-based theoretical Pc
    is_collision: bool  # True if distance < safety threshold
    timestamp: datetime
    alert_id: str


class ConjunctionDetector:
    """
    Detects conjunction events in real-time.
    Computes risk scores and estimates closest approach.
    """
    
    def __init__(self, 
                 distance_threshold_km: float = 10.0,
                 collision_threshold_km: float = 0.005,  # 5 meters — realistic hard-body radius
                 max_risk_score: float = 1.0):
        """
        Initialize conjunction detector.
        
        Args:
            distance_threshold_km: Alert distance threshold (km)
            collision_threshold_km: Collision detection threshold (km)
            max_risk_score: Maximum risk score (for normalization)
        """
        self.distance_threshold = distance_threshold_km
        self.collision_threshold = collision_threshold_km
        self.max_risk_score = max_risk_score
        self.alert_counter = 0

    @staticmethod
    def compute_collision_probability(miss_distance_km: float, time_to_ca_s: float) -> float:
        """
        Compute theoretical collision probability using an exponential Gaussian envelope.
        """
        d0 = 0.05  # 50m miss distance exponential scaling envelope
        t0 = 300.0  # 5 minute TCA urgency exponential scaling envelope
        return float(np.exp(-max(0.0, miss_distance_km) / d0) * np.exp(-max(0.0, time_to_ca_s) / t0))
        
    def detect(self, object_states: Dict[str, np.ndarray],
              timestamp: datetime,
              interest_ids: Optional[List[str]] = None) -> List[ConjunctionAlert]:
        """
        Detect conjunctions among objects.
        If interest_ids is provided, only checks pairs where at least one object is in the list.
        """
        alerts = []
        obj_ids = list(object_states.keys())
        
        if interest_ids is not None:
            # Optimized path: check only interest_ids against everyone else
            checked_pairs = set()
            for obj1_id in interest_ids:
                if obj1_id not in object_states:
                    continue
                state1 = object_states[obj1_id]
                pos1, vel1 = state1[:3], state1[3:]
                
                for obj2_id in obj_ids:
                    if obj1_id == obj2_id:
                        continue
                    
                    pair = tuple(sorted((obj1_id, obj2_id)))
                    if pair in checked_pairs:
                        continue
                    checked_pairs.add(pair)
                    
                    state2 = object_states[obj2_id]
                    rel_pos, rel_vel = state2[:3] - pos1, state2[3:] - vel1
                    
                    distance = np.linalg.norm(rel_pos)
                    if distance > self.distance_threshold:
                        continue
                    
                    miss_dist, time_to_ca = self._estimate_closest_approach(rel_pos, rel_vel)
                    rel_speed = np.linalg.norm(rel_vel)
                    risk_score = self._compute_risk_score(distance, miss_dist, rel_speed)
                    pc = self.compute_collision_probability(miss_dist, time_to_ca)
                    
                    self.alert_counter += 1
                    alerts.append(ConjunctionAlert(
                        object1_id=obj1_id,
                        object2_id=obj2_id,
                        distance_km=distance,
                        miss_distance_estimate_km=max(miss_dist, 0.0),
                        time_to_closest_approach_s=max(time_to_ca, 0.0),
                        relative_velocity_kms=rel_speed,
                        risk_score=risk_score,
                        collision_probability=pc,
                        is_collision=(distance < self.collision_threshold),
                        timestamp=timestamp,
                        alert_id=f"CONJ_{self.alert_counter:08d}"
                    ))
        else:
            # Original O(N^2) path
            for i, obj1_id in enumerate(obj_ids):
                for obj2_id in obj_ids[i+1:]:
                    state1, state2 = object_states[obj1_id], object_states[obj2_id]
                    rel_pos, rel_vel = state2[:3] - state1[:3], state2[3:] - state1[3:]
                    distance = np.linalg.norm(rel_pos)
                    if distance > self.distance_threshold:
                        continue
                    miss_dist, time_to_ca = self._estimate_closest_approach(rel_pos, rel_vel)
                    rel_speed = np.linalg.norm(rel_vel)
                    risk_score = self._compute_risk_score(distance, miss_dist, rel_speed)
                    pc = self.compute_collision_probability(miss_dist, time_to_ca)
                    self.alert_counter += 1
                    alerts.append(ConjunctionAlert(
                        object1_id=obj1_id,
                        object2_id=obj2_id,
                        distance_km=distance,
                        miss_distance_estimate_km=max(miss_dist, 0.0),
                        time_to_closest_approach_s=max(time_to_ca, 0.0),
                        relative_velocity_kms=rel_speed,
                        risk_score=risk_score,
                        collision_probability=pc,
                        is_collision=(distance < self.collision_threshold),
                        timestamp=timestamp,
                        alert_id=f"CONJ_{self.alert_counter:08d}"
                    ))
        return alerts

    def detect_for_object(self, obj_id: str, 
                         object_state: np.ndarray,
                         other_states: Dict[str, np.ndarray],
                         timestamp: datetime) -> List[ConjunctionAlert]:
        """Detect conjunctions for a specific object only."""
        alerts = []
        pos1, vel1 = object_state[:3], object_state[3:]
        for other_id, other_state in other_states.items():
            if other_id == obj_id: continue
            rel_pos, rel_vel = other_state[:3] - pos1, other_state[3:] - vel1
            distance = np.linalg.norm(rel_pos)
            if distance > self.distance_threshold: continue
            miss_dist, time_to_ca = self._estimate_closest_approach(rel_pos, rel_vel)
            rel_speed = np.linalg.norm(rel_vel)
            risk_score = self._compute_risk_score(distance, miss_dist, rel_speed)
            pc = self.compute_collision_probability(miss_dist, time_to_ca)
            self.alert_counter += 1
            alerts.append(ConjunctionAlert(
                object1_id=obj_id, object2_id=other_id, distance_km=distance,
                miss_distance_estimate_km=max(miss_dist, 0.0),
                time_to_closest_approach_s=max(time_to_ca, 0.0),
                relative_velocity_kms=rel_speed, risk_score=risk_score,
                collision_probability=pc, is_collision=(distance < self.collision_threshold),
                timestamp=timestamp, alert_id=f"CONJ_{self.alert_counter:08d}"
            ))
        return alerts

    @staticmethod
    def _estimate_closest_approach(rel_pos: np.ndarray, 
                                  rel_vel: np.ndarray) -> Tuple[float, float]:
        dot_rv = np.dot(rel_pos, rel_vel)
        dot_vv = np.dot(rel_vel, rel_vel)
        if dot_vv < 1e-10: return np.linalg.norm(rel_pos), 0.0
        time_to_ca = -dot_rv / dot_vv
        if time_to_ca < 0: return np.linalg.norm(rel_pos), 0.0
        pos_at_ca = rel_pos + rel_vel * time_to_ca
        return np.linalg.norm(pos_at_ca), time_to_ca
    
    def _compute_risk_score(self, current_distance: float, 
                           miss_distance: float,
                           relative_speed: float) -> float:
        proximity_risk = 1.0 - np.clip(current_distance / self.distance_threshold, 0, 1)
        trajectory_risk = 1.0 - np.clip(miss_distance / self.distance_threshold, 0, 1)
        velocity_risk = np.clip(relative_speed / 15.0, 0, 1)
        risk_score = (0.4 * proximity_risk + 0.4 * trajectory_risk + 0.2 * velocity_risk)
        return float(np.clip(risk_score, 0, self.max_risk_score))
    
    def get_alerts_by_risk(self, alerts: List[ConjunctionAlert],
                          min_risk: float = 0.5) -> List[ConjunctionAlert]:
        return [a for a in alerts if a.risk_score >= min_risk]
    
    def get_imminent_alerts(self, alerts: List[ConjunctionAlert],
                           time_threshold_s: float = 3600) -> List[ConjunctionAlert]:
        return [a for a in alerts if a.time_to_closest_approach_s <= time_threshold_s]
