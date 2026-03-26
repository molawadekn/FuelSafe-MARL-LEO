"""
MODULE 3: Conjunction Detection
Detects conjunction events in real-time during simulation.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
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
                 collision_threshold_km: float = 0.025,  # 25 meters
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
        
    def detect(self, object_states: Dict[str, np.ndarray],
              timestamp: datetime) -> List[ConjunctionAlert]:
        """
        Detect conjunctions among all objects.
        
        Args:
            object_states: Dict mapping object_id to [pos, vel] state
            timestamp: Current simulation time
            
        Returns:
            List of ConjunctionAlert objects
        """
        alerts = []
        obj_ids = list(object_states.keys())
        
        # Check all pairs
        for i, obj1_id in enumerate(obj_ids):
            for obj2_id in obj_ids[i+1:]:
                state1 = object_states[obj1_id]
                state2 = object_states[obj2_id]
                
                # Extract position and velocity
                pos1 = state1[:3]
                vel1 = state1[3:]
                pos2 = state2[:3]
                vel2 = state2[3:]
                
                # Compute relative state
                rel_pos = pos2 - pos1
                rel_vel = vel2 - vel1
                
                # Current distance
                distance = np.linalg.norm(rel_pos)
                
                # Skip if too far
                if distance > self.distance_threshold:
                    continue
                
                # Estimate closest approach
                miss_dist, time_to_ca = self._estimate_closest_approach(
                    rel_pos, rel_vel
                )
                
                # Compute risk score
                rel_speed = np.linalg.norm(rel_vel)
                risk_score = self._compute_risk_score(
                    distance, miss_dist, rel_speed
                )
                
                # Check if collision
                is_collision = distance < self.collision_threshold
                
                self.alert_counter += 1
                alert = ConjunctionAlert(
                    object1_id=obj1_id,
                    object2_id=obj2_id,
                    distance_km=distance,
                    miss_distance_estimate_km=max(miss_dist, 0.0),
                    time_to_closest_approach_s=max(time_to_ca, 0.0),
                    relative_velocity_kms=rel_speed,
                    risk_score=risk_score,
                    is_collision=is_collision,
                    timestamp=timestamp,
                    alert_id=f"CONJ_{self.alert_counter:08d}"
                )
                alerts.append(alert)
        
        return alerts
    
    def detect_for_object(self, obj_id: str, 
                         object_state: np.ndarray,
                         other_states: Dict[str, np.ndarray],
                         timestamp: datetime) -> List[ConjunctionAlert]:
        """
        Detect conjunctions for a specific object only.
        
        Args:
            obj_id: Target object ID
            object_state: Target object state [pos, vel]
            other_states: Dict of other object states
            timestamp: Current simulation time
            
        Returns:
            List of ConjunctionAlert objects
        """
        alerts = []
        pos1 = object_state[:3]
        vel1 = object_state[3:]
        
        for other_id, other_state in other_states.items():
            if other_id == obj_id:
                continue
            
            pos2 = other_state[:3]
            vel2 = other_state[3:]
            
            rel_pos = pos2 - pos1
            rel_vel = vel2 - vel1
            distance = np.linalg.norm(rel_pos)
            
            if distance > self.distance_threshold:
                continue
            
            miss_dist, time_to_ca = self._estimate_closest_approach(
                rel_pos, rel_vel
            )
            
            rel_speed = np.linalg.norm(rel_vel)
            risk_score = self._compute_risk_score(distance, miss_dist, rel_speed)
            is_collision = distance < self.collision_threshold
            
            self.alert_counter += 1
            alert = ConjunctionAlert(
                object1_id=obj_id,
                object2_id=other_id,
                distance_km=distance,
                miss_distance_estimate_km=max(miss_dist, 0.0),
                time_to_closest_approach_s=max(time_to_ca, 0.0),
                relative_velocity_kms=rel_speed,
                risk_score=risk_score,
                is_collision=is_collision,
                timestamp=timestamp,
                alert_id=f"CONJ_{self.alert_counter:08d}"
            )
            alerts.append(alert)
        
        return alerts
    
    @staticmethod
    def _estimate_closest_approach(rel_pos: np.ndarray, 
                                  rel_vel: np.ndarray) -> Tuple[float, float]:
        """
        Estimate miss distance and time to closest approach.
        Assumes linear relative motion (valid for short time scales).
        
        Args:
            rel_pos: Relative position vector [km]
            rel_vel: Relative velocity vector [km/s]
            
        Returns:
            (miss_distance_km, time_to_closest_approach_s)
        """
        # Time to closest approach: t = -(r·v) / (v·v)
        dot_rv = np.dot(rel_pos, rel_vel)
        dot_vv = np.dot(rel_vel, rel_vel)
        
        if dot_vv < 1e-10:  # Nearly zero relative velocity
            return np.linalg.norm(rel_pos), 0.0
        
        time_to_ca = -dot_rv / dot_vv
        
        if time_to_ca < 0:  # Closest approach in past
            return np.linalg.norm(rel_pos), 0.0
        
        # Position at closest approach
        pos_at_ca = rel_pos + rel_vel * time_to_ca
        miss_distance = np.linalg.norm(pos_at_ca)
        
        return miss_distance, time_to_ca
    
    def _compute_risk_score(self, current_distance: float, 
                           miss_distance: float,
                           relative_speed: float) -> float:
        """
        Compute normalized risk score (0-1).
        
        Args:
            current_distance: Current distance [km]
            miss_distance: Estimated miss distance [km]
            relative_speed: Relative speed [km/s]
            
        Returns:
            Risk score (0-1)
        """
        # Components of risk
        # 1. Proximity risk: how close currently
        proximity_risk = 1.0 - np.clip(
            current_distance / self.distance_threshold, 0, 1
        )
        
        # 2. Trajectory risk: how close will get
        trajectory_risk = 1.0 - np.clip(
            miss_distance / self.distance_threshold, 0, 1
        )
        
        # 3. Velocity risk: speed of approach
        velocity_risk = np.clip(relative_speed / 15.0, 0, 1)  # 15 km/s as nominal relative speed
        
        # Weighted combination
        risk_score = (0.4 * proximity_risk + 
                     0.4 * trajectory_risk + 
                     0.2 * velocity_risk)
        
        return float(np.clip(risk_score, 0, self.max_risk_score))
    
    def get_alerts_by_risk(self, alerts: List[ConjunctionAlert],
                          min_risk: float = 0.5) -> List[ConjunctionAlert]:
        """Get alerts above risk threshold."""
        return [a for a in alerts if a.risk_score >= min_risk]
    
    def get_imminent_alerts(self, alerts: List[ConjunctionAlert],
                           time_threshold_s: float = 3600) -> List[ConjunctionAlert]:
        """Get alerts with closest approach within time threshold."""
        return [a for a in alerts if a.time_to_closest_approach_s <= time_threshold_s]
