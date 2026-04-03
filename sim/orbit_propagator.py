"""
MODULE 1: Orbit Propagation Engine (SGP4)
Propagates satellite and debris orbits using SGP4 model.
"""

import hashlib
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
from datetime import datetime, timezone
from sgp4.api import Satrec


@dataclass
class OrbitalState:
    """Represents orbital state at a given time."""
    position: np.ndarray  # [x, y, z] in km
    velocity: np.ndarray  # [vx, vy, vz] in km/s
    timestamp: datetime
    object_id: str
    
    def to_array(self) -> np.ndarray:
        """Convert to [pos, vel] array."""
        return np.concatenate([self.position, self.velocity])


class OrbitPropagator:
    """
    Propagates orbits using SGP4 model.
    Handles satellite and debris trajectory computation.
    """
    
    def __init__(self):
        """Initialize the propagator."""
        self.satellites: Dict[str, Satrec] = {}
        self.tle_lines: Dict[str, Tuple[str, str]] = {}
        
    def load_tle(self, object_id: str, tle_line1: str, tle_line2: str) -> None:
        """
        Load TLE (Two-Line Element) for a satellite/debris.
        
        Args:
            object_id: Unique identifier for the object
            tle_line1: TLE line 1
            tle_line2: TLE line 2
        """
        sat = Satrec.twoline2rv(tle_line1, tle_line2)
        self.satellites[object_id] = sat
        self.tle_lines[object_id] = (tle_line1, tle_line2)
        
    def propagate(self, object_id: str, timestamp: datetime) -> OrbitalState:
        """
        Propagate orbital state to a given time.
        
        Args:
            object_id: Object identifier
            timestamp: Target propagation time
            
        Returns:
            OrbitalState at the given time
        """
        if object_id not in self.satellites:
            raise ValueError(f"Object {object_id} not loaded")
        
        sat = self.satellites[object_id]
        
        # SGP4 expects JD (Julian Date)
        jd, fr = self._datetime_to_jd(timestamp)
        
        err, r, v = sat.sgp4(jd, fr)
        
        if err != 0:
            raise RuntimeError(f"SGP4 error code {err} for {object_id}")
        
        return OrbitalState(
            position=np.array(r),  # km
            velocity=np.array(v),  # km/s
            timestamp=timestamp,
            object_id=object_id
        )
    
    def propagate_batch(self, object_id: str, times: List[datetime]) -> List[OrbitalState]:
        """
        Propagate orbital state to multiple times (vectorized).
        
        Args:
            object_id: Object identifier
            times: List of target times
            
        Returns:
            List of OrbitalState at each time
        """
        states = []
        for t in times:
            states.append(self.propagate(object_id, t))
        return states
    
    def propagate_all(self, timestamp: datetime) -> Dict[str, OrbitalState]:
        """
        Propagate all loaded objects to a given time.
        
        Args:
            timestamp: Target propagation time
            
        Returns:
            Dict mapping object_id to OrbitalState
        """
        states = {}
        for obj_id in self.satellites.keys():
            states[obj_id] = self.propagate(obj_id, timestamp)
        return states
    
    @staticmethod
    def _datetime_to_jd(dt: datetime) -> Tuple[float, float]:
        """Convert datetime to JD and fraction."""
        # Normalize to UTC naive datetime for JD calculation
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)

        # Reference epoch (naive UTC)
        ref_dt = datetime(1858, 11, 17, 0, 0, 0)
        delta = dt - ref_dt
        jd_start = 2400000.5

        # Compute JD (including fractional day)
        total_days = delta.days + delta.seconds / 86400.0 + delta.microseconds / 86400.0 / 1e6
        jd = jd_start + total_days

        # Separate integer and fractional parts
        jd_int = int(jd)
        jd_frac = jd - jd_int

        return float(jd_int), float(jd_frac)
    
    def relative_state(self, obj1_state: OrbitalState, 
                      obj2_state: OrbitalState) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute relative position and velocity between two objects.
        
        Args:
            obj1_state: State of object 1
            obj2_state: State of object 2
            
        Returns:
            (relative_position, relative_velocity)
        """
        rel_pos = obj2_state.position - obj1_state.position
        rel_vel = obj2_state.velocity - obj1_state.velocity
        return rel_pos, rel_vel
    
    def generate_sample_tle(self, object_id: str,
                           semi_major_axis_km: float = 6800,
                           inclination_deg: float = 51.6,
                           eccentricity: float = 0.0001,
                           epoch_datetime: Optional[datetime] = None,
                           mean_anomaly_deg: float = 0.0,
                           raan_deg: float = 0.0) -> None:
        """
        Generate a sample TLE for testing (circular orbit).
        
        Args:
            object_id: Object identifier
            semi_major_axis_km: Semi-major axis in km
            inclination_deg: Inclination in degrees
            eccentricity: Eccentricity
            epoch_datetime: Fixed epoch to keep scenarios reproducible
            mean_anomaly_deg: Initial mean anomaly in degrees
            raan_deg: Initial RAAN in degrees
        """
        # Create a simplified TLE (note: proper TLE generation is complex)
        # This creates a basic LEO TLE
        anomaly_motion = self._compute_mean_motion(semi_major_axis_km)
        
        # Generate deterministic satellite number (Python hash is randomized per-process)
        satnum = self._stable_satnum(object_id)

        # Current epoch (year and day of year) - overridden for reproducibility.
        now = epoch_datetime or datetime.now(timezone.utc)
        if now.tzinfo is not None:
            now = now.astimezone(timezone.utc)
        year = now.year % 100  # Last two digits
        day_of_year = now.timetuple().tm_yday
        epoch = f"{year:02d}{day_of_year:03d}.00000000"
        
        # TLE fields
        ecc_field = f"{int(eccentricity * 1e7):07d}"
        
        # Simplified TLE lines (following standard format)
        line1 = f"1 {satnum:05d}U 00000    {epoch}  .00000000  00000-0  00000-0 0     9"
        line2 = f"2 {satnum:05d} {inclination_deg:8.4f} {raan_deg:8.4f} {ecc_field} {0.0:8.4f} {mean_anomaly_deg:8.4f} {anomaly_motion:11.8f} {0:5d}"
        
        self.load_tle(object_id, line1, line2)

    @staticmethod
    def _stable_satnum(object_id: str) -> int:
        """Deterministic satnum derived from object_id."""
        digest = hashlib.sha256(object_id.encode("utf-8")).hexdigest()
        # 1..99999 (fits the simplified satnum formatting used above)
        return int(digest[:8], 16) % 99999 + 1
    
    @staticmethod
    def _compute_mean_motion(semi_major_axis_km: float) -> float:
        """Compute mean motion (revolutions/day) from semi-major axis."""
        mu = 398600.4418  # Earth's gravitational constant (km^3/s^2)
        a = semi_major_axis_km
        n = np.sqrt(mu / (a ** 3)) * 86400 / (2 * np.pi)  # rev/day
        return n
