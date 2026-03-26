"""
MODULE 2: ESA Conjunction Data Message (CDM) Ingestion
Loads and processes ESA CDM data for collision events.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
import json


@dataclass
class CDMEvent:
    """Represents a conjunction event from ESA CDM."""
    cdm_id: str
    object1_id: str
    object2_id: str
    time_of_closest_approach: datetime
    miss_distance_km: float  # Closest approach distance
    relative_velocity_kms: float
    collision_probability: float  # Probability of collision
    time_to_closest_approach_s: float  # Seconds from now to TCA
    conjunction_id: str
    
    def __hash__(self):
        return hash(self.cdm_id)


class CDMLoader:
    """
    Loads and processes ESA CDM data.
    Provides filtering and event extraction utilities.
    """
    
    def __init__(self):
        """Initialize CDM loader."""
        self.events: Dict[str, CDMEvent] = {}
        
    def load_from_json(self, filepath: str) -> List[CDMEvent]:
        """
        Load CDM events from JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            List of CDMEvent objects
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        events = []
        for record in data.get('conjunctions', []):
            event = self._parse_cdm_record(record)
            if event:
                events.append(event)
                self.events[event.cdm_id] = event
        
        return events
    
    def load_from_csv(self, filepath: str) -> List[CDMEvent]:
        """
        Load CDM events from CSV file.
        Expected columns: object1_id, object2_id, tca, miss_distance_km,
                         relative_velocity_kms, collision_probability
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            List of CDMEvent objects
        """
        df = pd.read_csv(filepath)
        events = []
        
        for idx, row in df.iterrows():
            event = CDMEvent(
                cdm_id=f"CDM_{idx:05d}",
                object1_id=str(row['object1_id']),
                object2_id=str(row['object2_id']),
                time_of_closest_approach=pd.to_datetime(row['tca']),
                miss_distance_km=float(row['miss_distance_km']),
                relative_velocity_kms=float(row.get('relative_velocity_kms', 0.0)),
                collision_probability=float(row.get('collision_probability', 0.0)),
                time_to_closest_approach_s=float(row.get('time_to_closest_approach_s', 0.0)),
                conjunction_id=f"{row['object1_id']}_{row['object2_id']}_{idx}"
            )
            events.append(event)
            self.events[event.cdm_id] = event
        
        return events
    
    def generate_sample_events(self, num_events: int = 5) -> List[CDMEvent]:
        """
        Generate sample CDM events for testing.
        
        Args:
            num_events: Number of events to generate
            
        Returns:
            List of CDMEvent objects
        """
        events = []
        now = datetime.utcnow()
        
        for i in range(num_events):
            # Create realistic sample events
            time_to_tca = 3600 + i * 7200  # Hours to closest approach
            event = CDMEvent(
                cdm_id=f"CDM_SAMPLE_{i:04d}",
                object1_id=f"SAT_{i:03d}",
                object2_id=f"DEBRIS_{i:03d}",
                time_of_closest_approach=now + pd.Timedelta(seconds=time_to_tca),
                miss_distance_km=0.5 + i * 0.1,  # 0.5 to 0.9 km
                relative_velocity_kms=8.0 + np.random.randn() * 0.5,
                collision_probability=1e-5 + np.random.rand() * 1e-3,
                time_to_closest_approach_s=float(time_to_tca),
                conjunction_id=f"CONJ_{i:04d}"
            )
            events.append(event)
            self.events[event.cdm_id] = event
        
        return events
    
    def filter_by_probability(self, min_prob: float = 1e-4) -> List[CDMEvent]:
        """
        Filter events by collision probability.
        
        Args:
            min_prob: Minimum collision probability threshold
            
        Returns:
            List of filtered CDMEvent objects
        """
        return [e for e in self.events.values() 
                if e.collision_probability >= min_prob]
    
    def filter_by_miss_distance(self, max_distance_km: float = 1.0) -> List[CDMEvent]:
        """
        Filter events by miss distance.
        
        Args:
            max_distance_km: Maximum miss distance in km
            
        Returns:
            List of filtered CDMEvent objects
        """
        return [e for e in self.events.values() 
                if e.miss_distance_km <= max_distance_km]
    
    def filter_by_time_window(self, start_time: datetime, 
                             end_time: datetime) -> List[CDMEvent]:
        """
        Filter events by time window.
        
        Args:
            start_time: Start of time window
            end_time: End of time window
            
        Returns:
            List of filtered CDMEvent objects
        """
        return [e for e in self.events.values() 
                if start_time <= e.time_of_closest_approach <= end_time]
    
    def filter_by_object(self, object_id: str) -> List[CDMEvent]:
        """
        Filter events involving a specific object.
        
        Args:
            object_id: Object identifier
            
        Returns:
            List of filtered CDMEvent objects
        """
        return [e for e in self.events.values() 
                if e.object1_id == object_id or e.object2_id == object_id]
    
    def get_critical_events(self, prob_threshold: float = 1e-3,
                           time_threshold_hours: float = 24) -> List[CDMEvent]:
        """
        Get critical conjunction events (high probability + imminent).
        
        Args:
            prob_threshold: Probability threshold
            time_threshold_hours: Time window in hours
            
        Returns:
            List of critical CDMEvent objects
        """
        now = datetime.utcnow()
        future = now + pd.Timedelta(hours=time_threshold_hours)
        
        critical = []
        for event in self.events.values():
            if (event.collision_probability >= prob_threshold and 
                now <= event.time_of_closest_approach <= future):
                critical.append(event)
        
        return sorted(critical, key=lambda e: e.collision_probability, reverse=True)
    
    def export_to_csv(self, filepath: str) -> None:
        """
        Export loaded events to CSV.
        
        Args:
            filepath: Output CSV filepath
        """
        data = []
        for event in self.events.values():
            data.append({
                'cdm_id': event.cdm_id,
                'object1_id': event.object1_id,
                'object2_id': event.object2_id,
                'tca': event.time_of_closest_approach,
                'miss_distance_km': event.miss_distance_km,
                'relative_velocity_kms': event.relative_velocity_kms,
                'collision_probability': event.collision_probability,
                'time_to_tca_s': event.time_to_closest_approach_s
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
    
    def _parse_cdm_record(self, record: dict) -> Optional[CDMEvent]:
        """Parse a CDM record from dict."""
        try:
            return CDMEvent(
                cdm_id=record.get('cdm_id', 'UNKNOWN'),
                object1_id=str(record['object1_id']),
                object2_id=str(record['object2_id']),
                time_of_closest_approach=pd.to_datetime(record['tca']),
                miss_distance_km=float(record['miss_distance_km']),
                relative_velocity_kms=float(record.get('relative_velocity_kms', 0.0)),
                collision_probability=float(record.get('collision_probability', 0.0)),
                time_to_closest_approach_s=float(record.get('time_to_tca_s', 0.0)),
                conjunction_id=record.get('conjunction_id', '')
            )
        except (KeyError, ValueError, TypeError):
            return None
