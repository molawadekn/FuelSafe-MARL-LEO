"""
CSV data loader for ESA CDM conjunction datasets.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class CSVDataLoader:
    """Load and preprocess ESA CDM conjunction data from CSV files."""

    def __init__(self, csv_path: str, verbose: bool = True):
        self.csv_path = Path(csv_path)
        self.verbose = verbose
        self.data: Optional[pd.DataFrame] = None
        self.events: List[Dict] = []
        self.event_count = 0

        if self.verbose:
            print(f"Initializing CSVDataLoader for: {self.csv_path}")

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

    def load(self, max_rows: Optional[int] = None) -> pd.DataFrame:
        """Load CSV data into memory."""
        if self.verbose:
            print(f"Loading CSV: {self.csv_path}")

        try:
            self.data = pd.read_csv(self.csv_path, nrows=max_rows)
            self.event_count = len(self.data)

            if self.verbose:
                print(f"[OK] Loaded {self.event_count} conjunction events")
                print(f"  Columns: {len(self.data.columns)}")
                print(f"  Memory: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

            return self.data
        except Exception as exc:
            print(f"[ERR] Error loading CSV: {exc}")
            raise

    def get_summary_stats(self) -> Dict:
        """Get basic statistics about loaded data."""
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        return {
            "total_events": len(self.data),
            "columns": len(self.data.columns),
            "date_range": f"{self.data['time_to_tca'].min():.2f} to {self.data['time_to_tca'].max():.2f} hours",
            "risk_range": f"{self.data['risk'].min():.6f} to {self.data['risk'].max():.6f}",
            "miss_distance_range": f"{self.data['miss_distance'].min():.0f} to {self.data['miss_distance'].max():.0f} m",
            "relative_speed_range": f"{self.data['relative_speed'].min():.0f} to {self.data['relative_speed'].max():.0f} m/s",
            "memory_mb": self.data.memory_usage(deep=True).sum() / 1024**2,
        }

    def filter_by_risk(self, min_risk: float = -8.0, max_risk: float = -5.0) -> pd.DataFrame:
        """Filter events by risk score range."""
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        filtered = self.data[(self.data["risk"] >= min_risk) & (self.data["risk"] <= max_risk)]

        if self.verbose:
            print(f"[INFO] Filtered by risk [{min_risk}, {max_risk}]: {len(filtered)} events")

        return filtered

    def filter_by_miss_distance(self, min_dist: float = 0, max_dist: float = 50000) -> pd.DataFrame:
        """Filter events by miss distance in meters."""
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        filtered = self.data[
            (self.data["miss_distance"] >= min_dist) & (self.data["miss_distance"] <= max_dist)
        ]

        if self.verbose:
            print(f"[INFO] Filtered by miss distance [{min_dist}, {max_dist}]m: {len(filtered)} events")

        return filtered

    def filter_by_time_to_tca(self, min_hours: float = 0, max_hours: float = 24) -> pd.DataFrame:
        """Filter events by time to closest approach."""
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        filtered = self.data[
            (self.data["time_to_tca"] >= min_hours) & (self.data["time_to_tca"] <= max_hours)
        ]

        if self.verbose:
            print(f"[INFO] Filtered by time to TCA [{min_hours}, {max_hours}] hours: {len(filtered)} events")

        return filtered

    def extract_high_risk_events(self, risk_threshold: float = -7.0, count: int = 100) -> pd.DataFrame:
        """Extract high-risk conjunction events sorted by descending risk."""
        if self.data is None:
            raise RuntimeError("Data not loaded. Call load() first.")

        high_risk = self.data[self.data["risk"] >= risk_threshold].sort_values("risk", ascending=False)
        high_risk = high_risk.head(count)

        if self.verbose:
            print(f"[INFO] Extracted {len(high_risk)} high-risk events (threshold: {risk_threshold})")

        return high_risk

    def extract_features_for_simulation(self, event_row: pd.Series, normalize: bool = True) -> Dict:
        """Extract simulation-ready features from one conjunction event row."""
        features = {
            "event_id": event_row["event_id"],
            "time_to_tca": event_row["time_to_tca"],
            "risk_score": event_row["risk"],
            "miss_distance": event_row["miss_distance"],
            "relative_speed": event_row["relative_speed"],
            "relative_position_r": event_row.get("relative_position_r", 0),
            "relative_position_t": event_row.get("relative_position_t", 0),
            "relative_position_n": event_row.get("relative_position_n", 0),
            "relative_velocity_r": event_row.get("relative_velocity_r", 0),
            "relative_velocity_t": event_row.get("relative_velocity_t", 0),
            "relative_velocity_n": event_row.get("relative_velocity_n", 0),
            "t_sma": event_row.get("t_j2k_sma", 7000),
            "t_ecc": event_row.get("t_j2k_ecc", 0.001),
            "t_inc": event_row.get("t_j2k_inc", 98.2),
            "c_sma": event_row.get("c_j2k_sma", 7000),
            "c_ecc": event_row.get("c_j2k_ecc", 0.001),
            "c_inc": event_row.get("c_j2k_inc", 98.2),
        }

        if normalize:
            features["miss_distance"] = np.clip(event_row["miss_distance"] / 50000.0, -1, 1)
            features["relative_speed"] = np.clip(event_row["relative_speed"] / 10000.0, -1, 1)
            features["time_to_tca"] = np.clip(event_row["time_to_tca"] / 24.0, -1, 1)

        return features

    def create_scenario_from_event(self, event_row: pd.Series, duration_hours: float = 1.0) -> Dict:
        """Create a simulation scenario dictionary from one conjunction event."""
        features = self.extract_features_for_simulation(event_row)

        scenario = {
            "name": f"Event_{event_row['event_id']}_Risk_{event_row['risk']:.2f}",
            "duration_hours": duration_hours,
            "risk_level": "HIGH"
            if event_row["risk"] > -7.0
            else "MEDIUM"
            if event_row["risk"] > -8.0
            else "LOW",
            "target_features": {
                "sma": features["t_sma"],
                "ecc": features["t_ecc"],
                "inc": features["t_inc"],
            },
            "chaser_features": {
                "sma": features["c_sma"],
                "ecc": features["c_ecc"],
                "inc": features["c_inc"],
            },
            "conjunction_info": {
                "miss_distance": event_row["miss_distance"],
                "relative_speed": event_row["relative_speed"],
                "time_to_tca": event_row["time_to_tca"],
                "risk_score": event_row["risk"],
            },
            "raw_features": features,
        }

        return scenario

    def get_batch_scenarios(self, events_df: pd.DataFrame, max_scenarios: int = 10) -> List[Dict]:
        """Create a bounded batch of simulation scenarios from a DataFrame of events."""
        scenarios = []
        for idx, (_, event_row) in enumerate(events_df.iterrows()):
            if idx >= max_scenarios:
                break

            scenario = self.create_scenario_from_event(event_row)
            scenarios.append(scenario)

        if self.verbose:
            print(f"[INFO] Created {len(scenarios)} scenarios")

        return scenarios

    def get_statistics_report(self) -> str:
        """Generate a formatted statistics report."""
        if self.data is None:
            return "No data loaded"

        stats = self.get_summary_stats()

        return f"""
============================================================
            ESA CDM CSV Dataset Statistics Report
============================================================

Total Events:        {stats['total_events']:,}
Columns:             {stats['columns']}
Memory Usage:        {stats['memory_mb']:.2f} MB

Time to TCA Range:   {stats['date_range']}
Risk Score Range:    {stats['risk_range']}
Miss Distance:       {stats['miss_distance_range']}
Relative Speed:      {stats['relative_speed_range']}

Key Columns:
  - event_id, time_to_tca, risk, miss_distance
  - relative_speed, relative_position_*
  - relative_velocity_*
  - t_j2k_sma, t_j2k_ecc, t_j2k_inc (target orbital elements)
  - c_j2k_sma, c_j2k_ecc, c_j2k_inc (chaser orbital elements)
  - F10, F3M, SSN, AP (solar activity)

Dataset Location:    {self.csv_path}
"""

    def print_statistics(self):
        """Print the formatted statistics report to stdout."""
        print(self.get_statistics_report())


if __name__ == "__main__":
    csv_path = Path(__file__).resolve().parent.parent / "data" / "test_data.csv"

    if not csv_path.exists():
        raise SystemExit(f"Dataset not found at {csv_path}. Place a CSV in data/ to run this example.")

    loader = CSVDataLoader(str(csv_path), verbose=True)

    print("\n" + "=" * 70)
    print("Loading Dataset...")
    print("=" * 70)
    data = loader.load(max_rows=1000)

    print("\n" + "=" * 70)
    print("Dataset Statistics")
    print("=" * 70)
    loader.print_statistics()

    print("\n" + "=" * 70)
    print("Extracting High-Risk Events")
    print("=" * 70)
    high_risk = loader.extract_high_risk_events(risk_threshold=-7.0, count=10)
    print("\nTop 5 High-Risk Events:")
    print(
        high_risk[["event_id", "risk", "miss_distance", "relative_speed"]]
        .head(5)
        .to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("Filtering Events")
    print("=" * 70)
    close_approaches = loader.filter_by_miss_distance(min_dist=0, max_dist=10000)
    print(f"Close approaches (<10km): {len(close_approaches)}")

    print("\n" + "=" * 70)
    print("Creating Simulation Scenarios")
    print("=" * 70)
    scenarios = loader.get_batch_scenarios(high_risk.head(3), max_scenarios=3)

    for i, scenario in enumerate(scenarios):
        print(f"\nScenario {i + 1}: {scenario['name']}")
        print(f"  Risk Level: {scenario['risk_level']}")
        print(f"  Duration: {scenario['duration_hours']} hours")
        print(f"  Miss Distance: {scenario['conjunction_info']['miss_distance']:.0f} m")
        print(f"  Relative Speed: {scenario['conjunction_info']['relative_speed']:.0f} m/s")
