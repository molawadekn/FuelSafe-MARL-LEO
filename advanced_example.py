#!/usr/bin/env python3
"""
ADVANCED EXAMPLE: Safety Filter Demonstration and Scalability Testing
Shows:
- Control Barrier Function safety filter in action
- Impact of safety filter on collision avoidance
- Performance scaling with number of agents
- Detailed metrics and logging
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from sim.conjunction_detector import ConjunctionDetector
from sim.orbit_propagator import OrbitPropagator
from sim.simulator import SimulationRunner


def demo_safety_filter_effectiveness():
    """
    Demonstrate effectiveness of CBF safety filter.
    Compare with and without safety filter.
    """
    print("\n" + "=" * 70)
    print("ADVANCED DEMO 1: Safety Filter Effectiveness")
    print("=" * 70)

    scenarios = [
        {"num_sats": 5, "num_debris": 20, "use_filter": False},
        {"num_sats": 5, "num_debris": 20, "use_filter": True},
    ]

    results = []

    for scenario in scenarios:
        print(
            f"\nTesting: {scenario['num_sats']} satellites, "
            f"{scenario['num_debris']} debris, "
            f"Safety filter: {scenario['use_filter']}"
        )

        runner = SimulationRunner(
            num_satellites=scenario["num_sats"],
            num_debris=scenario["num_debris"],
            use_safety_filter=scenario["use_filter"],
            policy_type="baseline",
            enable_logging=False,
        )

        stats_list = runner.run_multiple_episodes(
            num_episodes=3,
            max_steps=500,
            verbose=False,
            save_logs=False,
        )

        aggregated = runner._aggregate_stats(stats_list)
        aggregated["use_filter"] = scenario["use_filter"]
        results.append(aggregated)

        print(f"  Mean collisions: {aggregated['mean_collisions']:.2f}")
        print(f"  Mean fuel: {aggregated['mean_fuel']:.2f} kg")
        print(f"  Success rate: {aggregated['success_rate'] * 100:.1f}%")

    print("\n" + "-" * 70)
    print("SAFETY FILTER IMPACT:")
    without = results[0]
    with_filter = results[1]

    collision_reduction = (
        (without["mean_collisions"] - with_filter["mean_collisions"])
        / (without["mean_collisions"] + 1e-6)
        * 100
    )

    fuel_delta = with_filter["mean_fuel"] - without["mean_fuel"]
    if abs(without["mean_fuel"]) < 1e-6:
        fuel_delta_pct = float("nan")
    else:
        fuel_delta_pct = fuel_delta / without["mean_fuel"] * 100

    print(f"  Collision reduction: {collision_reduction:+.1f}%")
    print(f"  Fuel overhead: {fuel_delta:.2f} kg ({fuel_delta_pct:.1f}%)")
    print()


def demo_scalability():
    """
    Test scalability with increasing number of objects.
    """
    print("\n" + "=" * 70)
    print("ADVANCED DEMO 2: Scalability Analysis")
    print("=" * 70)

    configs = [
        {"num_sats": 3, "num_debris": 10},
        {"num_sats": 5, "num_debris": 50},
        {"num_sats": 10, "num_debris": 100},
    ]

    scalability_results = []

    for config in configs:
        print(f"\nConfig: {config['num_sats']} satellites, {config['num_debris']} debris")

        runner = SimulationRunner(
            num_satellites=config["num_sats"],
            num_debris=config["num_debris"],
            use_safety_filter=True,
            policy_type="rule_based",
            enable_logging=False,
        )

        stats_list = runner.run_multiple_episodes(
            num_episodes=2,
            max_steps=300,
            verbose=False,
            save_logs=False,
        )

        aggregated = runner._aggregate_stats(stats_list)

        result = {
            "num_sats": config["num_sats"],
            "num_debris": config["num_debris"],
            "total_objects": config["num_sats"] + config["num_debris"],
            **aggregated,
        }
        scalability_results.append(result)

        print(f"  Mean collisions: {aggregated['mean_collisions']:.2f}")
        print(f"  Mean fuel: {aggregated['mean_fuel']:.2f} kg")

    print("\n" + "-" * 70)
    print("SCALABILITY SUMMARY:")
    print("-" * 70)
    df = pd.DataFrame(scalability_results)
    print(
        df[
            [
                "num_sats",
                "num_debris",
                "total_objects",
                "mean_collisions",
                "mean_fuel",
                "success_rate",
            ]
        ].to_string(index=False)
    )
    print()


def demo_orbit_propagation():
    """
    Demonstrate SGP4 orbit propagation accuracy.
    """
    print("\n" + "=" * 70)
    print("ADVANCED DEMO 3: Orbit Propagation (SGP4)")
    print("=" * 70)

    propagator = OrbitPropagator()

    print("\nCreating sample satellites in LEO...")
    for i in range(3):
        propagator.generate_sample_tle(
            f"SAT_{i}",
            semi_major_axis_km=6800 + i * 100,
            inclination_deg=51.6 + i * 0.2,
        )

    start_time = datetime.now(timezone.utc)
    propagation_times = [start_time + timedelta(minutes=t) for t in range(0, 100, 10)]

    print(
        f"\nPropagating {len(propagator.satellites)} satellites "
        f"for {len(propagation_times)} timesteps..."
    )

    for obj_id in list(propagator.satellites.keys())[:1]:
        positions = []

        for current_time in propagation_times:
            state = propagator.propagate(obj_id, current_time)
            positions.append(state.position)

        positions = np.array(positions)

        distances_from_earth = np.linalg.norm(positions, axis=1)
        mean_distance = np.mean(distances_from_earth)
        altitude = mean_distance - 6371

        print(f"\n{obj_id} Statistics:")
        print(f"  Mean altitude: {altitude:.2f} km")
        print(f"  Altitude variation: {np.std(distances_from_earth):.3f} km")
        print(
            "  Trajectory length: "
            f"{np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)):.2f} km"
        )


def demo_conjunction_detection():
    """
    Demonstrate conjunction detection and risk scoring.
    """
    print("\n" + "=" * 70)
    print("ADVANCED DEMO 4: Conjunction Detection & Risk Scoring")
    print("=" * 70)

    detector = ConjunctionDetector(
        distance_threshold_km=20.0,
        collision_threshold_km=0.025,
    )

    propagator = OrbitPropagator()

    print("\nCreating objects with designed close approach...")
    propagator.generate_sample_tle("SAT_1", semi_major_axis_km=6800)
    propagator.generate_sample_tle("SAT_2", semi_major_axis_km=6805, inclination_deg=51.65)

    current_time = datetime.now(timezone.utc)
    detections = []

    print("\nDetecting conjunctions over 24 hours...")

    for hour in range(0, 24, 1):
        sample_time = current_time + timedelta(hours=hour)
        states = propagator.propagate_all(sample_time)
        object_states = {obj_id: state.to_array() for obj_id, state in states.items()}
        alerts = detector.detect(object_states, sample_time)

        if alerts:
            for alert in alerts:
                detections.append(
                    {
                        "hour": hour,
                        "distance_km": alert.distance_km,
                        "risk_score": alert.risk_score,
                        "time_to_tca": alert.time_to_closest_approach_s / 3600,
                    }
                )

                if alert.risk_score > 0.5:
                    print(f"  Hour {hour}: HIGH RISK conjunction detected!")
                    print(f"    Distance: {alert.distance_km:.3f} km")
                    print(f"    Risk score: {alert.risk_score:.3f}")
                    print(f"    Time to TCA: {alert.time_to_closest_approach_s / 60:.1f} min")

    if detections:
        df = pd.DataFrame(detections)
        print(f"\nDetected {len(detections)} conjunction events")
        print(f"Max risk score: {df['risk_score'].max():.3f}")
    else:
        print("\nNo conjunctions detected (objects too far apart)")


def main():
    """Run advanced demos."""
    print("\n" + "=" * 70)
    print("Running Advanced Example Demos")
    print("=" * 70)

    demos = [
        ("Safety Filter Effectiveness", demo_safety_filter_effectiveness),
        ("Scalability Analysis", demo_scalability),
        ("Orbit Propagation", demo_orbit_propagation),
        ("Conjunction Detection", demo_conjunction_detection),
    ]

    for i, (name, demo_func) in enumerate(demos, 1):
        print(f"\n--- {i}/{len(demos)} {name} ---")
        try:
            demo_func()
        except Exception as exc:
            print(f"\n[ERR] Error in {name}: {exc}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print("All demonstrations completed!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
