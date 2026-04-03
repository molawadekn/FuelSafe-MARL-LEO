#!/usr/bin/env python3
"""
MODULE 12: Main Entry Point
Fuel-Constrained Multi-Agent RL Simulator for Orbital Collision Avoidance in LEO

This script demonstrates the complete system with:
- SGP4 orbit propagation
- ESA CDM ingestion and conjunction detection
- multi-agent RL environment
- MAPPO training layer
- Control Barrier Function safety filter
- plug-and-play policy comparison

Usage:
    python main.py --demo
    python main.py --experiment --dataset data/test_data.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from marl.marl_trainer import MARLTrainer
from sim.simulator import SimulationRunner


def run_demo(
    num_episodes: int = 2,
    max_steps: int = 500,
    num_satellites: int = 3,
    num_debris: int = 5,
    safety_threshold_km: float = 0.5,
    distance_threshold_km: float = 250.0,
    collision_threshold_km: float = 1.0,
    high_risk_mode: bool = True,
    baseline_risk_threshold: float = 0.5,
    rule_based_aggression: float = 0.6,
    include_marl: bool = False,
    marl_model_path: Optional[str] = None,
) -> None:
    """Run the quick baseline, rule-based, and optional MARL comparison."""
    print("\n" + "=" * 70)
    print("Fuel-Constrained Multi-Agent Collision Avoidance Demo")
    print("=" * 70)

    marl_trainer = None
    if include_marl:
        marl_trainer = MARLTrainer(num_agents=num_satellites)
        if marl_model_path:
            try:
                marl_trainer.load(marl_model_path)
                print(f"Loaded MARL weights from {marl_model_path}")
            except Exception as exc:
                print(f"Warning: failed to load MARL model from {marl_model_path}: {exc}")
                print("Proceeding with untrained MARL weights.")

    policy_kwargs = {
        "baseline_risk_threshold": baseline_risk_threshold,
        "rule_based_aggression": rule_based_aggression,
    }

    runner = SimulationRunner(
        num_satellites=num_satellites,
        num_debris=num_debris,
        use_safety_filter=True,
        safety_threshold_km=safety_threshold_km,
        distance_threshold_km=distance_threshold_km,
        collision_threshold_km=collision_threshold_km,
        high_risk_mode=high_risk_mode,
        policy_type="baseline",
        marl_trainer=marl_trainer,
        policy_kwargs=policy_kwargs,
        enable_logging=True,
    )

    comparison_policies = ["baseline", "rule_based"]
    if include_marl:
        comparison_policies.append("marl")

    results = runner.compare_policies(
        comparison_policies,
        num_episodes=num_episodes,
        max_steps=max_steps,
    )

    print("\n" + "=" * 70)
    print("FINAL POLICY COMPARISON")
    print("=" * 70)

    for policy_name, stats in results.items():
        print(
            f"{policy_name:12} | collisions {stats['mean_collisions']:.3f} "
            f"+/- {stats['std_collisions']:.3f} | fuel {stats['mean_fuel']:.3f} "
            f"+/- {stats['std_fuel']:.3f} | success0 {stats['success_rate'] * 100:.1f}% "
            f"| success<=1 {stats.get('success_rate_<=1_collision', 0) * 100:.1f}%"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="FuelSafe-MARL-LEO Demo")
    parser.add_argument("--demo", action="store_true", help="Run demo and policy comparison")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes per policy")
    parser.add_argument("--steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--sats", type=int, default=3, help="Number of satellites")
    parser.add_argument("--debris", type=int, default=5, help="Number of debris objects")
    parser.add_argument(
        "--safety-threshold",
        type=float,
        default=0.5,
        help="Minimum safe separation distance used by the CBF filter",
    )
    parser.add_argument("--baseline-risk", type=float, default=0.5, help="Baseline risk threshold")
    parser.add_argument("--rule-aggression", type=float, default=0.6, help="Rule-based aggression level")
    parser.add_argument(
        "--include-marl",
        action="store_true",
        help="Include MARL policy (untrained unless model is provided).",
    )
    parser.add_argument(
        "--marl-model-path",
        type=str,
        default=None,
        help="Optional path to load MARL policy weights.",
    )

    parser.add_argument("--experiment", action="store_true", help="Run experiment on scenarios from dataset")
    parser.add_argument("--dataset", type=str, default=None, help="Path to ESA CDM CSV for scenario-driven experiments")
    parser.add_argument(
        "--risk-threshold",
        type=float,
        default=-7.0,
        help="Risk threshold for scenario selection (experiment mode)",
    )
    parser.add_argument("--max-scenarios", type=int, default=10, help="Max scenarios to run in experiment mode")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.demo:
        run_demo(
            num_episodes=args.episodes,
            max_steps=args.steps,
            num_satellites=args.sats,
            num_debris=args.debris,
            safety_threshold_km=args.safety_threshold,
            baseline_risk_threshold=args.baseline_risk,
            rule_based_aggression=args.rule_aggression,
            include_marl=args.include_marl,
            marl_model_path=args.marl_model_path,
        )
        return

    if args.experiment:
        if not args.dataset:
            print("Error: --dataset must be specified in experiment mode.")
            return

        from sim.dataset_integration import DatasetIntegration

        print("\n" + "=" * 70)
        print("Fuel-Constrained Multi-Agent Collision Avoidance Experiment")
        print("=" * 70)

        integration = DatasetIntegration(args.dataset, verbose=True)
        integration.load_dataset()
        scenarios = integration.create_scenarios_from_dataset(
            risk_threshold=args.risk_threshold,
            max_scenarios=args.max_scenarios,
        )

        policy_types = ["baseline", "rule_based"]
        marl_trainer = None
        if args.include_marl:
            policy_types.append("marl")
            marl_trainer = MARLTrainer(num_agents=args.sats)
            if args.marl_model_path:
                try:
                    marl_trainer.load(args.marl_model_path)
                    print(f"Loaded MARL weights from {args.marl_model_path}")
                except Exception as exc:
                    print(f"Warning: failed to load MARL model from {args.marl_model_path}: {exc}")
                    print("Proceeding with untrained MARL weights.")

        for policy in policy_types:
            print(f"\n{'=' * 60}\nTesting policy: {policy}\n{'=' * 60}")
            for i, scenario in enumerate(scenarios):
                print(f"\nScenario {i + 1}/{len(scenarios)}: {scenario['name']} (Risk: {scenario['risk_level']})")
                runner = SimulationRunner(
                    num_satellites=args.sats,
                    num_debris=args.debris,
                    use_safety_filter=True,
                    safety_threshold_km=args.safety_threshold,
                    distance_threshold_km=250.0,
                    collision_threshold_km=10.0,
                    high_risk_mode=True,
                    policy_type=policy,
                    marl_trainer=marl_trainer if policy == "marl" else None,
                    policy_kwargs={
                        "baseline_risk_threshold": args.baseline_risk,
                        "rule_based_aggression": args.rule_aggression,
                    },
                    enable_logging=False,
                )
                runner.run_scenario(scenario, max_steps=args.steps)

        print("\nExperiment complete.")
        return

    print(
        "Run with --demo or --experiment. "
        "--demo compares baseline and rule_based policies, with optional MARL. "
        "--experiment runs scenario-driven experiments from a dataset."
    )


if __name__ == "__main__":
    main()
