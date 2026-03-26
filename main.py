#!/usr/bin/env python3
"""
MODULE 12: Main Entry Point
Fuel-Constrained Multi-Agent RL Simulator for Orbital Collision Avoidance in LEO

This script demonstrates the complete system with:
- SGP4 orbit propagation
- ESA CDM ingestion and conjunction detection
- Multi-agent RL environment
- MAPPO training layer
- Control Barrier Function safety filter
- Plug-and-play policy comparison

Usage:
    python main.py --demo              # Run quick demo
    python main.py --experiment        # Run full experiment
    python main.py --train-marl        # Train MARL policy
    python main.py --compare           # Compare all policies
"""

import argparse
import sys
from pathlib import Path

# Add project modules to path
sys.path.insert(0, str(Path(__file__).parent))

from sim.simulator import SimulationRunner


def run_demo():
    """
    Core collision avoidance scenario: 3 satellites vs 5 debris, baseline and rule-based.
    """
    print("\n" + "="*70)
    print("Fuel-Constrained Multi-Agent Collision Avoidance Demo")
    print("="*70)

    results = {}
    for policy_type in ['baseline', 'rule_based']:
        print(f"\n{'─'*70}")
        print(f"Policy: {policy_type.upper()}")
        print(f"{'─'*70}\n")

        runner = SimulationRunner(
            num_satellites=3,
            num_debris=5,
            use_safety_filter=True,
            safety_threshold_km=0.5,
            distance_threshold_km=250.0,
            collision_threshold_km=10.0,
            high_risk_mode=True,
            policy_type=policy_type,
            enable_logging=True
        )

        stats_list = runner.run_multiple_episodes(num_episodes=2, max_steps=500, verbose=True)
        aggregated = runner._aggregate_stats(stats_list)
        results[policy_type] = aggregated

        # Save per-policy log
        if runner.logger:
            runner.logger.save_to_csv(f'{policy_type}_simulation_log.csv')

        print(f"\nResults for {policy_type}:")
        print(f"  Mean Collisions: {aggregated['mean_collisions']:.2f}")
        print(f"  Mean Fuel Used: {aggregated['mean_fuel']:.2f} kg")
        print(f"  Success Rate: {aggregated['success_rate']*100:.1f}%")

    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    for policy, stats in results.items():
        print(f"{policy:15} - Collisions: {stats['mean_collisions']:6.2f}, Fuel: {stats['mean_fuel']:8.2f} kg")
    print()


def main():
    """Main entry point for the core use case."""
    run_demo()


if __name__ == '__main__':
    main()

