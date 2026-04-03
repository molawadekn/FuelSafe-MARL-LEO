# Project Overview - FuelSafe-MARL-LEO

FuelSafe is a research simulator for fuel-constrained orbital collision avoidance in LEO.

## Core Capabilities

1. SGP4 reference orbit generation and propagation in `sim/orbit_propagator.py`
2. Persistent maneuver execution with fuel accounting in `sim/maneuver_engine.py` and `env/ma_env.py`
3. Pairwise conjunction detection and collision labeling in `sim/conjunction_detector.py`
4. Multi-agent rollout environment in `env/ma_env.py`
5. Centralized-training / decentralized-execution MARL in `marl/marl_trainer.py`
6. Policy comparison and experiment orchestration in `sim/simulator.py`
7. Reproducible Monte Carlo evaluations in `experiments/run_collision_avoidance_tests.py`
8. ESA CDM-driven scenario seeding in `sim/csv_data_loader.py` and `sim/dataset_integration.py`
9. Dataset-backed MARL train/validate workflow using `data/train_data.csv` and `data/test_data.csv`

## Policy Families

- `no_op`: never maneuvers
- `baseline`: simple heuristic
- `rule_based`: deterministic threat-geometry rule
- `threshold_rule`: burn when a distance threshold is crossed
- `fuel_aware_threshold_rule`: threshold rule with a minimum remaining fuel gate
- `marl`: learned joint policy using the MAPPO-style trainer

## Evaluation Metrics

- total collisions
- total fuel used
- total maneuvers executed
- total secondary conjunctions
- total near misses
- minimum separation distance
- success rate with zero collisions
- success rate with at most one collision

## Important Modeling Choice

The simulator uses SGP4 as a reference trajectory and keeps maneuver effects alive through persistent position and velocity offsets. That keeps experiments fast and makes policy comparison meaningful after burns, while staying simpler than full high-fidelity post-burn orbit re-estimation.
