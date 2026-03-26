# Project Overview - FuelSafe-MARL-LEO

FuelSafe is a simulation and evaluation framework for fuel-constrained orbital collision avoidance in LEO.

## Core capabilities

1. Orbit propagation with SGP4 (`sim/orbit_propagator.py`)
2. Pairwise conjunction detection and collision labeling (`sim/conjunction_detector.py`)
3. Maneuver execution with fuel accounting (`sim/maneuver_engine.py`)
4. Multi-agent environment for policy rollout (`env/ma_env.py`)
5. Safety filtering via CBF (`safety/cbf_filter.py`)
6. Policy comparison (`policies/policy_interface.py`, `sim/simulator.py`)
7. Reproducible test-case framework (`experiments/run_collision_avoidance_tests.py`)

## Policy families

- `no_op`: never maneuvers (worst-case reference)
- `baseline`: simple heuristic
- `rule_based`: existing TCA-based deterministic rule
- `threshold_rule`: distance-threshold deterministic rule
- `fuel_aware_threshold_rule`: threshold rule with fuel gate
- `marl`: optional learned policy wrapper

## Evaluation metrics

- total collisions
- total fuel used
- maneuvers executed
- secondary conjunctions
- near misses
- minimum separation distance

## Reproducibility features

- deterministic scenario setup support (fixed epoch, stable object IDs)
- consistent scenario generation across policy runs
- Monte Carlo execution with seed control in test framework script

## Typical workflow

```bash
python main.py --demo
python experiments/run_collision_avoidance_tests.py --mc-runs 30 --max-debris 200
```

Outputs are saved under `outputs/`.
