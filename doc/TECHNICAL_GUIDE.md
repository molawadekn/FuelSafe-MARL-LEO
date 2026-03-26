# Technical Guide - FuelSafe-MARL-LEO

## Architecture (current codebase)

### Core runtime flow

1. `OrbitPropagator` generates and propagates object states (`sim/orbit_propagator.py`)
2. `MultiAgentOrbitalEnv.step()`:
   - propagates objects
   - applies satellite maneuvers
   - detects conjunctions/collisions
   - updates rewards and episode metrics
3. `SimulationRunner`:
   - gets actions from selected policy
   - optionally filters actions through CBF
   - logs and aggregates run statistics

## Main components

- `sim/orbit_propagator.py`: SGP4 state propagation and sample TLE creation
- `sim/conjunction_detector.py`: pairwise distance, risk score, collision labeling
- `sim/maneuver_engine.py`: discrete action to ΔV, fuel consumption
- `env/ma_env.py`: environment state, step logic, rewards, episode metrics
- `safety/cbf_filter.py`: action safety projection
- `policies/policy_interface.py`: policy abstractions and implementations
- `sim/simulator.py`: policy execution loop, CBF integration, aggregation
- `experiments/run_collision_avoidance_tests.py`: reproducible test-case framework

## Policies

Implemented policy keys:
- `no_op`
- `baseline`
- `rule_based`
- `threshold_rule`
- `fuel_aware_threshold_rule`
- `marl` (optional wrapper, requires trainer/model injection)

## Metrics tracked in framework

- `total_collisions`
- `total_fuel_used`
- `total_maneuvers_executed`
- `total_secondary_conjunctions`
- `total_near_misses`
- `min_separation_distance_km`

## Reproducibility and fairness

- Fixed epoch support in scenario generation
- Stable object ID to satnum mapping for deterministic sample TLE generation
- Same scenario setup reused across policy variants per Monte Carlo index

## Running technical evaluations

```bash
# quick validation
python experiments/run_collision_avoidance_tests.py --quick --mc-runs 3 --max-debris 200

# full Monte Carlo with significance outputs
python experiments/run_collision_avoidance_tests.py --mc-runs 50 --max-debris 500
```

When `--mc-runs >= 2`, t-test outputs are generated automatically if SciPy is available.
