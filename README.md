# FuelSafe-MARL-LEO

Fuel-constrained multi-agent collision-avoidance simulator for LEO.

The project includes:
- SGP4-based orbit propagation
- conjunction detection and collision counting
- maneuver execution with fuel tracking
- CBF safety filtering
- pluggable policies (`no_op`, `baseline`, `rule_based`, `threshold_rule`, `fuel_aware_threshold_rule`, optional `marl`)
- reproducible test-case framework for policy comparison

## Quick Start

```bash
pip install -r requirements.txt
python main.py --demo
```

## Test-Case Framework (new)

Run reproducible evaluation and generate CSV + plots:

```bash
python experiments/run_collision_avoidance_tests.py --quick --mc-runs 3 --max-debris 200
```

Outputs are written under `outputs/test_framework*/`, including:
- `test_runs_per_policy.csv`
- `aggregated_summary.csv`
- `plot_mean_collisions.png`, `plot_mean_fuel.png`, `plot_mean_maneuvers.png`, `plot_mean_secondary_conjunctions.png`
- optional bonus outputs: `ttest_collisions.csv` (when `--mc-runs >= 2`) and `pareto_frontier_fuel_vs_collisions.csv`

Run a single test case:

```bash
python experiments/run_collision_avoidance_tests.py --test-cases TC1_no_maneuver --mc-runs 30 --max-debris 200
```

## Repository Layout

- `sim/`: orbit propagation, conjunction detection, maneuver engine, simulation runner
- `env/`: multi-agent orbital environment
- `policies/`: policy interface and rule implementations
- `safety/`: control barrier filter
- `marl/`: MAPPO trainer
- `experiments/`: experiment utilities and the test-case framework
- `doc/`: project documentation

## Notes

- Units are kilometers (`km`) for distances and `km/s` for velocities.
- Collision events are counted when pairwise separation falls below `collision_threshold_km`.
- For fair policy comparison, use fixed seeds and identical scenario settings.
