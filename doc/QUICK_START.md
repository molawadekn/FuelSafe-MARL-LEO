# Quick Start - FuelSafe-MARL-LEO

## 1) Setup

```bash
cd "c:\Users\molaw\code\Final Year Project\FuelSafe-MARL-LEO"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Run the demo

```bash
python main.py --demo
```

This runs a small collision-avoidance comparison and writes logs in `outputs/`.

## 3) Run the reproducible test-case framework

```bash
python experiments/run_collision_avoidance_tests.py --quick --mc-runs 3 --max-debris 200
```

### Useful options

```bash
# single test case
python experiments/run_collision_avoidance_tests.py --test-cases TC1_no_maneuver --mc-runs 30 --max-debris 200

# include MARL policy (requires model or --marl-untrained)
python experiments/run_collision_avoidance_tests.py --include-marl --marl-untrained --quick
```

## 4) Understand outputs

Each run folder under `outputs/test_framework*` contains:
- `test_runs_per_policy.csv`: one row per Monte Carlo run and policy
- `aggregated_summary.csv`: mean/std summary by test case and policy
- `plot_mean_collisions.png`
- `plot_mean_fuel.png`
- `plot_mean_maneuvers.png`
- `plot_mean_secondary_conjunctions.png`
- optional: `ttest_collisions.csv` and `pareto_frontier_fuel_vs_collisions.csv`

## 5) Core policies currently available

- `no_op` (worst-case, no maneuver)
- `baseline` (heuristic)
- `rule_based` (existing TCA-based logic)
- `threshold_rule` (if distance < threshold => burn)
- `fuel_aware_threshold_rule`
- `marl` (optional, when trainer/model is provided)

## 6) Important metric notes

- Distances are in `km`.
- Collision events are counted when `distance < collision_threshold_km`.
- For fair comparison, use same seed/scenario settings across policies.
