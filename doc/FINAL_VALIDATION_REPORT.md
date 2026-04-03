# Validation Report - April 3, 2026

## Summary

The repository now supports a complete fuel-constrained MARL workflow:
- joint MARL action selection
- centralized-critic training with stored executed-action log-probs
- persistent maneuver effects
- reward penalties based on actual fuel consumed
- dataset-driven scenario seeding
- MARL evaluation in `TC6_fuel_constrained`
- dataset-backed MARL train/validate runs from local `data/` CSVs
- a local Streamlit UI for launching scenarios and inspecting charts
- a synthetic `TC8_hypothetical_collision_cluster` comparison case

This remains a research simulator, not an operational flight-dynamics stack.

## Findings Addressed

1. Centralized critic mismatch
   Fixed in `marl/marl_trainer.py` so the critic now consumes concatenated multi-agent observations.

2. Invalid PPO bookkeeping
   Fixed by storing the log-probabilities of the actions actually executed and computing GAE-style returns.

3. Per-agent MARL inference
   Fixed by adding joint MARL action selection in `policies/policy_interface.py` and using it from `sim/simulator.py`.

4. No MARL in `TC6_fuel_constrained`
   Fixed in `experiments/run_collision_avoidance_tests.py`.

5. Dataset integration only labeling scenarios
   Fixed by injecting dataset-derived orbital elements and relative offsets into the actual simulated conjunction pair.

6. Fuel constraint only partially enforced
   Fixed by using actual fuel burn in rewards and terminating when all controlled satellites are out of fuel.

## Additional Runtime Corrections

- Maneuver effects were previously lost after one step because the environment snapped objects back to the reference orbit. This is now fixed with persistent position and velocity offsets.
- The dataset train/validate pipeline previously selected only the top `N` risk rows, which over-sampled repeated catastrophic events. It now uses risk-stratified, unique-event selection.
- The dataset train/validate pipeline now uses a tighter `0.5 km` collision threshold than the coarse `1.0 km` stress-test setup so sub-km CDM miss distances do not all terminate immediately as collisions.
- Demo and experiment logging was cleaned up so stale root-level output CSVs are no longer mixed across unrelated runs.
- Interactive Plotly report generation was added for both the test framework and the dataset train/validate workflow.

## Checks Run

The following checks were executed in the repository virtual environment:

```powershell
.venv\Scripts\python.exe -m compileall main.py env marl policies sim experiments advanced_example.py
```

```powershell
.venv\Scripts\python.exe experiments/run_collision_avoidance_tests.py --test-cases TC4_marl,TC6_fuel_constrained --mc-runs 1 --max-debris 20 --include-marl --marl-untrained --output-dir outputs/tc4_tc6_check
```

```powershell
.venv\Scripts\python.exe sim/dataset_integration.py --train-csv data\train_data.csv --test-csv data\test_data.csv --output-dir outputs\marl_train_validation --train-max-rows 162634 --test-max-rows 24484 --risk-threshold -7.0 --train-scenarios 12 --test-scenarios 8 --episodes-per-scenario 4 --max-steps 120 --num-satellites 3 --num-debris 10 --marl-epochs-per-batch 3
```

## Dataset-Backed Training And Validation

Local datasets used:
- `data/train_data.csv`: 162,634 rows
- `data/test_data.csv`: 24,484 rows

Saved artifacts:
- `outputs/marl_train_validation/marl_trained_from_train_dataset.pth`
- `outputs/marl_train_validation/train_metrics.csv`
- `outputs/marl_train_validation/validation_episode_metrics.csv`
- `outputs/marl_train_validation/validation_policy_summary.csv`
- `outputs/marl_train_validation/train_validation_report.json`

Training summary:
- scenarios: 12
- episodes per scenario: 4
- mean collisions: 0.0
- mean fuel used: 7.1375 kg
- mean steps: 60.0

Validation summary from `validation_policy_summary.csv`:
- `no_op`: `mean_collisions=0.0`, `mean_fuel=0.0`, `mean_maneuvers=0.0`
- `fuel_aware_threshold_rule`: `mean_collisions=0.0`, `mean_fuel=0.05`, `mean_maneuvers=1.0`
- `rule_based`: `mean_collisions=0.0`, `mean_fuel=0.90`, `mean_maneuvers=18.0`
- `marl`: `mean_collisions=0.0`, `mean_fuel=7.60`, `mean_maneuvers=152.0`

Synthetic comparison summary from `outputs/test_framework_full_validation` for `TC8_hypothetical_collision_cluster`:
- `no_op`: `mean_collisions=1.0`, `mean_fuel=0.0`, `mean_maneuvers=0.0`
- `baseline`: `mean_collisions=0.0`, `mean_fuel=0.25`, `mean_maneuvers=5.0`
- `rule_based`: `mean_collisions=0.0`, `mean_fuel=0.35`, `mean_maneuvers=7.0`
- `marl`: `mean_collisions=0.0`, `mean_fuel=1.50`, `mean_maneuvers=30.0`

## Interpretation

The implementation is now genuinely trainable and validated from local train/test datasets, but the currently trained MARL policy is not yet competitive with the simpler deterministic rules on fuel efficiency. That is a modeling and training-quality result, not a plumbing failure.

In other words:
- the fuel-constrained MARL path is now wired correctly end-to-end
- the dataset-driven validation path is now real and reproducible
- the present learned policy still needs reward tuning, curriculum changes, or longer training to outperform the hand-crafted baselines

## Remaining Limitations

- The simulator is still a lightweight research environment rather than a mission-operations-grade orbital dynamics stack.
- Dataset scenario injection still maps one historical event into a simplified first-pair encounter instead of reconstructing an entire historical CDM scene.
- The current MARL reward and training budget encourage too many maneuvers on the test split, so additional training work is still needed for competitive policy quality.
