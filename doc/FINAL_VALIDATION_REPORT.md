# Validation Report - April 3, 2026

## Summary

The repository now supports the upgraded fuel-constrained MARL workflow:
- risk-aware observations with miss distance, TCA, and risk score
- pre-propagation maneuver execution
- dense reward shaping
- an emergency maneuver option
- a threat-aware CBF safety filter
- dataset-backed curriculum training from local `data/` CSVs
- named test cases `TC1` through `TC8`
- a local Streamlit UI for launching scenarios and inspecting results

This remains a research simulator, not an operational flight-dynamics stack.

## Findings Addressed

1. Centralized critic mismatch
   Fixed in `marl/marl_trainer.py` so the critic consumes concatenated multi-agent observations.

2. Invalid PPO bookkeeping
   Fixed by storing the log-probabilities of the actions actually executed and computing GAE-style returns.

3. Per-agent MARL inference
   Fixed by adding joint MARL action selection in `policies/policy_interface.py` and using it from `sim/simulator.py`.

4. Late action execution
   Fixed in `env/ma_env.py` by applying maneuvers before the next propagation step.

5. Sparse collision-only reward
   Fixed by adding dense risk, proximity, TCA, and risk-reduction shaping in `env/ma_env.py`.

6. Weak hard-case exposure during training
   Fixed in `sim/dataset_integration.py` with `easy`, `medium`, `hard`, and `tc8_cluster` curriculum stages.

7. No emergency action
   Fixed in `sim/maneuver_engine.py` with a 7th discrete emergency burn option.

8. Underpowered safety layer
   Fixed in `safety/cbf_filter.py` and `sim/simulator.py` so the barrier filter runs on true relative geometry and TCA-aware active threats.

9. Reactive geometric filtering
   Fixed by adding exponential predictive probability of collision (Pc) mappings, expanding the observation to 70 dimensions, and rewarding based strictly on physics delta logic avoiding threshold-tripping.

## Checks Run

The following checks were executed in the repository virtual environment:

```powershell
.venv\Scripts\python.exe -m compileall main.py env marl policies safety sim experiments advanced_example.py
```

```powershell
.venv\Scripts\python.exe train.py --max-episodes 8000 --max-steps 120 --update-every 5 --eval-interval 50 --eval-episodes 5 --tc8-ratio 0.25 --use-dataset true --dataset-path data\train_data.jsonl --dataset-eval-path data\test_data.jsonl --save-dir policies\saved_models
```

```powershell
.venv\Scripts\python.exe experiments/run_collision_avoidance_tests.py --test-cases TC8_hypothetical_collision_cluster --mc-runs 3 --max-debris 200 --include-marl --marl-model-path policies\saved_models\mppo_final.pt --output-dir outputs\tc8_validation
```

```powershell
.venv\Scripts\python.exe experiments/run_collision_avoidance_tests.py --mc-runs 1 --max-debris 200 --include-marl --marl-model-path policies\saved_models\mppo_final.pt --output-dir outputs\test_framework_full_validation
```

## Dataset-Backed Training And Validation

Local datasets used:
- `data/train_data.csv`: 162,634 rows loaded
- `data/test_data.csv`: 24,484 rows loaded

Saved artifacts:
- `outputs/marl_train_validation/marl_trained_from_train_dataset.pth`
- `outputs/marl_train_validation/train_metrics.csv`
- `outputs/marl_train_validation/validation_episode_metrics.csv`
- `outputs/marl_train_validation/validation_policy_summary.csv`
- `outputs/marl_train_validation/train_validation_report.json`

Training summary from the latest full-dataset run:
- scenarios: 8
- curriculum scenarios: 32
- episodes per scenario: 3
- mean collisions: 0.0
- mean fuel used: 11.8781 kg
- mean steps: 60.0

Validation summary from `outputs/marl_train_validation/validation_policy_summary.csv`:
- `no_op`: `mean_collisions=0.0`, `mean_fuel=0.0`, `mean_maneuvers=0.0`
- `fuel_aware_threshold_rule`: `mean_collisions=0.0`, `mean_fuel=0.10`, `mean_maneuvers=1.0`
- `rule_based`: `mean_collisions=0.0`, `mean_fuel=12.00`, `mean_maneuvers=120.0`
- `marl`: `mean_collisions=0.0`, `mean_fuel=12.4875`, `mean_maneuvers=166.375`

## Test Framework Results

All named test cases completed successfully in `outputs/test_framework_full_validation/`.

Selected MARL rows from the latest suite:
- `TC4_marl`: `mean_collisions=970.0`, `mean_fuel=0.10`
- `TC5_high_density_stress`: `mean_collisions=3940.0`, `mean_fuel=3.25`
- `TC6_fuel_constrained`: `mean_collisions=970.0`, `mean_fuel=0.60`
- `TC7_secondary_conjunctions`: `mean_collisions=355.0`, `mean_fuel=0.20`
- `TC8_hypothetical_collision_cluster`: `mean_collisions=1.0`, `mean_fuel=1.50`

`TC8` summary from `outputs/tc8_validation/aggregated_summary.csv`:
- `no_op`: `mean_collisions=1.0`, `mean_fuel=0.0`, `mean_maneuvers=0.0`
- `baseline`: `mean_collisions=1.0`, `mean_fuel=1.2`, `mean_maneuvers=12.0`
- `rule_based`: `mean_collisions=1.0`, `mean_fuel=1.05`, `mean_maneuvers=10.67`
- `marl`: `mean_collisions=1.0`, `mean_fuel=1.50`, `mean_maneuvers=25.0`

## Interpretation

The implementation is now architecturally consistent end-to-end:
- the upgraded observation, action, reward, safety, and training paths all run together
- the full dataset train/validate workflow is reproducible
- all named experiment scenarios complete successfully with the current code

The current learned policy is still not competitive on efficiency, and the hardest synthetic case remains unsolved. That is now a policy-quality limitation, not a missing-wiring problem.

## Remaining Limitations

- The simulator is still a lightweight research environment rather than a mission-operations-grade orbital dynamics stack.
- The test framework’s high-density cases remain extremely collision-heavy under the present scenario definitions.
- The current MARL checkpoint still over-maneuvers and does not yet beat the simpler deterministic rules on fuel use or the hardest-case collision metric.
