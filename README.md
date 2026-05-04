# FuelSafe-MARL-LEO

FuelSafe-MARL-LEO is a research simulator for predictive, fuel-constrained satellite collision avoidance in LEO and MEO-style conjunction scenarios.

The current implementation combines:
- SGP4 reference orbit propagation with persistent post-burn state offsets
- physics-based predictive collision probability `Pc = exp(-dist_m / 50) * exp(-tca_s / 300)`
- multi-object risk aggregation using `total_risk = sum(Pc_i)`
- risk-ranked top-k observations with distance, relative velocity, TCA, risk score, and `Pc`
- dense reward shaping for risk reduction, urgency, fuel use, maneuver stability, and safe `NO_OP`
- physical realism layer with SSN tracking noise ($\sigma = 20$ m), $\pm10\%$ thruster execution noise, tracking catalog dropouts, and action-delay buffering
- pluggable policies: `no_op`, `baseline`, `rule_based`, `threshold_rule`, `fuel_aware_threshold_rule`, and `marl`
- a MAPPO-style trainer with a centralized critic, PPO clipping, advantage normalization, entropy regularization, and gradient clipping
- deterministic `TC1` to `TC8` experiment cases aligned with the current predictive policy and `Pc` logic, with realistic 5-meter hard-body collision thresholds
- both JSONL-driven MARL training and CSV-driven dataset integration utilities

This is still a lightweight research platform, not an operational flight-dynamics system.

## Current Implementation

Key implementation details:
- observation size is `94`
- each observation contains 10 self features plus 7 threat blocks x 12 values each
- threats are ranked by `Pc`, then TCA, then current distance
- the reward uses aggregated `total_risk` delta, explicit collision/near-miss penalties, TCA urgency, maneuver penalties, action-switching penalties, and a stable safe `NO_OP` reward
- a Realism Layer handles observation noise, partial observability, and maneuver delays.
- collision threshold is physically calibrated to `5` meters
- training updates every `5` episodes by default
- hard-scenario injection defaults to `40%` via `--tc8-ratio 0.40`
- the experiment runner now uses structured logging and supports `--log-level`

## Setup

```powershell
cd "c:\Users\molaw\code\Final Year Project\FuelSafe-MARL-LEO"
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Quick Commands

Run a demo comparison:

```powershell
.venv\Scripts\python.exe main.py --demo --include-marl --episodes 1 --steps 50
```

Launch the Streamlit UI:

```powershell
.venv\Scripts\streamlit.exe run ui\streamlit_app.py
```

Run the predictive `TC1` to `TC8` experiment framework:

```powershell
.venv\Scripts\python.exe experiments/run_collision_avoidance_tests.py --quick --mc-runs 3 --max-debris 200 --realism true --log-level INFO
```

Run the aligned local test suite:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

Run the short-warning `TC8` case with MARL:

```powershell
.venv\Scripts\python.exe experiments/run_collision_avoidance_tests.py --test-cases TC8_hypothetical_collision_cluster --mc-runs 3 --max-debris 200 --include-marl --marl-model-path policies\saved_models\mppo_final.pt --realism true --output-dir outputs\tc8_validation --log-level INFO
```

Train the JSONL-based MARL pipeline:

```powershell
.venv\Scripts\python.exe train.py --max-episodes 8000 --max-steps 120 --update-every 5 --eval-interval 50 --eval-episodes 5 --tc8-ratio 0.40 --entropy-start 0.02 --entropy-end 0.005 --realism true --use-dataset true --dataset-path data\train_data.jsonl --dataset-eval-path data\test_data.jsonl --save-dir policies\saved_models
```

Generate the combined synthetic JSONL dataset:

```powershell
.venv\Scripts\python.exe data\generate_final_dataset.py --count 100000 --output data\final_dataset.jsonl
```

## Dataset Paths

There are two data workflows in the repository:

- `train.py` expects JSONL scenario files and defaults to `data/train_data.jsonl` and `data/test_data.jsonl`
- `sim/dataset_integration.py`, `main.py --experiment`, and parts of the Streamlit UI still use CSV-based ESA CDM inputs such as `data/train_data.csv` and `data/test_data.csv`

The synthetic generator writes `data/final_dataset.jsonl`. For quick local experimentation you can point both training and evaluation to that file, but for meaningful validation you should keep separate train and eval splits.

## Experiment Cases

The named experiment suite now covers:
- `TC1_no_maneuver`: no-op baseline in a predictive normal cluster
- `TC2_threshold_rule`: threshold and emergency escalation benchmark
- `TC3_fuel_aware_rule`: predictive sequential-threat fuel-management case
- `TC4_marl`: mixed-warning MARL comparison benchmark
- `TC5_high_density_stress`: large-fleet dense-traffic stress case
- `TC6_fuel_constrained`: short-warning fuel-limited hybrid case
- `TC7_secondary_conjunctions`: maneuver-linked secondary conjunction benchmark
- `TC8_hypothetical_collision_cluster`: TC8-style hard short-warning cluster with sub-100 m misses

## Latest Verified Local Checks

On April 22, 2026, the following checks were rerun against the current implementation:
- `python -m unittest tests.test_latest_implementation -v`
- a direct smoke using `build_test_cases()` plus `run_policy_on_scenario()` for the rewritten experiment runner

Performance numbers remain checkpoint-dependent and should be regenerated locally rather than copied from stale reports.

## Repository Layout

- `env/`: multi-agent orbital environment and reward logic
- `experiments/`: reproducible policy-comparison framework
- `marl/`: MAPPO-style trainer and curriculum helpers
- `policies/`: heuristic and MARL policy interfaces
- `safety/`: CBF safety filter
- `sim/`: propagation, conjunction detection, observation encoding, maneuvers, dataset integration, reporting
- `ui/`: Streamlit simulator UI
- `data/`: CSV inputs, JSONL datasets, and synthetic dataset generator
- `doc/`: project guides and validation notes

## Outputs

The experiment framework writes results under `outputs/`, including:
- `aggregated_summary.csv`
- `test_runs_per_policy.csv`
- `pareto_frontier_fuel_vs_collisions.csv`
- `plot_*.png`
- `interactive_summary_*.html`
- `interactive_runs_*.html`

## Documentation

See:
- `doc/FUELSAFE_SYSTEM_GUIDE.md`
- `doc/QUICK_START.md`
- `doc/PROJECT_OVERVIEW.md`
- `doc/TECHNICAL_GUIDE.md`
- `doc/IMPLEMENTATION_GUIDE.md`
- `doc/FINAL_VALIDATION_REPORT.md`
