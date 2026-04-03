# FuelSafe-MARL-LEO

Fuel-constrained multi-agent collision-avoidance simulator for Low Earth Orbit.

The repository combines:
- SGP4-based reference orbit propagation
- persistent maneuver effects via state offsets on top of the reference orbit
- conjunction detection and collision counting
- discrete burn execution with fuel accounting
- a CBF-based safety filter
- pluggable policies (`no_op`, `baseline`, `rule_based`, `threshold_rule`, `fuel_aware_threshold_rule`, `marl`)
- a MAPPO-style trainer with centralized critic and joint MARL action selection
- reproducible test cases, including fuel-constrained MARL evaluation

## Current Status

The codebase now supports the full loop needed for your project:
- fuel is part of the observation and the reward uses actual fuel consumed
- maneuver effects persist across steps instead of disappearing after one tick
- MARL actions are selected jointly across agents
- the critic trains on centralized observations, with stored action log-probs and GAE-style returns
- dataset-derived scenarios now influence the actual encounter geometry
- `TC6_fuel_constrained` can evaluate `marl`

This is still a lightweight research simulator, not a flight-dynamics-grade operations tool. The runtime uses SGP4 as the reference trajectory and applies maneuver effects as persistent state offsets, which is a practical approximation for comparative experiments.

## Setup

```powershell
cd "c:\Users\molaw\code\Final Year Project\FuelSafe-MARL-LEO"
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Quick Commands

Run a small demo:

```powershell
.venv\Scripts\python.exe main.py --demo --include-marl --episodes 1 --steps 50
```

Launch the local simulator UI:

```powershell
.venv\Scripts\streamlit.exe run ui\streamlit_app.py
```

Run the reproducible test framework:

```powershell
.venv\Scripts\python.exe experiments/run_collision_avoidance_tests.py --quick --mc-runs 3 --max-debris 200
```

Run focused MARL and fuel-constrained checks:

```powershell
.venv\Scripts\python.exe experiments/run_collision_avoidance_tests.py --test-cases TC4_marl,TC6_fuel_constrained --mc-runs 1 --max-debris 50 --include-marl --marl-untrained --output-dir outputs/tc4_tc6_check
```

Run the synthetic high-collision comparison case:

```powershell
.venv\Scripts\python.exe experiments/run_collision_avoidance_tests.py --test-cases TC8_hypothetical_collision_cluster --mc-runs 1 --max-debris 200 --include-marl --marl-model-path outputs\marl_train_validation\marl_trained_from_train_dataset.pth --output-dir outputs\tc8_validation
```

Train and validate MARL on the local ESA CDM train/test datasets:

```powershell
.venv\Scripts\python.exe sim/dataset_integration.py --train-csv data\train_data.csv --test-csv data\test_data.csv --output-dir outputs\marl_train_validation --risk-threshold -7.0 --train-scenarios 12 --test-scenarios 8 --episodes-per-scenario 4 --max-steps 120 --num-satellites 3 --num-debris 10
```

That workflow writes:
- `outputs/marl_train_validation/marl_trained_from_train_dataset.pth`
- `outputs/marl_train_validation/train_metrics.csv`
- `outputs/marl_train_validation/validation_episode_metrics.csv`
- `outputs/marl_train_validation/validation_policy_summary.csv`
- `outputs/marl_train_validation/train_validation_report.json`

## Repository Layout

- `env/`: multi-agent orbital environment
- `experiments/`: reproducible policy-comparison scripts
- `marl/`: MAPPO-style trainer
- `policies/`: policy abstractions and implementations
- `safety/`: CBF safety filter
- `sim/`: propagation, conjunction detection, maneuvers, simulation orchestration, dataset integration
- `doc/`: project guides and validation notes

## Key Outputs

The experiment framework writes results under `outputs/`, including:
- scenario-test summaries such as `aggregated_summary.csv`
- plots such as `plot_mean_collisions.png`, `plot_mean_fuel.png`, and `plot_mean_secondary_conjunctions.png`
- interactive HTML charts such as `interactive_summary_mean_fuel.html` and `interactive_runs_total_collisions.html`
- optional per-policy demo logs such as `baseline_simulation_log.csv`, `rule_based_simulation_log.csv`, and `marl_simulation_log.csv`
- dataset train/validate artifacts under `outputs/marl_train_validation/`

## Documentation

See:
- `doc/QUICK_START.md`
- `doc/PROJECT_OVERVIEW.md`
- `doc/TECHNICAL_GUIDE.md`
- `doc/IMPLEMENTATION_GUIDE.md`
- `doc/FINAL_VALIDATION_REPORT.md`
