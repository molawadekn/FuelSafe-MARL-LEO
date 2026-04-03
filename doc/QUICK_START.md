# Quick Start - FuelSafe-MARL-LEO

## 1. Install

```powershell
cd "c:\Users\molaw\code\Final Year Project\FuelSafe-MARL-LEO"
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. Run a Demo

```powershell
.venv\Scripts\python.exe main.py --demo --include-marl --episodes 1 --steps 50
```

This runs baseline, rule-based, and MARL policy comparisons and writes logs under `outputs/`.

## 3. Launch the Local UI

```powershell
.venv\Scripts\streamlit.exe run ui\streamlit_app.py
```

The UI lets you:
- run demos and experiments
- train and validate MARL from local CSVs
- run any named test case from `TC1` to `TC8`
- inspect generated CSVs with interactive Plotly charts

## 4. Run the Reproducible Test Framework

```powershell
.venv\Scripts\python.exe experiments/run_collision_avoidance_tests.py --quick --mc-runs 3 --max-debris 200
```

Useful variants:

```powershell
# Single scenario family
.venv\Scripts\python.exe experiments/run_collision_avoidance_tests.py --test-cases TC1_no_maneuver --mc-runs 10 --max-debris 200

# Include MARL in the dedicated MARL and fuel-constrained cases
.venv\Scripts\python.exe experiments/run_collision_avoidance_tests.py --test-cases TC4_marl,TC6_fuel_constrained --mc-runs 3 --max-debris 200 --include-marl --marl-untrained

# Synthetic high-collision comparison case
.venv\Scripts\python.exe experiments/run_collision_avoidance_tests.py --test-cases TC8_hypothetical_collision_cluster --mc-runs 1 --max-debris 200 --include-marl --marl-model-path outputs\marl_train_validation\marl_trained_from_train_dataset.pth
```

## 5. Train And Validate MARL From ESA CDM Data

```powershell
.venv\Scripts\python.exe sim/dataset_integration.py --train-csv data\train_data.csv --test-csv data\test_data.csv --output-dir outputs\marl_train_validation --risk-threshold -7.0 --train-scenarios 12 --test-scenarios 8 --episodes-per-scenario 4 --max-steps 120 --num-satellites 3 --num-debris 10
```

Dataset-derived scenarios now parameterize the actual first satellite-debris encounter used during training and validation.

Key outputs:
- `outputs/marl_train_validation/marl_trained_from_train_dataset.pth`
- `outputs/marl_train_validation/train_metrics.csv`
- `outputs/marl_train_validation/validation_policy_summary.csv`
- `outputs/marl_train_validation/train_validation_report.json`
- `outputs/marl_train_validation/interactive_validation_summary_mean_fuel.html`

## 6. Outputs

Common files written by experiments:
- `test_runs_per_policy.csv`
- `aggregated_summary.csv`
- `plot_mean_collisions.png`
- `plot_mean_fuel.png`
- `plot_mean_maneuvers.png`
- `plot_mean_secondary_conjunctions.png`
- `interactive_summary_mean_collisions.html`
- `interactive_runs_total_fuel_used.html`

## 7. Notes

- Distances are in `km`.
- Velocities are in `km/s`.
- Fuel use is penalized using actual fuel burned per step, not just a flat action penalty.
- Maneuver effects persist across steps through state offsets relative to the SGP4 reference orbit.
