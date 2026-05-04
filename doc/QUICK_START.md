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

## 3. Launch the Local UI

```powershell
.venv\Scripts\streamlit.exe run ui\streamlit_app.py
```

The UI lets you:
- run demos and experiments
- train and validate MARL from local CSVs
- run any named test case from `TC1` to `TC8`
- inspect generated CSVs with interactive Plotly charts

## 4. Train And Validate MARL Pipeline

```powershell
.venv\Scripts\python.exe train.py --max-episodes 8000 --max-steps 120 --update-every 5 --eval-interval 50 --eval-episodes 5 --tc8-ratio 0.25 --use-dataset true --dataset-path data\train_data.jsonl --dataset-eval-path data\test_data.jsonl --save-dir policies\saved_models
```

This uses:
- curriculum stages with periodic hard TC8-like sampling
- gradient updates every `--update-every` episodes
- multi-objective evaluation every `--eval-interval` episodes (collisions, fuel, secondary conjunctions)

Key outputs:
- `policies/saved_models/mppo_final.pt`
- intermediate checkpoints (e.g. `mppo_checkpoint_50.pt`)

## 5. Run The Reproducible Test Framework

```powershell
.venv\Scripts\python.exe experiments/run_collision_avoidance_tests.py --quick --mc-runs 3 --max-debris 200
```

Useful variants:

```powershell
# Include MARL in the dedicated MARL and fuel-constrained cases
.venv\Scripts\python.exe experiments/run_collision_avoidance_tests.py --test-cases TC4_marl,TC6_fuel_constrained --mc-runs 3 --max-debris 200 --include-marl --marl-model-path policies\saved_models\mppo_final.pt

# Synthetic high-collision comparison case
.venv\Scripts\python.exe experiments/run_collision_avoidance_tests.py --test-cases TC8_hypothetical_collision_cluster --mc-runs 3 --max-debris 200 --include-marl --marl-model-path policies\saved_models\mppo_final.pt --output-dir outputs\tc8_validation

# Full named suite
.venv\Scripts\python.exe experiments/run_collision_avoidance_tests.py --mc-runs 1 --max-debris 200 --include-marl --marl-model-path policies\saved_models\mppo_final.pt --output-dir outputs\test_framework_full_validation
```

## 6. Notes

- Observation size is `70`, heavily optimized mapping predictive geometry variables across a normalized array.
- The discrete action space now has `7` actions and combines direction pushes using a hybrid continuous parameter generation limit.
- Maneuvers are applied before propagation inside each step.
- The reward mixes Probability of Collision (Pc) computing continuous geometric risk deltas + exponential TCA scaling.
