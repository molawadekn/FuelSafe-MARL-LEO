# Codex Instructions - FuelSafe-MARL-LEO

These instructions are for coding agents working in this repository.

## 1) Mission

Maintain and extend a research simulator for fuel-constrained multi-agent orbital collision avoidance in LEO, while preserving reproducibility of policy comparisons and test-case behavior.

## 2) Environment And Setup

- OS assumptions: Windows + PowerShell.
- Python env: local `.venv`.
- Install:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 3) High-Value Commands

- Quick demo:

```powershell
.venv\Scripts\python.exe main.py --demo --include-marl --episodes 1 --steps 50
```

- Repro test framework (runtime-capped):

```powershell
.venv\Scripts\python.exe experiments/run_collision_avoidance_tests.py --quick --mc-runs 3 --max-debris 200
```

- Unit tests (primary regression check):

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

- Train MARL (expensive; do not run unless requested):

```powershell
.venv\Scripts\python.exe train.py --max-episodes 5000 --max-steps 120 --save-dir policies\saved_models
```

## 4) Architecture Map

- `env/`: core multi-agent environment (`MultiAgentOrbitalEnv`).
- `sim/`: propagation, conjunction detection, maneuvers, observations, reporting.
- `policies/`: policy interface + heuristic and MARL policy wrappers.
- `marl/`: MAPPO-style trainer + curriculum manager.
- `safety/`: CBF-based safety filter.
- `experiments/`: reproducible TC1-TC8 evaluation framework.
- `tests/`: aligned regression and scenario coverage tests.
- `ui/`: Streamlit app for local runs and visualization.

## 5) Invariants To Preserve

- Observation encoding/decoding contract (`sim/observation_utils.py`) must stay backward-compatible with tests.
- Action-space constants (`ACTION_COUNT`, emergency action index) must remain consistent across env, policies, and trainer.
- Named scenario keys in test framework must remain:
  - `TC1_no_maneuver`
  - `TC2_threshold_rule`
  - `TC3_fuel_aware_rule`
  - `TC4_marl`
  - `TC5_high_density_stress`
  - `TC6_fuel_constrained`
  - `TC7_secondary_conjunctions`
  - `TC8_hypothetical_collision_cluster`
- `policies/saved_models/mppo_final.pt` is the default trained-model path used by README commands.

## 6) Editing Guidelines

- Prefer minimal, targeted changes over broad refactors.
- Keep public function signatures stable unless explicitly asked to change API.
- Maintain deterministic behavior where seeded runs are expected.
- Do not modify generated artifacts in `outputs/` unless task explicitly targets outputs.
- Do not delete or rename model checkpoints in `policies/saved_models/`.
- If adjusting policy logic, verify `tests/test_latest_implementation.py` still passes.

## 7) Validation Strategy After Changes

Run the smallest relevant checks first:

1. Targeted test(s) for touched module(s).
2. Full unit test discovery:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

3. If experiment logic changed, run:

```powershell
.venv\Scripts\python.exe experiments/run_collision_avoidance_tests.py --quick --mc-runs 1 --max-debris 50
```

4. If UI logic changed, smoke test:

```powershell
.venv\Scripts\streamlit.exe run ui\streamlit_app.py
```

## 8) Performance And Safety Guardrails

- Avoid defaulting to full training or full Monte Carlo sweeps during routine validation.
- Use `--quick`, smaller `--mc-runs`, and `--max-debris` caps for iterative checks.
- Treat this as a research simulator, not a flight-dynamics operational stack.

## 9) Definition Of Done For Code Tasks

- Requested behavior implemented.
- Relevant tests pass (or failures are clearly explained if blocked).
- No unnecessary changes to unrelated modules or large artifact directories.
- README/doc command paths remain valid on Windows PowerShell.
