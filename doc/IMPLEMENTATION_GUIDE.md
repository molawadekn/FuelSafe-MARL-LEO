# Implementation Guide - FuelSafe-MARL-LEO

## What Is Implemented

The current codebase supports an end-to-end comparative workflow:

1. Build orbital scenarios with SGP4 reference states.
2. Inject burns through a discrete maneuver engine with fuel accounting.
3. Preserve burn effects across timesteps using persistent state offsets.
4. Roll out heuristic and MARL policies in the same environment.
5. Apply an optional CBF safety filter before execution.
6. Aggregate collisions, fuel, maneuver counts, and secondary-conjunction metrics.

## MARL Pipeline

### Training

`marl/marl_trainer.py` now implements a consistent MAPPO-style flow:
- actors operate on local observations
- the critic operates on concatenated observations from all controlled satellites
- the trainer stores the log-probability of the action that was actually taken
- advantages are computed with GAE-style bootstrapping
- the critic trains on centralized observations instead of per-agent local vectors

### Inference

Joint MARL action selection is done in one call. The simulator no longer asks the MARL policy one agent at a time with partial observations.

## Environment Model

### Reference Orbit

Each object has a reference trajectory from SGP4.

### Actual Runtime State

The environment keeps:
- the current reference state
- a persistent position offset
- a persistent velocity offset

When a maneuver is executed, the offsets are updated and carried forward to future steps. This fixes the earlier behavior where burns only affected the current step.

## Fuel Handling

Fuel appears in three places:
- state observation as `fuel / max_fuel`
- maneuver feasibility checks in `sim/maneuver_engine.py`
- reward penalty using actual fuel burned in the step

This makes the environment fuel-aware and fuel-constrained in both dynamics and reward shaping.

## Dataset Integration

`sim/csv_data_loader.py` extracts conjunction features from ESA CDM data.

`sim/dataset_integration.py` now passes scenario metadata into `env/ma_env.py`, which uses:
- target orbital elements for `SAT_000`
- chaser orbital elements for `DEB_000`
- relative position and velocity features to seed the encounter geometry

This is still a simplified mapping, but it is no longer dataset-in-name-only.

The repository-level train/validate entry point is:

```powershell
.venv\Scripts\python.exe sim/dataset_integration.py --train-csv data\train_data.csv --test-csv data\test_data.csv --output-dir outputs\marl_train_validation
```

That flow trains on the train split, saves the learned weights, and then validates MARL against deterministic baselines on the test split.

For the dataset-backed train/validate path, scenario selection is risk-stratified across unique events and uses a tighter collision threshold than the coarse stress-test harness so the CSV-driven runs are not dominated by immediate sub-kilometer collision labels.

## Synthetic Comparison Case

`TC8_hypothetical_collision_cluster` is a handcrafted stress case added to `experiments/run_collision_avoidance_tests.py`.

It uses:
- 3 satellites
- 1 synthetic imminent conjunction debris object
- low available fuel
- a close dataset-style relative geometry seed

The goal is not realism. It is a controlled comparison case where:
- `no_op` collides
- heuristic policies avoid the collision with moderate fuel use
- the trained MARL policy also avoids the collision, but with much higher fuel usage

## Local UI

Run:

```powershell
.venv\Scripts\streamlit.exe run ui\streamlit_app.py
```

The UI can launch all major repo workflows and render interactive comparison charts directly from the generated CSV outputs.

## Reproducible Evaluation

Use the test framework for controlled policy comparison:

```powershell
.venv\Scripts\python.exe experiments/run_collision_avoidance_tests.py --quick --mc-runs 3 --max-debris 200
```

Use this for the key MARL and fuel-constrained comparison:

```powershell
.venv\Scripts\python.exe experiments/run_collision_avoidance_tests.py --test-cases TC4_marl,TC6_fuel_constrained --mc-runs 3 --max-debris 200 --include-marl --marl-untrained
```

## Practical Caveats

- The simulator is suitable for research comparison and thesis experimentation, not for operational flight software.
- Maneuver persistence is modeled with offsets relative to an SGP4 reference orbit, not full post-burn orbit determination.
- Untrained MARL policies are supported for plumbing checks, but meaningful evaluation should load trained weights.
