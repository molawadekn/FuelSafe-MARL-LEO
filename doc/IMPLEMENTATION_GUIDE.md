# Implementation Guide - FuelSafe-MARL-LEO

## What Is Implemented

The current codebase supports this end-to-end workflow:

1. Build orbital scenarios with SGP4 reference states.
2. Encode risk-aware observations with relative geometry, miss distance, TCA, and risk score.
3. Select heuristic or MARL actions jointly across satellites.
4. Apply an optional CBF safety filter before execution.
5. Execute pre-propagation maneuvers with real fuel accounting.
6. Roll out dense reward shaping and aggregate evaluation metrics.

## MARL Pipeline

### Training

`marl/marl_trainer.py` implements a MAPPO-style flow:
- actors operate on local 70-d predictive Pc-aware observations
- the critic operates on concatenated observations from all controlled satellites
- the trainer stores the log-probability of the action actually executed
- advantages are computed with GAE-style bootstrapping
- checkpoint loading now tolerates partial shape mismatches so older checkpoints can still warm-start the new architecture

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

When a maneuver is executed, the velocity offset is applied immediately and the position offset is carried into subsequent propagation steps.

## Fuel And Reward Handling

Fuel appears in three places:
- state observation as `fuel / max_fuel`
- maneuver feasibility checks in `sim/maneuver_engine.py`
- dense reward shaping using actual fuel burned, risk level, proximity pressure, urgency, and risk reduction

This makes the environment fuel-aware and fuel-constrained in both dynamics and reward shaping.

## Emergency Maneuver Option

The discrete action space now includes:
- standard maneuvers `0` through `5`
- `6` as an emergency radial-out burn

Heuristic policies and the CBF safety layer can both escalate to this action when TCA or miss distance becomes critical.

## Dataset Integration And Curriculum

`sim/csv_data_loader.py` extracts conjunction features from ESA CDM data.

`sim/dataset_integration.py` now:
- maps target orbital elements to `SAT_000`
- maps chaser orbital elements to `DEB_000`
- injects relative position and velocity features into the encounter geometry
- supports stage-based sampling for progressively harder scenarios

The repository-level train entry point is:

```powershell
.venv\Scripts\python.exe train.py --max-episodes 8000 --max-steps 120 --update-every 5 --eval-interval 50 --eval-episodes 5 --tc8-ratio 0.25 --use-dataset true --dataset-path data\train_data.jsonl --dataset-eval-path data\test_data.jsonl --save-dir policies\saved_models
```

## Synthetic Comparison Case

`TC8_hypothetical_collision_cluster` is the handcrafted short-warning stress case in `experiments/run_collision_avoidance_tests.py`.

It is designed to:
- keep collision pressure high
- expose whether early action and emergency maneuvers are working
- compare baseline, rule-based, and MARL policies under the same close conjunction

Use this case as a regression benchmark after each major training change.
Always regenerate metrics locally from the latest checkpoint instead of relying on stale historical results.

## Reproducible Evaluation

Use the test framework for controlled policy comparison:

```powershell
.venv\Scripts\python.exe experiments/run_collision_avoidance_tests.py --mc-runs 1 --max-debris 200 --include-marl --marl-model-path policies\saved_models\mppo_final.pt --output-dir outputs\test_framework_full_validation
```

## Practical Caveats

- The simulator is suitable for research comparison and thesis experimentation, not for operational flight software.
- Maneuver persistence is modeled with offsets relative to an SGP4 reference orbit, not full post-burn orbit determination.
- Policy quality remains checkpoint-dependent; track collisions, fuel, and secondary conjunctions together when comparing against heuristics.
