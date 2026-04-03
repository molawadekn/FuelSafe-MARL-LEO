# Technical Guide - FuelSafe-MARL-LEO

## Runtime Architecture

1. `sim/orbit_propagator.py`
   Generates reference orbital states with SGP4.
2. `env/ma_env.py`
   Maintains actual agent state, applies persistent maneuver offsets, computes rewards, and exposes observations.
3. `sim/conjunction_detector.py`
   Scores pairwise encounters and labels collisions.
4. `sim/maneuver_engine.py`
   Converts discrete actions into `delta-v` burns and fuel use.
5. `marl/marl_trainer.py`
   Trains a MAPPO-style policy with:
   - per-agent actors
   - a centralized critic over concatenated observations
   - stored action log-probs
   - GAE-style returns
6. `policies/policy_interface.py`
   Exposes both per-agent and joint-action policy APIs.
7. `sim/simulator.py`
   Runs episodes, applies the CBF filter, and aggregates metrics.
8. `sim/reporting.py`
   Builds interactive Plotly charts for summaries, raw runs, and training progress.

## Observation and Action Spaces

Observation size: `64`

Layout:
- own position: 3
- own velocity: 3
- fuel ratio: 1
- normalized step count: 1
- up to 7 nearby objects:
  each contributes 8 values
  `position(3) + velocity(3) + miss_distance_norm + tca_norm`

Action space: `Discrete(6)`
- `0`: `NO_OP`
- `1`: `PROGRADE`
- `2`: `RETROGRADE`
- `3`: `RADIAL_OUT`
- `4`: `RADIAL_IN`
- `5`: `NORMAL`

## Reward Model

The default reward combines:
- collision penalty
- actual fuel burned that step
- safe-separation reward when the agent is not in a high-risk alert
- secondary conjunction penalty when close non-collision alerts appear after maneuver activity

Default weights in `env/ma_env.py`:

```python
{
    "collision": -10000.0,
    "fuel": -1.0,
    "safe_separation": 1.0,
    "secondary_conjunction": -100.0,
}
```

## Termination Conditions

Episodes end when one of the following happens:
- collisions in the episode reach 5
- step count reaches 1000
- all satellites are out of fuel

## Dataset-Driven Scenarios

`sim/csv_data_loader.py` extracts:
- target orbital elements
- chaser orbital elements
- relative position and velocity features
- conjunction metadata such as miss distance, relative speed, and time to TCA

`sim/dataset_integration.py` passes those features into the environment so the first satellite-debris pair is initialized from dataset-derived geometry instead of a generic placeholder configuration.

The same module now exposes a train/validate CLI that:
- trains MARL on `data/train_data.csv`
- selects risk-stratified unique conjunction events instead of only the top few worst rows
- uses a tighter `0.5 km` collision threshold for CDM-driven train/validate runs
- saves weights and per-episode training metrics
- evaluates `no_op`, `fuel_aware_threshold_rule`, `rule_based`, and `marl` on `data/test_data.csv`
- writes summary artifacts under `outputs/marl_train_validation/`
- writes interactive HTML charts for validation and training progress

## Local UI

`ui/streamlit_app.py` provides a local UI for:
- demos
- dataset experiments
- dataset train/validate runs
- named test cases `TC1` through `TC8`
- chart-based result exploration from generated output directories

## Fuel-Constrained MARL Evaluation

`experiments/run_collision_avoidance_tests.py` now supports `marl` in:
- `TC4_marl`
- `TC5_high_density_stress`
- `TC6_fuel_constrained`
- `TC7_secondary_conjunctions`

That means the dedicated low-fuel scenario can now compare MARL against the deterministic baselines directly.
