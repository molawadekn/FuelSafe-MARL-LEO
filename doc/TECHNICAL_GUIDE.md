# Technical Guide - FuelSafe-MARL-LEO

## Runtime Architecture

1. `sim/orbit_propagator.py`
   Generates reference orbital states with SGP4.
2. `env/ma_env.py`
   Maintains actual agent state, applies pre-propagation maneuvers, computes dense rewards, and exposes risk-aware observations.
3. `sim/conjunction_detector.py`
   Scores pairwise encounters, estimates miss distance/TCA, and computes collision probability (Pc).
4. `sim/maneuver_engine.py`
   Converts actions into Delta-V burns, including emergency maneuvers and optional continuous magnitudes.
5. `marl/marl_trainer.py`
   Trains a MAPPO-style policy with per-agent actors and a centralized critic.
6. `policies/policy_interface.py`
   Exposes heuristic and MARL policies behind a common interface.
7. `safety/cbf_filter.py`
   Projects candidate Delta-V actions into a barrier-safe set using relative geometry.
8. `sim/simulator.py`
   Runs episodes, applies CBF filtering, and aggregates metrics.
9. `train.py`
   Runs curriculum/dataset training with periodic evaluation and checkpointing.

## Observation And Action Spaces

Observation size: `70`

Layout:
- own position: 3
- own velocity: 3
- fuel ratio: 1
- normalized step count: 1
- normalized minimum predicted miss distance: 1
- maximum local risk score: 1
- up to 6 nearby threats:
  each contributes 10 values
  `rel_pos(3) + rel_vel(3) + miss_distance_norm + tca_norm + risk_score + collision_probability_pc`

Action space:
- discrete direction index in `[0..6]` (`NO_OP`, `PROGRADE`, `RETROGRADE`, `RADIAL_OUT`, `RADIAL_IN`, `NORMAL`, `EMERGENCY_RADIAL_OUT`)
- MARL actor also predicts a continuous magnitude for hybrid execution `(direction, magnitude)`

## Step Timing

The environment applies maneuvers before propagation each step:
1. summarize current risk
2. apply selected maneuver
3. advance time
4. propagate reference orbit plus persistent offsets
5. detect conjunctions and compute rewards

## Reward Model

The default reward combines:
- collision penalty
- near-miss penalty
- fuel burn penalty
- Pc risk-delta shaping: `risk_delta_weight * (risk_now - risk_next)`
- residual urgency pressure: `tca_weight * risk_next * exp(-min_tca_s / 600.0)`
- predicted miss-distance pressure
- secondary conjunction penalty (only for alerts linked to maneuvering satellites and above Pc threshold)
- per-step maneuver-count penalty

Default weights in `env/ma_env.py`:

```python
{
    "collision": -1000.0,
    "near_miss": -50.0,
    "fuel": -0.1,
    "miss_distance": -2.0,
    "tca": -2.0,
    "risk_delta": 5.0,
    "secondary_conjunction": -5.0,
    "maneuver_count": -0.5,
}
```

## CBF Safety Filter

`safety/cbf_filter.py` uses:
- relative position from controlled satellite to threat
- relative closing rate
- minimum safe distance with risk-aware margin
- a linearized barrier condition on radial closing speed

The filter is applied to current geometry, then mapped back to executable action space.

## Dataset-Driven Curriculum

`train.py` supports:
- curriculum stages (low to high density/risk)
- periodic TC8-like hard scenario sampling via `--tc8-ratio`
- optional JSONL dataset sampling (`--use-dataset true`)
- periodic deterministic evaluation (`--eval-interval`, `--eval-episodes`)
- curriculum progression driven by a multi-objective score (collisions, fuel, secondary conjunctions)

## Local UI

`ui/streamlit_app.py` supports:
- demos
- dataset experiments
- training/validation runs
- named test cases `TC1` through `TC8`
- interactive chart exploration from `outputs/`
