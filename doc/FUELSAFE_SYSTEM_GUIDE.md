# Fuel-Constrained Multi-Agent Reinforcement Learning for Autonomous Collision Avoidance in Low Earth Orbit

This guide documents the current FuelSafe-MARL-LEO implementation as a research and prototyping platform for autonomous collision avoidance in Low Earth Orbit (LEO). It is written for two audiences:

- developers who need to understand architecture, data flow, interfaces, reward logic, and deployment tradeoffs
- users, operators, and reviewers who need a clear explanation of what the system does, how to run it, and how to interpret outputs

The implementation aligns conceptually with common Space Situational Awareness (SSA) workflows used by organizations such as ESA and NASA: screen conjunctions, characterize risk, rank threats, plan or recommend maneuvers, reassess post-maneuver risk, and track fuel and mission safety. It is not an operational flight-dynamics or certified conjunction assessment system.

---

## Part I. Developer Guide

### 1. Purpose and Scope

FuelSafe-MARL-LEO is a multi-agent orbital simulation and policy evaluation platform designed to study how fuel-constrained satellites can autonomously avoid collisions with debris and other threatening objects. The current implementation combines:

- SGP4-based reference orbit propagation
- CDM-style event ingestion and dataset normalization
- a multi-agent environment with controllable satellites and passive debris
- heuristic and MARL policies
- predictive collision probability estimation
- dense reward shaping based on risk reduction rather than collision-only outcomes
- curriculum learning with progressively harder cases, including TC8 short-warning clusters
- Monte Carlo evaluation with mission-oriented metrics such as fuel usage and secondary conjunctions

The system is most appropriate for research, algorithm comparison, thesis work, and early-stage mission concept studies.

### 2. System Architecture

#### 2.1 High-level architecture

```text
                 +----------------------+
                 |  CDM / Scenario Data |
                 | JSON, CSV, JSONL     |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | Data Ingestion Layer |
                 | CDMLoader            |
                 | CSVDataLoader        |
                 | DatasetIntegration   |
                 +----------+-----------+
                            |
                            v
+----------------+   +----------------------+   +----------------------+
| OrbitPropagator|-->| MultiAgentOrbitalEnv |<--| ManeuverEngine       |
| SGP4 reference |   | state, step, reward  |   | delta-v and fuel     |
+----------------+   +----------+-----------+   +----------------------+
                            |
                            v
                 +----------------------+
                 | ConjunctionDetector  |
                 | TCA, miss distance,  |
                 | risk score, Pc       |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | Policy Layer         |
                 | heuristic + MARL     |
                 +----------+-----------+
                            |
                            v
                 +----------------------+
                 | Training / Evaluation|
                 | train.py             |
                 | run_collision...py   |
                 | reporting.py         |
                 +----------------------+
```

#### 2.2 Main modules and responsibilities

- `sim/orbit_propagator.py`
  - wraps `sgp4` propagation
  - loads TLEs and returns Cartesian position and velocity in km and km/s
  - generates deterministic sample TLEs for synthetic scenarios

- `sim/conjunction_detector.py`
  - computes relative geometry, estimated closest approach, detector risk score, and predictive `Pc`
  - emits `ConjunctionAlert` records for satellite-relevant close approaches

- `sim/cdm_loader.py`
  - loads CDM-like JSON or CSV event records into `CDMEvent` objects

- `sim/csv_data_loader.py`
  - canonicalizes ESA-style or synthetic CSV inputs into a consistent scenario schema
  - extracts relative geometry and orbital features for simulation

- `sim/dataset_integration.py`
  - transforms dataset rows into training and experiment scenarios
  - can expand base scenarios into curriculum variants and TC8-style hard cases

- `env/ma_env.py`
  - central multi-agent environment
  - owns reset logic, step transitions, observation encoding, risk summarization, reward computation, and termination

- `sim/maneuver_engine.py`
  - maps discrete or hybrid actions into delta-v vectors
  - enforces per-step delta-v limits and fuel consumption

- `policies/policy_interface.py`
  - provides common policy interface
  - includes heuristic baselines and MARL wrapper

- `marl/marl_trainer.py`
  - implements MAPPO-style centralized training with decentralized execution
  - stores per-agent rollouts, computes GAE, and updates actor and critic networks

- `marl/curriculum_manager.py`
  - governs progression from easy long-warning scenarios to TC8-like hard conditions

- `safety/cbf_filter.py`
  - projects candidate delta-v commands into a threat-aware safe set using a linearized control barrier function

- `sim/simulator.py`
  - connects environment, policy manager, and safety filter for repeatable simulation runs

- `experiments/run_collision_avoidance_tests.py`
  - runs deterministic TC1-TC8 evaluation suites with Monte Carlo repetition
  - writes CSV, PNG, and HTML artifacts for comparison

- `sim/reporting.py`
  - generates interactive Plotly outputs for summary, Pareto, and run-distribution views

### 3. End-to-End Data Flow

#### 3.1 Online simulation flow

```text
reset()
  -> create satellites and debris
  -> generate or load scenario geometry
  -> propagate reference orbits
  -> encode per-satellite observations

step(actions)
  -> summarize pre-action risk
  -> apply maneuvers and burn fuel
  -> propagate all objects forward
  -> detect conjunctions
  -> summarize post-action risk
  -> compute rewards
  -> compute done flags
  -> emit observations, rewards, info
```

#### 3.2 Dataset-driven flow

```text
CSV / JSONL scenario
  -> schema normalization
  -> relative state extraction
  -> scenario object creation
  -> environment reset with overrides
  -> training or evaluation rollout
  -> metrics + plots + comparison reports
```

### 4. Environment Design

#### 4.1 Agents and objects

- controllable agents are satellites named like `SAT_000`
- passive objects are debris named like `DEB_000`
- each satellite tracks:
  - current orbital state
  - reference orbital state
  - persistent position and velocity offsets after maneuvers
  - remaining fuel
  - previous action
  - maneuver and collision counts

This lets the environment separate nominal orbital evolution from maneuver-induced deviations, which is closer to operational SSA thinking than treating every step as a fully re-sampled state.

#### 4.2 Observation design

The observation size is `94`. Each satellite receives:

- 10 self features
  - own position `(x, y, z)`
  - own velocity `(vx, vy, vz)`
  - fuel ratio
  - normalized step count
  - normalized minimum miss distance
  - maximum current risk

- 7 threat blocks x 12 features each
  - relative position `(r, t, n)`
  - relative velocity `(vr, vt, vn)`
  - normalized current distance
  - normalized relative speed
  - normalized miss distance estimate
  - normalized time to closest approach
  - risk score
  - predictive collision probability `Pc`

Threats are ranked by:

1. highest `Pc`
2. shortest time to closest approach
3. shortest current distance

This is consistent with operational screening logic where the most actionable threat is not simply the nearest object, but the one combining likelihood and urgency.

#### 4.3 Action design

The maneuver engine supports a discrete action space with seven actions:

- `0` `NO_OP`
- `1` `PROGRADE`
- `2` `RETROGRADE`
- `3` `RADIAL_OUT`
- `4` `RADIAL_IN`
- `5` `NORMAL`
- `6` `EMERGENCY_RADIAL_OUT`

The MARL actor uses a hybrid action output:

- a categorical head selects direction
- a continuous magnitude head outputs a delta-v magnitude in `[0, 5]`

Heuristic policies use the discrete action set. The safety filter can further modify these commands before they are executed.

#### 4.4 Reward design

The reward is dense and predictive. It is not based only on whether a collision occurred. Key components include:

- collision penalty
- near-miss penalty
- fuel penalty
- time-to-closest-approach urgency penalty
- reward for reducing aggregated predictive risk
- penalty for maneuver-linked secondary conjunctions
- per-step maneuver penalty
- jitter penalty when actions change unnecessarily
- reward for safe `NO_OP` when the environment is already low-risk

Conceptually:

```text
reward
= fuel term
+ risk-delta term
+ secondary conjunction term
+ maneuver / jitter terms
+ collision or near-miss terms
+ short-TCA urgency term
+ safe no-op bonus
```

The most distinctive term is the risk-delta component:

```text
risk_delta = total_risk_before - total_risk_after
reward_risk = w_risk_delta * risk_delta
```

This encourages the agent to reduce future collision likelihood, not merely react to collisions after it is too late.

#### 4.5 Termination logic

An episode ends when one of the following happens:

- satellite collisions exceed a fleet-scaled limit
- the step count reaches 1000
- all satellites are out of fuel

Scaling the collision threshold with constellation size is important for fair TC5-style large-fleet evaluation.

### 5. Collision Probability and Risk Estimation

#### 5.1 Predictive collision probability

The implementation uses a simplified physics-inspired predictive collision probability:

```text
Pc = exp(-dist_m / 100) * exp(-tca_s / 600)
```

Where:

- `dist_m` is estimated miss distance in meters
- `tca_s` is time to closest approach in seconds

Interpretation:

- risk increases sharply as miss distance approaches zero
- risk increases as TCA becomes more imminent
- the model is intentionally simple, smooth, and differentiable enough to support reward shaping

This is not a full covariance-based operational `Pc` computation of the type used in high-fidelity conjunction assessment pipelines. Instead, it functions as a research approximation for ranking and learning.

#### 5.2 Detector risk score vs `Pc`

The environment maintains both:

- a detector risk score driven by distance, miss distance, and relative speed
- a predictive `Pc`

Threat-level reward and prioritization lean heavily on `Pc`, while the detector score remains useful for screening and safety gating.

Within the threat collector, a combined score is formed as:

```text
risk_score = 0.35 * detector_risk + 0.65 * Pc
```

This design gives more weight to predicted collision likelihood while retaining geometric intuition.

#### 5.3 Secondary conjunction logic

Secondary conjunctions are counted only when a near-miss:

- is not a direct collision
- is below the near-miss threshold
- exceeds the configured `Pc` threshold
- directly involves a satellite that maneuvered this step or the previous step

This is important because in realistic operations, a maneuver is only blamed for creating a follow-on hazard if there is a plausible causal link.

### 6. Training Pipeline

#### 6.1 Learning approach

The trainer implements a lightweight MAPPO-style workflow:

- decentralized actors, one per satellite
- centralized critic over concatenated observations
- PPO clipping for stable updates
- GAE for advantage estimation
- entropy regularization for exploration
- gradient clipping for numerical stability

This follows the centralized-training, decentralized-execution pattern often used in cooperative multi-agent systems.

#### 6.2 Training loop

The default training script in `train.py`:

1. samples a scenario from either dataset-derived or curriculum-generated sources
2. resets the environment
3. rolls out up to `max_steps`
4. stores per-agent experience in PPO buffers
5. updates the model every `update_every` episodes
6. evaluates on deterministic rollouts every `eval_interval` episodes
7. updates curriculum stage based on evaluation performance
8. writes checkpoints and a final model

#### 6.3 Curriculum learning

The built-in curriculum manager defines three stages:

- Stage 1: long TCA, low density
- Stage 2: moderate risk, medium density
- Stage 3: short TCA, high density, aligned with TC8-style difficulty

Progression requires three consecutive evaluation results above a stage threshold. The dataset integration path can additionally synthesize:

- easy variants
- medium variants
- hard variants
- explicit TC8-style cluster variants

This mirrors the real-world need to train autonomy first on sparse, easier screening conditions before exposing it to dense, ambiguous, short-warning conjunctions.

#### 6.4 Evaluation during training

The training evaluator reports:

- mean collisions
- collision rate
- mean fuel
- mean maneuvers
- mean secondary conjunctions
- mean near misses
- mean objective score
- curriculum score

The curriculum score prioritizes:

1. collision reduction
2. lower collision rate
3. fuel efficiency
4. lower secondary conjunction count

### 7. Evaluation Framework

#### 7.1 Deterministic test catalog

The `experiments/run_collision_avoidance_tests.py` suite includes:

- `TC1_no_maneuver`
- `TC2_threshold_rule`
- `TC3_fuel_aware_rule`
- `TC4_marl`
- `TC5_high_density_stress`
- `TC6_fuel_constrained`
- `TC7_secondary_conjunctions`
- `TC8_hypothetical_collision_cluster`

These cases cover worst-case no-op behavior, threshold policies, fuel-limited response, MARL comparison, high-density fleets, maneuver-linked secondary risk, and very short-warning cluster events.

#### 7.2 Output artifacts

The experiment framework writes:

- `test_runs_per_policy.csv`
- `aggregated_summary.csv`
- `pareto_frontier_fuel_vs_collisions.csv`
- `plot_*.png`
- `interactive_summary_*.html`
- `interactive_runs_*.html`

These support both research reporting and operator-style comparison reviews.

### 8. Hyperparameters and Defaults

#### 8.1 Model and PPO defaults

From `marl/marl_trainer.py`:

- actor hidden size: `128`
- critic hidden size: `128`
- learning rate: `3e-4`
- gamma: `0.99`
- GAE lambda: `0.95`
- entropy coefficient: `0.01`
- value loss coefficient: `0.5`
- max gradient norm: `0.5`
- PPO clip ratio: `0.2`
- actor output magnitude cap: `5.0`

#### 8.2 Training defaults

From `train.py`:

- max steps per episode: `120`
- max episodes: `8000`
- update every: `5` episodes
- evaluation interval: `50`
- evaluation episodes: `5`
- entropy start/end: `0.01 / 0.01`
- TC8 hard-scenario mix ratio: `0.3`
- terminate on collision: `true`
- PPO update call: `num_epochs=10`, `batch_size=64`

#### 8.3 Environment defaults

Representative defaults include:

- time step `dt = 60 s`
- initial and maximum fuel `1000 kg`
- near-miss threshold `2 x collision_threshold`
- predictive distance scale `100 m`
- predictive TCA scale `600 s`

Reward weights default to:

- collision: `-1000`
- near miss: `-50`
- fuel: `-0.1`
- miss distance: `-2`
- TCA urgency: `-5`
- risk delta: `+50`
- secondary conjunction: `-5`
- maneuver count: `-0.2`
- jitter: `-0.1`
- safe no-op: `+1`

### 9. Deployment Considerations

For real mission integration, the current implementation should be treated as a decision-support prototype rather than an autonomous flight product. Practical deployment would require:

- certified orbit determination and covariance handling
- high-fidelity `Pc` models rather than the exponential approximation
- trusted CDM ingestion, validation, and time synchronization
- maneuver authorization and human-in-the-loop approval
- commandability constraints, keep-out zones, and mission-specific rules
- post-maneuver screening and coordination with external operators
- fail-safe logic if the learned policy behaves out of family

Recommended operational deployment pattern:

```text
SSA screening -> candidate maneuver generation -> safety filter / rules ->
human approval -> execution -> post-burn reassessment -> archive and learn
```

### 10. Scaling to Multi-Satellite Constellations

The codebase already supports higher object counts and includes a large-fleet stress case. Still, scaling has distinct technical considerations.

#### 10.1 What scales well

- per-agent observations are fixed size due to top-k threat encoding
- the centralized critic is manageable for small to medium fleets
- fleet-size-aware collision termination is already implemented
- the test suite includes a 50-satellite stress scenario

#### 10.2 What becomes difficult

- conjunction screening is still fundamentally pairwise, so total object interactions grow rapidly
- centralized critic input grows linearly with the number of agents
- training data requirements increase with fleet diversity
- secondary conjunction effects become harder to attribute cleanly
- a single policy may not generalize across mixed orbital shells without domain randomization

#### 10.3 Practical scaling strategies

- shard large constellations into local decision neighborhoods
- replace the concatenated critic with graph or attention-based critics
- separate tactical collision avoidance from strategic traffic management
- cache screening results and use asynchronous threat updates
- use curriculum schedules that explicitly increase fleet size over time

### 11. Limitations and Assumptions

This guide should be read with the following limits in mind:

- `Pc` is simplified and does not use covariance ellipsoids or full conjunction uncertainty models
- the environment uses synthetic TLE generation for many scenarios
- maneuver dynamics are intentionally simplified and do not model full orbital control authority, attitude, or actuator latency
- debris are passive and non-cooperative
- reward design reflects research priorities, not flight certification
- the current MARL implementation assumes a small fixed number of controllable agents during training
- the CBF safety layer is linearized and local, not a proof of global mission safety
- the system evaluates tactical collision avoidance, not end-to-end mission planning or catalog maintenance

### 12. Real-World Alignment with ESA and NASA Practice

The implementation is broadly aligned with how real SSA operations are structured:

- conjunctions are screened continuously
- events are prioritized by risk and time criticality
- maneuver decisions must trade safety, fuel, and operational burden
- secondary effects of maneuvers matter
- outputs must be explainable enough for review boards and mission operators

Where it differs from operational practice:

- operational agencies use more rigorous orbit determination, covariance propagation, and `Pc` estimation
- maneuver approval is procedural and mission-specific
- external coordination across operators and regulatory frameworks is out of scope here
- safety cases must be demonstrated far beyond reinforcement learning reward curves

---

## Part II. User Guide

### 13. System Overview

FuelSafe-MARL-LEO is a software system that helps simulate how satellites can avoid collisions in orbit while using as little fuel as possible. It does this by combining orbital motion models, conjunction warnings, and decision-making policies, including machine learning.

In simple terms, the system watches for dangerous close approaches, estimates how risky they are, and tests whether a satellite should maneuver now, maneuver later, or conserve fuel and do nothing.

### 14. The Problem Being Solved

Satellites in LEO operate in increasingly crowded environments. Operators face a difficult tradeoff:

- maneuver too rarely and collision risk rises
- maneuver too often and fuel is wasted, shortening mission life
- maneuver aggressively and you may create new secondary conjunctions

The system is designed to explore that trade space automatically and consistently.

### 15. How the System Works

At a high level:

1. the system reads or generates orbital situations and conjunction events
2. it predicts where satellites and debris will be over time
3. it identifies which nearby objects are most dangerous
4. it estimates collision risk using miss distance and time to closest approach
5. it lets a selected policy decide whether to maneuver
6. it measures outcomes such as safety, fuel use, maneuver count, and follow-on risk

Conceptual view:

```text
Threat appears
   -> system estimates risk
   -> policy decides action
   -> maneuver is checked for safety
   -> orbit evolves
   -> risk is recomputed
   -> results are logged and compared
```

### 16. Inputs and Outputs

#### 16.1 What you can provide

The system supports several input styles:

- synthetic scenarios generated internally
- CDM-like JSON or CSV event files
- canonicalized CSV datasets
- JSONL scenario files for MARL training

Typical inputs contain:

- object identifiers
- time to closest approach
- miss distance
- relative velocity
- collision probability or equivalent risk proxy
- orbital features for primary and secondary objects

#### 16.2 What the system produces

Common outputs include:

- per-run CSV logs
- aggregated policy comparison tables
- Pareto views of fuel versus collisions
- static PNG plots
- interactive HTML charts
- trained MARL checkpoints

### 17. Running Simulations

#### 17.1 Quick demo

Use the demo entry point to compare heuristic policies and optional MARL:

```powershell
.venv\Scripts\python.exe main.py --demo --include-marl --episodes 1 --steps 50
```

#### 17.2 Full experiment suite

Run the deterministic TC1-TC8 test framework:

```powershell
.venv\Scripts\python.exe experiments/run_collision_avoidance_tests.py --quick --mc-runs 3 --max-debris 200 --log-level INFO
```

For a detailed short-warning MARL stress case:

```powershell
.venv\Scripts\python.exe experiments/run_collision_avoidance_tests.py --test-cases TC8_hypothetical_collision_cluster --mc-runs 3 --max-debris 200 --include-marl --marl-model-path policies\saved_models\mppo_final.pt --output-dir outputs\tc8_validation --log-level INFO
```

For verbose diagnostics:

```powershell
.venv\Scripts\python.exe experiments/run_collision_avoidance_tests.py --mc-runs 1 --max-debris 200 --include-marl --marl-model-path policies\saved_models\mppo_final.pt --output-dir outputs\test_framework_full_validation --verbose --log-file run.log
```

#### 17.3 Training a MARL policy

```powershell
.venv\Scripts\python.exe train.py --max-episodes 8000 --max-steps 120 --update-every 5 --eval-interval 50 --eval-episodes 5 --tc8-ratio 0.3 --use-dataset true --dataset-path data\train_data.jsonl --dataset-eval-path data\test_data.jsonl --save-dir policies\saved_models
```

### 18. Interpreting Results

#### 18.1 Key metrics

- `collisions`
  - direct safety failures involving satellites

- `fuel`
  - total propellant spent on avoidance action

- `maneuvers`
  - number of avoidance commands executed

- `secondary conjunctions`
  - follow-on close approaches plausibly linked to a maneuver

- `near misses`
  - close approaches below the configured near-miss threshold

- `collision rate`
  - fraction of runs with at least one collision

- `mean minimum separation`
  - average closest separation achieved across runs

#### 18.2 What good performance looks like

The best policy is usually not the one with the fewest maneuvers or the lowest fuel in isolation. A strong policy typically shows:

- low or zero collisions
- low collision rate across Monte Carlo runs
- moderate fuel use rather than reckless overreaction
- controlled maneuver count
- low secondary conjunction count
- stable performance across multiple test cases, especially TC6-TC8

#### 18.3 Reading the plots

- summary bar charts
  - compare average policy performance by test case

- run distribution plots
  - show whether a policy is stable or highly variable

- Pareto chart
  - highlights whether a policy offers a good safety-fuel tradeoff

A policy far to the lower-left of the Pareto plot is typically desirable because it uses less fuel and experiences fewer collisions.

### 19. Business and Mission Value

Even as a research platform, the system demonstrates several forms of value:

- fuel savings
  - avoids unnecessary burns and helps protect mission lifetime

- safety improvement
  - emphasizes proactive risk reduction before conjunctions become critical

- automation
  - reduces operator workload in high-volume screening environments

- consistency
  - applies the same decision logic across many satellites and events

- what-if analysis
  - allows operators and analysts to compare policy families before deployment

For constellation operators, fuel-efficient collision avoidance directly affects revenue by preserving usable mission life. For agencies and defense users, it also supports resilience and traffic-management readiness.

### 20. Real-World Usage Scenarios

Examples of realistic users include:

- civil space agencies
  - evaluating autonomous maneuver support concepts for Earth observation or science missions

- commercial constellation operators
  - screening thousands of conjunctions and prioritizing which ones merit action

- SSA and STM research groups
  - testing multi-agent decision logic under crowded LEO conditions

- university and thesis teams
  - studying explainable MARL, safety layers, and fuel-aware autonomy

- mission operations trainers
  - using synthetic hard cases such as TC8 to rehearse short-warning response logic

### 21. Risks and Safeguards

#### 21.1 Key risks

- learned policies may exploit reward structure in unrealistic ways
- simplified `Pc` may not match operational collision probability estimates
- aggressive maneuvering can create secondary hazards
- dataset bias can cause poor generalization
- large-constellation behavior may differ from small-team training behavior

#### 21.2 Safeguards in the current system

- CBF safety filtering can modify unsafe actions
- heuristic baselines provide sanity-check comparisons
- deterministic test cases expose edge cases such as fuel limits and short-warning clusters
- secondary conjunctions are tracked explicitly
- verbose logging and per-run outputs support review and diagnosis

#### 21.3 Recommended operational safeguards

- require human approval for real maneuvers
- compare learned policy output against rule-based guardrails
- rerun screening after every candidate maneuver
- maintain mission-specific fuel reserves and keep-out constraints
- use independent verification tools before execution

### 22. Practical Adoption Guidance

If you are introducing this system into an operations-adjacent workflow, a sensible path is:

1. start with offline replay on historical or synthetic conjunction sets
2. compare MARL against heuristic policies on TC1-TC8 and dataset-derived scenarios
3. tune reward weights and safety thresholds to mission policy
4. enable the model first as an advisory recommender
5. only later evaluate partial autonomy under strict oversight

This mirrors the way many aerospace autonomy systems move from simulation to trusted operational use.

---

## Appendix A. Repository Mapping

- `env/` multi-agent environment and reward logic
- `experiments/` deterministic evaluation suites and Monte Carlo validation
- `marl/` trainer and curriculum manager
- `policies/` heuristic and learned policy interfaces
- `safety/` control barrier safety filter
- `sim/` propagation, conjunction detection, ingestion, reporting, and simulation glue
- `data/` datasets and synthetic scenario generation
- `outputs/` evaluation artifacts and logs

## Appendix B. Recommended Reading Order

For developers:

1. `env/ma_env.py`
2. `sim/observation_utils.py`
3. `marl/marl_trainer.py`
4. `train.py`
5. `experiments/run_collision_avoidance_tests.py`

For users and reviewers:

1. this guide, Part II
2. `README.md`
3. the latest files under `outputs/`

