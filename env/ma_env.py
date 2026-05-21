"""
env/ma_env.py — MultiAgentOrbitalEnv
=====================================
Core multi-agent Gymnasium-compatible environment for fuel-constrained
satellite collision avoidance in LEO.

Architecture
------------
* Satellites  : num_satellites agents, each with position, velocity, fuel.
* Debris      : num_debris passive objects on crossing trajectories.
* Physics     : Two-body Euler integration (Newtonian gravity) per step;
                SGP4 used only to seed initial orbital states.
* Perception  : Each agent sees a 96-dim observation (own-state + up to 7
                ranked threat blocks) built by observation_utils.encode_observation.
* Safety      : CBF filter projects unsafe actions back to the safe set
                before the maneuver engine applies them.
* Realism     : Configurable noise, partial observability, maneuver delay,
                reaction delay, and decision noise via RealismLayer.

Episode lifecycle
-----------------
    obs = env.reset()
    while not done["__all__"]:
        actions = {agent_id: (dir_idx, magnitude_kms), ...}
        obs, rewards, done, info = env.step(actions)

Constructor kwargs (match train.py usage exactly)
--------------------------------------------------
    num_satellites              int   = 3
    num_debris                  int   = 10
    collision_threshold_km      float = 0.005   (5 m hard-body radius)
    distance_threshold_km       float = 10.0    (alert / observation range)
    high_risk_mode              bool  = True
    scenario_config             dict  = None    (reserved for dataset scenarios)
    hard_scenario_probability   float = 0.0
    secondary_conjunction_risk_threshold float = 0.01
    realism_config              RealismConfig = None
    max_steps                   int   = 120
    initial_fuel_kg             float = 100.0
    dt                          float = 60.0   (seconds per step)
    seed                        int   = None
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from safety.cbf_filter import CBFSafetyFilter
from sim.conjunction_detector import ConjunctionDetector, ConjunctionAlert
from sim.maneuver_engine import ManeuverEngine, ManeuverType
from sim.observation_utils import (
    OBS_SIZE,
    ThreatObservation,
    encode_observation,
    rank_threats,
)
from sim.orbit_propagator import OrbitPropagator
from sim.realism import RealismConfig, RealismLayer


# ─────────────────────────────────────────────────────────────────────────────
# Physical constants
# ─────────────────────────────────────────────────────────────────────────────
MU_EARTH            = 398600.4418   # km³/s²  Earth gravitational parameter
LEO_SEMI_MAJOR_KM   = 6800.0       # km       ~420 km altitude
LEO_INCLINATION_DEG = 51.6         # deg      ISS-like inclination

# Fixed reproducible epoch for all TLE generation inside an episode
_SCENARIO_EPOCH = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

# ─────────────────────────────────────────────────────────────────────────────
# Reward weights  (dense shaping)
# ─────────────────────────────────────────────────────────────────────────────
RW_COLLISION        = -1000.0   # per collision event
RW_NEAR_MISS        = -50.0     # per near-miss alert
RW_FUEL_PER_KG      = -0.1     # per kg fuel burned
RW_MANEUVER         = -0.5     # per non-NO_OP action
RW_RISK_REDUCTION   = +5.0     # per unit of peak-risk reduction
RW_SECONDARY        = -5.0     # per secondary conjunction induced
RW_TCA_URGENCY      = -2.0     # scaled by TCA urgency [0,1]
RW_MISS_DIST        = -2.0     # scaled by miss-distance proximity [0,1]

# Operational thresholds
NEAR_MISS_THRESHOLD_KM = 10.0   # km  — below this → near-miss penalty
INITIAL_FUEL_KG        = 100.0  # kg  — starting fuel per satellite


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

class MultiAgentOrbitalEnv:
    """
    Multi-agent orbital collision avoidance environment.

    Observation space  (per agent) : np.ndarray[float32, shape=(96,)]
    Action space       (per agent) : Tuple[int, float]
                                     int   — ManeuverType index  0..6
                                     float — burn magnitude km/s 0..0.005
    Reward             (per agent) : float  (dense shaped, per step)
    """

    # ──────────────────────────────────────────── construction ───────────────

    def __init__(
        self,
        num_satellites: int = 3,
        num_debris: int = 10,
        collision_threshold_km: float = 0.005,
        distance_threshold_km: float = 10.0,
        high_risk_mode: bool = True,
        scenario_config: Optional[Dict[str, Any]] = None,
        hard_scenario_probability: float = 0.0,
        secondary_conjunction_risk_threshold: float = 0.01,
        realism_config: Optional[RealismConfig] = None,
        max_steps: int = 120,
        initial_fuel_kg: float = INITIAL_FUEL_KG,
        dt: float = 60.0,
        seed: Optional[int] = None,
    ) -> None:
        # ── configuration ────────────────────────────────────────────────────
        self.num_satellites         = int(num_satellites)
        self.num_debris             = int(num_debris)
        self.collision_threshold    = float(collision_threshold_km)
        self.distance_threshold     = float(distance_threshold_km)
        self.high_risk_mode         = bool(high_risk_mode)
        self.scenario_config        = scenario_config
        self.hard_scenario_prob     = float(np.clip(hard_scenario_probability, 0.0, 1.0))
        self.secondary_risk_thr     = float(secondary_conjunction_risk_threshold)
        self.max_steps              = int(max_steps)
        self.initial_fuel_kg        = float(initial_fuel_kg)
        self.dt                     = float(dt)

        if seed is not None:
            np.random.seed(seed)

        # ── agent / debris IDs ───────────────────────────────────────────────
        self.agent_ids: List[str]  = [f"SAT_{i:03d}" for i in range(self.num_satellites)]
        self.debris_ids: List[str] = []

        # ── sub-modules ──────────────────────────────────────────────────────
        _realism_cfg = realism_config if realism_config is not None else RealismConfig(enabled=False)
        self.realism   = RealismLayer(_realism_cfg)
        self.detector  = ConjunctionDetector(
            distance_threshold_km=self.distance_threshold,
            collision_threshold_km=self.collision_threshold,
        )
        self.engine    = ManeuverEngine()
        self.cbf       = CBFSafetyFilter(min_safe_distance_km=0.1)
        self.propagator = OrbitPropagator()

        # ── episode state (populated by reset) ───────────────────────────────
        self._sat_pos:   Dict[str, np.ndarray] = {}
        self._sat_vel:   Dict[str, np.ndarray] = {}
        self._sat_fuel:  Dict[str, float]       = {}
        self._deb_pos:   Dict[str, np.ndarray] = {}
        self._deb_vel:   Dict[str, np.ndarray] = {}
        self._prev_risks: Dict[str, float]     = {}

        # ── episode counters (read by train.py) ──────────────────────────────
        self.episode_collisions:             int   = 0
        self.episode_fuel_used:              float = 0.0
        self.episode_maneuvers_executed:     int   = 0
        self.episode_secondary_conjunctions: int   = 0
        self.episode_near_misses:            int   = 0
        self._episode_min_separation_km:     float = float("inf")
        self._step:                          int   = 0
        self._collided_satellites:           set   = set()
        self._is_hard:                       bool  = False

    # ──────────────────────────────────────────────────────── reset ──────────

    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset environment for a new episode.

        Returns
        -------
        observations : Dict[agent_id → np.ndarray[float32, (96,)]]
        """
        # Reset realism stateful components (covariance, delay buffers)
        self.realism.reset()

        # Reset step counter and episode metrics
        self._step                          = 0
        self.episode_collisions             = 0
        self.episode_fuel_used              = 0.0
        self.episode_maneuvers_executed     = 0
        self.episode_secondary_conjunctions = 0
        self.episode_near_misses            = 0
        self._episode_min_separation_km     = float("inf")
        self._collided_satellites           = set()
        self._prev_risks                    = {}

        # Decide difficulty tier for this episode
        self._is_hard = (
            self.num_debris >= 15
            or (self.hard_scenario_prob > 0.0 and np.random.rand() < self.hard_scenario_prob)
        )

        # Generate orbital scenario
        self._generate_scenario()

        return self._build_observations()

    # ──────────────────────────────────────────────────────── step ───────────

    def step(
        self,
        actions: Dict[str, Tuple[int, float]],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, Any],
    ]:
        """
        Advance the environment by one timestep (dt seconds).

        Parameters
        ----------
        actions : {agent_id: (direction_idx: int, magnitude_kms: float)}

        Returns
        -------
        observations : Dict[agent_id → np.ndarray]
        rewards      : Dict[agent_id → float]
        dones        : Dict[agent_id → bool]  (includes "__all__" key)
        info         : Dict[str → Any]
        """
        self._step += 1
        rewards: Dict[str, float] = {aid: 0.0 for aid in self.agent_ids}

        # ── Phase 1: Apply maneuvers ─────────────────────────────────────────
        for agent_id in self.agent_ids:
            if agent_id in self._collided_satellites:
                continue

            raw = actions.get(agent_id, (0, 0.0))
            dir_idx   = int(raw[0])
            magnitude = float(raw[1]) if len(raw) > 1 else 0.0

            # Realism: 1-step maneuver delay (slew / pointing time)
            dir_idx = int(self.realism.delay.submit(agent_id, dir_idx))

            # Realism: reaction delay — force NO_OP near imminent TCA
            min_tca = self._min_tca(agent_id)
            if self.realism.reaction_delay.should_delay(min_tca):
                dir_idx = 0

            # Realism: decision noise — random action substitution
            dir_idx = self.realism.decision_noise.maybe_randomize(dir_idx)

            maneuver_type = ManeuverType(dir_idx)
            if maneuver_type == ManeuverType.NO_OP:
                continue

            # CBF safety projection before executing
            cbf_threats = self._cbf_threat_list(agent_id)
            if cbf_threats:
                direction_unit = self.engine._get_maneuver_direction(
                    self._sat_vel[agent_id], maneuver_type
                )
                proposed_dv = direction_unit * magnitude
                safe_dv = self.cbf.filter_action(
                    state=np.concatenate(
                        [self._sat_pos[agent_id], self._sat_vel[agent_id]]
                    ),
                    action_dv=proposed_dv,
                    threats=cbf_threats,
                    max_dv=self.engine.max_delta_v,
                )
                magnitude = float(np.linalg.norm(safe_dv))
                if magnitude < 1e-10:
                    continue  # CBF projected to zero → skip

            # Execute maneuver
            noise_scale = (
                self.realism.config.maneuver_noise_scale
                if self.realism.config.enabled and self.realism.config.maneuver_noise
                else 0.0
            )
            result = self.engine.apply_discrete_maneuver(
                position=self._sat_pos[agent_id],
                velocity=self._sat_vel[agent_id],
                maneuver_type=maneuver_type,
                fuel_available=self._sat_fuel[agent_id],
                dt=self.dt,
                propagate_position=False,
                magnitude=magnitude,
                noise_scale=noise_scale,
            )

            if result.success:
                self._sat_vel[agent_id]  = result.new_velocity
                self._sat_fuel[agent_id] = max(0.0, self._sat_fuel[agent_id] - result.fuel_consumed)
                self.episode_fuel_used  += result.fuel_consumed
                self.episode_maneuvers_executed += 1

                rewards[agent_id] += RW_FUEL_PER_KG * result.fuel_consumed
                rewards[agent_id] += RW_MANEUVER

                # Tracking update resets covariance growth for this agent
                self.realism.covariance.reset_agent(agent_id)

        # ── Phase 2: Propagate all orbits ────────────────────────────────────
        for agent_id in self.agent_ids:
            if agent_id not in self._collided_satellites:
                p, v = self._two_body_step(
                    self._sat_pos[agent_id], self._sat_vel[agent_id], self.dt
                )
                self._sat_pos[agent_id] = p
                self._sat_vel[agent_id] = v

        for deb_id in self.debris_ids:
            p, v = self._two_body_step(
                self._deb_pos[deb_id], self._deb_vel[deb_id], self.dt
            )
            self._deb_pos[deb_id] = p
            self._deb_vel[deb_id] = v

        # ── Phase 3: Detect conjunctions & shape rewards ─────────────────────
        state_dict    = self._full_state_dict()
        alerts        = self.detector.detect(state_dict, _SCENARIO_EPOCH, interest_ids=self.agent_ids)
        current_risks = {aid: 0.0 for aid in self.agent_ids}

        for alert in alerts:
            # Identify which satellite is involved
            sat_id = self._sat_in_alert(alert)
            if sat_id is None or sat_id in self._collided_satellites:
                continue

            self._episode_min_separation_km = min(
                self._episode_min_separation_km, alert.distance_km
            )
            current_risks[sat_id] = max(current_risks[sat_id], alert.risk_score)

            if alert.is_collision:
                # Satellite destroyed
                self._collided_satellites.add(sat_id)
                self.episode_collisions += 1
                rewards[sat_id] += RW_COLLISION

            elif alert.distance_km < NEAR_MISS_THRESHOLD_KM:
                self.episode_near_misses += 1
                rewards[sat_id] += RW_NEAR_MISS

                # TCA urgency penalty: higher penalty when conjunction is imminent
                urgency = float(
                    np.clip(1.0 - alert.time_to_closest_approach_s / 3600.0, 0.0, 1.0)
                )
                rewards[sat_id] += RW_TCA_URGENCY * urgency

                # Miss-distance proximity penalty: closer miss → bigger penalty
                prox = float(
                    np.clip(1.0 - alert.miss_distance_estimate_km / NEAR_MISS_THRESHOLD_KM, 0.0, 1.0)
                )
                rewards[sat_id] += RW_MISS_DIST * prox

                # Secondary conjunction: penalize if this agent's risk is high enough
                # to suggest our maneuver worsened the situation
                if alert.risk_score > self.secondary_risk_thr:
                    self.episode_secondary_conjunctions += 1
                    rewards[sat_id] += RW_SECONDARY

        # Risk-delta reward: reward satellites that reduced their peak risk
        for agent_id in self.agent_ids:
            if agent_id in self._collided_satellites:
                continue
            prev = self._prev_risks.get(agent_id, 0.0)
            curr = current_risks[agent_id]
            if curr < prev:
                rewards[agent_id] += RW_RISK_REDUCTION * (prev - curr)

        self._prev_risks = current_risks

        # ── Phase 4: Update covariance growth ───────────────────────────────
        for agent_id in self.agent_ids:
            if agent_id not in self._collided_satellites:
                self.realism.covariance.update(agent_id, self.dt)

        # ── Phase 5: Build next observations ────────────────────────────────
        observations = self._build_observations()

        # ── Phase 6: Termination ─────────────────────────────────────────────
        all_destroyed = len(self._collided_satellites) >= self.num_satellites
        all_fuel_gone = all(self._sat_fuel.get(aid, 0.0) <= 0.0 for aid in self.agent_ids)
        timeout       = self._step >= self.max_steps
        ep_done       = all_destroyed or all_fuel_gone or timeout

        dones: Dict[str, bool] = {
            aid: (aid in self._collided_satellites or ep_done)
            for aid in self.agent_ids
        }
        dones["__all__"] = ep_done

        info: Dict[str, Any] = {
            # Fields read directly by train.py _run_episode
            "episode_collisions": self.episode_collisions,
            # Fields read by evaluation.compute_detailed_metrics
            "total_collisions":          self.episode_collisions,
            "total_fuel_used":           self.episode_fuel_used,
            "total_maneuvers_executed":  self.episode_maneuvers_executed,
            "total_near_misses":         self.episode_near_misses,
            "min_separation_distance_km":
                self._episode_min_separation_km
                if self._episode_min_separation_km < float("inf")
                else 0.0,
            "episode_min_separation_distance_km":
                self._episode_min_separation_km
                if self._episode_min_separation_km < float("inf")
                else 0.0,
            "tc8_active":         self._is_tc8_like_scenario(),
            "step":               self._step,
            "fuel_remaining":     {aid: self._sat_fuel.get(aid, 0.0) for aid in self.agent_ids},
            "collided_satellites": list(self._collided_satellites),
        }

        return observations, rewards, dones, info

    # ────────────────────────────────────────── public query helpers ──────────

    def _is_tc8_like_scenario(self) -> bool:
        """Return True if this episode is a hard / TC8-like scenario."""
        return self._is_hard

    # ──────────────────────────────────────── scenario generation ────────────

    def _generate_scenario(self) -> None:
        """
        Initialise orbital states for all satellites and debris.

        Satellites: equally spaced around a LEO ring via distinct RAAN / mean
        anomaly offsets so they are well-separated at episode start.

        Debris: each piece is placed close to one of the satellites with a
        relative velocity component aimed to produce a conjunction within
        the episode horizon.  In high-risk / hard mode the offset is smaller
        to guarantee close approaches.
        """
        self.debris_ids = [f"DEB_{j:04d}" for j in range(self.num_debris)]

        # ── Satellites ───────────────────────────────────────────────────────
        for i, agent_id in enumerate(self.agent_ids):
            raan_deg = float(i) * (360.0 / max(self.num_satellites, 1))
            ma_deg   = float(i) * (360.0 / max(self.num_satellites, 1))

            self.propagator.generate_sample_tle(
                object_id=agent_id,
                semi_major_axis_km=LEO_SEMI_MAJOR_KM,
                inclination_deg=LEO_INCLINATION_DEG,
                eccentricity=0.0001,
                epoch_datetime=_SCENARIO_EPOCH,
                mean_anomaly_deg=ma_deg,
                raan_deg=raan_deg,
            )
            state = self.propagator.propagate(agent_id, _SCENARIO_EPOCH)
            self._sat_pos[agent_id] = state.position.copy()
            self._sat_vel[agent_id] = state.velocity.copy()
            self._sat_fuel[agent_id] = self.initial_fuel_kg

        # ── Debris ───────────────────────────────────────────────────────────
        for j, deb_id in enumerate(self.debris_ids):
            target_id = self.agent_ids[j % self.num_satellites]
            dpos, dvel = self._place_debris(
                sat_pos=self._sat_pos[target_id],
                sat_vel=self._sat_vel[target_id],
                debris_index=j,
            )
            self._deb_pos[deb_id] = dpos
            self._deb_vel[deb_id] = dvel

    def _place_debris(
        self,
        sat_pos: np.ndarray,
        sat_vel: np.ndarray,
        debris_index: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (position, velocity) for one debris object in ECI frame.

        Geometry: debris is offset from the satellite by `offset_km` in a
        random direction, then given a relative velocity that brings it
        toward the satellite (conjunction geometry).  A perpendicular
        component prevents perfectly head-on collisions (adds variety).
        """
        rng = np.random.default_rng(seed=debris_index + 1000)

        # Distance to satellite at episode start
        if self._is_hard or self.high_risk_mode:
            # Close initial offset → guaranteed near approach within episode
            offset_km = float(rng.uniform(
                self.distance_threshold * 0.3,
                self.distance_threshold * 4.0,
            ))
        else:
            # Wider spread: approaches may or may not happen in time
            offset_km = float(rng.uniform(
                self.distance_threshold * 2.0,
                self.distance_threshold * 15.0,
            ))

        # Random unit offset direction
        rand_dir = rng.standard_normal(3)
        rand_dir = rand_dir / (np.linalg.norm(rand_dir) + 1e-12)
        deb_pos  = sat_pos + rand_dir * offset_km

        # Relative velocity: approaching along -rand_dir + small perpendicular
        approach_dir = -rand_dir
        perp = rng.standard_normal(3)
        perp -= np.dot(perp, approach_dir) * approach_dir
        perp = perp / (np.linalg.norm(perp) + 1e-12)

        # Hard scenario: faster approach; normal: gentle closure
        if self._is_hard:
            rel_speed = float(rng.uniform(0.003, 0.010))   # 3..10 m/s
        else:
            rel_speed = float(rng.uniform(0.001, 0.005))   # 1..5 m/s

        perp_speed = float(rng.uniform(0.0, rel_speed * 0.3))
        rel_vel    = approach_dir * rel_speed + perp * perp_speed
        deb_vel    = sat_vel + rel_vel

        return deb_pos.astype(np.float64), deb_vel.astype(np.float64)

    # ──────────────────────────────────────── observation builder ─────────────

    def _build_observations(self) -> Dict[str, np.ndarray]:
        """
        Build the 96-dim observation vector for every agent.

        For collided satellites the observation is a zero vector (they are
        excluded from the episode but their slot is kept for the centralized
        critic's fixed-size input).
        """
        observations: Dict[str, np.ndarray] = {}
        state_dict   = self._full_state_dict()

        # Partial observability: some debris are dropped from the catalog
        visible_debris = set(
            self.realism.observability.filter_visible_debris(self.debris_ids)
        )

        for agent_id in self.agent_ids:
            if agent_id in self._collided_satellites:
                observations[agent_id] = np.zeros(OBS_SIZE, dtype=np.float32)
                continue

            # Own (possibly noisy) state
            own_state = state_dict[agent_id]
            noisy     = self.realism.noise.apply(own_state)
            noisy_pos = noisy[:3]
            noisy_vel = noisy[3:6]

            fuel_ratio      = float(np.clip(
                self._sat_fuel[agent_id] / max(self.initial_fuel_kg, 1e-6), 0.0, 1.0
            ))
            step_normalized = float(self._step / max(self.max_steps, 1))

            # Build threat observations from visible debris and other satellites
            threat_source_ids = visible_debris | (
                set(self.agent_ids) - {agent_id} - self._collided_satellites
            )
            other_states = {
                k: state_dict[k]
                for k in threat_source_ids
                if k in state_dict
            }

            raw_alerts = self.detector.detect_for_object(
                agent_id, own_state, other_states, _SCENARIO_EPOCH
            )

            threats: List[ThreatObservation] = []
            for alert in raw_alerts:
                other_id   = (
                    alert.object2_id
                    if alert.object1_id == agent_id
                    else alert.object1_id
                )
                other_raw  = state_dict.get(other_id, np.zeros(6))
                noisy_other = self.realism.noise.apply(other_raw)

                rel_pos = noisy_other[:3] - noisy_pos
                rel_vel = noisy_other[3:6] - noisy_vel

                threats.append(ThreatObservation(
                    rel_pos=rel_pos,
                    rel_vel=rel_vel,
                    distance_km=alert.distance_km,
                    miss_distance_estimate_km=alert.miss_distance_estimate_km,
                    time_to_closest_approach_s=alert.time_to_closest_approach_s,
                    risk_score=alert.risk_score,
                    collision_probability=alert.collision_probability,
                    relative_speed_kms=alert.relative_velocity_kms,
                ))

            ranked = rank_threats(threats)

            # Summary scalars for the own-state block
            if ranked:
                min_miss_km        = min(t.miss_distance_estimate_km for t in ranked)
                max_risk           = max(t.risk_score for t in ranked)
                top3               = ranked[:3]
                combined_risk_top3 = float(
                    np.clip(sum(t.risk_score for t in top3) / max(len(top3), 1), 0.0, 1.0)
                )
                min_tca_top3_s     = min(t.time_to_closest_approach_s for t in top3)
            else:
                min_miss_km        = None
                max_risk           = 0.0
                combined_risk_top3 = 0.0
                min_tca_top3_s     = 3600.0

            observations[agent_id] = encode_observation(
                own_state=np.concatenate([noisy_pos, noisy_vel]),
                fuel_ratio=fuel_ratio,
                step_normalized=step_normalized,
                min_miss_distance_km=min_miss_km,
                max_risk=max_risk,
                combined_risk_top3=combined_risk_top3,
                min_tca_top3_s=min_tca_top3_s,
                threats=ranked,
            )

        return observations

    # ───────────────────────────────────── physics helpers ───────────────────

    @staticmethod
    def _two_body_step(
        pos: np.ndarray, vel: np.ndarray, dt: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advance one object by dt seconds using two-body Euler integration.

        Uses the full Newtonian gravitational acceleration:
            a = -μ/r³ * r_vec
        Position in km, velocity in km/s, dt in seconds.
        """
        r      = np.linalg.norm(pos)
        acc    = -(MU_EARTH / (r ** 3 + 1e-30)) * pos   # km/s²
        new_vel = vel + acc * dt
        new_pos = pos + vel * dt + 0.5 * acc * (dt * dt)
        return new_pos, new_vel

    # ───────────────────────────────────── state dict helpers ────────────────

    def _full_state_dict(self) -> Dict[str, np.ndarray]:
        """Return {id: [pos(3), vel(3)]} for all active objects."""
        states: Dict[str, np.ndarray] = {}
        for agent_id in self.agent_ids:
            if agent_id not in self._collided_satellites:
                states[agent_id] = np.concatenate(
                    [self._sat_pos[agent_id], self._sat_vel[agent_id]]
                )
        for deb_id in self.debris_ids:
            states[deb_id] = np.concatenate(
                [self._deb_pos[deb_id], self._deb_vel[deb_id]]
            )
        return states

    def _sat_in_alert(self, alert: ConjunctionAlert) -> Optional[str]:
        """Return the satellite agent_id involved in a conjunction alert, or None."""
        if alert.object1_id in self.agent_ids:
            return alert.object1_id
        if alert.object2_id in self.agent_ids:
            return alert.object2_id
        return None

    # ───────────────────────────────────── CBF / realism helpers ─────────────

    def _cbf_threat_list(self, agent_id: str) -> List[Dict[str, Any]]:
        """
        Build the threat-descriptor list consumed by CBFSafetyFilter.filter_action.
        Only debris (not other satellites) are considered for CBF projection.
        """
        pos = self._sat_pos[agent_id]
        vel = self._sat_vel[agent_id]
        sat_state = np.concatenate([pos, vel])

        other_states = {
            deb_id: np.concatenate([self._deb_pos[deb_id], self._deb_vel[deb_id]])
            for deb_id in self.debris_ids
        }
        alerts = self.detector.detect_for_object(
            agent_id, sat_state, other_states, _SCENARIO_EPOCH
        )

        threats = []
        for alert in alerts:
            other_id    = (
                alert.object2_id if alert.object1_id == agent_id else alert.object1_id
            )
            other_state = other_states.get(other_id, np.zeros(6))
            threats.append({
                "rel_pos":                  other_state[:3] - pos,
                "rel_vel":                  other_state[3:6] - vel,
                "distance_km":              alert.distance_km,
                "risk_score":               alert.risk_score,
                "time_to_closest_approach_s": alert.time_to_closest_approach_s,
            })
        return threats

    def _min_tca(self, agent_id: str) -> float:
        """Return minimum TCA (seconds) across all debris threats for agent."""
        threats = self._cbf_threat_list(agent_id)
        if not threats:
            return float("inf")
        return float(min(t["time_to_closest_approach_s"] for t in threats))
