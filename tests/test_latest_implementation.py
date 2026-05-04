import unittest
from dataclasses import replace
from datetime import datetime

import numpy as np

from env.ma_env import MultiAgentOrbitalEnv
from experiments.run_collision_avoidance_tests import build_test_cases, run_policy_on_scenario
from policies.policy_interface import FuelAwareThresholdRulePolicy, ThresholdRulePolicy
from sim.conjunction_detector import ConjunctionAlert
from sim.maneuver_engine import EMERGENCY_ACTION_INDEX
from sim.observation_utils import OBS_SIZE, ThreatObservation, decode_observation, encode_observation, rank_threats
from sim.orbit_propagator import OrbitalState
from sim.simulator import SimulationRunner


def _make_observation(
    *,
    threats,
    fuel_ratio=1.0,
    max_risk=0.0,
):
    own_state = np.array([7000.0, 0.0, 0.0, 0.0, 7.5, 0.0], dtype=np.float64)
    ranked = rank_threats(list(threats))
    return encode_observation(
        own_state=own_state,
        fuel_ratio=fuel_ratio,
        step_normalized=0.25,
        min_miss_distance_km=min((t.miss_distance_estimate_km for t in threats), default=100.0),
        max_risk=max_risk,
        combined_risk_top3=sum(t.collision_probability for t in ranked[:3]),
        min_tca_top3_s=min((t.time_to_closest_approach_s for t in ranked[:3]), default=3600.0),
        threats=list(threats),
    )


def _threat(
    *,
    rel_pos,
    rel_vel,
    miss_km,
    tca_s,
    risk,
    pc,
):
    rel_pos_arr = np.asarray(rel_pos, dtype=np.float64)
    rel_vel_arr = np.asarray(rel_vel, dtype=np.float64)
    return ThreatObservation(
        rel_pos=rel_pos_arr,
        rel_vel=rel_vel_arr,
        distance_km=float(np.linalg.norm(rel_pos_arr)),
        miss_distance_estimate_km=float(miss_km),
        time_to_closest_approach_s=float(tca_s),
        risk_score=float(risk),
        collision_probability=float(pc),
        relative_speed_kms=float(np.linalg.norm(rel_vel_arr)),
    )


class ObservationAndPolicyTests(unittest.TestCase):
    def test_encode_decode_round_trip_schema(self):
        primary = _threat(
            rel_pos=[1.0, 2.0, 0.0],
            rel_vel=[0.01, -0.02, 0.0],
            miss_km=0.8,
            tca_s=120.0,
            risk=0.92,
            pc=0.81,
        )
        secondary = _threat(
            rel_pos=[6.0, 0.0, 0.0],
            rel_vel=[0.0, -0.01, 0.0],
            miss_km=4.0,
            tca_s=900.0,
            risk=0.2,
            pc=0.1,
        )
        obs = _make_observation(threats=[primary, secondary], fuel_ratio=0.7, max_risk=0.92)

        self.assertEqual(obs.shape[0], OBS_SIZE)

        decoded = decode_observation(obs)
        self.assertAlmostEqual(decoded["fuel_ratio"], 0.7, places=5)
        self.assertAlmostEqual(decoded["max_risk"], 0.92, places=5)
        self.assertAlmostEqual(decoded["combined_risk_top3"], 0.91, places=5)
        self.assertAlmostEqual(decoded["min_tca_top3_s"], 120.0, places=4)
        self.assertEqual(len(decoded["threats"]), 2)
        np.testing.assert_allclose(decoded["threats"][0].rel_vel, primary.rel_vel, atol=1e-6)

        ranked = rank_threats(decoded["threats"])
        self.assertGreaterEqual(ranked[0].collision_probability, ranked[1].collision_probability)

    def test_threshold_rule_uses_emergency_action_for_urgent_threat(self):
        urgent = _threat(
            rel_pos=[0.7, 0.0, 0.0],
            rel_vel=[-0.03, 0.0, 0.0],
            miss_km=0.4,
            tca_s=90.0,
            risk=0.99,
            pc=0.95,
        )
        obs = _make_observation(threats=[urgent], fuel_ratio=1.0, max_risk=0.99)
        policy = ThresholdRulePolicy(threshold_km=5.0, dv_action=1)

        action = policy.select_action(obs, "SAT_000")
        self.assertEqual(action, EMERGENCY_ACTION_INDEX)

    def test_fuel_aware_threshold_policy_respects_fuel_gate(self):
        close_non_emergency = _threat(
            rel_pos=[1.0, 0.0, 0.0],
            rel_vel=[-0.005, 0.0, 0.0],
            miss_km=1.0,
            tca_s=300.0,
            risk=0.4,
            pc=0.2,
        )
        policy = FuelAwareThresholdRulePolicy(threshold_km=1.5, dv_action=4, min_fuel_ratio=0.2)

        low_fuel_obs = _make_observation(threats=[close_non_emergency], fuel_ratio=0.1, max_risk=0.4)
        high_fuel_obs = _make_observation(threats=[close_non_emergency], fuel_ratio=0.8, max_risk=0.4)

        self.assertEqual(policy.select_action(low_fuel_obs, "SAT_000"), 0)
        self.assertEqual(policy.select_action(high_fuel_obs, "SAT_000"), 4)


class RegressionTests(unittest.TestCase):
    def test_predictive_risk_decreases_with_distance_and_tca(self):
        env = MultiAgentOrbitalEnv(epoch_datetime=datetime(2020, 1, 1))

        high_risk = env.compute_collision_risk(10.0, 60.0)
        lower_distance_risk = env.compute_collision_risk(100.0, 60.0)
        lower_tca_risk = env.compute_collision_risk(10.0, 600.0)

        self.assertGreater(high_risk, lower_distance_risk)
        self.assertGreater(high_risk, lower_tca_risk)

    def test_high_risk_initialization_avoids_sat_debris_overlap(self):
        env = MultiAgentOrbitalEnv(
            num_satellites=3,
            num_debris=20,
            collision_threshold_km=5.0,
            distance_threshold_km=250.0,
            high_risk_mode=True,
            dt=60.0,
            orbit_altitude_km=600.0,
            epoch_datetime=datetime(2020, 1, 1),
        )
        env.reset()

        states = env._current_object_states()
        sat_ids = [k for k in states if k.startswith("SAT_")]
        deb_ids = [k for k in states if k.startswith("DEB_")]

        overlapping_pairs = []
        for sat_id in sat_ids:
            for deb_id in deb_ids:
                distance_km = float(np.linalg.norm(states[sat_id][:3] - states[deb_id][:3]))
                if distance_km < env.detector.collision_threshold:
                    overlapping_pairs.append((sat_id, deb_id, distance_km))

        self.assertEqual(
            overlapping_pairs,
            [],
            msg=f"Found sat-debris pairs inside collision threshold at reset: {overlapping_pairs}",
        )

    def test_collision_limit_scales_with_fleet_size(self):
        env = MultiAgentOrbitalEnv(
            num_satellites=50,
            num_debris=1,
            epoch_datetime=datetime(2020, 1, 1),
        )
        env.reset()
        env.step_count = 1

        env.episode_collisions = 5
        self.assertFalse(env._compute_dones()["__all__"])

        env.episode_collisions = 100
        self.assertTrue(env._compute_dones()["__all__"])

    def test_collision_counter_ignores_debris_only_collisions(self):
        env = MultiAgentOrbitalEnv(
            num_satellites=1,
            num_debris=2,
            collision_threshold_km=5.0,
            distance_threshold_km=10.0,
            epoch_datetime=datetime(2020, 1, 1),
        )
        env.reset()

        env.agents["SAT_000"].orbital_state = OrbitalState(
            position=np.array([100.0, 0.0, 0.0], dtype=np.float64),
            velocity=np.zeros(3, dtype=np.float64),
            timestamp=env.current_time,
            object_id="SAT_000",
        )
        env.agents["DEB_000"].orbital_state = OrbitalState(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            velocity=np.zeros(3, dtype=np.float64),
            timestamp=env.current_time,
            object_id="DEB_000",
        )
        env.agents["DEB_001"].orbital_state = OrbitalState(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            velocity=np.zeros(3, dtype=np.float64),
            timestamp=env.current_time,
            object_id="DEB_001",
        )

        env.episode_collisions = 0
        alerts = env._detect_conjunctions()
        self.assertEqual(env.episode_collisions, 0)

        env.agents["SAT_000"].orbital_state = OrbitalState(
            position=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            velocity=np.zeros(3, dtype=np.float64),
            timestamp=env.current_time,
            object_id="SAT_000",
        )
        env.agents["DEB_001"].orbital_state = OrbitalState(
            position=np.array([20.0, 0.0, 0.0], dtype=np.float64),
            velocity=np.zeros(3, dtype=np.float64),
            timestamp=env.current_time,
            object_id="DEB_001",
        )

        env.episode_collisions = 0
        env._detect_conjunctions()
        self.assertGreaterEqual(env.episode_collisions, 1)

    def test_secondary_conjunctions_only_count_for_maneuvering_satellite_alerts(self):
        env = MultiAgentOrbitalEnv(
            num_satellites=1,
            num_debris=2,
            near_miss_distance_km=10.0,
            epoch_datetime=datetime(2020, 1, 1),
        )
        env.reset()
        env._maneuvering_satellites_this_step = {"SAT_000"}
        env._maneuvering_satellites_prev_step = set()

        debris_only_alert = ConjunctionAlert(
            object1_id="DEB_000",
            object2_id="DEB_001",
            distance_km=5.0,
            miss_distance_estimate_km=4.5,
            time_to_closest_approach_s=120.0,
            relative_velocity_kms=0.8,
            risk_score=0.3,
            collision_probability=0.05,
            is_collision=False,
            timestamp=env.current_time,
            alert_id="ALERT_DEB_DEB",
        )
        sat_alert = ConjunctionAlert(
            object1_id="SAT_000",
            object2_id="DEB_000",
            distance_km=5.0,
            miss_distance_estimate_km=4.0,
            time_to_closest_approach_s=90.0,
            relative_velocity_kms=1.0,
            risk_score=0.4,
            collision_probability=0.08,
            is_collision=False,
            timestamp=env.current_time,
            alert_id="ALERT_SAT_DEB",
        )

        env.detector.detect = lambda object_states, timestamp, interest_ids=None: [debris_only_alert]
        env._detect_conjunctions()
        self.assertEqual(env._secondary_conjunctions_this_step, 0)

        env.detector.detect = lambda object_states, timestamp, interest_ids=None: [sat_alert]
        env._detect_conjunctions()
        self.assertEqual(env._secondary_conjunctions_this_step, 1)

    def test_reward_maneuver_penalty_is_applied_per_step_not_cumulative(self):
        env = MultiAgentOrbitalEnv(
            num_satellites=1,
            num_debris=1,
            epoch_datetime=datetime(2020, 1, 1),
        )
        env.reset()

        sat_id = "SAT_000"
        env.agents[sat_id].maneuvers_executed = 50

        pre_summary = {
            sat_id: {
                "min_miss_distance_km": 0.05,
                "min_distance_km": 0.05,
                "min_tca_s": 120.0,
            }
        }
        post_summary = {
            sat_id: {
                "min_miss_distance_km": 0.05,
                "min_distance_km": 0.05,
                "min_tca_s": 120.0,
            }
        }

        env._maneuvering_satellites_this_step = {sat_id}
        reward_with_step_maneuver = env._compute_rewards(
            actions={sat_id: 1},
            alerts=[],
            pre_risk_summary=pre_summary,
            post_risk_summary=post_summary,
        )[sat_id]

        env._maneuvering_satellites_this_step = set()
        reward_without_step_maneuver = env._compute_rewards(
            actions={sat_id: 0},
            alerts=[],
            pre_risk_summary=pre_summary,
            post_risk_summary=post_summary,
        )[sat_id]

        env.agents[sat_id].prev_action = None
        self.assertAlmostEqual(
            reward_with_step_maneuver - reward_without_step_maneuver,
            env.reward_weights["maneuver_count"],
            places=6,
        )

    def test_reward_uses_aggregated_total_risk_delta(self):
        env = MultiAgentOrbitalEnv(
            num_satellites=1,
            num_debris=2,
            epoch_datetime=datetime(2020, 1, 1),
        )
        env.reset()

        sat_id = "SAT_000"
        pre_summary = {
            sat_id: {
                "total_risk": 0.7,
                "min_miss_distance_km": 10.0,
                "min_distance_km": 10.0,
                "min_tca_s": 1200.0,
            }
        }
        post_summary = {
            sat_id: {
                "total_risk": 0.2,
                "min_miss_distance_km": 10.0,
                "min_distance_km": 10.0,
                "min_tca_s": 1200.0,
            }
        }

        reward = env._compute_rewards(
            actions={sat_id: 0},
            alerts=[],
            pre_risk_summary=pre_summary,
            post_risk_summary=post_summary,
        )[sat_id]

        self.assertAlmostEqual(
            reward,
            (
                env.reward_weights["risk_delta"]
                + env.reward_weights["high_risk_risk_delta"]
            )
            * (0.7 - 0.2),
            places=6,
        )

    def test_safe_no_op_reward_and_jitter_penalty_are_applied(self):
        env = MultiAgentOrbitalEnv(
            num_satellites=1,
            num_debris=1,
            epoch_datetime=datetime(2020, 1, 1),
        )
        env.reset()

        sat_id = "SAT_000"
        env.agents[sat_id].prev_action = 1

        safe_summary = {
            sat_id: {
                "min_miss_distance_km": 2.5,
                "min_distance_km": 2.5,
                "min_tca_s": 1200.0,
            }
        }

        reward = env._compute_rewards(
            actions={sat_id: 0},
            alerts=[],
            pre_risk_summary=safe_summary,
            post_risk_summary=safe_summary,
        )[sat_id]

        expected = env.reward_weights["safe_no_op"] + env.reward_weights["jitter"]
        self.assertAlmostEqual(reward, expected, places=6)

    def test_secondary_penalty_uses_maneuver_link_and_pc_threshold(self):
        env = MultiAgentOrbitalEnv(
            num_satellites=2,
            num_debris=1,
            near_miss_distance_km=10.0,
            secondary_conjunction_risk_threshold=0.1,
            epoch_datetime=datetime(2020, 1, 1),
        )
        env.reset()

        sat0 = "SAT_000"
        sat1 = "SAT_001"

        base_alert = ConjunctionAlert(
            object1_id=sat0,
            object2_id="DEB_000",
            distance_km=5.0,
            miss_distance_estimate_km=4.0,
            time_to_closest_approach_s=120.0,
            relative_velocity_kms=1.0,
            risk_score=0.5,
            collision_probability=0.2,
            is_collision=False,
            timestamp=env.current_time,
            alert_id="ALERT_BASE",
        )

        pre_summary = {sat0: {"total_risk": 0.0}, sat1: {"total_risk": 0.0}}
        post_summary = {sat0: {"total_risk": 0.0}, sat1: {"total_risk": 0.0}}

        env._maneuvering_satellites_this_step = {sat1}
        env._maneuvering_satellites_prev_step = set()
        reward_unlinked = env._compute_rewards(
            actions={sat0: 0, sat1: 1},
            alerts=[base_alert],
            pre_risk_summary=pre_summary,
            post_risk_summary=post_summary,
        )[sat0]

        env._maneuvering_satellites_this_step = {sat1}
        env._maneuvering_satellites_prev_step = {sat0}
        low_pc_alert = ConjunctionAlert(
            object1_id=sat0,
            object2_id="DEB_000",
            distance_km=5.0,
            miss_distance_estimate_km=4.0,
            time_to_closest_approach_s=120.0,
            relative_velocity_kms=1.0,
            risk_score=0.5,
            collision_probability=0.05,
            is_collision=False,
            timestamp=env.current_time,
            alert_id="ALERT_LOW_PC",
        )
        reward_low_pc = env._compute_rewards(
            actions={sat0: 0, sat1: 1},
            alerts=[low_pc_alert],
            pre_risk_summary=pre_summary,
            post_risk_summary=post_summary,
        )[sat0]

        high_pc_alert = ConjunctionAlert(
            object1_id=sat0,
            object2_id="DEB_000",
            distance_km=5.0,
            miss_distance_estimate_km=4.0,
            time_to_closest_approach_s=120.0,
            relative_velocity_kms=1.0,
            risk_score=0.5,
            collision_probability=0.2,
            is_collision=False,
            timestamp=env.current_time,
            alert_id="ALERT_HIGH_PC",
        )
        reward_high_pc = env._compute_rewards(
            actions={sat0: 0, sat1: 1},
            alerts=[high_pc_alert],
            pre_risk_summary=pre_summary,
            post_risk_summary=post_summary,
        )[sat0]

        self.assertAlmostEqual(reward_unlinked, reward_low_pc, places=6)
        self.assertAlmostEqual(
            reward_high_pc - reward_low_pc,
            env.reward_weights["secondary_conjunction"],
            places=6,
        )


class ScenarioCoverageTests(unittest.TestCase):
    EXPECTED_CASES = {
        "TC1_no_maneuver",
        "TC2_threshold_rule",
        "TC3_fuel_aware_rule",
        "TC4_marl",
        "TC5_high_density_stress",
        "TC6_fuel_constrained",
        "TC7_secondary_conjunctions",
        "TC8_hypothetical_collision_cluster",
    }

    REPRESENTATIVE_POLICIES = {
        "TC1_no_maneuver": "no_op",
        "TC2_threshold_rule": "threshold_rule",
        "TC3_fuel_aware_rule": "fuel_aware_threshold_rule",
        "TC4_marl": "rule_based",
        "TC5_high_density_stress": "baseline",
        "TC6_fuel_constrained": "fuel_aware_threshold_rule",
        "TC7_secondary_conjunctions": "threshold_rule",
        "TC8_hypothetical_collision_cluster": "fuel_aware_threshold_rule",
    }

    def test_all_expected_scenarios_exist(self):
        scenarios = build_test_cases(max_debris=5)
        self.assertEqual(set(scenarios.keys()), self.EXPECTED_CASES)

    def test_each_scenario_executes_with_representative_policy(self):
        scenarios = build_test_cases(max_debris=5)
        base_epoch = datetime(2020, 1, 1)

        for case_name in sorted(self.EXPECTED_CASES):
            with self.subTest(case_name=case_name):
                scenario = replace(scenarios[case_name], dt_sec=3600.0)
                policy = self.REPRESENTATIVE_POLICIES[case_name]

                stats = run_policy_on_scenario(
                    scenario=scenario,
                    policy_type=policy,
                    mc_idx=0,
                    base_epoch=base_epoch,
                    run_seed=123,
                    include_marl=False,
                    marl_trainer=None,
                )

                self.assertIn("total_collisions", stats)
                self.assertIn("total_fuel_used", stats)
                self.assertIn("total_maneuvers_executed", stats)
                self.assertIn("total_secondary_conjunctions", stats)
                self.assertIn("total_near_misses", stats)
                self.assertIn("min_separation_distance_km", stats)
                self.assertIn("final_step", stats)
                self.assertGreaterEqual(stats["final_step"], 0)
                self.assertTrue(np.isfinite(float(stats["total_fuel_used"])))

    def test_all_cases_use_predictive_multi_object_scenario_configs(self):
        scenarios = build_test_cases(max_debris=5)

        for case_name, scenario in scenarios.items():
            with self.subTest(case_name=case_name):
                self.assertTrue(bool(scenario.description))
                self.assertIsNotNone(scenario.scenario_config)
                self.assertIn(scenario.scenario_family, {"normal", "hybrid", "hard"})
                self.assertTrue(scenario.scenario_config.get("multi_object", False))
                self.assertGreaterEqual(len(scenario.scenario_config.get("cluster_offsets", [])), 1)
                self.assertIn("conjunction_info", scenario.scenario_config)
                self.assertIn("raw_features", scenario.scenario_config)

    def test_tc1_to_tc6_have_predictive_warning_windows(self):
        scenarios = build_test_cases(max_debris=6)
        normal_or_hybrid_cases = [
            "TC1_no_maneuver",
            "TC2_threshold_rule",
            "TC3_fuel_aware_rule",
            "TC4_marl",
            "TC5_high_density_stress",
            "TC6_fuel_constrained",
        ]

        for case_name in normal_or_hybrid_cases:
            with self.subTest(case_name=case_name):
                scenario = scenarios[case_name]
                conjunction = scenario.scenario_config["conjunction_info"]
                self.assertGreaterEqual(conjunction["miss_distance"], 150.0)
                self.assertGreaterEqual(conjunction["time_to_tca"], 360.0)
                self.assertGreaterEqual(conjunction["risk_score"], 1e-4)
                self.assertLessEqual(conjunction["risk_score"], 0.1)

    def test_tc5_stress_and_tc6_fuel_constraints_match_latest_intent(self):
        scenarios = build_test_cases(max_debris=6)
        tc5 = scenarios["TC5_high_density_stress"]
        tc6 = scenarios["TC6_fuel_constrained"]

        self.assertEqual(tc5.num_satellites, 50)
        self.assertGreaterEqual(tc5.num_debris, 6)
        self.assertTrue(tc5.use_high_risk_mode)
        self.assertGreaterEqual(tc5.distance_threshold_km, 400.0)

        self.assertEqual(tc6.scenario_family, "hybrid")
        self.assertLessEqual(tc6.policy_params["fuel_kg"], 1.0)
        self.assertGreaterEqual(tc6.policy_params["min_fuel_ratio"], 0.35)
        self.assertLessEqual(tc6.collision_threshold_km, 0.10)

    def test_tc7_secondary_case_uses_multi_object_pc_thresholds(self):
        scenario = build_test_cases(max_debris=5)["TC7_secondary_conjunctions"]
        cluster_offsets = scenario.scenario_config.get("cluster_offsets", [])

        self.assertAlmostEqual(scenario.secondary_conjunction_risk_threshold, 0.01, places=6)
        self.assertIsNotNone(scenario.scenario_config)
        self.assertEqual(scenario.scenario_family, "normal")
        self.assertTrue(scenario.scenario_config.get("multi_object", False))
        self.assertGreaterEqual(len(cluster_offsets), 2)
        self.assertLessEqual(scenario.collision_threshold_km, 0.05)
        for entry in cluster_offsets:
            self.assertGreater(entry["conjunction_info"]["miss_distance"], scenario.collision_threshold_km * 1000.0)

    def test_tc8_case_matches_predictive_hard_cluster_profile(self):
        scenario = build_test_cases(max_debris=5)["TC8_hypothetical_collision_cluster"]
        cluster_offsets = scenario.scenario_config.get("cluster_offsets", [])

        self.assertGreaterEqual(scenario.num_debris, 3)
        self.assertEqual(scenario.scenario_family, "hard")
        self.assertAlmostEqual(scenario.secondary_conjunction_risk_threshold, 0.01, places=6)
        self.assertGreaterEqual(scenario.distance_threshold_km, 500.0)
        self.assertLessEqual(scenario.collision_threshold_km, 0.025)
        self.assertTrue(scenario.scenario_config.get("multi_object", False))
        self.assertGreaterEqual(len(cluster_offsets), 1)

        for entry in cluster_offsets:
            conjunction = entry["conjunction_info"]
            self.assertLessEqual(conjunction["miss_distance"], 50.0)
            self.assertLessEqual(conjunction["time_to_tca"], 300.0)
            self.assertGreaterEqual(conjunction["relative_speed"], 2500.0)
            self.assertGreaterEqual(conjunction["risk_score"], 1e-3)

    def test_reset_can_inject_hard_cluster_scenario(self):
        env = MultiAgentOrbitalEnv(
            num_satellites=3,
            num_debris=30,
            hard_scenario_probability=1.0,
            epoch_datetime=datetime(2020, 1, 1),
        )

        env.reset()

        self.assertIsNotNone(env.active_scenario_config)
        self.assertTrue(env.active_scenario_config.get("multi_object", False))
        self.assertIn("cluster_offsets", env.active_scenario_config)
        self.assertGreaterEqual(len(env.active_scenario_config["cluster_offsets"]), 10)
        self.assertLessEqual(len(env.active_scenario_config["cluster_offsets"]), 25)

    def test_hard_reset_override_can_supersede_normal_scenario(self):
        env = MultiAgentOrbitalEnv(
            num_satellites=3,
            num_debris=30,
            scenario_config={"name": "normal_scenario", "high_risk_mode": True},
            hard_scenario_probability=1.0,
            epoch_datetime=datetime(2020, 1, 1),
        )

        env.reset()

        self.assertTrue(env.active_scenario_config.get("multi_object", False))
        self.assertAlmostEqual(env.detector.collision_threshold, 0.005, places=6)
        self.assertGreaterEqual(env.detector.distance_threshold, 500.0)

    def test_simulation_runner_forwards_hard_scenario_probability(self):
        runner = SimulationRunner(
            num_satellites=3,
            num_debris=20,
            policy_type="no_op",
            enable_logging=False,
            hard_scenario_probability=1.0,
        )

        stats = runner.run_episode(max_steps=1, verbose=False)

        self.assertTrue(stats["tc8_active"])
        self.assertTrue(runner.env.active_scenario_config.get("multi_object", False))
        self.assertGreaterEqual(len(runner.env.active_scenario_config.get("cluster_offsets", [])), 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
