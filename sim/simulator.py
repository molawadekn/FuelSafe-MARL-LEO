"""
MODULE 9: Core Simulation Loop
Orchestrates the full simulation with orbit propagation, collision detection,
maneuvers, reward computation, and logging.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

from env.ma_env import MultiAgentOrbitalEnv
from safety.cbf_filter import CBFSafetyFilter
from policies.policy_interface import (
    PolicyManager,
    BaselinePolicy,
    RuleBasedPolicy,
    NoOpPolicy,
    ThresholdRulePolicy,
    FuelAwareThresholdRulePolicy,
    MARLPolicy,
)


class SimulationLogger:
    """Logs simulation data for analysis."""
    
    def __init__(self, output_dir: str = 'outputs'):
        """
        Initialize logger.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.timesteps = []
        self.collisions = []
        self.fuel_used = []
        self.alerts = []
        self.maneuvers = []

    def reset(self) -> None:
        """Clear in-memory log buffers before a fresh run."""
        self.timesteps = []
        self.collisions = []
        self.fuel_used = []
        self.alerts = []
        self.maneuvers = []
    
    def log_timestep(self, timestep: int, collisions: int, 
                    fuel_used: float, num_alerts: int) -> None:
        """Log one timestep."""
        self.timesteps.append(timestep)
        self.collisions.append(collisions)
        self.fuel_used.append(fuel_used)
        self.alerts.append(num_alerts)
    
    def log_maneuver(self, agent_id: str, action: int, delta_v: float,
                    fuel_consumed: float) -> None:
        """Log maneuver execution."""
        self.maneuvers.append({
            'agent_id': agent_id,
            'action': action,
            'delta_v': delta_v,
            'fuel_consumed': fuel_consumed,
            'timestamp': datetime.utcnow()
        })
    
    def save_to_csv(self, filename: str = 'simulation_log.csv') -> None:
        """Save logs to CSV."""
        filepath = self.output_dir / filename
        
        df = pd.DataFrame({
            'timesteps': self.timesteps,
            'collisions': self.collisions,
            'fuel_used': self.fuel_used,
            'alerts': self.alerts
        })
        
        df.to_csv(filepath, index=False)
        print(f"Saved simulation log to {filepath}")
    
    def save_maneuvers_to_csv(self, filename: str = 'maneuvers_log.csv') -> None:
        """Save maneuver logs to CSV."""
        if not self.maneuvers:
            return
        
        filepath = self.output_dir / filename
        df = pd.DataFrame(self.maneuvers)
        df.to_csv(filepath, index=False)
        print(f"Saved maneuvers log to {filepath}")


class SimulationRunner:
    """
    Main simulation runner.
    Coordinates environment, policies, safety filters, and logging.
    """
    
    def __init__(self,
                 num_satellites: int = 3,
                 num_debris: int = 5,
                 use_safety_filter: bool = True,
                 safety_threshold_km: float = 0.1,
                 distance_threshold_km: float = 50.0,
                 collision_threshold_km: float = 1.0,
                 high_risk_mode: bool = False,
                 policy_type: str = 'baseline',
                 enable_logging: bool = True,
                 dt_sec: float = 60.0,
                 orbit_altitude_km: float = 600.0,
                 epoch_datetime: Optional[datetime] = None,
                 initial_fuel_kg: float = 1000.0,
                 max_fuel_kg: float = 1000.0,
                 near_miss_distance_km: Optional[float] = None,
                 secondary_conjunction_risk_threshold: float = 0.5,
                 policy_kwargs: Optional[Dict] = None,
                 marl_trainer: Optional[object] = None,
                 scenario_config: Optional[Dict[str, Any]] = None):
        """
        Initialize simulation runner.
        
        Args:
            num_satellites: Number of satellites
            num_debris: Number of debris objects
            use_safety_filter: Whether to use CBF safety filter
            safety_threshold_km: Safety distance threshold
            policy_type: Type of policy ('baseline', 'rule_based', 'marl', 'random')
            enable_logging: Whether to enable logging
        """
        self.num_satellites = num_satellites
        self.num_debris = num_debris
        self.policy_type = policy_type
        self.policy_kwargs = policy_kwargs or {}
        self.marl_trainer = marl_trainer
        self.scenario_config = scenario_config
        
        # Environment
        self.env = MultiAgentOrbitalEnv(
            num_satellites=num_satellites,
            num_debris=num_debris,
            distance_threshold_km=distance_threshold_km,
            collision_threshold_km=collision_threshold_km,
            high_risk_mode=high_risk_mode,
            dt=dt_sec,
            initial_fuel_kg=initial_fuel_kg,
            max_fuel_kg=max_fuel_kg,
            epoch_datetime=epoch_datetime,
            orbit_altitude_km=orbit_altitude_km,
            near_miss_distance_km=near_miss_distance_km,
            secondary_conjunction_risk_threshold=secondary_conjunction_risk_threshold,
            scenario_config=scenario_config,
        )
        
        # Safety filter
        self.use_safety_filter = use_safety_filter
        self.safety_filter = None
        if use_safety_filter:
            self.safety_filter = CBFSafetyFilter(
                min_safe_distance_km=safety_threshold_km
            )
        
        # Policy manager
        self.policy_manager = PolicyManager()
        self._setup_policies()
        self.policy_manager.use_policy(policy_type)
        
        # Logging
        self.logger = SimulationLogger() if enable_logging else None
    
    def _setup_policies(self) -> None:
        """Setup available policies."""
        self.policy_manager.register_policy(
            'baseline',
            BaselinePolicy(risk_threshold=float(self.policy_kwargs.get('baseline_risk_threshold', 0.5))),
        )
        self.policy_manager.register_policy(
            'rule_based',
            RuleBasedPolicy(aggression=float(self.policy_kwargs.get('rule_based_aggression', 0.5))),
        )

        # Deterministic policies for worst-case and threshold-based tests.
        self.policy_manager.register_policy('no_op', NoOpPolicy())
        self.policy_manager.register_policy(
            'threshold_rule',
            ThresholdRulePolicy(
                threshold_km=float(self.policy_kwargs.get('threshold_km', 5.0)),
                dv_action=int(self.policy_kwargs.get('dv_action', 1)),
            ),
        )
        self.policy_manager.register_policy(
            'fuel_aware_threshold_rule',
            FuelAwareThresholdRulePolicy(
                threshold_km=float(self.policy_kwargs.get('threshold_km', 5.0)),
                dv_action=int(self.policy_kwargs.get('dv_action', 1)),
                min_fuel_ratio=float(self.policy_kwargs.get('min_fuel_ratio', 0.1)),
            ),
        )

        if self.marl_trainer is not None:
            self.policy_manager.register_policy('marl', MARLPolicy(self.marl_trainer))
    
    def run_episode(self, max_steps: int = 1000,
                   verbose: bool = True) -> Dict:
        """
        Run one complete episode.
        
        Args:
            max_steps: Maximum steps per episode
            verbose: Print progress
            
        Returns:
            Episode statistics
        """
        # Reset environment
        observations = self.env.reset()
        
        episode_stats = {
            'total_collisions': 0,
            'total_fuel_used': 0.0,
            'total_alerts': 0,
            'total_maneuvers_executed': 0,
            'total_secondary_conjunctions': 0,
            'total_near_misses': 0,
            'min_separation_distance_km': float("inf"),
            'final_step': 0,
        }
        
        for step in range(max_steps):
            # Get joint actions from policy
            actions = self.policy_manager.select_actions(observations)

            # Keep action dict scoped to active satellite agents only.
            actions = {
                agent_id: int(actions.get(agent_id, 0))
                for agent_id in self.env.agent_ids_ordered[: self.env.num_satellites]
                if agent_id in observations
            }
            
            # Apply safety filter if enabled
            if self.use_safety_filter:
                actions = self._apply_safety_filter(actions, observations)
            
            # Step environment
            next_obs, rewards, dones, info = self.env.step(actions)
            
            # Log
            if self.logger:
                self.logger.log_timestep(
                    step,
                    info['collisions_this_step'],
                    self.env.episode_fuel_used,
                    info['alerts_count']
                )
            
            # Update stats
            episode_stats['total_collisions'] = info['episode_collisions']
            episode_stats['total_fuel_used'] = self.env.episode_fuel_used
            episode_stats['total_alerts'] += info['alerts_count']
            episode_stats['total_maneuvers_executed'] = self.env.episode_maneuvers_executed
            episode_stats['total_secondary_conjunctions'] = self.env.episode_secondary_conjunctions
            episode_stats['total_near_misses'] = self.env.episode_near_misses
            episode_stats['min_separation_distance_km'] = self.env.episode_min_separation_distance_km
            
            # Update observations
            observations = next_obs
            
            # Check termination
            if dones.get('__all__', False):
                episode_stats['final_step'] = step
                break
            else:
                episode_stats['final_step'] = max_steps - 1
            
            if verbose and (step + 1) % 100 == 0:
                print(f"Step {step + 1}/{max_steps} - "
                      f"Collisions: {info['episode_collisions']}, "
                      f"Fuel: {self.env.episode_fuel_used:.2f} kg, "
                      f"Alerts: {info['alerts_count']}")
        
        return episode_stats

    def run_scenario(
        self,
        scenario: Dict[str, Any],
        max_steps: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Run a single episode using a dataset-derived scenario configuration.
        """
        self.scenario_config = scenario
        self.env.set_scenario_config(scenario)

        if max_steps is None:
            duration_hours = float(scenario.get("duration_hours", 1.0))
            max_steps = max(1, int(np.ceil(duration_hours * 3600.0 / self.env.dt)))

        return self.run_episode(max_steps=max_steps, verbose=verbose)
    
    def run_multiple_episodes(
        self,
        num_episodes: int,
        max_steps: int = 1000,
        verbose: bool = True,
        save_logs: bool = True,
        log_filename: str = "simulation_log.csv",
        maneuver_log_filename: str = "maneuvers_log.csv",
    ) -> List[Dict]:
        """
        Run multiple episodes.
        
        Args:
            num_episodes: Number of episodes
            max_steps: Max steps per episode
            verbose: Print progress
            
        Returns:
            List of episode statistics
        """
        all_stats = []

        if self.logger:
            self.logger.reset()
        
        for ep in range(num_episodes):
            if verbose:
                print(f"\nEpisode {ep + 1}/{num_episodes}")
            
            stats = self.run_episode(max_steps, verbose)
            all_stats.append(stats)
        
        if self.logger and save_logs:
            # Save logs for post-run visualization
            self.logger.save_to_csv(log_filename)
            self.logger.save_maneuvers_to_csv(maneuver_log_filename)

        return all_stats
    
    def _apply_safety_filter(self, actions: Dict[str, int],
                            observations: Dict[str, np.ndarray]) -> Dict[str, int]:
        """
        Apply CBF safety filter to actions.
        
        Args:
            actions: Raw policy actions
            observations: Current observations
            
        Returns:
            Filtered actions
        """
        filtered_actions = {}
        
        for agent_id, action in actions.items():
            if agent_id not in observations:
                filtered_actions[agent_id] = action
                continue
            
            obs = observations[agent_id]
            
            # Extract state
            own_state = obs[:6]  # [x, y, z, vx, vy, vz]
            
            # Get nearby objects
            nearby_start = 8
            nearby_objects = {}
            for i in range(7):
                obj_start = nearby_start + i * 8
                if obj_start + 8 <= len(obs):
                    obj_state = obs[obj_start:obj_start+6]
                    if not (np.allclose(obj_state[:3], 0) and np.allclose(obj_state[3:], 0)):
                        nearby_objects[f"OBJ_{i}"] = obj_state
            
            # Convert action to delta-v
            action_dv = self.env.maneuver_engine.action_index_to_delta_v(
                action, own_state[3:]
            )
            
            # Filter through CBF
            safe_dv = self.safety_filter.filter_action(
                own_state, action_dv, nearby_objects
            )
            
            # Check if action changed
            if not np.allclose(safe_dv, action_dv):
                # Action was modified - find closest valid discrete action
                best_action = 0
                min_dist = float('inf')
                for a_idx in range(6):
                    test_dv = self.env.maneuver_engine.action_index_to_delta_v(
                        a_idx, own_state[3:]
                    )
                    dist = np.linalg.norm(safe_dv - test_dv)
                    if dist < min_dist:
                        min_dist = dist
                        best_action = a_idx
                filtered_actions[agent_id] = best_action
            else:
                filtered_actions[agent_id] = action
        
        return filtered_actions
    
    def compare_policies(self, policies: List[str],
                        num_episodes: int = 5,
                        max_steps: int = 1000) -> Dict:
        """
        Compare multiple policies.
        
        Args:
            policies: List of policy names
            num_episodes: Episodes per policy
            max_steps: Max steps per episode
            
        Returns:
            Comparison results
        """
        results = {}
        
        for policy_name in policies:
            print(f"\n{'='*60}")
            print(f"Testing policy: {policy_name}")
            print('='*60)

            if policy_name not in self.policy_manager.get_available_policies():
                print(f"Policy '{policy_name}' is not available in the policy manager. Skipping.")
                continue

            self.policy_manager.use_policy(policy_name)

            # Reset policy state when available
            policy = self.policy_manager.policies.get(policy_name)
            if policy is not None and hasattr(policy, 'reset'):
                policy.reset()

            stats_list = self.run_multiple_episodes(
                num_episodes,
                max_steps,
                verbose=True,
                save_logs=self.logger is not None,
                log_filename=f"{policy_name}_simulation_log.csv",
                maneuver_log_filename=f"{policy_name}_maneuvers_log.csv",
            )

            # Aggregate statistics
            results[policy_name] = self._aggregate_stats(stats_list)

        return results
    
    def _aggregate_stats(self, stats_list: List[Dict]) -> Dict:
        """Aggregate episode statistics."""
        if not stats_list:
            return {}
        
        collisions = [s['total_collisions'] for s in stats_list]
        fuel = [s['total_fuel_used'] for s in stats_list]
        maneuvers = [s.get('total_maneuvers_executed', 0) for s in stats_list]
        secondary = [s.get('total_secondary_conjunctions', 0) for s in stats_list]
        
        collision_free = np.array([1 if c == 0 else 0 for c in collisions], dtype=float)
        collision_light = np.array([1 if c <= 1 else 0 for c in collisions], dtype=float)

        return {
            'mean_collisions': np.mean(collisions),
            'std_collisions': np.std(collisions),
            'mean_fuel': np.mean(fuel),
            'std_fuel': np.std(fuel),
            'mean_maneuvers_executed': np.mean(maneuvers),
            'mean_secondary_conjunctions': np.mean(secondary),
            'success_rate': float(collision_free.mean()),
            'success_rate_<=1_collision': float(collision_light.mean()),
            'avg_episode_length': np.mean([s['final_step'] for s in stats_list]),
        }
    
    def save_results(self, filename: str = 'simulation_results.csv') -> None:
        """Save simulation results."""
        if self.logger:
            self.logger.save_to_csv(filename)
