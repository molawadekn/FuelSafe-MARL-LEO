"""
MODULE 9: Core Simulation Loop
Orchestrates the full simulation with orbit propagation, collision detection,
maneuvers, reward computation, and logging.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from pathlib import Path

from env.ma_env import MultiAgentOrbitalEnv
from safety.cbf_filter import CBFSafetyFilter
from policies.policy_interface import PolicyManager, BaselinePolicy, RuleBasedPolicy


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
                 enable_logging: bool = True):
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
        
        # Environment
        self.env = MultiAgentOrbitalEnv(
            num_satellites=num_satellites,
            num_debris=num_debris,
            distance_threshold_km=distance_threshold_km,
            collision_threshold_km=collision_threshold_km,
            high_risk_mode=high_risk_mode
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
        self.policy_manager.register_policy('baseline', BaselinePolicy())
        self.policy_manager.register_policy('rule_based', RuleBasedPolicy())
        # MARL policy would be added when trainer is ready
    
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
            'total_maneuvers': 0,
            'final_step': 0,
        }
        
        for step in range(max_steps):
            # Get actions from policy
            actions = {}
            for agent_id in self.env.agent_ids_ordered[:self.env.num_satellites]:
                if agent_id in observations:
                    actions[agent_id] = self.policy_manager.select_action(
                        observations[agent_id], agent_id
                    )
            
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
            episode_stats['total_maneuvers'] += sum(
                1 for a in actions.values() if a != 0
            )
            
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
    
    def run_multiple_episodes(self, num_episodes: int,
                            max_steps: int = 1000,
                            verbose: bool = True) -> List[Dict]:
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
        
        for ep in range(num_episodes):
            if verbose:
                print(f"\nEpisode {ep + 1}/{num_episodes}")
            
            stats = self.run_episode(max_steps, verbose)
            all_stats.append(stats)
        
        if self.logger:
            # Save logs for post-run visualization
            self.logger.save_to_csv('simulation_log.csv')
            self.logger.save_maneuvers_to_csv('maneuvers_log.csv')

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
                obj_start = nearby_start + i * 6
                if obj_start + 6 <= len(obs):
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
                # Action was modified - convert back to discrete if possible
                filtered_actions[agent_id] = 0  # NO_OP on safety violation
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
            
            self.policy_manager.use_policy(policy_name)
            
            stats_list = self.run_multiple_episodes(
                num_episodes, max_steps, verbose=True
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
        
        return {
            'mean_collisions': np.mean(collisions),
            'std_collisions': np.std(collisions),
            'mean_fuel': np.mean(fuel),
            'std_fuel': np.std(fuel),
            'success_rate': np.mean([1 if c == 0 else 0 for c in collisions]),
            'avg_episode_length': np.mean([s['final_step'] for s in stats_list]),
        }
    
    def save_results(self, filename: str = 'simulation_results.csv') -> None:
        """Save simulation results."""
        if self.logger:
            self.logger.save_to_csv(filename)
