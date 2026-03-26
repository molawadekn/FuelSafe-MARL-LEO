"""
MODULE 10: Experiment Framework
Orchestrates full experiments comparing policies and evaluating performance.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

from sim.simulator import SimulationRunner


class ExperimentConfig:
    """Configuration for experiments."""
    
    def __init__(self):
        """Initialize default config."""
        self.num_satellites_list = [3, 10, 50]
        self.num_debris_list = [5, 100, 500]
        self.policies = ['baseline', 'rule_based']
        self.num_episodes_per_config = 5
        self.max_steps_per_episode = 1000
        self.use_safety_filter = True
        self.output_dir = 'outputs'
    
    def save(self, filepath: str) -> None:
        """Save config to JSON."""
        config_dict = {
            'num_satellites_list': self.num_satellites_list,
            'num_debris_list': self.num_debris_list,
            'policies': self.policies,
            'num_episodes_per_config': self.num_episodes_per_config,
            'max_steps_per_episode': self.max_steps_per_episode,
            'use_safety_filter': self.use_safety_filter,
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str):
        """Load config from JSON."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        config.num_satellites_list = config_dict.get('num_satellites_list', config.num_satellites_list)
        config.num_debris_list = config_dict.get('num_debris_list', config.num_debris_list)
        config.policies = config_dict.get('policies', config.policies)
        config.num_episodes_per_config = config_dict.get('num_episodes_per_config', config.num_episodes_per_config)
        config.max_steps_per_episode = config_dict.get('max_steps_per_episode', config.max_steps_per_episode)
        config.use_safety_filter = config_dict.get('use_safety_filter', config.use_safety_filter)
        
        return config


class ExperimentRunner:
    """
    Runs comprehensive experiments with multiple configurations.
    Enables large-scale policy comparison and evaluation.
    """
    
    def __init__(self, config: Optional[ExperimentConfig] = None):
        """
        Initialize experiment runner.
        
        Args:
            config: Experiment configuration
        """
        self.config = config or ExperimentConfig()
        self.results_dir = Path(self.config.output_dir) / 'experiments'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results = {}
    
    def run_full_experiment(self, verbose: bool = True) -> Dict:
        """
        Run full experiment grid.
        
        Args:
            verbose: Print progress
            
        Returns:
            Complete results
        """
        if verbose:
            print(f"Starting experiment {self.experiment_id}")
            print(f"Configurations: {len(self.config.num_satellites_list) * len(self.config.num_debris_list)}")
            print(f"Policies: {len(self.config.policies)}")
            print(f"Total runs: {len(self.config.num_satellites_list) * len(self.config.num_debris_list) * len(self.config.policies)}")
        
        config_id = 0
        
        for num_sats in self.config.num_satellites_list:
            for num_debs in self.config.num_debris_list:
                config_id += 1
                config_key = f"SAT{num_sats}_DEB{num_debs}"
                
                if verbose:
                    print(f"\n{'='*70}")
                    print(f"Config {config_id}: {config_key}")
                    print(f"{'='*70}")
                
                config_results = self.run_configuration(
                    num_sats, num_debs, verbose=verbose
                )
                self.results[config_key] = config_results
        
        return self.results
    
    def run_configuration(self, num_satellites: int,
                         num_debris: int,
                         verbose: bool = True) -> Dict:
        """
        Run one configuration (all policies).
        
        Args:
            num_satellites: Number of satellites
            num_debris: Number of debris
            verbose: Print progress
            
        Returns:
            Configuration results
        """
        config_results = {
            'num_satellites': num_satellites,
            'num_debris': num_debris,
            'policies': {}
        }
        
        for policy in self.config.policies:
            if verbose:
                print(f"\n  Testing policy: {policy}")
            
            runner = SimulationRunner(
                num_satellites=num_satellites,
                num_debris=num_debris,
                use_safety_filter=self.config.use_safety_filter,
                policy_type=policy,
                enable_logging=False
            )
            
            stats_list = runner.run_multiple_episodes(
                num_episodes=self.config.num_episodes_per_config,
                max_steps=self.config.max_steps_per_episode,
                verbose=False
            )
            
            # Aggregate
            aggregated = runner._aggregate_stats(stats_list)
            config_results['policies'][policy] = aggregated
            
            if verbose:
                print(f"    Mean collisions: {aggregated['mean_collisions']:.2f} ±{aggregated['std_collisions']:.2f}")
                print(f"    Mean fuel: {aggregated['mean_fuel']:.2f} ±{aggregated['std_fuel']:.2f} kg")
                print(f"    Success rate: {aggregated['success_rate']*100:.1f}%")
        
        return config_results
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """
        Save results to files.
        
        Args:
            filename: Output filename prefix
            
        Returns:
            Path to saved results
        """
        if not filename:
            filename = f"experiment_{self.experiment_id}"
        
        filepath = self.results_dir / filename
        
        # Save as JSON
        json_path = filepath.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save as CSV (flattened)
        csv_path = filepath.with_suffix('.csv')
        self._save_results_csv(csv_path)
        
        print(f"\nResults saved to:")
        print(f"  JSON: {json_path}")
        print(f"  CSV: {csv_path}")
        
        return str(filepath)
    
    def _save_results_csv(self, filepath: str) -> None:
        """Save results as CSV."""
        rows = []
        
        for config_key, config_results in self.results.items():
            num_sats = config_results['num_satellites']
            num_deb = config_results['num_debris']
            
            for policy, agg_stats in config_results['policies'].items():
                row = {
                    'config': config_key,
                    'num_satellites': num_sats,
                    'num_debris': num_deb,
                    'policy': policy,
                    'mean_collisions': agg_stats['mean_collisions'],
                    'std_collisions': agg_stats['std_collisions'],
                    'mean_fuel': agg_stats['mean_fuel'],
                    'std_fuel': agg_stats['std_fuel'],
                    'success_rate': agg_stats['success_rate'],
                    'avg_episode_length': agg_stats['avg_episode_length'],
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
    
    def generate_report(self) -> str:
        """
        Generate human-readable report.
        
        Returns:
            Report text
        """
        report = f"{'='*70}\n"
        report += f"Orbital Collision Avoidance Experiment Report\n"
        report += f"Experiment ID: {self.experiment_id}\n"
        report += f"Timestamp: {datetime.now().isoformat()}\n"
        report += f"{'='*70}\n\n"
        
        report += "CONFIGURATION:\n"
        report += f"  Satellites: {self.config.num_satellites_list}\n"
        report += f"  Debris: {self.config.num_debris_list}\n"
        report += f"  Policies: {self.config.policies}\n"
        report += f"  Episodes per config: {self.config.num_episodes_per_config}\n"
        report += f"  Max steps per episode: {self.config.max_steps_per_episode}\n"
        report += f"  Safety filter: {self.config.use_safety_filter}\n\n"
        
        report += "RESULTS:\n"
        report += f"{'Config':<20} {'Policy':<15} {'Collisions':<20} {'Fuel (kg)':<20} {'Success %':<15}\n"
        report += "-" * 90 + "\n"
        
        for config_key, config_results in sorted(self.results.items()):
            for policy, agg_stats in config_results['policies'].items():
                collisions = f"{agg_stats['mean_collisions']:.2f}±{agg_stats['std_collisions']:.2f}"
                fuel = f"{agg_stats['mean_fuel']:.2f}±{agg_stats['std_fuel']:.2f}"
                success = f"{agg_stats['success_rate']*100:.1f}%"
                
                report += f"{config_key:<20} {policy:<15} {collisions:<20} {fuel:<20} {success:<15}\n"
        
        report += "\n" + "="*70 + "\n"
        return report
    
    def print_report(self) -> None:
        """Print report to stdout."""
        print(self.generate_report())
    
    def save_report(self, filename: Optional[str] = None) -> None:
        """Save report to file."""
        if not filename:
            filename = f"experiment_{self.experiment_id}_report.txt"
        
        filepath = self.results_dir / filename
        with open(filepath, 'w') as f:
            f.write(self.generate_report())
        
        print(f"Report saved to {filepath}")
