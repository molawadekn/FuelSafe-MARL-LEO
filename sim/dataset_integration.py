"""
Dataset integration utilities for ESA CDM-driven MARL training and validation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from marl.marl_trainer import MARLTrainer
from sim.csv_data_loader import CSVDataLoader
from sim.reporting import save_summary_charts, save_run_distribution_charts, save_training_progress_charts
from sim.simulator import SimulationRunner


class DatasetIntegration:
    """Interface for integrating ESA CDM CSV data with the FuelSafe simulator."""

    def __init__(self, csv_path: str, verbose: bool = True):
        self.csv_path = str(csv_path)
        self.verbose = verbose
        self.loader = CSVDataLoader(csv_path, verbose=verbose)
        self.data: Optional[pd.DataFrame] = None
        self.scenarios: List[Dict] = []

    def load_dataset(self, max_rows: Optional[int] = None) -> pd.DataFrame:
        self.data = self.loader.load(max_rows=max_rows)
        if self.verbose:
            print(f"[OK] Loaded {len(self.data)} events from CSV")
        return self.data

    def get_risk_distribution(self) -> Dict[str, int]:
        if self.data is None:
            raise RuntimeError("Dataset not loaded. Call load_dataset() first.")

        return {
            "critical": int((self.data["risk"] > -5.0).sum()),
            "high": int(((self.data["risk"] > -7.0) & (self.data["risk"] <= -5.0)).sum()),
            "medium": int(((self.data["risk"] > -9.0) & (self.data["risk"] <= -7.0)).sum()),
            "low": int((self.data["risk"] <= -9.0).sum()),
        }

    def create_scenarios_from_dataset(
        self, risk_threshold: float = -7.0, max_scenarios: int = 10
    ) -> List[Dict]:
        if self.data is None:
            raise RuntimeError("Dataset not loaded. Call load_dataset() first.")

        high_risk = self.data[self.data["risk"] >= risk_threshold].sort_values("risk", ascending=False)
        if "event_id" in high_risk.columns:
            high_risk = high_risk.drop_duplicates(subset=["event_id"], keep="first")

        if high_risk.empty:
            scenarios = []
        else:
            selection_count = min(max_scenarios, len(high_risk))
            selection_indices = np.linspace(0, len(high_risk) - 1, num=selection_count, dtype=int)
            selected_events = high_risk.iloc[np.unique(selection_indices)]
            scenarios = self.loader.get_batch_scenarios(selected_events, max_scenarios=max_scenarios)

        self.scenarios = scenarios

        if self.verbose:
            print(
                f"[OK] Created {len(scenarios)} scenarios from dataset "
                f"(risk-stratified unique-event selection from {len(high_risk)} candidates)"
            )

        return scenarios

    def generate_integration_report(self) -> str:
        if self.data is None:
            return "No data loaded"

        risk_dist = self.get_risk_distribution()
        total = max(len(self.data), 1)

        return (
            "FuelSafe Dataset Integration Report\n"
            f"CSV: {self.csv_path}\n"
            f"Rows loaded: {len(self.data)}\n"
            f"Scenarios prepared: {len(self.scenarios)}\n"
            f"Risk distribution: critical={risk_dist['critical']} "
            f"high={risk_dist['high']} medium={risk_dist['medium']} low={risk_dist['low']}\n"
            f"Miss distance range (m): {self.data['miss_distance'].min():.0f} .. "
            f"{self.data['miss_distance'].max():.0f}\n"
            f"Relative speed range (m/s): {self.data['relative_speed'].min():.0f} .. "
            f"{self.data['relative_speed'].max():.0f}\n"
            f"Time to TCA range (hours): {self.data['time_to_tca'].min():.2f} .. "
            f"{self.data['time_to_tca'].max():.2f}\n"
            f"Data completeness: {100 * (1 - self.data.isnull().sum().sum() / (total * len(self.data.columns))):.2f}%"
        )

    def print_report(self) -> None:
        print(self.generate_integration_report())

    @staticmethod
    def _scenario_max_steps(scenario: Dict, fallback_max_steps: int) -> int:
        duration_hours = float(scenario.get("duration_hours", 0.0) or 0.0)
        if duration_hours <= 0:
            return int(fallback_max_steps)
        steps_from_duration = int((duration_hours * 3600.0) // 60.0)
        return max(1, min(int(fallback_max_steps), steps_from_duration if steps_from_duration > 0 else fallback_max_steps))

    @staticmethod
    def _build_runner(
        *,
        scenario: Dict,
        policy_type: str,
        max_steps: int,
        num_satellites: int,
        num_debris: int,
        initial_fuel_kg: float,
        marl_trainer: Optional[MARLTrainer] = None,
    ) -> Tuple[SimulationRunner, int]:
        runner = SimulationRunner(
            num_satellites=num_satellites,
            num_debris=num_debris,
            use_safety_filter=True,
            safety_threshold_km=0.5,
            distance_threshold_km=50.0,
            # Use a tighter collision threshold for CDM-driven validation than the
            # coarse Monte Carlo stress tests, otherwise sub-km miss distances are
            # all labeled as immediate collisions.
            collision_threshold_km=0.5,
            high_risk_mode=True,
            policy_type=policy_type,
            enable_logging=False,
            initial_fuel_kg=initial_fuel_kg,
            max_fuel_kg=initial_fuel_kg,
            scenario_config=scenario,
            marl_trainer=marl_trainer,
            policy_kwargs={
                "threshold_km": 5.0,
                "dv_action": 1,
                "min_fuel_ratio": 0.2,
                "baseline_risk_threshold": 0.5,
                "rule_based_aggression": 0.5,
            },
        )
        return runner, DatasetIntegration._scenario_max_steps(scenario, max_steps)

    def train_marl_from_dataset(
        self,
        *,
        max_rows: Optional[int] = 1000,
        risk_threshold: float = -7.0,
        max_scenarios: int = 10,
        episodes_per_scenario: int = 5,
        max_steps: int = 200,
        num_satellites: int = 3,
        num_debris: int = 5,
        initial_fuel_kg: float = 1000.0,
        marl_epochs_per_batch: int = 3,
        save_model_path: Optional[str] = None,
    ) -> Dict:
        self.load_dataset(max_rows=max_rows)
        scenarios = self.create_scenarios_from_dataset(
            risk_threshold=risk_threshold,
            max_scenarios=max_scenarios,
        )

        trainer = MARLTrainer(num_agents=num_satellites)
        metrics: List[Dict] = []

        for scenario in scenarios:
            if self.verbose:
                print(f"\n[TRAIN] Scenario: {scenario['name']} ({scenario['risk_level']})")

            runner, scenario_steps = self._build_runner(
                scenario=scenario,
                policy_type="marl",
                max_steps=max_steps,
                num_satellites=num_satellites,
                num_debris=num_debris,
                initial_fuel_kg=initial_fuel_kg,
                marl_trainer=trainer,
            )

            env = runner.env

            for episode_idx in range(episodes_per_scenario):
                observations = env.reset()
                done = False
                step = 0

                while not done and step < scenario_steps:
                    actions, log_probs, central_value = trainer.get_action_details(observations)
                    next_obs, rewards, dones, _info = env.step(actions)
                    trainer.collect_experience(
                        observations,
                        rewards,
                        next_obs,
                        dones,
                        actions,
                        log_probs=log_probs,
                        central_value=central_value,
                    )
                    observations = next_obs
                    done = bool(dones.get("__all__", False))
                    step += 1

                train_stats = trainer.train(num_epochs=marl_epochs_per_batch)
                row = {
                    "scenario": scenario["name"],
                    "risk_level": scenario["risk_level"],
                    "episode": episode_idx + 1,
                    "final_collisions": env.episode_collisions,
                    "final_fuel_used": env.episode_fuel_used,
                    "final_steps": env.step_count,
                    **train_stats,
                }
                metrics.append(row)

                if self.verbose:
                    print(
                        f"  ep{episode_idx + 1}/{episodes_per_scenario} "
                        f"collisions={env.episode_collisions} "
                        f"fuel={env.episode_fuel_used:.3f} "
                        f"steps={env.step_count} "
                        f"actor_loss={train_stats.get('actor_loss', 0.0):.4f}"
                    )

        if save_model_path:
            save_path = Path(save_model_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save(str(save_path))
            if self.verbose:
                print(f"[TRAIN] Saved MARL model to {save_path}")

        metrics_df = pd.DataFrame(metrics)
        summary = {
            "trained_model_path": save_model_path,
            "num_scenarios": len(scenarios),
            "episodes_per_scenario": episodes_per_scenario,
            "mean_collisions": float(metrics_df["final_collisions"].mean()) if not metrics_df.empty else 0.0,
            "mean_fuel_used": float(metrics_df["final_fuel_used"].mean()) if not metrics_df.empty else 0.0,
            "mean_steps": float(metrics_df["final_steps"].mean()) if not metrics_df.empty else 0.0,
            "metrics": metrics,
        }
        return summary

    def evaluate_policies_on_dataset(
        self,
        *,
        policy_types: List[str],
        max_rows: Optional[int] = 1000,
        risk_threshold: float = -7.0,
        max_scenarios: int = 10,
        max_steps: int = 200,
        num_satellites: int = 3,
        num_debris: int = 5,
        initial_fuel_kg: float = 1000.0,
        marl_model_path: Optional[str] = None,
    ) -> Dict:
        self.load_dataset(max_rows=max_rows)
        scenarios = self.create_scenarios_from_dataset(
            risk_threshold=risk_threshold,
            max_scenarios=max_scenarios,
        )

        marl_trainer: Optional[MARLTrainer] = None
        if "marl" in policy_types:
            marl_trainer = MARLTrainer(num_agents=num_satellites)
            if marl_model_path is None:
                raise ValueError("marl_model_path must be provided when evaluating the MARL policy.")
            marl_trainer.load(marl_model_path)

        rows: List[Dict] = []
        for scenario in scenarios:
            for policy_type in policy_types:
                runner, scenario_steps = self._build_runner(
                    scenario=scenario,
                    policy_type=policy_type,
                    max_steps=max_steps,
                    num_satellites=num_satellites,
                    num_debris=num_debris,
                    initial_fuel_kg=initial_fuel_kg,
                    marl_trainer=marl_trainer if policy_type == "marl" else None,
                )
                stats = runner.run_episode(max_steps=scenario_steps, verbose=False)
                rows.append(
                    {
                        "scenario": scenario["name"],
                        "risk_level": scenario["risk_level"],
                        "policy": policy_type,
                        **stats,
                    }
                )

        episodes_df = pd.DataFrame(rows)
        summary_df = (
            episodes_df.groupby("policy", dropna=False)
            .agg(
                mean_collisions=("total_collisions", "mean"),
                mean_fuel=("total_fuel_used", "mean"),
                mean_maneuvers=("total_maneuvers_executed", "mean"),
                mean_secondary_conjunctions=("total_secondary_conjunctions", "mean"),
                mean_near_misses=("total_near_misses", "mean"),
                mean_min_separation_km=("min_separation_distance_km", "mean"),
            )
            .reset_index()
            .sort_values("policy")
        )

        return {
            "num_scenarios": len(scenarios),
            "episode_metrics": episodes_df.to_dict(orient="records"),
            "policy_summary": summary_df.to_dict(orient="records"),
        }


def train_and_validate_marl(
    *,
    train_csv: str,
    test_csv: str,
    output_dir: str,
    train_max_rows: int,
    test_max_rows: int,
    risk_threshold: float,
    train_scenarios: int,
    test_scenarios: int,
    episodes_per_scenario: int,
    max_steps: int,
    num_satellites: int,
    num_debris: int,
    initial_fuel_kg: float,
    marl_epochs_per_batch: int,
    verbose: bool = True,
) -> Dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / "marl_trained_from_train_dataset.pth"
    train_metrics_path = output_path / "train_metrics.csv"
    validation_metrics_path = output_path / "validation_episode_metrics.csv"
    validation_summary_path = output_path / "validation_policy_summary.csv"
    report_path = output_path / "train_validation_report.json"

    train_integration = DatasetIntegration(train_csv, verbose=verbose)
    train_result = train_integration.train_marl_from_dataset(
        max_rows=train_max_rows,
        risk_threshold=risk_threshold,
        max_scenarios=train_scenarios,
        episodes_per_scenario=episodes_per_scenario,
        max_steps=max_steps,
        num_satellites=num_satellites,
        num_debris=num_debris,
        initial_fuel_kg=initial_fuel_kg,
        marl_epochs_per_batch=marl_epochs_per_batch,
        save_model_path=str(model_path),
    )
    train_metrics_df = pd.DataFrame(train_result["metrics"])
    train_metrics_df.to_csv(train_metrics_path, index=False)

    test_integration = DatasetIntegration(test_csv, verbose=verbose)
    validation_result = test_integration.evaluate_policies_on_dataset(
        policy_types=["no_op", "fuel_aware_threshold_rule", "rule_based", "marl"],
        max_rows=test_max_rows,
        risk_threshold=risk_threshold,
        max_scenarios=test_scenarios,
        max_steps=max_steps,
        num_satellites=num_satellites,
        num_debris=num_debris,
        initial_fuel_kg=initial_fuel_kg,
        marl_model_path=str(model_path),
    )
    validation_metrics_df = pd.DataFrame(validation_result["episode_metrics"])
    validation_summary_df = pd.DataFrame(validation_result["policy_summary"])
    validation_metrics_df.to_csv(validation_metrics_path, index=False)
    validation_summary_df.to_csv(validation_summary_path, index=False)

    save_training_progress_charts(train_metrics_df, output_path, prefix="interactive_train_progress")
    save_summary_charts(validation_summary_df, output_path, prefix="interactive_validation_summary")
    save_run_distribution_charts(validation_metrics_df, output_path, prefix="interactive_validation_runs")

    report = {
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "model_path": str(model_path),
        "train_summary": {
            k: v for k, v in train_result.items() if k != "metrics"
        },
        "validation_summary": validation_result["policy_summary"],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train MARL on an ESA CDM training set and validate on a test set."
    )
    parser.add_argument("--train-csv", type=str, required=True, help="Path to training CSV")
    parser.add_argument("--test-csv", type=str, required=True, help="Path to test CSV")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/marl_train_validation",
        help="Directory for model and validation artifacts",
    )
    parser.add_argument("--train-max-rows", type=int, default=5000)
    parser.add_argument("--test-max-rows", type=int, default=2000)
    parser.add_argument("--risk-threshold", type=float, default=-7.0)
    parser.add_argument("--train-scenarios", type=int, default=12)
    parser.add_argument("--test-scenarios", type=int, default=8)
    parser.add_argument("--episodes-per-scenario", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--num-satellites", type=int, default=3)
    parser.add_argument("--num-debris", type=int, default=10)
    parser.add_argument("--initial-fuel-kg", type=float, default=1000.0)
    parser.add_argument("--marl-epochs-per-batch", type=int, default=3)
    args = parser.parse_args()

    result = train_and_validate_marl(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        output_dir=args.output_dir,
        train_max_rows=args.train_max_rows,
        test_max_rows=args.test_max_rows,
        risk_threshold=args.risk_threshold,
        train_scenarios=args.train_scenarios,
        test_scenarios=args.test_scenarios,
        episodes_per_scenario=args.episodes_per_scenario,
        max_steps=args.max_steps,
        num_satellites=args.num_satellites,
        num_debris=args.num_debris,
        initial_fuel_kg=args.initial_fuel_kg,
        marl_epochs_per_batch=args.marl_epochs_per_batch,
        verbose=True,
    )

    print("\nTrain/validation report:")
    print(json.dumps(result, indent=2))
