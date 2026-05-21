#!/usr/bin/env python3
"""
Training script for MARL collision avoidance with curriculum and dataset support.

Phase 1 upgrades
----------------
* Loads hyperparameters from config/default.yaml (CLI flags override YAML).
* Supports --actor-type: mlp | attention | recurrent | ensemble.
* Optional W&B logging via --use-wandb flag.
* Optional MLflow logging via --use-mlflow flag.
* Tsiolkovsky fuel model and Foster Pc active by default (controlled via YAML).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Add local path for imports
sys.path.insert(0, str(Path(__file__).parent))

from env.ma_env import MultiAgentOrbitalEnv
from marl.curriculum_manager import CurriculumManager
from marl.marl_trainer import MARLTrainer
from sim.evaluation import compute_efficiency_score, compute_tc8_success_rate, save_pareto_artifacts
from sim.maneuver_engine import ACTION_COUNT
from sim.observation_utils import OBS_SIZE
from sim.realism import RealismConfig


# ─────────────────────────────────────────────────────────────────────────────
# YAML config loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_yaml_config(path: str) -> Dict[str, Any]:
    """Load YAML config; returns empty dict if file missing or PyYAML absent."""
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg
    except FileNotFoundError:
        return {}
    except ImportError:
        print("[WARNING] PyYAML not installed — skipping YAML config. "
              "Run: pip install pyyaml")
        return {}


def _cfg(yaml_cfg: Dict, *keys, default=None):
    """Safely navigate nested YAML dict: _cfg(cfg, 'ppo', 'learning_rate')."""
    node = yaml_cfg
    for k in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(k, default)
        if node is default:
            return default
    return node


# ─────────────────────────────────────────────────────────────────────────────
# Experiment tracker (W&B + MLflow, both optional)
# ─────────────────────────────────────────────────────────────────────────────

class ExperimentTracker:
    """
    Thin wrapper around W&B and/or MLflow.
    Both are optional — the tracker degrades gracefully if not installed.
    """

    def __init__(self, use_wandb: bool, use_mlflow: bool,
                 project: str, entity: str, experiment: str, config: Dict):
        self._wandb   = None
        self._mlflow  = None

        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project=project,
                    entity=entity or None,
                    config=config,
                    reinit=True,
                )
                self._wandb = wandb
                print(f"[W&B] Tracking enabled → project: {project}")
            except ImportError:
                print("[WARNING] wandb not installed. Run: pip install wandb")

        if use_mlflow:
            try:
                import mlflow
                mlflow.set_experiment(experiment)
                mlflow.start_run()
                mlflow.log_params(config)
                self._mlflow = mlflow
                print(f"[MLflow] Tracking enabled → experiment: {experiment}")
            except ImportError:
                print("[WARNING] mlflow not installed. Run: pip install mlflow")

    def log(self, metrics: Dict[str, float], step: int) -> None:
        if self._wandb:
            self._wandb.log(metrics, step=step)
        if self._mlflow:
            self._mlflow.log_metrics(metrics, step=step)

    def finish(self) -> None:
        if self._wandb:
            self._wandb.finish()
        if self._mlflow:
            self._mlflow.end_run()

def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return rows

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _sample_training_config(
    *,
    use_dataset: bool,
    dataset_scenarios: List[Dict[str, Any]],
    curriculum: CurriculumManager,
    tc8_ratio: float,
) -> Dict[str, Any]:
    hard_probability = float(np.clip(tc8_ratio, 0.0, 1.0))
    hard_min_debris = 15 if hard_probability > 0.0 else 1

    if use_dataset and dataset_scenarios:
        scenario = random.choice(dataset_scenarios)
        return {
            "name": str(scenario.get("name", "dataset_box")),
            "max_debris": int(max(hard_min_debris, scenario.get("num_debris", 10))),
            "collision_threshold_km": float(scenario.get("collision_threshold_km", 1.0)),
            "distance_threshold_km": float(scenario.get("distance_threshold_km", 250.0)),
            "high_risk_mode": bool(scenario.get("use_high_risk_mode", False)),
            "scenario_config": scenario,
            "hard_scenario_probability": hard_probability,
        }

    stage_cfg = curriculum.get_current_config()
    return {
        "name": str(stage_cfg.get("name", "curriculum_stage")),
        "max_debris": int(max(hard_min_debris, stage_cfg.get("max_debris", 10))),
        "collision_threshold_km": float(stage_cfg.get("collision_threshold_km", 1.0)),
        "distance_threshold_km": float(stage_cfg.get("distance_threshold_km", 250.0)),
        "high_risk_mode": True,
        "scenario_config": None,
        "hard_scenario_probability": hard_probability,
    }


def _run_episode(
    *,
    trainer: MARLTrainer,
    env: MultiAgentOrbitalEnv,
    max_steps: int,
    deterministic: bool,
    collect_experience: bool,
    terminate_on_collision: bool,
) -> Dict[str, float]:
    obs = env.reset()
    done = {"__all__": False}
    step = 0

    while step < max_steps and not done["__all__"]:
        actions, log_probs, value = trainer.get_action_details(obs, deterministic=deterministic)
        next_obs, rewards, done, info = env.step(actions)

        forced_done = terminate_on_collision and info.get("episode_collisions", 0) > 0
        done_for_buffer = done
        if forced_done:
            done_for_buffer = {agent_id: True for agent_id in obs.keys()}
            done_for_buffer["__all__"] = True

        if collect_experience:
            trainer.collect_experience(
                observations=obs,
                rewards=rewards,
                next_observations=next_obs,
                dones=done_for_buffer,
                actions=actions,
                log_probs=log_probs,
                central_value=value,
            )

        obs = next_obs
        step += 1

        if forced_done:
            break

    return {
        "collisions": float(env.episode_collisions),
        "fuel": float(env.episode_fuel_used),
        "maneuvers": float(env.episode_maneuvers_executed),
        "secondary": float(env.episode_secondary_conjunctions),
        "near_misses": float(env.episode_near_misses),
        "final_step": float(step),
        "tc8_active": bool(env._is_tc8_like_scenario()),
        "score": compute_efficiency_score(
            env.episode_collisions,
            env.episode_fuel_used,
            env.episode_maneuvers_executed,
        ),
    }


def _evaluate_policy(
    *,
    trainer: MARLTrainer,
    curriculum: CurriculumManager,
    eval_scenarios: List[Dict[str, Any]],
    use_dataset: bool,
    tc8_ratio: float,
    eval_episodes: int,
    max_steps: int,
    num_satellites: int,
    terminate_on_collision: bool,
    realism_config: Optional[RealismConfig] = None,
) -> Dict[str, float]:
    eval_stats: List[Dict[str, float]] = []

    for _ in range(eval_episodes):
        config = _sample_training_config(
            use_dataset=use_dataset,
            dataset_scenarios=eval_scenarios,
            curriculum=curriculum,
            tc8_ratio=tc8_ratio,
        )
        env = MultiAgentOrbitalEnv(
            num_satellites=num_satellites,
            num_debris=int(config["max_debris"]),
            collision_threshold_km=float(config["collision_threshold_km"]),
            distance_threshold_km=float(config.get("distance_threshold_km", 250.0)),
            high_risk_mode=bool(config.get("high_risk_mode", True)),
            scenario_config=config.get("scenario_config"),
            hard_scenario_probability=float(config.get("hard_scenario_probability", 0.0)),
            secondary_conjunction_risk_threshold=0.01,
            realism_config=realism_config,
        )
        eval_stats.append(
            _run_episode(
                trainer=trainer,
                env=env,
                max_steps=max_steps,
                deterministic=True,
                collect_experience=False,
                terminate_on_collision=terminate_on_collision,
            )
        )

    mean_collisions = float(np.mean([s["collisions"] for s in eval_stats])) if eval_stats else 0.0
    collision_rate = float(np.mean([1.0 if s["collisions"] > 0.0 else 0.0 for s in eval_stats])) if eval_stats else 0.0
    mean_fuel = float(np.mean([s["fuel"] for s in eval_stats])) if eval_stats else 0.0
    mean_maneuvers = float(np.mean([s["maneuvers"] for s in eval_stats])) if eval_stats else 0.0
    mean_secondary = float(np.mean([s["secondary"] for s in eval_stats])) if eval_stats else 0.0
    mean_near_misses = float(np.mean([s["near_misses"] for s in eval_stats])) if eval_stats else 0.0
    mean_score = float(np.mean([s["score"] for s in eval_stats])) if eval_stats else 0.0
    tc8_stats = [s for s in eval_stats if bool(s.get("tc8_active", False))]
    tc8_runs = len(tc8_stats)
    tc8_collisions = float(sum(s["collisions"] for s in tc8_stats)) if tc8_stats else 0.0
    tc8_success_rate = compute_tc8_success_rate(tc8_collisions, tc8_runs) if tc8_runs > 0 else float("nan")

    # Multi-objective curriculum score:
    # prioritize collision reduction, then fuel efficiency, then secondary conjunction risk.
    collision_score = float(np.exp(-mean_collisions))
    collision_rate_score = float(np.clip(1.0 - collision_rate, 0.0, 1.0))
    fuel_scale = max(1.0, 0.2 * 1000.0 * num_satellites)
    fuel_score = float(np.exp(-mean_fuel / fuel_scale))
    secondary_score = float(np.exp(-mean_secondary))
    curriculum_score = float(
        np.clip(
            0.55 * collision_score
            + 0.2 * collision_rate_score
            + 0.15 * fuel_score
            + 0.1 * secondary_score,
            0.0,
            1.0,
        )
    )

    return {
        "mean_collisions": mean_collisions,
        "collision_rate": collision_rate,
        "mean_fuel": mean_fuel,
        "mean_maneuvers": mean_maneuvers,
        "mean_secondary": mean_secondary,
        "mean_near_misses": mean_near_misses,
        "mean_score": mean_score,
        "curriculum_score": curriculum_score,
        "tc8_success_rate": tc8_success_rate,
    }


def train() -> None:
    # ── 1. Load YAML config (base layer) ─────────────────────────────────────
    yaml_cfg = _load_yaml_config(
        str(Path(__file__).parent / "config" / "default.yaml")
    )

    parser = argparse.ArgumentParser(
        description="Train MARL Policy  (Phase 1: attention actor, Tsiolkovsky fuel, Foster Pc)"
    )
    # Training
    parser.add_argument("--max-steps",    type=int,   default=_cfg(yaml_cfg,"training","max_steps",    default=120))
    parser.add_argument("--max-episodes", type=int,   default=_cfg(yaml_cfg,"training","max_episodes", default=8000))
    parser.add_argument("--save-dir",     type=str,   default=_cfg(yaml_cfg,"output","save_dir",       default="policies/saved_models"))
    parser.add_argument("--update-every", type=int,   default=_cfg(yaml_cfg,"training","update_every", default=5))
    parser.add_argument("--eval-interval",type=int,   default=_cfg(yaml_cfg,"training","eval_interval",default=50))
    parser.add_argument("--eval-episodes",type=int,   default=_cfg(yaml_cfg,"training","eval_episodes",default=5))
    parser.add_argument("--seed",         type=int,   default=_cfg(yaml_cfg,"training","seed",         default=123))
    # PPO
    parser.add_argument("--entropy-start",type=float, default=_cfg(yaml_cfg,"entropy","start",         default=0.02))
    parser.add_argument("--entropy-end",  type=float, default=_cfg(yaml_cfg,"entropy","end",           default=0.005))
    parser.add_argument("--ppo-epochs",   type=int,   default=_cfg(yaml_cfg,"ppo","ppo_epochs",        default=10))
    parser.add_argument("--batch-size",   type=int,   default=_cfg(yaml_cfg,"ppo","batch_size",        default=128))
    parser.add_argument("--lr",           type=float, default=_cfg(yaml_cfg,"ppo","learning_rate",     default=3e-4))
    # Phase 1: actor architecture
    parser.add_argument("--actor-type",       type=str, default=_cfg(yaml_cfg,"actor","type",              default="attention"),
                        help="mlp | attention | recurrent | ensemble")
    parser.add_argument("--hidden-size",      type=int, default=_cfg(yaml_cfg,"actor","hidden_size",        default=128))
    parser.add_argument("--num-heads",        type=int, default=_cfg(yaml_cfg,"actor","num_heads",          default=4))
    parser.add_argument("--num-tf-layers",    type=int, default=_cfg(yaml_cfg,"actor","num_transformer_layers", default=2))
    parser.add_argument("--ensemble-size",    type=int, default=_cfg(yaml_cfg,"actor","ensemble_size",      default=5))
    # Environment
    parser.add_argument("--tc8-ratio",    type=float, default=_cfg(yaml_cfg,"environment","tc8_ratio",  default=0.40))
    parser.add_argument("--realism",      type=str,   default=str(_cfg(yaml_cfg,"environment","realism",default=True)))
    parser.add_argument("--terminate-on-collision", type=str, default=str(_cfg(yaml_cfg,"training","terminate_on_collision", default=True)))
    # Dataset
    parser.add_argument("--use-dataset",       type=str, default="false")
    parser.add_argument("--dataset-path",      type=str, default=_cfg(yaml_cfg,"dataset","train_path", default="data/train_data.jsonl"))
    parser.add_argument("--dataset-eval-path", type=str, default=_cfg(yaml_cfg,"dataset","eval_path",  default="data/test_data.jsonl"))
    # Phase 1: experiment tracking
    parser.add_argument("--use-wandb",  type=str, default=str(_cfg(yaml_cfg,"tracking","use_wandb",  default=False)))
    parser.add_argument("--use-mlflow", type=str, default=str(_cfg(yaml_cfg,"tracking","use_mlflow", default=False)))
    args = parser.parse_args()

    # ── 2. Resolve settings ──────────────────────────────────────────────────
    use_dataset            = _as_bool(args.use_dataset)
    terminate_on_collision = _as_bool(args.terminate_on_collision)
    use_realism            = _as_bool(args.realism)
    use_wandb              = _as_bool(args.use_wandb)
    use_mlflow             = _as_bool(args.use_mlflow)
    realism_config         = RealismConfig(enabled=use_realism)
    os.makedirs(args.save_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── 3. Build trainer with Phase 1 actor ──────────────────────────────────
    num_satellites = _cfg(yaml_cfg, "environment", "num_satellites", default=3)
    trainer = MARLTrainer(
        num_agents=num_satellites,
        obs_size=OBS_SIZE,
        action_size=ACTION_COUNT,
        learning_rate=args.lr,
        hidden_size=args.hidden_size,
        actor_type=args.actor_type,
        ensemble_size=args.ensemble_size,
        num_heads=args.num_heads,
        num_transformer_layers=args.num_tf_layers,
    )
    curriculum = CurriculumManager()

    # ── 4. Experiment tracker ────────────────────────────────────────────────
    tracker_config = {
        "actor_type": args.actor_type,
        "max_episodes": args.max_episodes,
        "max_steps": args.max_steps,
        "ppo_epochs": args.ppo_epochs,
        "batch_size": args.batch_size,
        "tc8_ratio": args.tc8_ratio,
        "realism": use_realism,
        "ensemble_size": args.ensemble_size,
        "num_heads": args.num_heads,
        "num_tf_layers": args.num_tf_layers,
    }
    tracker = ExperimentTracker(
        use_wandb=use_wandb,
        use_mlflow=use_mlflow,
        project=_cfg(yaml_cfg, "tracking", "wandb_project", default="fuelsafe-marl-leo"),
        entity=_cfg(yaml_cfg, "tracking", "wandb_entity", default=""),
        experiment=_cfg(yaml_cfg, "tracking", "mlflow_experiment", default="FuelSafe-MARL-LEO"),
        config=tracker_config,
    )

    train_scenarios: List[Dict[str, Any]] = []
    eval_scenarios: List[Dict[str, Any]] = []
    if use_dataset:
        train_scenarios = _load_jsonl(args.dataset_path)
        eval_scenarios = _load_jsonl(args.dataset_eval_path)
        if not eval_scenarios:
            eval_scenarios = list(train_scenarios)

    print("\n" + "=" * 70)
    print("STARTING MARL TRAINING  [Phase 1]")
    print(
        f"Actor:  {args.actor_type.upper()} | "
        f"Episodes: {args.max_episodes} | "
        f"TC8: {args.tc8_ratio:.2f} | "
        f"Realism: {use_realism} | "
        f"W&B: {use_wandb} | MLflow: {use_mlflow}"
    )
    if use_dataset:
        print(f"Train scenarios: {len(train_scenarios)} | Eval scenarios: {len(eval_scenarios)}")
    print("=" * 70)

    global_episode = 0
    episodes_since_update = 0
    latest_train_metrics: Dict[str, float] = {}
    eval_history: List[Dict[str, float]] = []

    while global_episode < args.max_episodes:
        denom = max(args.max_episodes - 1, 1)
        entropy_coeff = args.entropy_start - (
            (args.entropy_start - args.entropy_end) * (global_episode / denom)
        )
        trainer.set_entropy(entropy_coeff)

        config = _sample_training_config(
            use_dataset=use_dataset,
            dataset_scenarios=train_scenarios,
            curriculum=curriculum,
            tc8_ratio=args.tc8_ratio,
        )

        env = MultiAgentOrbitalEnv(
            num_satellites=num_satellites,
            num_debris=int(config["max_debris"]),
            collision_threshold_km=float(config["collision_threshold_km"]),
            distance_threshold_km=float(config.get("distance_threshold_km", 250.0)),
            high_risk_mode=bool(config.get("high_risk_mode", True)),
            scenario_config=config.get("scenario_config"),
            hard_scenario_probability=float(config.get("hard_scenario_probability", 0.0)),
            secondary_conjunction_risk_threshold=0.01,
            realism_config=realism_config,
        )

        # Reset recurrent hidden states at episode start
        if args.actor_type == "recurrent":
            trainer.reset_hidden_states()

        episode_stats = _run_episode(
            trainer=trainer,
            env=env,
            max_steps=args.max_steps,
            deterministic=False,
            collect_experience=True,
            terminate_on_collision=terminate_on_collision,
        )

        episodes_since_update += 1
        global_episode += 1

        if episodes_since_update >= args.update_every:
            latest_train_metrics = trainer.train(num_epochs=args.ppo_epochs, batch_size=args.batch_size)
            episodes_since_update = 0

        if global_episode % 10 == 0:
            # Epistemic uncertainty (ensemble mode)
            mean_uncertainty = float(np.mean(list(
                getattr(trainer, "_last_uncertainties", {}).values()
            ))) if hasattr(trainer, "_last_uncertainties") else 0.0

            print(
                f"Ep {global_episode:4d} | "
                f"Actor: {args.actor_type[:3].upper()} | "
                f"Coll: {episode_stats['collisions']:.1f} | "
                f"Fuel: {episode_stats['fuel']:6.2f} | "
                f"Man: {episode_stats['maneuvers']:5.1f} | "
                f"Score: {episode_stats['score']:7.2f} | "
                f"Unc: {mean_uncertainty:.3f} | "
                f"Loss: {latest_train_metrics.get('actor_loss', 0.0):.4f}"
            )
            # Log to tracker
            tracker.log({
                "episode/collisions":  episode_stats["collisions"],
                "episode/fuel":        episode_stats["fuel"],
                "episode/maneuvers":   episode_stats["maneuvers"],
                "episode/score":       episode_stats["score"],
                "episode/near_misses": episode_stats["near_misses"],
                "train/actor_loss":    latest_train_metrics.get("actor_loss", 0.0),
                "train/critic_loss":   latest_train_metrics.get("critic_loss", 0.0),
                "train/entropy":       latest_train_metrics.get("entropy", 0.0),
                "train/entropy_coeff": entropy_coeff,
                "train/uncertainty":   mean_uncertainty,
                "curriculum/stage":    curriculum.current_stage_idx + 1,
            }, step=global_episode)

        if global_episode % args.eval_interval == 0:
            eval_metrics = _evaluate_policy(
                trainer=trainer,
                curriculum=curriculum,
                eval_scenarios=eval_scenarios,
                use_dataset=use_dataset,
                tc8_ratio=args.tc8_ratio,
                eval_episodes=args.eval_episodes,
                max_steps=args.max_steps,
                num_satellites=num_satellites,
                terminate_on_collision=terminate_on_collision,
                realism_config=realism_config,
            )
            curriculum.update_performance(eval_metrics["curriculum_score"])
            eval_history.append(eval_metrics)

            print(
                f"[Eval @ Ep {global_episode}] "
                f"Coll={eval_metrics['mean_collisions']:.3f}, "
                f"CollRate={eval_metrics['collision_rate']:.3f}, "
                f"Fuel={eval_metrics['mean_fuel']:.2f}, "
                f"Man={eval_metrics['mean_maneuvers']:.2f}, "
                f"Sec={eval_metrics['mean_secondary']:.3f}, "
                f"Near={eval_metrics['mean_near_misses']:.3f}, "
                f"Obj={eval_metrics['mean_score']:.2f}, "
                f"Curr={eval_metrics['curriculum_score']:.3f}, "
                f"TC8={eval_metrics['tc8_success_rate']:.3f}"
            )
            # Log eval metrics to tracker
            tracker.log({
                f"eval/mean_collisions":   eval_metrics["mean_collisions"],
                f"eval/collision_rate":    eval_metrics["collision_rate"],
                f"eval/mean_fuel":         eval_metrics["mean_fuel"],
                f"eval/mean_maneuvers":    eval_metrics["mean_maneuvers"],
                f"eval/mean_score":        eval_metrics["mean_score"],
                f"eval/curriculum_score":  eval_metrics["curriculum_score"],
                f"eval/tc8_success_rate":  eval_metrics.get("tc8_success_rate", 0.0) or 0.0,
            }, step=global_episode)

            save_path = os.path.join(args.save_dir, f"mppo_checkpoint_{global_episode}.pt")
            trainer.save(save_path)
            save_pareto_artifacts(eval_history, args.save_dir, "periodic")

    if episodes_since_update > 0:
        trainer.train(num_epochs=args.ppo_epochs, batch_size=args.batch_size)

    print("\nTRAINING COMPLETE")
    final_path = os.path.join(args.save_dir, "mppo_final.pt")
    trainer.save(final_path)
    save_pareto_artifacts(eval_history, args.save_dir, "final")
    print(f"Final model saved to {final_path}")
    tracker.finish()


if __name__ == "__main__":
    train()
