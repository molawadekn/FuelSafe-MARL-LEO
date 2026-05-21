"""
MODULE 6: Multi-Agent Reinforcement Learning Training Layer
Implements a lightweight MAPPO-style trainer with centralized critic input.

Phase 1 upgrades
----------------
* actor_type parameter selects: "mlp" | "attention" | "recurrent" | "ensemble".
* ensemble_size controls how many members EnsembleActor uses.
* Recurrent mode: hidden states per agent are managed by the trainer and
  reset at episode boundaries via reset_hidden_states().
* get_action_details now returns per-agent epistemic_uncertainty when using
  an EnsembleActor.
* Uncertainty is logged in training_stats for W&B / MLflow export.

Phase 2 upgrades
----------------
* actor_type="shared"       — single SharedAttentionActor + agent-ID embedding.
* actor_type="hierarchical" — FleetCommander + GoalConditionedActor sub-policies.
* use_comm=True             — TarMAC inter-satellite communication round before
                              action selection (CommAwareAttentionActor).
* use_graph_critic=True     — GAT centralized critic (GraphAttentionCritic)
                              instead of flat MLP concatenation critic.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sim.maneuver_engine import ACTION_COUNT, MAX_DELTA_V_PER_STEP_KMS
from sim.observation_utils import OBS_SIZE


class ActorNetwork(nn.Module):
    """Actor network (policy network) for MAPPO, updated for Hybrid Action Space."""

    def __init__(self, input_size: int = OBS_SIZE, output_size: int = ACTION_COUNT, hidden_size: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.direction_head = nn.Linear(hidden_size, output_size)
        self.magnitude_mean_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        self.magnitude_logstd = nn.Parameter(torch.tensor([np.log(5.0e-4)], dtype=torch.float32))
        self.max_dv = float(MAX_DELTA_V_PER_STEP_KMS)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(x)
        logits = self.direction_head(features)
        mag_mean = self.magnitude_mean_head(features) * self.max_dv
        return logits, mag_mean

    def distribution(self, x: torch.Tensor) -> Tuple[torch.distributions.Categorical, torch.distributions.Normal]:
        logits, mag_mean = self.forward(x)
        dir_dist = torch.distributions.Categorical(logits=logits)
        mag_std = torch.exp(self.magnitude_logstd).clamp(min=5.0e-5, max=1.5e-3).expand_as(mag_mean)
        mag_dist = torch.distributions.Normal(mag_mean, mag_std)
        return dir_dist, mag_dist

    def get_action(
        self, state: np.ndarray, device: str = "cpu", deterministic: bool = False
    ) -> Tuple[Tuple[int, float], float]:
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            dir_dist, mag_dist = self.distribution(state_tensor)

            if deterministic:
                dir_action = torch.argmax(dir_dist.probs, dim=-1)
                mag_action = mag_dist.mean
            else:
                dir_action = dir_dist.sample()
                mag_action = mag_dist.sample()

            mag_action = torch.clamp(mag_action, 0.0, self.max_dv)
            log_prob = dir_dist.log_prob(dir_action) + mag_dist.log_prob(mag_action).squeeze(-1)

        return (int(dir_action.item()), float(mag_action.item())), float(log_prob.item())


class CriticNetwork(nn.Module):
    """Centralized critic over concatenated agent observations."""

    def __init__(self, input_size: int = OBS_SIZE * 3, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PPOBuffer:
    """Per-agent trajectory buffer for PPO/MAPPO updates."""

    def __init__(self, buffer_size: int = 4096):
        self.buffer_size = buffer_size
        self.obs: Deque[np.ndarray] = deque(maxlen=buffer_size)
        self.central_obs: Deque[np.ndarray] = deque(maxlen=buffer_size)
        self.actions: Deque[np.ndarray] = deque(maxlen=buffer_size)
        self.rewards: Deque[float] = deque(maxlen=buffer_size)
        self.dones: Deque[float] = deque(maxlen=buffer_size)
        self.log_probs: Deque[float] = deque(maxlen=buffer_size)
        self.values: Deque[float] = deque(maxlen=buffer_size)
        self.next_values: Deque[float] = deque(maxlen=buffer_size)

    def store(
        self,
        obs: np.ndarray,
        central_obs: np.ndarray,
        action: Tuple[int, float],
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
        next_value: float,
    ) -> None:
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.central_obs.append(np.asarray(central_obs, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.next_values.append(float(next_value))

    def clear(self) -> None:
        self.obs.clear()
        self.central_obs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.next_values.clear()

    def __len__(self) -> int:
        return len(self.obs)

    def as_arrays(self) -> Dict[str, np.ndarray]:
        return {
            "obs": np.asarray(self.obs, dtype=np.float32),
            "central_obs": np.asarray(self.central_obs, dtype=np.float32),
            "actions": np.asarray(self.actions, dtype=np.float32),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "dones": np.asarray(self.dones, dtype=np.float32),
            "log_probs": np.asarray(self.log_probs, dtype=np.float32),
            "values": np.asarray(self.values, dtype=np.float32),
            "next_values": np.asarray(self.next_values, dtype=np.float32),
        }


class MARLTrainer:
    """
    MAPPO trainer using centralized training with decentralized execution.

    Phase 1: supports attention / recurrent / ensemble actor types.
    Phase 2: adds shared-parameter actors, hierarchical fleet controller,
             TarMAC inter-satellite communication, and GAT centralized critic.
    """

    def __init__(
        self,
        num_agents: int = 3,
        obs_size: int = OBS_SIZE,
        action_size: int = ACTION_COUNT,
        hidden_size: int = 128,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coeff: float = 0.01,
        value_loss_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
        clip_ratio: float = 0.2,
        device: str = "cpu",
        # Phase 1 additions
        actor_type: str = "mlp",          # "mlp" | "attention" | "recurrent" | "ensemble"
        ensemble_size: int = 5,           # used when actor_type == "ensemble"
        num_heads: int = 4,               # attention head count
        num_transformer_layers: int = 2,  # transformer depth
        # Phase 2 additions
        use_comm: bool = False,           # TarMAC inter-satellite communication
        use_graph_critic: bool = False,   # GAT critic instead of flat MLP
        message_size: int = 16,          # TarMAC message vector size
        sig_size: int = 16,              # TarMAC signature vector size
        comm_delay: bool = False,         # 1-step comm delay (realistic latency)
        goal_size: int = 16,             # commander goal vector size (hierarchical)
        commander_k: int = 5,            # commander fires every K steps (hierarchical)
    ):
        self.num_agents   = num_agents
        self.obs_size     = obs_size
        self.action_size  = action_size
        self.device       = device
        self.actor_type   = actor_type

        self.gamma             = gamma
        self.gae_lambda        = gae_lambda
        self.entropy_coeff     = entropy_coeff
        self.value_loss_coeff  = value_loss_coeff
        self.max_grad_norm     = max_grad_norm
        self.clip_ratio        = clip_ratio

        self._use_comm         = use_comm
        self._use_graph_critic = use_graph_critic
        self._goal_size        = goal_size

        # Episode step counter (used by hierarchical controller)
        self._episode_step: int = 0

        # Build agent IDs
        agent_ids = [f"SAT_{i:03d}" for i in range(num_agents)]

        # ── Build actors ──────────────────────────────────────────────────────
        self.hierarchical_controller = None
        self._shared_optimizer       = None
        self._commander_optimizer    = None
        self._is_shared_mode         = False

        if actor_type == "shared":
            # Phase 2: single SharedAttentionActor for all satellites
            from marl.shared_actor import SharedAttentionActor, SharedMARLTrainerAdapter
            shared_actor = SharedAttentionActor(
                input_size=obs_size,
                output_size=action_size,
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_transformer_layers=num_transformer_layers,
                max_agents=max(16, num_agents),
                embed_size=8,
            ).to(device)
            self.actors = SharedMARLTrainerAdapter(shared_actor, agent_ids)
            self._shared_actor    = shared_actor
            self._is_shared_mode  = True
            # Single optimizer (all agents share the same weights)
            self._shared_optimizer = optim.Adam(shared_actor.parameters(), lr=learning_rate)
            self.actor_optimizers  = {}   # unused in shared mode

        elif actor_type == "hierarchical":
            # Phase 2: FleetCommander + per-agent GoalConditionedActor
            from marl.hierarchical import HierarchicalController
            self.hierarchical_controller = HierarchicalController(
                agent_ids=agent_ids,
                obs_size=obs_size,
                hidden_size=hidden_size,
                goal_size=goal_size,
                commander_k=commander_k,
                num_heads=num_heads,
                num_transformer_layers=num_transformer_layers,
                device=device,
            )
            self.actors = self.hierarchical_controller.satellite_actors
            # Per-actor optimizers for satellite policies
            self.actor_optimizers = {
                aid: optim.Adam(actor.parameters(), lr=learning_rate)
                for aid, actor in self.actors.items()
            }
            # Separate optimizer for the commander
            self._commander_optimizer = optim.Adam(
                self.hierarchical_controller.commander.parameters(), lr=learning_rate
            )

        else:
            # Phase 1 actor types: mlp | attention | recurrent | ensemble
            from marl.attention_actor import build_actor
            self.actors: Dict[str, nn.Module] = {
                f"SAT_{i:03d}": build_actor(
                    actor_type=actor_type,
                    input_size=obs_size,
                    output_size=action_size,
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    num_transformer_layers=num_transformer_layers,
                    ensemble_size=ensemble_size,
                ).to(device)
                for i in range(num_agents)
            }
            self.actor_optimizers = {
                agent_id: optim.Adam(actor.parameters(), lr=learning_rate)
                for agent_id, actor in self.actors.items()
            }

        # ── Build critic ──────────────────────────────────────────────────────
        if use_graph_critic:
            from marl.graph_critic import GraphAttentionCritic
            self.critic = GraphAttentionCritic(
                obs_size=obs_size,
                hidden_size=hidden_size,
                num_heads=num_heads,
            ).to(device)
        else:
            self.critic = CriticNetwork(obs_size * num_agents, hidden_size).to(device)

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        # ── Build communication module (Phase 2) ──────────────────────────────
        self._comm_actor     = None
        self._comm_optimizer = None
        if use_comm:
            from marl.communication import CommAwareAttentionActor
            self._comm_actor = CommAwareAttentionActor(
                input_size=obs_size,
                output_size=action_size,
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_transformer_layers=num_transformer_layers,
                message_size=message_size,
                sig_size=sig_size,
                use_gate=True,
                comm_delay=comm_delay,
            ).to(device)
            self._comm_optimizer = optim.Adam(
                self._comm_actor.parameters(), lr=learning_rate
            )

        # ── Buffers and stats ─────────────────────────────────────────────────
        self.buffers = {agent_id: PPOBuffer() for agent_id in agent_ids}
        self.training_stats: List[Dict[str, float]] = []

        # ── Recurrent hidden states (used when actor_type == "recurrent") ─────
        self._hidden_states: Dict[str, Optional[torch.Tensor]] = {
            aid: None for aid in agent_ids
        }

        # Latest per-agent epistemic uncertainties
        self._last_uncertainties: Dict[str, float] = {aid: 0.0 for aid in agent_ids}

    # ─────────────────────────────────────────────────────────────────────────
    # Episode management
    # ─────────────────────────────────────────────────────────────────────────

    def set_entropy(self, coeff: float) -> None:
        """Update entropy coefficient for scheduled annealing."""
        self.entropy_coeff = float(coeff)

    def reset_hidden_states(self) -> None:
        """
        Reset GRU hidden states for all agents.
        Call at the start of every episode when using actor_type='recurrent'.
        """
        for aid, actor in self.actors.items():
            if hasattr(actor, "initial_hidden"):
                self._hidden_states[aid] = actor.initial_hidden(self.device)
            else:
                self._hidden_states[aid] = None

    def reset_episode(self) -> None:
        """
        Reset all episode-level state.  Call at the start of each episode:
        - Clears step counter for hierarchical commander.
        - Clears comm delay buffers.
        - Resets hierarchical goal vectors.
        """
        self._episode_step = 0
        if self.hierarchical_controller is not None:
            self.hierarchical_controller.reset()
        if self._comm_actor is not None:
            self._comm_actor.reset_comm()

    # ─────────────────────────────────────────────────────────────────────────
    # Centralized value estimates
    # ─────────────────────────────────────────────────────────────────────────

    def _build_central_observation(self, observations: Dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate observations in a stable agent order for the centralized critic."""
        central_obs = []
        for agent_id in self.buffers.keys():
            obs = observations.get(agent_id)
            if obs is None:
                obs = np.zeros(self.obs_size, dtype=np.float32)
            central_obs.append(np.asarray(obs, dtype=np.float32))
        return np.concatenate(central_obs, axis=0).astype(np.float32)

    def _critic_value(
        self,
        central_obs: np.ndarray,
        observations: Optional[Dict[str, np.ndarray]] = None,
    ) -> float:
        """Compute V(s) — routes to GAT critic or flat MLP critic."""
        with torch.no_grad():
            if self._use_graph_critic and observations:
                return float(
                    self.critic.forward(observations, device=self.device)
                )
            central_obs_t = torch.as_tensor(
                central_obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            return float(self.critic(central_obs_t).item())

    def _log_prob_for_action(self, actor: nn.Module, obs: np.ndarray, action: Tuple[int, float]) -> float:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            dir_action_t = torch.as_tensor([action[0]], dtype=torch.long, device=self.device)
            mag_action_t = torch.as_tensor([[action[1]]], dtype=torch.float32, device=self.device)
            dir_dist, mag_dist = actor.distribution(obs_t)
            return float((dir_dist.log_prob(dir_action_t) + mag_dist.log_prob(mag_action_t).squeeze(-1)).item())

    # ─────────────────────────────────────────────────────────────────────────
    # Action selection
    # ─────────────────────────────────────────────────────────────────────────

    def get_action_details(
        self, observations: Dict[str, np.ndarray], deterministic: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, float], float]:
        """
        Return joint actions, per-agent log-probs, and centralized value estimate.

        Phase 1: handles recurrent (GRU hidden states) and ensemble actors.
        Phase 2: handles shared-parameter, hierarchical, and TarMAC comm actors.
        """
        central_obs = self._build_central_observation(observations)
        value       = self._critic_value(central_obs, observations)

        # ── Phase 2: Hierarchical controller ─────────────────────────────────
        if self.actor_type == "hierarchical" and self.hierarchical_controller is not None:
            actions, log_probs = self.hierarchical_controller.get_actions(
                observations,
                step=self._episode_step,
                deterministic=deterministic,
            )
            self._episode_step += 1
            self._last_uncertainties = {aid: 0.0 for aid in actions}
            return actions, log_probs, value

        # ── Phase 2: TarMAC comm actor ────────────────────────────────────────
        if self._use_comm and self._comm_actor is not None:
            all_obs_tensors: Dict[str, torch.Tensor] = {}
            for aid, obs in observations.items():
                if obs is not None:
                    all_obs_tensors[aid] = torch.as_tensor(
                        obs, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)

            outputs = self._comm_actor.forward_with_comm(all_obs_tensors)
            actions:   Dict[str, Any]   = {}
            log_probs: Dict[str, float] = {}

            for aid, (logits, mag_mean) in outputs.items():
                dir_dist = torch.distributions.Categorical(logits=logits)
                mag_std  = torch.exp(self._comm_actor.magnitude_logstd).clamp(5e-5, 1.5e-3).expand_as(mag_mean)
                mag_dist = torch.distributions.Normal(mag_mean, mag_std)

                if deterministic:
                    dir_action = torch.argmax(dir_dist.probs, dim=-1)
                    mag_action = mag_dist.mean
                else:
                    dir_action = dir_dist.sample()
                    mag_action = mag_dist.sample()

                mag_action = mag_action.clamp(0.0, self._comm_actor.max_dv)
                log_prob   = (
                    dir_dist.log_prob(dir_action)
                    + mag_dist.log_prob(mag_action).squeeze(-1)
                )
                actions[aid]   = (int(dir_action.item()), float(mag_action.item()))
                log_probs[aid] = float(log_prob.item())

            # Fill in any missing agents
            for aid in self.buffers:
                if aid not in actions:
                    actions[aid]   = (0, 0.0)
                    log_probs[aid] = 0.0

            self._last_uncertainties = {aid: 0.0 for aid in actions}
            self._episode_step += 1
            return actions, log_probs, value

        # ── Phase 1 / standard paths ──────────────────────────────────────────
        actions:       Dict[str, Any]   = {}
        log_probs:     Dict[str, float] = {}
        uncertainties: Dict[str, float] = {}

        for agent_id, actor in self.actors.items():
            obs = observations.get(agent_id)
            if obs is None:
                actions[agent_id]       = (0, 0.0)
                log_probs[agent_id]     = 0.0
                uncertainties[agent_id] = 0.0
                continue

            # Phase 2: shared actor proxy (agent_id-aware get_action)
            if self._is_shared_mode:
                action, log_prob = actor.get_action(obs, self.device, deterministic)
                uncertainties[agent_id] = 0.0

            # Phase 1: recurrent actor — thread hidden state
            elif hasattr(actor, "get_action") and hasattr(actor, "initial_hidden"):
                h = self._hidden_states.get(agent_id)
                if h is None:
                    h = actor.initial_hidden(self.device)
                action, log_prob, h_new = actor.get_action(
                    obs, self.device, deterministic=deterministic, hidden=h
                )
                self._hidden_states[agent_id] = h_new
                uncertainties[agent_id] = 0.0

            # Phase 1: ensemble actor — capture epistemic uncertainty
            elif hasattr(actor, "get_uncertainty"):
                action, log_prob = actor.get_action(obs, self.device, deterministic=deterministic)
                uncertainties[agent_id] = actor.get_uncertainty(obs, self.device)

            # Standard actor (MLP or attention)
            else:
                action, log_prob = actor.get_action(obs, self.device, deterministic=deterministic)
                uncertainties[agent_id] = 0.0

            actions[agent_id]   = action
            log_probs[agent_id] = log_prob

        self._last_uncertainties = uncertainties
        self._episode_step += 1
        return actions, log_probs, value

    def get_actions(
        self, observations: Dict[str, np.ndarray], deterministic: bool = False
    ) -> Dict[str, Any]:
        actions, _, _ = self.get_action_details(observations, deterministic=deterministic)
        return actions

    # ─────────────────────────────────────────────────────────────────────────
    # Experience collection
    # ─────────────────────────────────────────────────────────────────────────

    def collect_experience(
        self,
        observations: Dict[str, np.ndarray],
        rewards: Dict[str, float],
        next_observations: Dict[str, np.ndarray],
        dones: Dict[str, bool],
        actions: Dict[str, Any],
        log_probs: Optional[Dict[str, float]] = None,
        central_value: Optional[float] = None,
    ) -> None:
        """Store environment experience into per-agent replay buffers."""
        central_obs      = self._build_central_observation(observations)
        next_central_obs = self._build_central_observation(next_observations)

        if central_value is None:
            central_value = self._critic_value(central_obs, observations)
        next_value = self._critic_value(next_central_obs, next_observations)

        for agent_id in self.buffers:
            obs = observations.get(agent_id)
            if obs is None:
                continue

            actor  = self.actors[agent_id] if agent_id in self.actors else None
            action = actions.get(agent_id, (0, 0.0))
            reward = float(rewards.get(agent_id, 0.0))
            done   = bool(dones.get(agent_id, dones.get("__all__", False)))

            if log_probs is not None and agent_id in log_probs:
                log_prob = float(log_probs[agent_id])
            elif actor is not None:
                log_prob = self._log_prob_for_action(actor, obs, action)
            else:
                log_prob = 0.0

            self.buffers[agent_id].store(
                obs=obs,
                central_obs=central_obs,
                action=action,
                reward=reward,
                done=done,
                log_prob=log_prob,
                value=central_value,
                next_value=next_value,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # PPO / GAE computation
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_gae(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        next_values: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute generalized advantage estimates and critic targets."""
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(len(rewards))):
            mask  = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_values[t] * mask - values[t]
            gae   = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages.astype(np.float32), returns.astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────────

    def train(self, num_epochs: int = 10, batch_size: int = 128) -> Dict[str, float]:
        """Train all actors and the centralized critic using collected rollouts."""
        stats = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy": 0.0,
            "policy_loss": 0.0,
        }

        actor_updates  = 0
        critic_updates = 0
        critic_obs_batches:    List[np.ndarray] = []
        critic_return_batches: List[np.ndarray] = []

        # ── Actor training ────────────────────────────────────────────────────
        if self._is_shared_mode:
            # Single shared actor: combine all agents' data into one batch
            stats, actor_updates = self._train_shared_actor(
                num_epochs, batch_size, stats, actor_updates, critic_obs_batches, critic_return_batches
            )
        elif self._use_comm and self._comm_actor is not None:
            # TarMAC actor: train CommAwareAttentionActor with combined data
            stats, actor_updates = self._train_comm_actor(
                num_epochs, batch_size, stats, actor_updates, critic_obs_batches, critic_return_batches
            )
        else:
            # Standard per-agent training (Phase 1 path)
            for agent_id, actor in self.actors.items():
                buffer = self.buffers[agent_id]
                if len(buffer) == 0:
                    continue

                data = buffer.as_arrays()
                advantages, returns = self._compute_gae(
                    data["rewards"], data["dones"], data["values"], data["next_values"]
                )
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                obs_t           = torch.as_tensor(data["obs"],       dtype=torch.float32, device=self.device)
                actions_t       = torch.as_tensor(data["actions"],   dtype=torch.float32, device=self.device)
                old_log_probs_t = torch.as_tensor(data["log_probs"], dtype=torch.float32, device=self.device)
                advantages_t    = torch.as_tensor(advantages,        dtype=torch.float32, device=self.device)

                critic_obs_batches.append(data["central_obs"])
                critic_return_batches.append(returns)

                optimizer   = self.actor_optimizers[agent_id]
                num_samples = len(data["actions"])

                for _ in range(num_epochs):
                    permutation = np.random.permutation(num_samples)
                    for start in range(0, num_samples, batch_size):
                        idx = permutation[start: start + batch_size]

                        dir_dist, mag_dist = actor.distribution(obs_t[idx])
                        batch_dir = actions_t[idx, 0].long()
                        batch_mag = actions_t[idx, 1].unsqueeze(-1)

                        new_lp  = dir_dist.log_prob(batch_dir) + mag_dist.log_prob(batch_mag).squeeze(-1)
                        entropy = (dir_dist.entropy() + mag_dist.entropy().squeeze(-1)).mean()

                        ratio  = torch.exp(new_lp - old_log_probs_t[idx])
                        surr1  = ratio * advantages_t[idx]
                        surr2  = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages_t[idx]

                        policy_loss = -torch.min(surr1, surr2).mean()
                        actor_loss  = policy_loss - self.entropy_coeff * entropy

                        optimizer.zero_grad()
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
                        optimizer.step()

                        stats["actor_loss"]  += float(actor_loss.item())
                        stats["policy_loss"] += float(policy_loss.item())
                        stats["entropy"]     += float(entropy.item())
                        actor_updates += 1

        # ── Critic training ───────────────────────────────────────────────────
        if critic_obs_batches:
            critic_obs     = np.concatenate(critic_obs_batches,    axis=0)
            critic_returns = np.concatenate(critic_return_batches, axis=0)
            critic_obs_t     = torch.as_tensor(critic_obs,     dtype=torch.float32, device=self.device)
            critic_returns_t = torch.as_tensor(critic_returns, dtype=torch.float32, device=self.device)

            num_samples = critic_obs_t.shape[0]
            for _ in range(num_epochs):
                permutation = np.random.permutation(num_samples)
                for start in range(0, num_samples, batch_size):
                    idx         = permutation[start: start + batch_size]
                    batch_obs   = critic_obs_t[idx]
                    batch_ret   = critic_returns_t[idx]

                    if self._use_graph_critic:
                        # GraphAttentionCritic.forward_tensor accepts flat concat format
                        predicted_values = self.critic.forward_tensor(
                            batch_obs, num_agents=self.num_agents
                        ).squeeze(-1)
                    else:
                        predicted_values = self.critic(batch_obs).squeeze(-1)

                    critic_loss = nn.MSELoss()(predicted_values, batch_ret)

                    self.critic_optimizer.zero_grad()
                    (self.value_loss_coeff * critic_loss).backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optimizer.step()

                    stats["critic_loss"] += float(critic_loss.item())
                    critic_updates += 1

        # ── Normalise and clear ───────────────────────────────────────────────
        if actor_updates > 0:
            stats["actor_loss"]  /= actor_updates
            stats["policy_loss"] /= actor_updates
            stats["entropy"]     /= actor_updates
        if critic_updates > 0:
            stats["critic_loss"] /= critic_updates

        for buffer in self.buffers.values():
            buffer.clear()

        self.training_stats.append(stats)
        return stats

    def _train_shared_actor(
        self,
        num_epochs: int,
        batch_size: int,
        stats: Dict[str, float],
        actor_updates: int,
        critic_obs_batches: List,
        critic_return_batches: List,
    ) -> Tuple[Dict[str, float], int]:
        """
        Train the single shared actor using all agents' combined experience.
        Only ONE gradient update pass per epoch (avoids redundant weight updates).
        """
        all_obs:      List[np.ndarray] = []
        all_actions:  List[np.ndarray] = []
        all_lp:       List[float]      = []
        all_adv:      List[float]      = []

        for agent_id in self.buffers:
            buffer = self.buffers[agent_id]
            if len(buffer) == 0:
                continue
            data = buffer.as_arrays()
            advantages, returns = self._compute_gae(
                data["rewards"], data["dones"], data["values"], data["next_values"]
            )
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            all_obs.extend(data["obs"])
            all_actions.extend(data["actions"])
            all_lp.extend(data["log_probs"])
            all_adv.extend(advantages.tolist())

            critic_obs_batches.append(data["central_obs"])
            critic_return_batches.append(returns)

        if not all_obs:
            return stats, actor_updates

        obs_t        = torch.as_tensor(np.array(all_obs,     dtype=np.float32), dtype=torch.float32, device=self.device)
        actions_t    = torch.as_tensor(np.array(all_actions, dtype=np.float32), dtype=torch.float32, device=self.device)
        old_lp_t     = torch.as_tensor(np.array(all_lp,     dtype=np.float32), dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(np.array(all_adv,     dtype=np.float32), dtype=torch.float32, device=self.device)
        num_samples  = obs_t.shape[0]

        shared_actor = self._shared_actor
        optimizer    = self._shared_optimizer

        for _ in range(num_epochs):
            permutation = np.random.permutation(num_samples)
            for start in range(0, num_samples, batch_size):
                idx = permutation[start: start + batch_size]

                # SharedAttentionActor.distribution(obs, agent_idx=0) — use default idx
                dir_dist, mag_dist = shared_actor.distribution(obs_t[idx], agent_idx=0)
                batch_dir = actions_t[idx, 0].long()
                batch_mag = actions_t[idx, 1].unsqueeze(-1)

                new_lp  = dir_dist.log_prob(batch_dir) + mag_dist.log_prob(batch_mag).squeeze(-1)
                entropy = (dir_dist.entropy() + mag_dist.entropy().squeeze(-1)).mean()

                ratio  = torch.exp(new_lp - old_lp_t[idx])
                surr1  = ratio * advantages_t[idx]
                surr2  = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_t[idx]

                policy_loss = -torch.min(surr1, surr2).mean()
                actor_loss  = policy_loss - self.entropy_coeff * entropy

                optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(shared_actor.parameters(), self.max_grad_norm)
                optimizer.step()

                stats["actor_loss"]  += float(actor_loss.item())
                stats["policy_loss"] += float(policy_loss.item())
                stats["entropy"]     += float(entropy.item())
                actor_updates += 1

        return stats, actor_updates

    def _train_comm_actor(
        self,
        num_epochs: int,
        batch_size: int,
        stats: Dict[str, float],
        actor_updates: int,
        critic_obs_batches: List,
        critic_return_batches: List,
    ) -> Tuple[Dict[str, float], int]:
        """
        Train the CommAwareAttentionActor (single-agent forward, no comm context).
        Uses the compatibility shim `comm_actor.forward(obs)` for PPO update.
        """
        all_obs:     List[np.ndarray] = []
        all_actions: List[np.ndarray] = []
        all_lp:      List[float]      = []
        all_adv:     List[float]      = []

        for agent_id in self.buffers:
            buffer = self.buffers[agent_id]
            if len(buffer) == 0:
                continue
            data = buffer.as_arrays()
            advantages, returns = self._compute_gae(
                data["rewards"], data["dones"], data["values"], data["next_values"]
            )
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            all_obs.extend(data["obs"])
            all_actions.extend(data["actions"])
            all_lp.extend(data["log_probs"])
            all_adv.extend(advantages.tolist())

            critic_obs_batches.append(data["central_obs"])
            critic_return_batches.append(returns)

        if not all_obs:
            return stats, actor_updates

        obs_t        = torch.as_tensor(np.array(all_obs,     dtype=np.float32), dtype=torch.float32, device=self.device)
        actions_t    = torch.as_tensor(np.array(all_actions, dtype=np.float32), dtype=torch.float32, device=self.device)
        old_lp_t     = torch.as_tensor(np.array(all_lp,     dtype=np.float32), dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(np.array(all_adv,     dtype=np.float32), dtype=torch.float32, device=self.device)
        num_samples  = obs_t.shape[0]

        comm_actor = self._comm_actor
        optimizer  = self._comm_optimizer

        for _ in range(num_epochs):
            permutation = np.random.permutation(num_samples)
            for start in range(0, num_samples, batch_size):
                idx = permutation[start: start + batch_size]

                # CommAwareAttentionActor.distribution delegates to the encoder (no comm)
                dir_dist, mag_dist = comm_actor.distribution(obs_t[idx])
                batch_dir = actions_t[idx, 0].long()
                batch_mag = actions_t[idx, 1].unsqueeze(-1)

                new_lp  = dir_dist.log_prob(batch_dir) + mag_dist.log_prob(batch_mag).squeeze(-1)
                entropy = (dir_dist.entropy() + mag_dist.entropy().squeeze(-1)).mean()

                ratio  = torch.exp(new_lp - old_lp_t[idx])
                surr1  = ratio * advantages_t[idx]
                surr2  = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_t[idx]

                policy_loss = -torch.min(surr1, surr2).mean()
                actor_loss  = policy_loss - self.entropy_coeff * entropy

                optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(comm_actor.parameters(), self.max_grad_norm)
                optimizer.step()

                stats["actor_loss"]  += float(actor_loss.item())
                stats["policy_loss"] += float(policy_loss.item())
                stats["entropy"]     += float(entropy.item())
                actor_updates += 1

        return stats, actor_updates

    # ─────────────────────────────────────────────────────────────────────────
    # Checkpoint I/O
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, filepath: str) -> None:
        """Save trained weights and trainer metadata."""
        state_dict: Dict[str, Any] = {
            "num_agents":  self.num_agents,
            "obs_size":    self.obs_size,
            "action_size": self.action_size,
            "actor_type":  self.actor_type,
            "critic":      self.critic.state_dict(),
        }

        if self._is_shared_mode:
            state_dict["shared_actor"] = self._shared_actor.state_dict()
        elif self.actor_type == "hierarchical" and self.hierarchical_controller is not None:
            state_dict["commander"] = self.hierarchical_controller.commander.state_dict()
            state_dict["actors"]    = {
                aid: a.state_dict()
                for aid, a in self.hierarchical_controller.satellite_actors.items()
            }
        else:
            state_dict["actors"] = {aid: a.state_dict() for aid, a in self.actors.items()}

        if self._use_comm and self._comm_actor is not None:
            state_dict["comm_actor"] = self._comm_actor.state_dict()

        torch.save(state_dict, filepath)

    def load(self, filepath: str) -> None:
        """Load trained weights."""
        state_dict = torch.load(filepath, map_location=self.device)

        # Shared actor checkpoint
        if "shared_actor" in state_dict and self._is_shared_mode:
            self._load_partial_state_dict(
                self._shared_actor, state_dict["shared_actor"], label="shared_actor"
            )
        # Hierarchical checkpoint
        elif self.actor_type == "hierarchical" and self.hierarchical_controller is not None:
            if "commander" in state_dict:
                self._load_partial_state_dict(
                    self.hierarchical_controller.commander, state_dict["commander"], label="commander"
                )
            saved_actors = state_dict.get("actors", {})
            for aid, actor in self.hierarchical_controller.satellite_actors.items():
                if aid in saved_actors:
                    self._load_partial_state_dict(actor, saved_actors[aid], label=f"actor[{aid}]")
        # Standard per-agent or shared (broadcast) checkpoint
        else:
            saved_actors = state_dict.get("actors", {})
            if not saved_actors and "actor" in state_dict:
                print(f"[INFO] Found shared 'actor' key in {filepath}. Broadcasting to all agents.")
                shared_state = state_dict["actor"]
                for agent_id, actor in self.actors.items():
                    self._load_partial_state_dict(actor, shared_state, label=f"actor[{agent_id}]")
            else:
                saved_actor_ids = sorted(saved_actors.keys())
                reused = False
                for idx, (agent_id, actor) in enumerate(self.actors.items()):
                    src = agent_id if agent_id in saved_actors else saved_actor_ids[idx % len(saved_actor_ids)]
                    if src != agent_id:
                        reused = True
                    self._load_partial_state_dict(actor, saved_actors[src], label=f"actor[{agent_id}]")
                if reused:
                    print(f"[INFO] Reused saved actor weights from {len(saved_actor_ids)} agents "
                          f"to initialize {self.num_agents} agents.")

        # Comm actor
        if self._use_comm and "comm_actor" in state_dict and self._comm_actor is not None:
            self._load_partial_state_dict(self._comm_actor, state_dict["comm_actor"], label="comm_actor")

        # Critic
        critic_state = state_dict.get("critic")
        if critic_state:
            self._load_partial_state_dict(self.critic, critic_state, label="critic")
        else:
            print("[INFO] No critic weights found in checkpoint — skipping.")

    def _load_partial_state_dict(
        self,
        module: nn.Module,
        saved_state: Dict[str, torch.Tensor],
        label: str,
    ) -> None:
        """Load only shape-compatible parameters from a checkpoint."""
        current_state    = module.state_dict()
        compatible_state = {}
        skipped          = []

        for key, value in saved_state.items():
            if key in current_state and current_state[key].shape == value.shape:
                compatible_state[key] = value
            else:
                skipped.append(key)

        if compatible_state:
            merged = current_state.copy()
            merged.update(compatible_state)
            module.load_state_dict(merged)

        if skipped:
            print(
                f"[INFO] Partially loaded {label}: "
                f"{len(compatible_state)} matched, {len(skipped)} skipped (shape mismatch)."
            )
