"""
MODULE 6: Multi-Agent Reinforcement Learning Training Layer
Implements a lightweight MAPPO-style trainer with centralized critic input.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ActorNetwork(nn.Module):
    """Actor network (policy network) for MAPPO."""

    def __init__(self, input_size: int = 64, output_size: int = 6, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return action logits."""
        return self.net(x)

    def distribution(self, x: torch.Tensor) -> torch.distributions.Categorical:
        """Return a categorical distribution over discrete actions."""
        logits = self.forward(x)
        return torch.distributions.Categorical(logits=logits)

    def get_action(
        self, state: np.ndarray, device: str = "cpu", deterministic: bool = False
    ) -> Tuple[int, float]:
        """Sample or greedily select an action and return its log-probability."""
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            dist = self.distribution(state_tensor)

            if deterministic:
                action = torch.argmax(dist.probs, dim=-1)
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)

        return int(action.item()), float(log_prob.item())


class CriticNetwork(nn.Module):
    """Centralized critic over concatenated agent observations."""

    def __init__(self, input_size: int = 64 * 3, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
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
        self.actions: Deque[int] = deque(maxlen=buffer_size)
        self.rewards: Deque[float] = deque(maxlen=buffer_size)
        self.dones: Deque[float] = deque(maxlen=buffer_size)
        self.log_probs: Deque[float] = deque(maxlen=buffer_size)
        self.values: Deque[float] = deque(maxlen=buffer_size)
        self.next_values: Deque[float] = deque(maxlen=buffer_size)

    def store(
        self,
        obs: np.ndarray,
        central_obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
        next_value: float,
    ) -> None:
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.central_obs.append(np.asarray(central_obs, dtype=np.float32))
        self.actions.append(int(action))
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
            "actions": np.asarray(self.actions, dtype=np.int64),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "dones": np.asarray(self.dones, dtype=np.float32),
            "log_probs": np.asarray(self.log_probs, dtype=np.float32),
            "values": np.asarray(self.values, dtype=np.float32),
            "next_values": np.asarray(self.next_values, dtype=np.float32),
        }


class MARLTrainer:
    """
    MAPPO trainer using centralized training with decentralized execution.
    """

    def __init__(
        self,
        num_agents: int = 3,
        obs_size: int = 64,
        action_size: int = 6,
        hidden_size: int = 64,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coeff: float = 0.01,
        value_loss_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
        clip_ratio: float = 0.2,
        device: str = "cpu",
    ):
        self.num_agents = num_agents
        self.obs_size = obs_size
        self.action_size = action_size
        self.device = device

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.max_grad_norm = max_grad_norm
        self.clip_ratio = clip_ratio

        self.actors = {
            f"SAT_{i:03d}": ActorNetwork(obs_size, action_size, hidden_size).to(device)
            for i in range(num_agents)
        }
        self.critic = CriticNetwork(obs_size * num_agents, hidden_size).to(device)

        self.actor_optimizers = {
            agent_id: optim.Adam(actor.parameters(), lr=learning_rate)
            for agent_id, actor in self.actors.items()
        }
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.buffers = {agent_id: PPOBuffer() for agent_id in self.actors.keys()}
        self.training_stats: List[Dict[str, float]] = []

    def _build_central_observation(self, observations: Dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate observations in a stable agent order for the centralized critic."""
        central_obs = []
        for agent_id in self.actors.keys():
            obs = observations.get(agent_id)
            if obs is None:
                obs = np.zeros(self.obs_size, dtype=np.float32)
            central_obs.append(np.asarray(obs, dtype=np.float32))
        return np.concatenate(central_obs, axis=0).astype(np.float32)

    def _critic_value(self, central_obs: np.ndarray) -> float:
        with torch.no_grad():
            central_obs_t = torch.as_tensor(
                central_obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            return float(self.critic(central_obs_t).item())

    def _log_prob_for_action(self, actor: ActorNetwork, obs: np.ndarray, action: int) -> float:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            action_t = torch.as_tensor([action], dtype=torch.long, device=self.device)
            dist = actor.distribution(obs_t)
            return float(dist.log_prob(action_t).item())

    def get_action_details(
        self, observations: Dict[str, np.ndarray], deterministic: bool = False
    ) -> Tuple[Dict[str, int], Dict[str, float], float]:
        """
        Return joint actions, per-agent log-probs, and centralized value estimate.
        """
        actions: Dict[str, int] = {}
        log_probs: Dict[str, float] = {}
        central_obs = self._build_central_observation(observations)
        value = self._critic_value(central_obs)

        for agent_id, actor in self.actors.items():
            obs = observations.get(agent_id)
            if obs is None:
                actions[agent_id] = 0
                log_probs[agent_id] = 0.0
                continue

            action, log_prob = actor.get_action(obs, self.device, deterministic=deterministic)
            actions[agent_id] = action
            log_probs[agent_id] = log_prob

        return actions, log_probs, value

    def get_actions(
        self, observations: Dict[str, np.ndarray], deterministic: bool = False
    ) -> Dict[str, int]:
        actions, _, _ = self.get_action_details(observations, deterministic=deterministic)
        return actions

    def collect_experience(
        self,
        observations: Dict[str, np.ndarray],
        rewards: Dict[str, float],
        next_observations: Dict[str, np.ndarray],
        dones: Dict[str, bool],
        actions: Dict[str, int],
        log_probs: Optional[Dict[str, float]] = None,
        central_value: Optional[float] = None,
    ) -> None:
        """
        Collect environment experience using the actions actually executed.
        """
        central_obs = self._build_central_observation(observations)
        next_central_obs = self._build_central_observation(next_observations)

        if central_value is None:
            central_value = self._critic_value(central_obs)
        next_value = self._critic_value(next_central_obs)

        for agent_id, actor in self.actors.items():
            obs = observations.get(agent_id)
            if obs is None:
                continue

            action = int(actions.get(agent_id, 0))
            reward = float(rewards.get(agent_id, 0.0))
            done = bool(dones.get(agent_id, dones.get("__all__", False)))

            if log_probs is not None and agent_id in log_probs:
                log_prob = float(log_probs[agent_id])
            else:
                log_prob = self._log_prob_for_action(actor, obs, action)

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
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_values[t] * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages.astype(np.float32), returns.astype(np.float32)

    def train(self, num_epochs: int = 3, batch_size: int = 64) -> Dict[str, float]:
        """Train all actors and the centralized critic using collected rollouts."""
        stats = {
            "actor_loss": 0.0,
            "critic_loss": 0.0,
            "entropy": 0.0,
            "policy_loss": 0.0,
        }

        actor_updates = 0
        critic_updates = 0
        critic_obs_batches: List[np.ndarray] = []
        critic_return_batches: List[np.ndarray] = []

        for agent_id, actor in self.actors.items():
            buffer = self.buffers[agent_id]
            if len(buffer) == 0:
                continue

            data = buffer.as_arrays()
            advantages, returns = self._compute_gae(
                rewards=data["rewards"],
                dones=data["dones"],
                values=data["values"],
                next_values=data["next_values"],
            )
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            obs_t = torch.as_tensor(data["obs"], dtype=torch.float32, device=self.device)
            actions_t = torch.as_tensor(data["actions"], dtype=torch.long, device=self.device)
            old_log_probs_t = torch.as_tensor(
                data["log_probs"], dtype=torch.float32, device=self.device
            )
            advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)

            critic_obs_batches.append(data["central_obs"])
            critic_return_batches.append(returns)

            optimizer = self.actor_optimizers[agent_id]
            num_samples = len(data["actions"])

            for _ in range(num_epochs):
                permutation = np.random.permutation(num_samples)
                for start in range(0, num_samples, batch_size):
                    idx = permutation[start : start + batch_size]
                    batch_obs = obs_t[idx]
                    batch_actions = actions_t[idx]
                    batch_old_log_probs = old_log_probs_t[idx]
                    batch_advantages = advantages_t[idx]

                    dist = actor.distribution(batch_obs)
                    new_log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(
                        ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
                    ) * batch_advantages

                    policy_loss = -torch.min(surr1, surr2).mean()
                    actor_loss = policy_loss - self.entropy_coeff * entropy

                    optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
                    optimizer.step()

                    stats["actor_loss"] += float(actor_loss.item())
                    stats["policy_loss"] += float(policy_loss.item())
                    stats["entropy"] += float(entropy.item())
                    actor_updates += 1

        if critic_obs_batches:
            critic_obs = np.concatenate(critic_obs_batches, axis=0)
            critic_returns = np.concatenate(critic_return_batches, axis=0)
            critic_obs_t = torch.as_tensor(critic_obs, dtype=torch.float32, device=self.device)
            critic_returns_t = torch.as_tensor(
                critic_returns, dtype=torch.float32, device=self.device
            )

            num_samples = critic_obs_t.shape[0]
            for _ in range(num_epochs):
                permutation = np.random.permutation(num_samples)
                for start in range(0, num_samples, batch_size):
                    idx = permutation[start : start + batch_size]
                    batch_obs = critic_obs_t[idx]
                    batch_returns = critic_returns_t[idx]

                    predicted_values = self.critic(batch_obs).squeeze(-1)
                    critic_loss = nn.MSELoss()(predicted_values, batch_returns)

                    self.critic_optimizer.zero_grad()
                    (self.value_loss_coeff * critic_loss).backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.critic.parameters(), self.max_grad_norm
                    )
                    self.critic_optimizer.step()

                    stats["critic_loss"] += float(critic_loss.item())
                    critic_updates += 1

        if actor_updates > 0:
            stats["actor_loss"] /= actor_updates
            stats["policy_loss"] /= actor_updates
            stats["entropy"] /= actor_updates
        if critic_updates > 0:
            stats["critic_loss"] /= critic_updates

        for buffer in self.buffers.values():
            buffer.clear()

        self.training_stats.append(stats)
        return stats

    def save(self, filepath: str) -> None:
        """Save trained weights and trainer metadata."""
        state_dict = {
            "num_agents": self.num_agents,
            "obs_size": self.obs_size,
            "action_size": self.action_size,
            "actors": {aid: actor.state_dict() for aid, actor in self.actors.items()},
            "critic": self.critic.state_dict(),
        }
        torch.save(state_dict, filepath)

    def load(self, filepath: str) -> None:
        """Load trained weights."""
        state_dict = torch.load(filepath, map_location=self.device)
        saved_actors = state_dict.get("actors", {})
        saved_actor_ids = sorted(saved_actors.keys())

        if not saved_actor_ids:
            raise ValueError(f"No actor weights found in {filepath}")

        reused_actor_weights = False
        for idx, (agent_id, actor) in enumerate(self.actors.items()):
            if agent_id in saved_actors:
                source_id = agent_id
            else:
                source_id = saved_actor_ids[idx % len(saved_actor_ids)]
                reused_actor_weights = True
            actor.load_state_dict(saved_actors[source_id])

        if reused_actor_weights:
            print(
                f"[INFO] Reused saved actor weights from {len(saved_actor_ids)} agents "
                f"to initialize {self.num_agents} agents."
            )

        critic_state = state_dict.get("critic")
        current_critic_state = self.critic.state_dict()
        critic_compatible = bool(
            critic_state
            and current_critic_state.keys() == critic_state.keys()
            and all(current_critic_state[k].shape == critic_state[k].shape for k in current_critic_state)
        )

        if critic_compatible:
            self.critic.load_state_dict(critic_state)
        else:
            print("[INFO] Skipping critic weights because the saved centralized critic shape does not match this trainer.")
