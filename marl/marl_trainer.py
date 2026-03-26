"""
MODULE 6: Multi-Agent Reinforcement Learning Training Layer
Implements MAPPO (Multi-Agent PPO) training algorithm.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
from collections import deque
from datetime import datetime


class ActorNetwork(nn.Module):
    """Actor network (policy network) for MAPPO."""
    
    def __init__(self, input_size: int = 50, output_size: int = 6,
                 hidden_size: int = 64):
        """
        Initialize actor network.
        
        Args:
            input_size: Observation size
            output_size: Action size
            hidden_size: Hidden layer size
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        """Forward pass."""
        return self.net(x)
    
    def get_action(self, state: np.ndarray, device: str = 'cpu') -> Tuple[int, float]:
        """
        Sample action from policy.
        
        Args:
            state: Observation
            device: Device to use
            
        Returns:
            (action, log_probability)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            probs = self.forward(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item()


class CriticNetwork(nn.Module):
    """Critic network (value network) for MAPPO."""
    
    def __init__(self, input_size: int = 50 * 3,  # Centralized: all agents
                 hidden_size: int = 64):
        """
        Initialize critic network (centralized).
        
        Args:
            input_size: Total state size (concatenated observations)
            hidden_size: Hidden layer size
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        """Forward pass."""
        return self.net(x)


class PPOBuffer:
    """Experience replay buffer for PPO training."""
    
    def __init__(self, buffer_size: int = 2000):
        """Initialize buffer."""
        self.buffer_size = buffer_size
        self.states = deque(maxlen=buffer_size)
        self.actions = deque(maxlen=buffer_size)
        self.rewards = deque(maxlen=buffer_size)
        self.next_states = deque(maxlen=buffer_size)
        self.dones = deque(maxlen=buffer_size)
        self.log_probs = deque(maxlen=buffer_size)
        self.values = deque(maxlen=buffer_size)
    
    def store(self, state, action, reward, next_state, done, log_prob, value):
        """Store transition."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def clear(self):
        """Clear buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
    
    def __len__(self):
        return len(self.states)
    
    def get_batch(self, batch_size: int = 32) -> Tuple:
        """Get random batch."""
        indices = np.random.choice(len(self), batch_size, replace=False)
        
        states = np.array([self.states[i] for i in indices])
        actions = np.array([self.actions[i] for i in indices])
        rewards = np.array([self.rewards[i] for i in indices])
        next_states = np.array([self.next_states[i] for i in indices])
        dones = np.array([self.dones[i] for i in indices])
        log_probs = np.array([self.log_probs[i] for i in indices])
        values = np.array([self.values[i] for i in indices])
        
        return states, actions, rewards, next_states, dones, log_probs, values


class MARLTrainer:
    """
    MAPPO (Multi-Agent PPO) trainer.
    Uses centralized training with decentralized execution.
    """
    
    def __init__(self, num_agents: int = 3,
                 obs_size: int = 50,
                 action_size: int = 6,
                 hidden_size: int = 64,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 entropy_coeff: float = 0.01,
                 value_loss_coeff: float = 0.5,
                 max_grad_norm: float = 0.5,
                 device: str = 'cpu'):
        """
        Initialize MARL trainer.
        
        Args:
            num_agents: Number of agents
            obs_size: Observation size per agent
            action_size: Action size per agent
            hidden_size: Neural network hidden size
            learning_rate: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda
            entropy_coeff: Entropy coefficient
            value_loss_coeff: Value loss coefficient
            max_grad_norm: Max gradient norm
            device: Computation device
        """
        self.num_agents = num_agents
        self.obs_size = obs_size
        self.action_size = action_size
        self.device = device
        
        # Training parameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.max_grad_norm = max_grad_norm
        
        # Networks (one actor per agent, centralized critic)
        self.actors = {
            f"SAT_{i:03d}": ActorNetwork(obs_size, action_size, hidden_size)
            for i in range(num_agents)
        }
        
        self.critic = CriticNetwork(obs_size * num_agents, hidden_size)
        
        # Move to device
        for actor in self.actors.values():
            actor.to(device)
        self.critic.to(device)
        
        # Optimizers
        self.actor_optimizers = {
            agent_id: optim.Adam(actor.parameters(), lr=learning_rate)
            for agent_id, actor in self.actors.items()
        }
        
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Buffers (per agent)
        self.buffers = {
            agent_id: PPOBuffer()
            for agent_id in self.actors.keys()
        }
        
        # Logging
        self.training_stats = []
    
    def collect_experience(self, observations: Dict[str, np.ndarray],
                          rewards: Dict[str, float],
                          next_observations: Dict[str, np.ndarray],
                          dones: Dict[str, bool],
                          actions: Dict[str, int]) -> None:
        """
        Collect experience from environment interaction.
        
        Args:
            observations: Current observations
            rewards: Rewards from environment
            next_observations: Next observations
            dones: Done flags
            actions: Actions taken
        """
        with torch.no_grad():
            for agent_id in self.actors.keys():
                if agent_id not in observations:
                    continue
                
                obs = observations[agent_id]
                reward = rewards.get(agent_id, 0.0)
                next_obs = next_observations[agent_id]
                done = dones.get(agent_id, False)
                action = actions.get(agent_id, 0)
                
                # Get action probability and value
                actor = self.actors[agent_id]
                _, log_prob = actor.get_action(obs, self.device)
                
                # Get value
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                value = self.critic(obs_tensor).item()
                
                # Store in buffer
                self.buffers[agent_id].store(
                    obs, action, reward, next_obs, done, log_prob, value
                )
    
    def train(self, num_epochs: int = 3, batch_size: int = 32) -> Dict[str, float]:
        """
        Train the policy using collected experience.
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            Training statistics
        """
        stats = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'entropy': 0.0,
            'policy_loss': 0.0,
        }
        
        for epoch in range(num_epochs):
            epoch_stats = {'actor_loss': 0.0, 'critic_loss': 0.0, 'entropy': 0.0}
            
            for agent_id in self.actors.keys():
                buffer = self.buffers[agent_id]
                
                if len(buffer) < batch_size:
                    continue
                
                # Get batch
                states, actions, rewards, next_states, dones, log_probs, values = \
                    buffer.get_batch(batch_size)
                
                # Convert to tensors
                states_t = torch.FloatTensor(states).to(self.device)
                actions_t = torch.LongTensor(actions).to(self.device)
                rewards_t = torch.FloatTensor(rewards).to(self.device)
                values_t = torch.FloatTensor(values).to(self.device)
                old_log_probs_t = torch.FloatTensor(log_probs).to(self.device)
                
                # Compute advantages (GAE)
                advantages = rewards_t - values_t
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Actor update
                actor = self.actors[agent_id]
                actor_optimizer = self.actor_optimizers[agent_id]
                
                probs = actor(states_t)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(actions_t)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - old_log_probs_t)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
                
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coeff * entropy
                
                actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
                actor_optimizer.step()
                
                epoch_stats['actor_loss'] += actor_loss.item()
                epoch_stats['entropy'] += entropy.item()
            
            # Critic update (centralized - use all agent states)
            # Simplified: only update critic if we have enough data overall
            total_buffer_size = sum(len(self.buffers[aid]) for aid in self.actors.keys())
            
            if total_buffer_size > batch_size:
                self.critic_optimizer.zero_grad()
                
                # Average loss over agents (simplified)
                critic_loss_total = 0.0
                for agent_id in self.actors.keys():
                    buffer = self.buffers[agent_id]
                    if len(buffer) >= batch_size:
                        states, _, rewards, _, _, _, _ = buffer.get_batch(batch_size)
                        states_t = torch.FloatTensor(states).to(self.device)
                        rewards_t = torch.FloatTensor(rewards).to(self.device)
                        
                        predicted_values = self.critic(states_t).squeeze()
                        critic_loss = nn.MSELoss()(predicted_values, rewards_t)
                        critic_loss_total += critic_loss
                
                critic_loss_total.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                
                epoch_stats['critic_loss'] = critic_loss_total.item()
            
            # Accumulate stats
            for key in epoch_stats:
                stats[key] += epoch_stats[key] / num_epochs
        
        # Clear buffers
        for buffer in self.buffers.values():
            buffer.clear()
        
        self.training_stats.append(stats)
        return stats
    
    def get_actions(self, observations: Dict[str, np.ndarray]) -> Dict[str, int]:
        """
        Get actions from policy (inference).
        
        Args:
            observations: Observations from environment
            
        Returns:
            Dict mapping agent_id to action
        """
        actions = {}
        
        for agent_id, actor in self.actors.items():
            if agent_id not in observations:
                actions[agent_id] = 0
                continue
            
            obs = observations[agent_id]
            action, _ = actor.get_action(obs, self.device)
            actions[agent_id] = action
        
        return actions
    
    def save(self, filepath: str) -> None:
        """Save trained weights."""
        state_dict = {
            'actors': {aid: actor.state_dict() for aid, actor in self.actors.items()},
            'critic': self.critic.state_dict(),
        }
        torch.save(state_dict, filepath)
    
    def load(self, filepath: str) -> None:
        """Load trained weights."""
        state_dict = torch.load(filepath, map_location=self.device)
        
        for agent_id, actor in self.actors.items():
            if agent_id in state_dict['actors']:
                actor.load_state_dict(state_dict['actors'][agent_id])
        
        self.critic.load_state_dict(state_dict['critic'])
