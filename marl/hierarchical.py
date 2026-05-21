"""
marl/hierarchical.py — Hierarchical Fleet Controller
=====================================================

Two-level decision hierarchy for coordinated satellite collision avoidance:

Level 1 — Fleet Commander  (acts every K steps, meta-policy)
-------------------------------------------------------------
Input  : all satellite observations + pairwise conjunction risk graph
Output : per-satellite GOAL VECTOR  g_i ∈ R^{goal_size}
         encoding high-level intent (maneuver priority, avoidance direction,
         fuel budget signal, coordination token)

Level 2 — Satellite Actor  (acts every step, sub-policy)
---------------------------------------------------------
Input  : own observation (96-dim) + goal vector from commander (goal_size)
Output : (direction_idx, magnitude) — same action space as before

Why hierarchy?
--------------
Flat MARL treats every step as equally important.  In orbital mechanics:
* Maneuver TIMING matters more than direction: a burn 30 min before TCA
  costs 10× less fuel than one 2 min before.
* Satellites should coordinate BEFORE a conjunction window, not react step
  by step.  The commander sees the global picture and pre-assigns roles
  (e.g. "SAT_000 burns prograde, SAT_001 holds, SAT_002 burns normal").
* Sub-policies only need to optimise execution of the assigned goal —
  simpler credit assignment, faster convergence.

Commander architecture
----------------------
Input: concat of all agent observations → GAT encoding → global pool
       + pairwise risk matrix (N×N edge features)
Output: N × goal_size goal vectors (one per satellite)
Uses the GraphAttentionCritic's GAT backbone (shared weights optional).

Satellite actor architecture
----------------------------
ThreatAttentionActor (Phase 1) augmented with a goal-conditioned input:
own_state (12) + goal_vector (goal_size) → Linear → CLS token.
The rest of the attention over threats is unchanged.

Training
--------
* Commander is trained with a TEAM reward (mean across all satellites).
* Satellite actors are trained with individual rewards (standard PPO).
* Commander update frequency: every K*N environment steps (slower loop).
* Goals are DETACHED from the commander for satellite PPO updates
  (avoid double-gradient issues).

Commander update interval K
---------------------------
Default K=5 (commander updates every 5 satellite steps).
This matches orbital mechanics: at 60s/step, K=5 → 5 min horizon,
which is a natural pre-conjunction planning window.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sim.maneuver_engine import ACTION_COUNT, MAX_DELTA_V_PER_STEP_KMS
from sim.observation_utils import OBS_SIZE, MAX_RISK_INDEX


# ─────────────────────────────────────────────────────────────────────────────
# Fleet Commander
# ─────────────────────────────────────────────────────────────────────────────

class FleetCommander(nn.Module):
    """
    Global meta-policy.  Receives all satellite observations and outputs
    a per-satellite goal vector every K steps.

    Architecture
    ------------
    1. Project each agent's obs → node embedding (Linear + ReLU).
    2. Self-attention over all node embeddings (see all satellites at once).
    3. For each node: concat global mean-pool + node embedding → goal head.
    4. Output: N × goal_size goal vectors.

    The commander is NOT recurrent (stateless per K-step window) but can
    be extended to carry hidden state if desired.
    """

    def __init__(
        self,
        obs_size: int = OBS_SIZE,
        hidden_size: int = 128,
        goal_size: int = 16,
        num_heads: int = 4,
        num_layers: int = 2,
        max_agents: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.obs_size    = obs_size
        self.hidden_size = hidden_size
        self.goal_size   = goal_size

        # Node encoder: obs → hidden
        self.node_proj = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Global self-attention over all satellite nodes
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.global_attn = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Per-agent goal head: (node_h + global_pool) → goal_vector
        self.goal_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, goal_size),
            nn.Tanh(),   # bounded goal signal [-1, 1]
        )

        # Commander value head (for training with team reward)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        observations: Dict[str, np.ndarray],
        device: str = "cpu",
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """
        Parameters
        ----------
        observations : {agent_id → np.ndarray (96,)}

        Returns
        -------
        goals  : {agent_id → Tensor (goal_size,)}  — detached goal vectors
        value  : float  — commander's value estimate of current state
        """
        agent_ids = sorted(observations.keys())
        obs_list  = [
            torch.as_tensor(observations[aid], dtype=torch.float32, device=device)
            for aid in agent_ids
        ]

        if not obs_list:
            return {}, 0.0

        obs_tensor = torch.stack(obs_list, dim=0).unsqueeze(0)  # (1, N, 96)
        N = obs_tensor.size(1)

        # Node embeddings
        node_h = self.node_proj(obs_tensor)                     # (1, N, H)

        # Global self-attention
        attended = self.global_attn(node_h)                     # (1, N, H)

        # Global mean pool
        global_pool = attended.mean(dim=1, keepdim=True)        # (1, 1, H)
        global_pool = global_pool.expand(1, N, -1)              # (1, N, H)

        # Per-agent goal: concat(node_h, global_pool) → goal
        goal_input = torch.cat([attended, global_pool], dim=-1) # (1, N, 2H)
        goals_raw  = self.goal_head(goal_input).squeeze(0)      # (N, goal_size)

        # Commander value (from global pool)
        value = self.value_head(
            global_pool.squeeze(0).mean(dim=0, keepdim=True)   # (1, H)
        ).item()

        goals = {
            agent_ids[i]: goals_raw[i].detach()
            for i in range(N)
        }
        return goals, value

    def get_value(
        self,
        observations: Dict[str, np.ndarray],
        device: str = "cpu",
    ) -> float:
        _, value = self.forward(observations, device)
        return value


# ─────────────────────────────────────────────────────────────────────────────
# Goal-Conditioned Satellite Actor
# ─────────────────────────────────────────────────────────────────────────────

class GoalConditionedActor(nn.Module):
    """
    Satellite actor augmented with commander goal vector.

    The goal vector is concatenated to the own-state before the CLS token
    projection — the attention tower then has access to the commander's
    intent when deciding how to respond to threats.
    """

    NUM_TOKENS = 1 + 7   # CLS + 7 threat tokens

    def __init__(
        self,
        input_size: int = OBS_SIZE,
        output_size: int = ACTION_COUNT,
        hidden_size: int = 128,
        goal_size: int = 16,
        num_heads: int = 4,
        num_transformer_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        from sim.observation_utils import (
            OWN_FEATURE_COUNT, THREAT_FEATURE_COUNT,
            MAX_NEARBY_OBJECTS, THREATS_START_INDEX,
        )
        self.OWN_FEATURE_COUNT  = OWN_FEATURE_COUNT
        self.THREAT_FEATURE_COUNT = THREAT_FEATURE_COUNT
        self.MAX_NEARBY_OBJECTS = MAX_NEARBY_OBJECTS
        self.THREATS_START_INDEX = THREATS_START_INDEX

        self.hidden_size = hidden_size
        self.goal_size   = goal_size
        self.max_dv      = float(MAX_DELTA_V_PER_STEP_KMS)

        assert hidden_size % num_heads == 0

        # CLS projection: own_state (12) + goal (goal_size) → hidden
        self.own_proj    = nn.Linear(OWN_FEATURE_COUNT + goal_size, hidden_size)
        self.threat_proj = nn.Linear(THREAT_FEATURE_COUNT, hidden_size)

        self.pos_enc = nn.Embedding(self.NUM_TOKENS, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )

        self.direction_head = nn.Linear(hidden_size, output_size)
        self.magnitude_mean_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        self.magnitude_logstd = nn.Parameter(
            torch.tensor([math.log(5.0e-4)], dtype=torch.float32)
        )

        self._init_weights()
        # Default zero goal (used when commander hasn't fired yet)
        self._default_goal = torch.zeros(goal_size)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.direction_head.weight, gain=0.01)
        nn.init.orthogonal_(self.magnitude_mean_head[0].weight, gain=0.01)

    def _encode(
        self,
        obs: torch.Tensor,       # (B, 96)
        goal: torch.Tensor,      # (B, goal_size) or (goal_size,)
    ) -> torch.Tensor:
        B = obs.shape[0]
        if goal.dim() == 1:
            goal = goal.unsqueeze(0).expand(B, -1)

        own_state     = obs[:, :self.OWN_FEATURE_COUNT]          # (B, 12)
        threat_flat   = obs[:, self.THREATS_START_INDEX:]
        threat_tokens = threat_flat.reshape(
            B, self.MAX_NEARBY_OBJECTS, self.THREAT_FEATURE_COUNT
        )

        is_padding = (threat_tokens.abs().sum(dim=-1) < 1e-6)
        cls_pad    = torch.zeros(B, 1, dtype=torch.bool, device=obs.device)
        pad_mask   = torch.cat([cls_pad, is_padding], dim=1)

        # CLS = own_state + goal
        own_goal  = torch.cat([own_state, goal.to(obs.device)], dim=-1)
        cls_token = self.own_proj(own_goal).unsqueeze(1)         # (B, 1, H)

        threat_emb = self.threat_proj(threat_tokens)             # (B, 7, H)

        tokens    = torch.cat([cls_token, threat_emb], dim=1)   # (B, 8, H)
        positions = torch.arange(self.NUM_TOKENS, device=obs.device).unsqueeze(0)
        tokens    = tokens + self.pos_enc(positions)

        attended  = self.transformer(tokens, src_key_padding_mask=pad_mask)
        return attended[:, 0, :]                                 # (B, H)

    def forward(
        self,
        obs: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if goal is None:
            goal = self._default_goal
        h        = self._encode(obs, goal)
        logits   = self.direction_head(h)
        mag_mean = self.magnitude_mean_head(h) * self.max_dv
        return logits, mag_mean

    def distribution(
        self,
        obs: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.distributions.Categorical, torch.distributions.Normal]:
        logits, mag_mean = self.forward(obs, goal)
        dir_dist = torch.distributions.Categorical(logits=logits)
        mag_std  = torch.exp(self.magnitude_logstd).clamp(5e-5, 1.5e-3).expand_as(mag_mean)
        mag_dist = torch.distributions.Normal(mag_mean, mag_std)
        return dir_dist, mag_dist

    def get_action(
        self,
        state: np.ndarray,
        device: str = "cpu",
        deterministic: bool = False,
        goal: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[int, float], float]:
        with torch.no_grad():
            x = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            g = goal.to(device) if goal is not None else self._default_goal.to(device)
            dir_dist, mag_dist = self.distribution(x, g)

            if deterministic:
                dir_action = torch.argmax(dir_dist.probs, dim=-1)
                mag_action = mag_dist.mean
            else:
                dir_action = dir_dist.sample()
                mag_action = mag_dist.sample()

            mag_action = mag_action.clamp(0.0, self.max_dv)
            log_prob   = (
                dir_dist.log_prob(dir_action)
                + mag_dist.log_prob(mag_action).squeeze(-1)
            )
        return (int(dir_action.item()), float(mag_action.item())), float(log_prob.item())


# ─────────────────────────────────────────────────────────────────────────────
# HierarchicalController — bundles commander + actors
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalController:
    """
    Top-level wrapper for hierarchical MARL.

    Maintains:
      * FleetCommander  — fires every K steps, stores current goals.
      * Per-agent GoalConditionedActor  (one per satellite).
      * Step counter to know when to fire the commander.

    Usage in train.py
    -----------------
        hc = HierarchicalController(agent_ids, ...)
        obs = env.reset()
        for step in range(max_steps):
            actions, log_probs = hc.get_actions(obs, step)
            next_obs, rewards, dones, info = env.step(actions)
            hc.collect_experience(obs, rewards, next_obs, dones, actions, log_probs)
            obs = next_obs
        hc.train(...)
    """

    def __init__(
        self,
        agent_ids: List[str],
        obs_size: int = OBS_SIZE,
        hidden_size: int = 128,
        goal_size: int = 16,
        commander_k: int = 5,    # commander fires every K satellite steps
        num_heads: int = 4,
        num_transformer_layers: int = 2,
        dropout: float = 0.1,
        device: str = "cpu",
    ) -> None:
        self.agent_ids    = agent_ids
        self.goal_size    = goal_size
        self.commander_k  = commander_k
        self.device       = device

        self.commander = FleetCommander(
            obs_size=obs_size,
            hidden_size=hidden_size,
            goal_size=goal_size,
            num_heads=num_heads,
            num_layers=num_transformer_layers,
            dropout=dropout,
        ).to(device)

        self.satellite_actors: Dict[str, GoalConditionedActor] = {
            aid: GoalConditionedActor(
                input_size=obs_size,
                output_size=ACTION_COUNT,
                hidden_size=hidden_size,
                goal_size=goal_size,
                num_heads=num_heads,
                num_transformer_layers=num_transformer_layers,
                dropout=dropout,
            ).to(device)
            for aid in agent_ids
        }

        # Current goals from the commander (updated every K steps)
        self._current_goals: Dict[str, torch.Tensor] = {
            aid: torch.zeros(goal_size, device=device)
            for aid in agent_ids
        }

    def reset(self) -> None:
        """Reset goals at episode start."""
        self._current_goals = {
            aid: torch.zeros(self.goal_size, device=self.device)
            for aid in self.agent_ids
        }

    def get_actions(
        self,
        observations: Dict[str, np.ndarray],
        step: int,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """
        Get joint actions.  Commander fires every K steps to update goals.

        Parameters
        ----------
        observations : {agent_id → np.ndarray (96,)}
        step         : current episode step (0-indexed)
        deterministic: greedy action selection

        Returns
        -------
        actions   : {agent_id → (dir_idx, magnitude)}
        log_probs : {agent_id → float}
        """
        # Commander step: update goals every K steps
        if step % self.commander_k == 0:
            new_goals, _ = self.commander.forward(observations, self.device)
            self._current_goals.update(new_goals)

        # Satellite actors: execute with current goals
        actions:   Dict[str, Any]   = {}
        log_probs: Dict[str, float] = {}

        for aid in self.agent_ids:
            obs  = observations.get(aid)
            if obs is None:
                actions[aid]   = (0, 0.0)
                log_probs[aid] = 0.0
                continue

            goal   = self._current_goals.get(aid, torch.zeros(self.goal_size, device=self.device))
            actor  = self.satellite_actors[aid]
            action, lp = actor.get_action(obs, self.device, deterministic, goal=goal)
            actions[aid]   = action
            log_probs[aid] = lp

        return actions, log_probs

    def parameters(self):
        """All trainable parameters (commander + all satellite actors)."""
        params = list(self.commander.parameters())
        for actor in self.satellite_actors.values():
            params.extend(actor.parameters())
        return params
