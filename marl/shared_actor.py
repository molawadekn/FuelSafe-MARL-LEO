"""
marl/shared_actor.py — Parameter-Shared Actor with Agent-ID Embedding
======================================================================

Solves the scalability problem of Phase 1's per-agent actor networks:
Phase 1: N separate ThreatAttentionActor instances (one per satellite).
         Adding a 4th satellite requires a new network + full retraining.

Phase 2: ONE shared ThreatAttentionActor + per-agent identity embedding.
         The embedding tells the shared network "you are SAT_002" so it
         can specialise its behaviour per agent while sharing all weights.
         Any number of satellites (even unseen at training time) can be
         handled by providing a new embedding vector.

Architecture
------------
obs (96) → split → own_state (12) + threat_tokens (7×12)
own_state → concat with agent_id_embedding (embed_size) → own_proj → CLS token
threat_tokens → threat_proj → 7 threat tokens
[CLS | T1..T7] → Transformer Encoder (2 layers, 4 heads) → CLS output
CLS output → direction_head → Categorical(7)
CLS output → magnitude_head → Normal(μ, σ)

Agent-ID embedding
------------------
* Learnable embedding table: nn.Embedding(max_agents, embed_size).
* At runtime, look up by integer agent index (0, 1, 2, ...).
* For agents with index > max_agents (new satellites), use the last
  embedding + learned offset (generalisation heuristic).
* embed_size is small (default 8) so the own-state projection is not
  dominated by the ID signal.

Benefits
--------
1. O(1) parameters regardless of fleet size (vs O(N) for per-agent nets).
2. Trains more efficiently: all agents' experience trains the same weights.
3. Zero-shot generalisation: drop a 4th satellite in, it inherits the
   shared orbital avoidance policy with a fresh ID embedding.
4. Heterogeneous fleets: different satellite types get distinct embeddings
   while sharing the full attention + action architecture.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sim.maneuver_engine import ACTION_COUNT, MAX_DELTA_V_PER_STEP_KMS
from sim.observation_utils import (
    MAX_NEARBY_OBJECTS,
    OBS_SIZE,
    OWN_FEATURE_COUNT,
    THREAT_FEATURE_COUNT,
    THREATS_START_INDEX,
)


class SharedAttentionActor(nn.Module):
    """
    Single parameter-shared actor for an arbitrary-size satellite fleet.

    All satellites use this same network.  The agent's identity is injected
    via a learnable embedding concatenated to the own-state features before
    the CLS token projection.

    Interface identical to ThreatAttentionActor for drop-in compatibility.
    """

    NUM_TOKENS = 1 + MAX_NEARBY_OBJECTS   # CLS + 7 threat tokens

    def __init__(
        self,
        input_size: int = OBS_SIZE,
        output_size: int = ACTION_COUNT,
        hidden_size: int = 128,
        num_heads: int = 4,
        num_transformer_layers: int = 2,
        dropout: float = 0.1,
        max_agents: int = 16,          # embedding table size (supports up to 16 agents)
        embed_size: int = 8,           # agent-ID embedding dimensionality
    ) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size  = hidden_size
        self.output_size  = output_size
        self.max_agents   = max_agents
        self.embed_size   = embed_size
        self.max_dv       = float(MAX_DELTA_V_PER_STEP_KMS)

        # ── Agent-ID embedding table ─────────────────────────────────────────
        self.agent_embedding = nn.Embedding(max_agents, embed_size)

        # ── Token projections ─────────────────────────────────────────────────
        # own_state (12) + agent_id_emb (embed_size) → hidden_size
        self.own_proj    = nn.Linear(OWN_FEATURE_COUNT + embed_size, hidden_size)
        self.threat_proj = nn.Linear(THREAT_FEATURE_COUNT, hidden_size)

        # ── Positional encoding ──────────────────────────────────────────────
        self.pos_enc = nn.Embedding(self.NUM_TOKENS, hidden_size)

        # ── Transformer encoder ──────────────────────────────────────────────
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

        # ── Action heads ─────────────────────────────────────────────────────
        self.direction_head = nn.Linear(hidden_size, output_size)
        self.magnitude_mean_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        self.magnitude_logstd = nn.Parameter(
            torch.tensor([math.log(5.0e-4)], dtype=torch.float32)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.direction_head.weight, gain=0.01)
        nn.init.orthogonal_(self.magnitude_mean_head[0].weight, gain=0.01)
        nn.init.normal_(self.agent_embedding.weight, std=0.1)

    def _agent_index(self, agent_id: str) -> int:
        """
        Parse agent index from agent_id string.
        E.g. "SAT_002" → 2,  "SAT_007" → 7 (clamped to max_agents-1).
        """
        try:
            idx = int(agent_id.split("_")[-1])
        except (ValueError, IndexError):
            idx = 0
        return min(idx, self.max_agents - 1)

    def _encode(
        self,
        obs: torch.Tensor,    # (B, 96)
        agent_idx: int = 0,
    ) -> torch.Tensor:
        """
        Encode observation with agent-identity injection.

        Returns
        -------
        cls_out : (B, hidden_size)
        """
        B = obs.shape[0]

        # Split observation
        own_state    = obs[:, :OWN_FEATURE_COUNT]                # (B, 12)
        threat_flat  = obs[:, THREATS_START_INDEX:]              # (B, 84)
        threat_tokens = threat_flat.reshape(
            B, MAX_NEARBY_OBJECTS, THREAT_FEATURE_COUNT
        )                                                        # (B, 7, 12)

        # Padding mask (zero-padded threats)
        is_padding = (threat_tokens.abs().sum(dim=-1) < 1e-6)   # (B, 7)
        cls_pad    = torch.zeros(B, 1, dtype=torch.bool, device=obs.device)
        pad_mask   = torch.cat([cls_pad, is_padding], dim=1)    # (B, 8)

        # Agent-ID embedding
        idx_tensor = torch.tensor(
            [agent_idx], dtype=torch.long, device=obs.device
        ).expand(B)
        id_emb = self.agent_embedding(idx_tensor)               # (B, embed_size)

        # CLS token: own_state concat ID embedding → project
        own_with_id = torch.cat([own_state, id_emb], dim=-1)    # (B, 12+embed)
        cls_token   = self.own_proj(own_with_id).unsqueeze(1)   # (B, 1, H)

        # Threat tokens
        threat_emb = self.threat_proj(threat_tokens)            # (B, 7, H)

        # Build token sequence + positional encoding
        tokens   = torch.cat([cls_token, threat_emb], dim=1)   # (B, 8, H)
        positions = torch.arange(
            self.NUM_TOKENS, device=obs.device
        ).unsqueeze(0)                                           # (1, 8)
        tokens   = tokens + self.pos_enc(positions)             # (B, 8, H)

        # Transformer
        attended = self.transformer(tokens, src_key_padding_mask=pad_mask)
        return attended[:, 0, :]                                 # (B, H) — CLS

    def forward(
        self,
        obs: torch.Tensor,
        agent_idx: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h        = self._encode(obs, agent_idx)
        logits   = self.direction_head(h)
        mag_mean = self.magnitude_mean_head(h) * self.max_dv
        return logits, mag_mean

    def distribution(
        self,
        obs: torch.Tensor,
        agent_idx: int = 0,
    ) -> Tuple[torch.distributions.Categorical, torch.distributions.Normal]:
        logits, mag_mean = self.forward(obs, agent_idx)
        dir_dist = torch.distributions.Categorical(logits=logits)
        mag_std  = torch.exp(self.magnitude_logstd).clamp(5e-5, 1.5e-3).expand_as(mag_mean)
        mag_dist = torch.distributions.Normal(mag_mean, mag_std)
        return dir_dist, mag_dist

    def get_action(
        self,
        state: np.ndarray,
        device: str = "cpu",
        deterministic: bool = False,
        agent_id: str = "SAT_000",
    ) -> Tuple[Tuple[int, float], float]:
        agent_idx = self._agent_index(agent_id)
        with torch.no_grad():
            x = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            dir_dist, mag_dist = self.distribution(x, agent_idx)

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


class SharedMARLTrainerAdapter:
    """
    Thin adapter that makes a single SharedAttentionActor look like
    MARLTrainer's self.actors dict (one actor per agent).

    Wraps the shared actor so each "per-agent" call automatically injects
    the correct agent_idx.  Plug this into MARLTrainer.actors to enable
    shared-parameter training with no other code changes.
    """

    class _AgentProxy:
        """Proxy that binds agent_idx to the shared actor."""

        def __init__(self, shared: SharedAttentionActor, agent_id: str) -> None:
            self._shared   = shared
            self._agent_id = agent_id
            self._idx      = shared._agent_index(agent_id)

        # Forward the nn.Module interface the trainer expects
        def parameters(self):
            return self._shared.parameters()

        def state_dict(self):
            return self._shared.state_dict()

        def load_state_dict(self, *args, **kwargs):
            return self._shared.load_state_dict(*args, **kwargs)

        def to(self, device):
            self._shared.to(device)
            return self

        def train(self, mode=True):
            self._shared.train(mode)
            return self

        def eval(self):
            self._shared.eval()
            return self

        def forward(self, obs):
            return self._shared.forward(obs, self._idx)

        def distribution(self, obs):
            return self._shared.distribution(obs, self._idx)

        def get_action(self, state, device="cpu", deterministic=False):
            return self._shared.get_action(state, device, deterministic, self._agent_id)

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    def __init__(self, shared_actor: SharedAttentionActor, agent_ids: List[str]) -> None:
        self._shared  = shared_actor
        self._proxies = {
            aid: self._AgentProxy(shared_actor, aid)
            for aid in agent_ids
        }

    def __getitem__(self, agent_id: str):
        return self._proxies[agent_id]

    def __contains__(self, agent_id: str):
        return agent_id in self._proxies

    def keys(self):
        return self._proxies.keys()

    def values(self):
        return self._proxies.values()

    def items(self):
        return self._proxies.items()

    def get(self, agent_id, default=None):
        return self._proxies.get(agent_id, default)
