"""
marl/attention_actor.py — Futuristic Actor Architectures for MAPPO
====================================================================

Replaces the vanilla 3-layer MLP actor with three progressively advanced
network designs:

1. ThreatAttentionActor
   - Splits the 96-dim observation into own-state (12 dims) + up to 7 threat
     tokens (12 dims each).
   - Projects each to a shared embedding space (hidden_size).
   - Applies multi-head self-attention: the network *learns* which threats
     matter most rather than relying on the hard-coded Pc sort order.
   - Attention mask zeros out padding tokens (debris that weren't detected).
   - Output: CLS (own-state) token after attention → direction + magnitude heads.

2. RecurrentAttentionActor  (ThreatAttentionActor + GRU memory)
   - Adds a GRU layer on top of the attention encoder.
   - Hidden state (h_t) is carried across steps within an episode by
     MARLTrainer, giving the agent episodic memory of approach trajectories.
   - Fully compatible with the existing PPO buffer — hidden states are
     stored as extra fields per time-step.

3. EnsembleActor  (N × ThreatAttentionActor)
   - Maintains N independently initialised actor networks per agent.
   - Action selection: averages logits / magnitude across ensemble members.
   - Epistemic uncertainty: std-dev of direction logits across members.
     High uncertainty → agent is in an out-of-distribution situation.
   - Training: each member is updated independently with the same rollout
     data (implicit diversification through different random init + dropout).

All three classes expose the same interface as the original ActorNetwork:
    get_action(state, device, deterministic) → (action, log_prob)
    distribution(x)                          → (Categorical, Normal)

so MARLTrainer.actors can be swapped without changing any other code.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

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


# ─────────────────────────────────────────────────────────────────────────────
# Helper: split a flat observation into (own_state, threat_tokens, padding_mask)
# ─────────────────────────────────────────────────────────────────────────────

def _split_observation(
    obs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split the 96-dim observation vector into structured components.

    Parameters
    ----------
    obs : Tensor  shape (B, 96)

    Returns
    -------
    own_state    : Tensor  (B, OWN_FEATURE_COUNT=12)
    threat_tokens: Tensor  (B, MAX_NEARBY_OBJECTS=7, THREAT_FEATURE_COUNT=12)
    pad_mask     : BoolTensor  (B, 1 + MAX_NEARBY_OBJECTS)
                   True  → token is padding (should be masked out in attention)
                   False → token is valid
    """
    B = obs.shape[0]
    own_state = obs[:, :OWN_FEATURE_COUNT]                          # (B, 12)

    threat_flat = obs[:, THREATS_START_INDEX:]                      # (B, 84)
    threat_tokens = threat_flat.reshape(B, MAX_NEARBY_OBJECTS, THREAT_FEATURE_COUNT)  # (B,7,12)

    # A threat token is pure padding if ALL its features are zero
    is_padding = (threat_tokens.abs().sum(dim=-1) < 1e-6)          # (B, 7)

    # CLS (own-state) is never padding
    cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=obs.device)
    pad_mask = torch.cat([cls_pad, is_padding], dim=1)              # (B, 8)

    return own_state, threat_tokens, pad_mask


# ─────────────────────────────────────────────────────────────────────────────
# Learnable positional encoding (8 positions: 1 CLS + 7 threats)
# ─────────────────────────────────────────────────────────────────────────────

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, num_positions: int, hidden_size: int):
        super().__init__()
        self.encoding = nn.Embedding(num_positions, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, S, H) — add position embedding to each token."""
        B, S, H = x.shape
        positions = torch.arange(S, device=x.device).unsqueeze(0)  # (1, S)
        return x + self.encoding(positions)                         # (B, S, H)


# ─────────────────────────────────────────────────────────────────────────────
# 1. ThreatAttentionActor
# ─────────────────────────────────────────────────────────────────────────────

class ThreatAttentionActor(nn.Module):
    """
    Transformer-based policy network.

    Architecture
    ------------
    own_state (12)   → Linear → embedding (H)  = CLS token
    threat_i  (12)   → Linear → embedding (H)  = threat token i   [× 7]
    ─────────────────────────────────────────────────────────────────
    Sequence of 8 tokens → TransformerEncoder (L layers, nhead heads)
    ─────────────────────────────────────────────────────────────────
    CLS output → direction_head → Categorical(7 actions)
    CLS output → magnitude_head → Normal(μ, σ)
    """

    NUM_TOKENS = 1 + MAX_NEARBY_OBJECTS   # 1 CLS + 7 threats = 8

    def __init__(
        self,
        input_size: int = OBS_SIZE,          # kept for API compat, not directly used
        output_size: int = ACTION_COUNT,
        hidden_size: int = 128,
        num_heads: int = 4,
        num_transformer_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0, \
            f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"

        self.hidden_size = hidden_size
        self.max_dv      = float(MAX_DELTA_V_PER_STEP_KMS)

        # Token projections
        self.own_proj    = nn.Linear(OWN_FEATURE_COUNT,    hidden_size)
        self.threat_proj = nn.Linear(THREAT_FEATURE_COUNT, hidden_size)

        # Learnable positional encoding (8 positions)
        self.pos_enc = LearnablePositionalEncoding(self.NUM_TOKENS, hidden_size)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,          # Pre-LayerNorm (more stable training)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Action heads (same interface as original ActorNetwork)
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
                nn.init.zeros_(m.bias)
        # Smaller init for action heads (common in RL)
        nn.init.orthogonal_(self.direction_head.weight, gain=0.01)
        nn.init.orthogonal_(self.magnitude_mean_head[0].weight, gain=0.01)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of observations through the attention tower.

        Parameters
        ----------
        x : Tensor (B, 96)

        Returns
        -------
        cls_out : Tensor (B, hidden_size)  — CLS token after attention
        """
        own_state, threat_tokens, pad_mask = _split_observation(x)

        # Project own-state → CLS token  (B, 1, H)
        cls_token = self.own_proj(own_state).unsqueeze(1)

        # Project each threat → threat token  (B, 7, H)
        threat_emb = self.threat_proj(threat_tokens)

        # Concatenate → token sequence  (B, 8, H)
        tokens = torch.cat([cls_token, threat_emb], dim=1)

        # Add positional encoding
        tokens = self.pos_enc(tokens)

        # Transformer encoder with padding mask
        # PyTorch convention: True in src_key_padding_mask → ignore that position
        attended = self.transformer(tokens, src_key_padding_mask=pad_mask)

        # Use CLS token (position 0) as the aggregated representation
        return attended[:, 0, :]   # (B, H)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (direction_logits, magnitude_mean). Same signature as ActorNetwork."""
        h = self._encode(x)
        logits   = self.direction_head(h)
        mag_mean = self.magnitude_mean_head(h) * self.max_dv
        return logits, mag_mean

    def distribution(
        self, x: torch.Tensor
    ) -> Tuple[torch.distributions.Categorical, torch.distributions.Normal]:
        logits, mag_mean = self.forward(x)
        dir_dist = torch.distributions.Categorical(logits=logits)
        mag_std  = torch.exp(self.magnitude_logstd).clamp(5e-5, 1.5e-3).expand_as(mag_mean)
        mag_dist = torch.distributions.Normal(mag_mean, mag_std)
        return dir_dist, mag_dist

    def get_action(
        self,
        state: np.ndarray,
        device: str = "cpu",
        deterministic: bool = False,
    ) -> Tuple[Tuple[int, float], float]:
        """Drop-in replacement for ActorNetwork.get_action."""
        with torch.no_grad():
            x         = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            dir_dist, mag_dist = self.distribution(x)

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
# 2. RecurrentAttentionActor  (attention + GRU memory)
# ─────────────────────────────────────────────────────────────────────────────

class RecurrentAttentionActor(nn.Module):
    """
    ThreatAttentionActor augmented with a GRU layer for episodic memory.

    The hidden state h_t (shape: 1 × 1 × hidden_size) is maintained
    *externally* by MARLTrainer and threaded through each call to
    get_action_recurrent / distribution_recurrent.

    This lets the agent remember the trajectory of past conjunctions
    (approach geometry, relative velocities over time) without requiring
    changes to the PPO rollout buffer structure.
    """

    def __init__(
        self,
        input_size: int = OBS_SIZE,
        output_size: int = ACTION_COUNT,
        hidden_size: int = 128,
        num_heads: int = 4,
        num_transformer_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.max_dv      = float(MAX_DELTA_V_PER_STEP_KMS)

        # Attention encoder (same as ThreatAttentionActor)
        self.attention_encoder = ThreatAttentionActor(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_transformer_layers=num_transformer_layers,
            dropout=dropout,
        )

        # GRU memory layer
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # Action heads
        self.direction_head = nn.Linear(hidden_size, output_size)
        self.magnitude_mean_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        self.magnitude_logstd = nn.Parameter(
            torch.tensor([math.log(5.0e-4)], dtype=torch.float32)
        )

    def initial_hidden(self, device: str = "cpu") -> torch.Tensor:
        """Return a zero hidden state for episode start. Shape: (1, 1, H)."""
        return torch.zeros(1, 1, self.hidden_size, device=device)

    def forward_with_hidden(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor (B, 96)
        h : Tensor (1, B, H)   GRU hidden state

        Returns
        -------
        logits   : (B, 7)
        mag_mean : (B, 1)
        h_new    : (1, B, H)
        """
        # Attention encoding
        attn_out = self.attention_encoder._encode(x)   # (B, H)

        # GRU step: (B, 1, H) → (B, 1, H),  hidden: (1, B, H)
        gru_in  = attn_out.unsqueeze(1)                # (B, 1, H)
        gru_out, h_new = self.gru(gru_in, h)           # (B,1,H), (1,B,H)
        mem_out = gru_out.squeeze(1)                   # (B, H)

        logits   = self.direction_head(mem_out)
        mag_mean = self.magnitude_mean_head(mem_out) * self.max_dv
        return logits, mag_mean, h_new

    def distribution_recurrent(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
    ) -> Tuple[
        torch.distributions.Categorical,
        torch.distributions.Normal,
        torch.Tensor,
    ]:
        logits, mag_mean, h_new = self.forward_with_hidden(x, h)
        dir_dist = torch.distributions.Categorical(logits=logits)
        mag_std  = torch.exp(self.magnitude_logstd).clamp(5e-5, 1.5e-3).expand_as(mag_mean)
        mag_dist = torch.distributions.Normal(mag_mean, mag_std)
        return dir_dist, mag_dist, h_new

    # ── Compatibility shims (non-recurrent mode, h=zeros) ──────────────────
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        h = torch.zeros(1, B, self.hidden_size, device=x.device)
        logits, mag_mean, _ = self.forward_with_hidden(x, h)
        return logits, mag_mean

    def distribution(
        self, x: torch.Tensor
    ) -> Tuple[torch.distributions.Categorical, torch.distributions.Normal]:
        logits, mag_mean = self.forward(x)
        dir_dist = torch.distributions.Categorical(logits=logits)
        mag_std  = torch.exp(self.magnitude_logstd).clamp(5e-5, 1.5e-3).expand_as(mag_mean)
        mag_dist = torch.distributions.Normal(mag_mean, mag_std)
        return dir_dist, mag_dist

    def get_action(
        self,
        state: np.ndarray,
        device: str = "cpu",
        deterministic: bool = False,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[int, float], float, torch.Tensor]:
        """Returns action, log_prob, new_hidden."""
        with torch.no_grad():
            x = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            h = hidden if hidden is not None else self.initial_hidden(device)

            dir_dist, mag_dist, h_new = self.distribution_recurrent(x, h)

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

        return (
            (int(dir_action.item()), float(mag_action.item())),
            float(log_prob.item()),
            h_new,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. EnsembleActor  (N independent ThreatAttentionActor networks)
# ─────────────────────────────────────────────────────────────────────────────

class EnsembleActor(nn.Module):
    """
    Deep Ensemble policy: N independently trained ThreatAttentionActors.

    During inference:
      - Direction logits are averaged across members → single Categorical.
      - Magnitude mean/std are averaged across members.
      - Epistemic uncertainty = std of direction logits across members
        (high → agent is in an OOD / novel threat situation).

    During training:
      - All members share the same rollout data.
      - Each member's loss is computed independently and summed → the
        optimizer receives the full ensemble gradient in one backward pass.

    Properties
    ----------
    members : nn.ModuleList of ThreatAttentionActor (length N)
    """

    def __init__(
        self,
        input_size: int = OBS_SIZE,
        output_size: int = ACTION_COUNT,
        hidden_size: int = 128,
        num_heads: int = 4,
        num_transformer_layers: int = 2,
        dropout: float = 0.1,
        ensemble_size: int = 5,
    ) -> None:
        super().__init__()
        self.ensemble_size = ensemble_size
        self.max_dv        = float(MAX_DELTA_V_PER_STEP_KMS)

        self.members = nn.ModuleList([
            ThreatAttentionActor(
                input_size=input_size,
                output_size=output_size,
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_transformer_layers=num_transformer_layers,
                dropout=dropout,
            )
            for _ in range(ensemble_size)
        ])

    # ── Ensemble inference ──────────────────────────────────────────────────

    def forward_ensemble(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        mean_logits     : (B, ACTION_COUNT)  — averaged direction logits
        mean_mag        : (B, 1)             — averaged magnitude mean
        epistemic_std   : (B,)               — std of logits across members
        """
        all_logits: List[torch.Tensor] = []
        all_mags:   List[torch.Tensor] = []

        for member in self.members:
            logits, mag_mean = member.forward(x)
            all_logits.append(logits)
            all_mags.append(mag_mean)

        stacked_logits = torch.stack(all_logits, dim=0)   # (N, B, A)
        stacked_mags   = torch.stack(all_mags,   dim=0)   # (N, B, 1)

        mean_logits  = stacked_logits.mean(dim=0)          # (B, A)
        mean_mag     = stacked_mags.mean(dim=0)            # (B, 1)
        # Epistemic uncertainty: mean std of logits across ensemble members
        epistemic_std = stacked_logits.std(dim=0).mean(dim=-1)  # (B,)

        return mean_logits, mean_mag, epistemic_std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """API-compat forward → (mean_logits, mean_mag)."""
        mean_logits, mean_mag, _ = self.forward_ensemble(x)
        return mean_logits, mean_mag

    def distribution(
        self, x: torch.Tensor
    ) -> Tuple[torch.distributions.Categorical, torch.distributions.Normal]:
        mean_logits, mean_mag, _ = self.forward_ensemble(x)
        dir_dist = torch.distributions.Categorical(logits=mean_logits)
        # Average log_std across members
        avg_logstd = torch.stack(
            [m.magnitude_logstd for m in self.members], dim=0
        ).mean(dim=0)
        mag_std  = torch.exp(avg_logstd).clamp(5e-5, 1.5e-3).expand_as(mean_mag)
        mag_dist = torch.distributions.Normal(mean_mag, mag_std)
        return dir_dist, mag_dist

    def get_action(
        self,
        state: np.ndarray,
        device: str = "cpu",
        deterministic: bool = False,
    ) -> Tuple[Tuple[int, float], float]:
        """Drop-in replacement for ActorNetwork.get_action — also returns uncertainty."""
        with torch.no_grad():
            x = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            mean_logits, mean_mag, epistemic_std = self.forward_ensemble(x)

            dir_dist = torch.distributions.Categorical(logits=mean_logits)
            avg_logstd = torch.stack(
                [m.magnitude_logstd for m in self.members], dim=0
            ).mean(dim=0)
            mag_std  = torch.exp(avg_logstd).clamp(5e-5, 1.5e-3).expand_as(mean_mag)
            mag_dist = torch.distributions.Normal(mean_mag, mag_std)

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

    def get_uncertainty(self, state: np.ndarray, device: str = "cpu") -> float:
        """
        Return the epistemic uncertainty for a given observation.
        High values indicate an out-of-distribution or novel threat geometry.
        """
        with torch.no_grad():
            x = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            _, _, epistemic_std = self.forward_ensemble(x)
        return float(epistemic_std.item())

    # ── Training helper ─────────────────────────────────────────────────────

    def ensemble_loss(
        self,
        obs: torch.Tensor,
        dir_actions: torch.Tensor,
        mag_actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        clip_ratio: float = 0.2,
        entropy_coeff: float = 0.01,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the summed PPO actor loss across all ensemble members.

        Returns (total_loss, stats_dict).  Call total_loss.backward() once.
        """
        total_loss = torch.tensor(0.0, device=obs.device, requires_grad=True)
        stats: Dict[str, float] = {
            "actor_loss": 0.0, "policy_loss": 0.0, "entropy": 0.0
        }

        for member in self.members:
            dir_dist, mag_dist = member.distribution(obs)
            new_log_probs = (
                dir_dist.log_prob(dir_actions)
                + mag_dist.log_prob(mag_actions.unsqueeze(-1)).squeeze(-1)
            )
            entropy = (dir_dist.entropy() + mag_dist.entropy().squeeze(-1)).mean()

            ratio  = torch.exp(new_log_probs - old_log_probs)
            surr1  = ratio * advantages
            surr2  = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            loss   = policy_loss - entropy_coeff * entropy

            total_loss = total_loss + loss
            stats["actor_loss"]  += float(loss.item())
            stats["policy_loss"] += float(policy_loss.item())
            stats["entropy"]     += float(entropy.item())

        N = max(self.ensemble_size, 1)
        for k in stats:
            stats[k] /= N

        return total_loss / N, stats


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_actor(
    actor_type: str = "attention",
    input_size: int = OBS_SIZE,
    output_size: int = ACTION_COUNT,
    hidden_size: int = 128,
    num_heads: int = 4,
    num_transformer_layers: int = 2,
    dropout: float = 0.1,
    ensemble_size: int = 5,
) -> nn.Module:
    """
    Factory function — returns the requested actor type.

    actor_type options
    ------------------
    "mlp"        → original ActorNetwork (imported from marl_trainer)
    "attention"  → ThreatAttentionActor  (recommended)
    "recurrent"  → RecurrentAttentionActor  (attention + GRU memory)
    "ensemble"   → EnsembleActor  (N × ThreatAttentionActor)
    """
    if actor_type == "mlp":
        from marl.marl_trainer import ActorNetwork
        return ActorNetwork(input_size, output_size, hidden_size)

    if actor_type == "attention":
        return ThreatAttentionActor(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_transformer_layers=num_transformer_layers,
            dropout=dropout,
        )

    if actor_type == "recurrent":
        return RecurrentAttentionActor(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_transformer_layers=num_transformer_layers,
            dropout=dropout,
        )

    if actor_type == "ensemble":
        return EnsembleActor(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_transformer_layers=num_transformer_layers,
            dropout=dropout,
            ensemble_size=ensemble_size,
        )

    raise ValueError(
        f"Unknown actor_type '{actor_type}'. "
        "Choose from: mlp, attention, recurrent, ensemble"
    )
