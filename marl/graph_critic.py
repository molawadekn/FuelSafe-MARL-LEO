"""
marl/graph_critic.py — Graph Attention Network Centralized Critic
==================================================================

Replaces the flat-concatenation centralized critic
    (OBS_SIZE × N → MLP → scalar)
with a Graph Attention Network (GAT) critic that:

  * Treats each satellite as a GRAPH NODE with features = its observation.
  * Adds EDGES between every pair of satellites weighted by the conjunction
    risk between them (extracted from their observations).
  * Runs 2 GAT layers (multi-head attention over neighbours).
  * Global MEAN-POOLS node representations → single scalar value estimate.

Key advantages over the flat MLP critic
-----------------------------------------
1. Scale-invariant: handles N=3 or N=100 satellites without changing
   any weights — the graph pooling adapts to fleet size automatically.
2. Relational inductive bias: the critic explicitly reasons about pairwise
   interactions (who is close to whom, whose risk is highest).
3. Permutation equivariant: value estimate is unchanged if satellite IDs
   are shuffled (correct for homogeneous fleets).
4. Better credit assignment: each node's contribution to the global value
   is proportional to its role in the current joint situation.

Graph structure
---------------
  Nodes : one per active satellite, feature = obs (96-dim)
  Edges : fully connected (each satellite sees all others)
  Edge features : [relative risk, normalized distance] (2-dim)
                  extracted from the observations' own-state summaries.

GAT layer
---------
For each node i and neighbor j:
    e_{ij} = LeakyReLU( a^T [W h_i || W h_j || W_e edge_{ij}] )
    α_{ij} = softmax_j(e_{ij})
    h_i'   = σ( Σ_j α_{ij} W h_j )   (multi-head, concatenated)

Value head
----------
  global_repr = mean_pool( h_i' )  →  Linear → scalar
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sim.observation_utils import (
    MAX_RISK_INDEX,
    MIN_MISS_DISTANCE_INDEX,
    OBS_SIZE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Single GAT layer
# ─────────────────────────────────────────────────────────────────────────────

class GATLayer(nn.Module):
    """
    Multi-head Graph Attention layer.

    Parameters
    ----------
    in_features  : int   Input node feature size.
    out_features : int   Output node feature size (per head).
    num_heads    : int   Number of attention heads.
    edge_dim     : int   Edge feature size (default 2: risk + distance).
    concat       : bool  Concatenate heads (True) or average (False).
    dropout      : float Dropout on attention weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        edge_dim: int = 2,
        concat: bool = True,
        dropout: float = 0.1,
        leaky_slope: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.num_heads    = num_heads
        self.concat       = concat
        self.dropout      = dropout

        # Node feature projection (shared across heads)
        self.W = nn.Linear(in_features, out_features * num_heads, bias=False)

        # Edge feature projection
        self.W_edge = nn.Linear(edge_dim, out_features * num_heads, bias=False)

        # Attention coefficient vector  a ∈ R^{2*out + out}
        self.a = nn.Parameter(
            torch.empty(num_heads, 2 * out_features + out_features)
        )
        self.leaky = nn.LeakyReLU(leaky_slope)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.W_edge.weight)
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

    def forward(
        self,
        x: torch.Tensor,          # (N, in_features)
        edge_feats: torch.Tensor,  # (N, N, edge_dim)
    ) -> torch.Tensor:             # (N, out_features * num_heads) or (N, out_features)
        N = x.size(0)
        H = self.num_heads
        F_out = self.out_features

        # Project node features: (N, H*F_out)
        Wh = self.W(x).view(N, H, F_out)                        # (N, H, F_out)

        # Project edge features: (N, N, H*F_out)
        We = self.W_edge(edge_feats).view(N, N, H, F_out)       # (N, N, H, F_out)

        # Broadcast for pairwise combination
        Wh_i = Wh.unsqueeze(1).expand(N, N, H, F_out)           # (N, N, H, F_out)
        Wh_j = Wh.unsqueeze(0).expand(N, N, H, F_out)           # (N, N, H, F_out)

        # Attention input: [h_i || h_j || e_ij]  per head
        attn_input = torch.cat([Wh_i, Wh_j, We], dim=-1)        # (N, N, H, 3*F_out)

        # Attention coefficient: dot with a per head
        e = (attn_input * self.a.unsqueeze(0).unsqueeze(0)).sum(-1)  # (N, N, H)
        e = self.leaky(e)

        # Softmax over neighbours (dim=1 = source nodes)
        alpha = F.softmax(e, dim=1)                              # (N, N, H)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Aggregate: Σ_j α_{ij} Wh_j
        out = (alpha.unsqueeze(-1) * Wh_j).sum(dim=1)           # (N, H, F_out)

        if self.concat:
            return out.reshape(N, H * F_out)                     # (N, H*F_out)
        else:
            return out.mean(dim=1)                               # (N, F_out)


# ─────────────────────────────────────────────────────────────────────────────
# Graph Attention Critic
# ─────────────────────────────────────────────────────────────────────────────

class GraphAttentionCritic(nn.Module):
    """
    GAT-based centralized value function.

    Input  : dict {agent_id → observation (96-dim)}
    Output : scalar value estimate V(s)

    The graph is fully connected (all satellites see each other).
    Edge features are extracted from observations:
      [max_risk_i, max_risk_j]  (2-dim per edge, normalized to [0,1])

    Architecture
    ------------
    obs (96) → Linear(hidden) → GAT layer 1 (4 heads, concat) →
    GAT layer 2 (1 head, mean) → global mean pool → Linear(1) → V
    """

    def __init__(
        self,
        obs_size: int = OBS_SIZE,
        hidden_size: int = 128,
        num_heads: int = 4,
        edge_dim: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.obs_size    = obs_size
        self.hidden_size = hidden_size
        self.num_heads   = num_heads

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
        )

        # GAT layer 1: concat heads → hidden * num_heads
        gat1_out = hidden_size   # per head
        self.gat1 = GATLayer(
            in_features=hidden_size,
            out_features=gat1_out,
            num_heads=num_heads,
            edge_dim=edge_dim,
            concat=True,
            dropout=dropout,
        )
        gat1_total = gat1_out * num_heads   # e.g. 128*4=512

        # GAT layer 2: average heads → hidden_size
        self.gat2 = GATLayer(
            in_features=gat1_total,
            out_features=hidden_size,
            num_heads=1,
            edge_dim=edge_dim,
            concat=False,
            dropout=dropout,
        )

        # Value head: global mean pool → scalar
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
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)

    def _build_edge_features(
        self,
        obs_tensor: torch.Tensor,   # (N, OBS_SIZE)
    ) -> torch.Tensor:
        """
        Extract 2-dim edge features for all (i,j) pairs.

        Edge feature for (i→j): [max_risk_i, max_risk_j]
        Both are already normalized to [0,1] in the observation.
        """
        N = obs_tensor.size(0)
        # max_risk for each agent: index MAX_RISK_INDEX in obs
        risks = obs_tensor[:, MAX_RISK_INDEX]                    # (N,)
        risk_i = risks.unsqueeze(1).expand(N, N)                 # (N, N)
        risk_j = risks.unsqueeze(0).expand(N, N)                 # (N, N)
        edge_feats = torch.stack([risk_i, risk_j], dim=-1)       # (N, N, 2)
        return edge_feats

    def forward(self, observations: Dict[str, np.ndarray], device: str = "cpu") -> float:
        """
        Compute scalar value for the joint observation.

        Parameters
        ----------
        observations : {agent_id → np.ndarray (96,)}
        device       : torch device string

        Returns
        -------
        value : float   V(s) scalar
        """
        agent_ids  = sorted(observations.keys())
        obs_list   = [
            torch.as_tensor(observations[aid], dtype=torch.float32, device=device)
            for aid in agent_ids
        ]

        if not obs_list:
            return 0.0

        obs_tensor = torch.stack(obs_list, dim=0)                # (N, 96)
        return float(self._forward_tensor(obs_tensor).item())

    def forward_tensor(
        self,
        central_obs: torch.Tensor,    # (B, N * OBS_SIZE) — flat concat format
        num_agents: int = 3,
    ) -> torch.Tensor:
        """
        Batch forward compatible with MARLTrainer's PPO update loop.

        central_obs is the flat concatenated observation used by the
        original CriticNetwork — we reshape it back into (B, N, OBS_SIZE)
        then run the GAT.

        Returns: (B, 1) value predictions.
        """
        B  = central_obs.size(0)
        obs = central_obs.view(B, num_agents, self.obs_size)     # (B, N, OBS_SIZE)

        values = []
        for b in range(B):
            v = self._forward_tensor(obs[b])                     # scalar tensor
            values.append(v)

        return torch.stack(values, dim=0).unsqueeze(-1)          # (B, 1)

    def _forward_tensor(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Core GAT forward for a single graph (one episode step).

        Parameters
        ----------
        obs : (N, OBS_SIZE)

        Returns
        -------
        value : scalar tensor
        """
        # Input projection
        h = self.input_proj(obs)                                  # (N, H)

        # Edge features
        edge_feats = self._build_edge_features(obs)               # (N, N, 2)

        # GAT layers
        h = F.elu(self.gat1(h, edge_feats))                       # (N, H*heads)
        h = F.elu(self.gat2(h, edge_feats))                       # (N, H)

        # Global mean pool → value
        pooled = h.mean(dim=0, keepdim=True)                      # (1, H)
        value  = self.value_head(pooled).squeeze()                 # scalar

        return value
