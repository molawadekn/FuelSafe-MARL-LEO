"""
marl/communication.py — TarMAC Inter-Satellite Communication
=============================================================

Implements Targeted Multi-Agent Communication (TarMAC, Das et al. 2019)
adapted for bandwidth-constrained inter-satellite links in LEO.

Architecture
------------
Each satellite agent i:
  1. Encodes its observation into a MESSAGE vector  m_i ∈ R^{msg_size}
     and a SIGNATURE vector  s_i ∈ R^{sig_size}  (what "type" of info it has)

  2. BROADCASTS (m_i, s_i) to all other satellites in the fleet.

  3. RECEIVES messages from all other agents.  Computes attention weights
     over received messages using its own QUERY vector  q_i ∈ R^{sig_size}:
         α_{ij} = softmax_j( q_i · s_j / √sig_size )
     Aggregates: c_i = Σ_j α_{ij} · m_j   (context vector)

  4. GATES the communication: a scalar gate g_i ∈ {0,1} decides whether
     to actually use the context vector this step (straight-through
     estimator for differentiability).  Gate is open when the agent's
     peak risk exceeds a learned threshold.

  5. AUGMENTS its encoded state:  h_i_aug = concat(h_i, c_i · g_i)
     This augmented vector is what the action heads receive.

Bandwidth realism
-----------------
Inter-satellite links in LEO are typically 1–10 Mbps with latency of
tens of ms.  We model this as:
  * message_size: configurable (default 16 floats = 64 bytes/step)
  * A 1-step communication delay mirror to ManeuverDelay (optional)
  * Messages are only sent when the gate is open (reduces effective
    bandwidth usage during routine operations)

Key properties
--------------
* Permutation equivariant: the aggregation is order-invariant.
* Scalable: O(N) messages per agent, O(N²) total — fine for N ≤ 100.
* Fully differentiable end-to-end (gate uses STE).
* Drop-in: output shape is (hidden_size + message_size) per agent,
  which the action heads receive after a linear projection back to
  hidden_size.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class StraightThroughBinary(torch.autograd.Function):
    """Straight-Through Estimator for binary gating."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return (x > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return grad_output   # pass-through gradient unchanged


_ste_binary = StraightThroughBinary.apply


class TarMACModule(nn.Module):
    """
    Targeted Multi-Agent Communication module.

    Parameters
    ----------
    hidden_size  : int   Size of the encoded agent representation (input).
    message_size : int   Size of the broadcast message vector (bandwidth proxy).
    sig_size     : int   Size of signature / query vectors for attention.
    num_agents   : int   Maximum expected fleet size (used for buffer pre-alloc).
    use_gate     : bool  Whether to use the binary communication gate.
    comm_delay   : bool  Buffer messages for 1 step (realistic link latency).
    """

    def __init__(
        self,
        hidden_size: int = 128,
        message_size: int = 16,
        sig_size: int = 16,
        num_agents: int = 3,
        use_gate: bool = True,
        comm_delay: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size  = hidden_size
        self.message_size = message_size
        self.sig_size     = sig_size
        self.use_gate     = use_gate
        self.comm_delay   = comm_delay
        self.scale        = math.sqrt(sig_size)

        # Message encoder: hidden → message + signature
        self.msg_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, message_size + sig_size),
        )

        # Query encoder: hidden → query (used to attend over signatures)
        self.query_encoder = nn.Linear(hidden_size, sig_size)

        # Gate: hidden → scalar probability (STE binarised)
        if use_gate:
            self.gate_head = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        # Projection: (hidden + message) → hidden  (keeps downstream dims clean)
        self.output_proj = nn.Linear(hidden_size + message_size, hidden_size)

        self._init_weights()

        # 1-step delay buffer: {agent_id: (message, signature)}
        self._delay_buffer: Dict[str, Optional[Tuple[torch.Tensor, torch.Tensor]]] = {}

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)

    def reset(self) -> None:
        """Clear delay buffers at episode start."""
        self._delay_buffer.clear()

    def forward(
        self,
        hidden_states: Dict[str, torch.Tensor],   # {agent_id: (1, H)}
    ) -> Dict[str, torch.Tensor]:
        """
        Run one communication round.

        Parameters
        ----------
        hidden_states : {agent_id: Tensor (1, hidden_size)}
            Encoded observation representations for each active agent.

        Returns
        -------
        augmented : {agent_id: Tensor (1, hidden_size)}
            Hidden states augmented with aggregated peer messages,
            projected back to hidden_size.
        """
        agent_ids = list(hidden_states.keys())
        if len(agent_ids) <= 1:
            return hidden_states   # nothing to communicate with

        # ── Step 1: Encode messages and signatures ────────────────────────────
        messages:   Dict[str, torch.Tensor] = {}
        signatures: Dict[str, torch.Tensor] = {}
        queries:    Dict[str, torch.Tensor] = {}
        gates:      Dict[str, torch.Tensor] = {}

        for aid, h in hidden_states.items():
            ms_out = self.msg_encoder(h)                         # (1, msg+sig)
            msg    = ms_out[:, :self.message_size]               # (1, msg)
            sig    = ms_out[:, self.message_size:]               # (1, sig)
            qry    = self.query_encoder(h)                       # (1, sig)

            # Communication delay: use last step's message
            if self.comm_delay:
                prev = self._delay_buffer.get(aid)
                if prev is not None:
                    msg_to_send, sig_to_send = prev
                else:
                    msg_to_send = torch.zeros_like(msg)
                    sig_to_send = torch.zeros_like(sig)
                self._delay_buffer[aid] = (msg.detach(), sig.detach())
            else:
                msg_to_send, sig_to_send = msg, sig

            messages[aid]   = msg_to_send
            signatures[aid] = sig_to_send
            queries[aid]    = qry

            if self.use_gate:
                gates[aid] = _ste_binary(self.gate_head(h))     # (1, 1)

        # ── Step 2: Signature-based attention aggregation ─────────────────────
        augmented: Dict[str, torch.Tensor] = {}

        for aid, h in hidden_states.items():
            other_ids = [o for o in agent_ids if o != aid]
            if not other_ids:
                augmented[aid] = self.output_proj(
                    torch.cat([h, torch.zeros(1, self.message_size, device=h.device)], dim=-1)
                )
                continue

            # Stack other agents' signatures and messages
            other_sigs = torch.cat(
                [signatures[o] for o in other_ids], dim=0
            )                                                    # (N-1, sig)
            other_msgs = torch.cat(
                [messages[o] for o in other_ids], dim=0
            )                                                    # (N-1, msg)

            # Attention weights: q_i · S^T / √d
            q = queries[aid]                                     # (1, sig)
            attn_logits = (q @ other_sigs.T) / self.scale       # (1, N-1)
            attn_weights = F.softmax(attn_logits, dim=-1)        # (1, N-1)

            # Context vector: weighted sum of others' messages
            context = attn_weights @ other_msgs                  # (1, msg)

            # Apply gate
            if self.use_gate:
                context = context * gates[aid]                   # zero out if gate closed

            # Project augmented representation back to hidden_size
            h_aug = self.output_proj(torch.cat([h, context], dim=-1))  # (1, H)
            augmented[aid] = h_aug

        return augmented

    def get_gate_values(
        self,
        hidden_states: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Return gate open/close decision per agent (for logging)."""
        if not self.use_gate:
            return {aid: 1.0 for aid in hidden_states}
        with torch.no_grad():
            return {
                aid: float(_ste_binary(self.gate_head(h)).item())
                for aid, h in hidden_states.items()
            }


class CommAwareAttentionActor(nn.Module):
    """
    ThreatAttentionActor + TarMAC communication in a single forward pass.

    The actor:
      1. Encodes its observation through the attention encoder → h_i
      2. Participates in a TarMAC communication round → h_i_aug
      3. Passes h_i_aug to the direction + magnitude heads.

    During training, all agents' encodings are computed in one batched
    forward pass, then TarMAC aggregates across the batch.

    During rollout, hidden_states are accumulated across agents before
    calling forward() so that communication is synchronous.
    """

    def __init__(
        self,
        input_size: int = 96,
        output_size: int = 7,
        hidden_size: int = 128,
        num_heads: int = 4,
        num_transformer_layers: int = 2,
        dropout: float = 0.1,
        message_size: int = 16,
        sig_size: int = 16,
        use_gate: bool = True,
        comm_delay: bool = False,
    ) -> None:
        super().__init__()

        from marl.attention_actor import ThreatAttentionActor
        from sim.maneuver_engine import MAX_DELTA_V_PER_STEP_KMS

        self.max_dv = float(MAX_DELTA_V_PER_STEP_KMS)

        # Attention encoder (reuse Phase 1 architecture)
        self.encoder = ThreatAttentionActor(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_transformer_layers=num_transformer_layers,
            dropout=dropout,
        )

        # TarMAC communication module
        self.comm = TarMACModule(
            hidden_size=hidden_size,
            message_size=message_size,
            sig_size=sig_size,
            use_gate=use_gate,
            comm_delay=comm_delay,
        )

        # Action heads (operate on comm-augmented hidden state)
        self.direction_head = nn.Linear(hidden_size, output_size)
        self.magnitude_mean_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        self.magnitude_logstd = nn.Parameter(
            torch.tensor([math.log(5.0e-4)], dtype=torch.float32)
        )

    def encode_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode a single agent's observation → hidden vector (1, H)."""
        return self.encoder._encode(obs)   # (B, H)

    def forward_with_comm(
        self,
        all_obs: Dict[str, torch.Tensor],   # {agent_id: (1, 96)}
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Full forward pass with inter-agent communication.

        Returns {agent_id: (direction_logits, mag_mean)} after comms.
        """
        # Step 1: encode all observations
        hidden = {aid: self.encoder._encode(obs) for aid, obs in all_obs.items()}

        # Step 2: TarMAC round
        augmented = self.comm(hidden)

        # Step 3: action heads on augmented representation
        outputs = {}
        for aid, h_aug in augmented.items():
            logits   = self.direction_head(h_aug)
            mag_mean = self.magnitude_mean_head(h_aug) * self.max_dv
            outputs[aid] = (logits, mag_mean)
        return outputs

    # ── Compatibility shims for MARLTrainer ──────────────────────────────────

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-agent forward (no communication). Used in PPO update."""
        return self.encoder.forward(x)

    def distribution(
        self, x: torch.Tensor
    ) -> Tuple[torch.distributions.Categorical, torch.distributions.Normal]:
        return self.encoder.distribution(x)

    def get_action(
        self,
        state: np.ndarray,
        device: str = "cpu",
        deterministic: bool = False,
    ) -> Tuple[Tuple[int, float], float]:
        """Single-agent action (no comm context). Fallback for isolated inference."""
        return self.encoder.get_action(state, device, deterministic)

    def reset_comm(self) -> None:
        """Reset communication delay buffers at episode start."""
        self.comm.reset()
