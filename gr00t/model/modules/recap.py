# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RECAP (arXiv:2511.14759) building blocks: advantage conditioning + the
distributional value function used to derive it.

References the paper's equations directly:
  - Eq. 1 (distributional value function training / cross-entropy over bins)
  - Eq. 3 (advantage-conditioned policy training objective)
  - Eq. 5 (sparse terminal reward -> empirical return)
  - Appendix F (advantage estimation, threshold selection, CFG dropout)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
import torch.nn.functional as F
from transformers import PretrainedConfig


class AdvantageEmbedding(nn.Module):
    """Encodes the binary advantage/improvement indicator ``I_t`` (RECAP §IV-B).

    Three indices:
        NULL_IDX (0) - unconditional token (CFG dropout / unconditional inference pass).
        NEG_IDX  (1) - A(o, a) <= eps_l  ->  "Advantage: negative"
        POS_IDX  (2) - A(o, a)  > eps_l  ->  "Advantage: positive"

    The resulting token is appended to the VL encoder context so DiT cross-attention
    can condition the predicted velocity field on optimality.
    """

    NULL_IDX: int = 0
    NEG_IDX: int = 1
    POS_IDX: int = 2

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(3, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """labels: (B,) long in {NULL_IDX, NEG_IDX, POS_IDX} -> (B, 1, embedding_dim)."""
        return self.embedding(labels).unsqueeze(1)


@dataclass
class DistributionalValueHeadConfig(PretrainedConfig):
    """Config for the distributional value function p(V | o_t, l) (RECAP §IV-A).

    Kept as a lightweight head that reuses the shared VL backbone features the
    policy already computes (no separate value-function backbone), so it costs
    one extra small forward pass rather than a whole extra VLM.
    """

    backbone_embedding_dim: int = field(default=2048)
    state_dim: int = field(default=64)
    num_heads: int = field(default=8)
    dropout: float = field(default=0.1)
    hidden_dim: int = field(default=512)
    num_bins: int = field(default=201)  # B = 201, as in pi*0.6
    value_min: float = field(default=-1.0)  # normalised lower bound (failure)
    value_max: float = field(default=0.0)  # normalised upper bound (success, 0 steps left)
    value_loss_coeff: float = field(default=1.0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)


class DistributionalValueHead(nn.Module):
    """Distributional value function p(V | o_t, l) from RECAP §IV-A.

    Trained by minimising cross-entropy between the predicted bin distribution
    and the discretised empirical return R_t(tau) (Eq. 1):

        min_phi  E_{tau in D} [ sum_{o_t in tau}  H( R^B_t(tau),  p_phi(V | o_t, l) ) ]

    At inference, a scalar value is extracted as:

        V(o_t, l) = sum_b  p_phi(V=b | o_t) * v(b)

    where v(b) is the real-valued centre of bin b.
    """

    def __init__(self, config: DistributionalValueHeadConfig):
        super().__init__()
        num_heads = config.num_heads
        while config.backbone_embedding_dim % num_heads != 0 and num_heads > 1:
            num_heads //= 2

        self.norm_in = nn.LayerNorm(config.backbone_embedding_dim)

        # Learnable CLS token acting as a dedicated "task-completion query".
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.backbone_embedding_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.backbone_embedding_dim,
            num_heads=num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm_attn = nn.LayerNorm(config.backbone_embedding_dim)

        self.state_proj = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.backbone_embedding_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(config.backbone_embedding_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim // 2),
            nn.Linear(config.hidden_dim // 2, config.num_bins),
        )

        self.register_buffer(
            "bin_centres",
            torch.linspace(config.value_min, config.value_max, config.num_bins),
        )

        self.config = config
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.ones_(self.norm_in.weight)
        nn.init.zeros_(self.norm_in.bias)
        nn.init.ones_(self.norm_attn.weight)
        nn.init.zeros_(self.norm_attn.bias)
        for name, p in self.cross_attn.named_parameters():
            if "weight" in name:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(p)
        for module in list(self.state_proj.modules()) + list(self.classifier.modules()):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.zeros_(module.bias)

    def _encode(self, backbone_features: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            backbone_features: (B, S, D) VL token sequence (post vlln/self-attn).
            state: (B, state_dim) flattened proprioceptive state.
        Returns:
            (B, num_bins) unnormalised logits over value bins.
        """
        backbone_features = backbone_features.to(self.cls_token.dtype)
        cls = self.cls_token.expand(backbone_features.shape[0], -1, -1)  # (B, 1, D)
        normed = self.norm_in(backbone_features)
        attended, _ = self.cross_attn(query=cls, key=normed, value=normed)  # (B, 1, D)
        attended = self.norm_attn(attended.squeeze(1))  # (B, D)
        state_proj = self.state_proj(state.to(self.cls_token.dtype))  # (B, D)
        fused = torch.cat([attended, state_proj], dim=-1)  # (B, 2D)
        return self.classifier(fused)

    def predict_value(self, backbone_features: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(self._encode(backbone_features, state), dim=-1)
        return (probs * self.bin_centres.to(probs.dtype)).sum(dim=-1)  # (B,)

    def compute_loss(
        self,
        backbone_features: torch.Tensor,
        state: torch.Tensor,
        empirical_returns: torch.Tensor,
    ) -> torch.Tensor:
        logits = self._encode(backbone_features, state)
        bin_idx = self._discretise(empirical_returns)
        return F.cross_entropy(logits.float(), bin_idx) * self.config.value_loss_coeff

    def _discretise(self, returns: torch.Tensor) -> torch.Tensor:
        v_min, v_max = self.config.value_min, self.config.value_max
        r = returns.clamp(v_min, v_max)
        return ((r - v_min) / (v_max - v_min) * (self.config.num_bins - 1)).long()


def compute_normalised_returns(
    success: torch.Tensor,
    episode_lengths: torch.Tensor,
    t: torch.Tensor,
    max_episode_length: float = 1000.0,
    c_fail: float = 500.0,
) -> torch.Tensor:
    """Empirical return R_t(tau) used to train the value head (RECAP Eq. 1, Eq. 5, §V-C).

    Sparse terminal reward (Eq. 5):
        r_t' = 0        if t' == T and success
               -C_fail  if t' == T and failure
               -1       otherwise

    Return-to-go from step t:
        R_t(tau) = sum_{t'=t}^{T} r_t'
                 = -(T - t)           if success
                 = -(T - t) - C_fail  if failure

    Normalised to (-1, 0) (pi*0.6 §V-C):
        R_t_norm = R_t(tau) / (max_episode_length + C_fail)

    Args:
        success: (B,) bool - episode-level success flag.
        episode_lengths: (B,) long - T for each sampled episode.
        t: (B,) long - current timestep index within the episode.
        max_episode_length: normalisation constant.
        c_fail: penalty added to the denominator / failure return.

    Returns:
        (B,) float in [-1, 0].
    """
    steps_remaining = (episode_lengths - t).float()
    raw_return = -steps_remaining
    raw_return = torch.where(success, raw_return, raw_return - c_fail)
    norm_denom = float(max_episode_length + c_fail)
    return (raw_return / norm_denom).clamp(-1.0, 0.0)
