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

import logging
from typing import Any, Tuple

import torch
from torch import nn
from torch.distributions import Beta
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.modules.dit import AlternateVLDiT, DiT, SelfAttentionTransformer
from gr00t.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)
from gr00t.model.modules.recap import (
    AdvantageEmbedding,
    DistributionalValueHead,
    DistributionalValueHeadConfig,
    compute_normalised_returns,
)


logger = logging.getLogger(__name__)


class Gr00tN1d7ActionHead(nn.Module):
    """Action head component for flow matching diffusion policy."""

    supports_gradient_checkpointing = True

    def __init__(self, config: Gr00tN1d7Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
            logger.info("Using AlternateVLDiT for diffusion model")
        else:
            self.model = DiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
            )
            logger.info("Using DiT for diffusion model")
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim * config.state_history_length,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )

        vl_self_attention_cfg = getattr(config, "vl_self_attention_cfg", None)
        if vl_self_attention_cfg and vl_self_attention_cfg.get("num_layers", 0) > 0:
            self.vl_self_attention = SelfAttentionTransformer(**vl_self_attention_cfg)
        else:
            self.vl_self_attention = nn.Identity()

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # State dropout parameters
        self.state_dropout_prob = config.state_dropout_prob

        # Pin the time-sampling Beta to CPU/fp32 explicitly. The action head can
        # be instantiated under a meta / no_init_weights default-device context
        # (e.g. nested from_pretrained). A Beta built from bare Python floats
        # would then place its concentration tensors on the meta device (or in
        # the active default dtype, e.g. bf16). With validate_args enabled that
        # already fails here in __init__ (Beta's internal .item() check cannot
        # run on meta); even with validation off, sample_time would later raise
        # or return garbage. Explicit device/dtype here makes the sampler depend
        # only on the config, not on the construction-time device/dtype context,
        # so the noise schedule is identical across SDPA/FA2/FA4 and meta vs.
        # real-device loads. config is the canonical source for these values.
        self.beta_dist = Beta(
            torch.tensor(float(config.noise_beta_alpha), dtype=torch.float32, device="cpu"),
            torch.tensor(float(config.noise_beta_beta), dtype=torch.float32, device="cpu"),
        )
        self.num_timestep_buckets = config.num_timestep_buckets

        # --- RECAP (arXiv:2511.14759) advantage conditioning ---
        self.recap_enabled = config.recap_enabled
        self._phase = config.recap_stage if config.recap_enabled else "policy"
        if config.recap_enabled:
            self.advantage_embedding = AdvantageEmbedding(config.backbone_embedding_dim)
            self.value_head = DistributionalValueHead(
                DistributionalValueHeadConfig(
                    backbone_embedding_dim=config.backbone_embedding_dim,
                    state_dim=config.max_state_dim * config.state_history_length,
                    hidden_dim=config.value_head_hidden_dim,
                    num_heads=config.value_head_num_heads,
                    dropout=config.value_head_dropout,
                    num_bins=config.value_head_num_bins,
                    value_loss_coeff=config.value_loss_coeff,
                )
            )
        else:
            self.advantage_embedding = None
            self.value_head = None

        self.set_trainable_parameters(
            config.tune_projector, config.tune_diffusion_model, config.tune_vlln
        )

        if config.recap_enabled:
            if self._phase == "value_head":
                self.set_phase_value_head()
            elif self._phase == "policy":
                self.set_phase_policy()
            else:
                raise ValueError(
                    f"Unknown recap_stage={self._phase!r}; expected 'value_head' or 'policy'"
                )

    def set_trainable_parameters(
        self, tune_projector: bool, tune_diffusion_model: bool, tune_vlln: bool
    ):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        if not tune_vlln:
            self.vlln.requires_grad_(False)
            self.vl_self_attention.requires_grad_(False)
        logger.debug(f"Tune action head projector: {self.tune_projector}")
        logger.debug(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        logger.debug(f"Tune action head vlln: {self.tune_vlln}")
        # Check if any parameters are still trainable. If not, log a warning.
        if not tune_projector and not tune_diffusion_model and not tune_vlln:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    logger.debug(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            logger.warning("No action head trainable parameters found.")

    def set_phase_value_head(self):
        """RECAP Stage 1 (Eq. 1, Algorithm 1 lines 1/4/8): train V on D.

        Everything except the value head + advantage embedding is frozen —
        the policy (DiT, encoders, decoders, backbone-facing vlln) must stay
        fixed while the value head is fit.
        """
        assert self.recap_enabled, "set_phase_value_head requires config.recap_enabled=True"
        for p in self.parameters():
            p.requires_grad = False
        for p in self.value_head.parameters():
            p.requires_grad = True
        for p in self.advantage_embedding.parameters():
            p.requires_grad = True
        self._phase = "value_head"
        logger.info(
            "[RECAP] Phase 1 (value_head) — trainable params: %d",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def set_phase_policy(self):
        """RECAP Stage 2 (Eq. 3, Algorithm 1 lines 2/5/9): train pi on D given
        the frozen value head V.

        Restores normal policy trainability (tune_projector/tune_diffusion_model/
        tune_vlln), freezes the value head (used only to label advantages),
        and keeps the advantage embedding trainable.
        """
        assert self.recap_enabled, "set_phase_policy requires config.recap_enabled=True"
        self.set_trainable_parameters(
            self.tune_projector, self.tune_diffusion_model, self.tune_vlln
        )
        for p in self.value_head.parameters():
            p.requires_grad = False
        for p in self.advantage_embedding.parameters():
            p.requires_grad = True
        self._phase = "policy"
        logger.info(
            "[RECAP] Phase 2 (policy) — trainable params: %d",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()
            if not self.tune_vlln:
                self.vlln.eval()
                self.vl_self_attention.eval()
            if self.recap_enabled and self._phase == "policy":
                # Value head only labels advantages in the policy phase; never trained here.
                self.value_head.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.config.noise_s
        return sample

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def _apply_advantage_conditioning(
        self,
        vl_embeds: torch.Tensor,
        vl_attn_mask: torch.Tensor,
        image_mask: torch.Tensor | None,
        advantage_label: torch.Tensor | None,
        *,
        force_null: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Append a single advantage token to the VL encoder context (RECAP §V-B).

        Args:
            vl_embeds: (B, S, backbone_embedding_dim) VL token sequence.
            vl_attn_mask: (B, S) attention mask over ``vl_embeds``.
            image_mask: (B, S) bool, True for image tokens, or None when the
                model doesn't use AlternateVLDiT (mask is unused there).
            advantage_label: (B,) long in {NEG_IDX, POS_IDX}, or None to force
                the unconditional (NULL) token for every sample.
            force_null: If True, ignore ``advantage_label`` and always inject
                the NULL token (used for the unconditional CFG branch).

        Returns:
            (vl_embeds_aug, vl_attn_mask_aug, image_mask_aug) each with one
            extra token appended along the sequence dimension. The appended
            token is marked as non-image (attended, ``image_mask=False``) so
            it participates in the text/non-image cross-attention blocks.
        """
        B = vl_embeds.shape[0]
        device = vl_embeds.device

        if advantage_label is None or force_null:
            labels = torch.full((B,), AdvantageEmbedding.NULL_IDX, dtype=torch.long, device=device)
        else:
            labels = advantage_label.to(device=device)
            if self.training:
                drop = torch.rand(B, device=device) < self.config.advantage_cfg_dropout_prob
                labels = labels.masked_fill(drop, AdvantageEmbedding.NULL_IDX)

        adv_token = self.advantage_embedding(labels).to(vl_embeds.dtype)  # (B, 1, D)
        vl_embeds_aug = torch.cat([vl_embeds, adv_token], dim=1)

        extra_mask = torch.ones(B, 1, dtype=vl_attn_mask.dtype, device=device)
        vl_attn_mask_aug = torch.cat([vl_attn_mask, extra_mask], dim=1)

        image_mask_aug = None
        if image_mask is not None:
            extra_img = torch.zeros(B, 1, dtype=image_mask.dtype, device=device)
            image_mask_aug = torch.cat([image_mask, extra_img], dim=1)

        return vl_embeds_aug, vl_attn_mask_aug, image_mask_aug

    @torch.no_grad()
    def compute_advantage_labels(
        self,
        backbone_feats: torch.Tensor,
        state: torch.Tensor,
        reward: torch.Tensor,
        percentile: float,
    ) -> torch.Tensor:
        """Derive the per-sample advantage indicator I_t from the frozen value head.

        RECAP App. F: eps_l is set at a percentile of the values predicted by
        the value head over the current batch, so a target fraction of steps
        get positive advantage. Failure steps (reward < 0) are always labelled
        NEG regardless of predicted value — failure is never positive advantage.

        Returns:
            (B,) long tensor in {AdvantageEmbedding.NEG_IDX, AdvantageEmbedding.POS_IDX}.
        """
        V = self.value_head.predict_value(backbone_feats, state)  # (B,)
        epsilon = torch.quantile(V.float(), percentile)
        is_failure = reward < 0
        above_threshold = V > epsilon
        labels = torch.where(
            above_threshold & ~is_failure,
            torch.full_like(V, AdvantageEmbedding.POS_IDX, dtype=torch.long),
            torch.full_like(V, AdvantageEmbedding.NEG_IDX, dtype=torch.long),
        )
        return labels

    def forward_value_head(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        """RECAP Stage 1 forward (Eq. 1): cross-entropy loss for the distributional
        value head against the discretised empirical return."""
        self.set_frozen_modules_to_eval_mode()
        backbone_output = self.process_backbone_output(backbone_output)
        vl_embeds = backbone_output.backbone_features

        if logger.isEnabledFor(logging.DEBUG):
            with torch.no_grad():
                logger.debug(
                    "[RECAP value_head] vl_embeds stats: "
                    f"abs_max={vl_embeds.abs().max().item():.3g} "
                    f"abs_mean={vl_embeds.abs().mean().item():.3g} "
                    f"std={vl_embeds.float().std().item():.3g}"
                )

        assert "reward" in action_input, (
            "RECAP value-head training requires 'reward', 'reward.current_frame_idx' and "
            f"'reward.episode_lengths' in action_input; got keys={list(action_input.keys())}. "
            "Ensure the dataset has a 'next.done' column (see LeRobotEpisodeLoader)."
        )
        reward = action_input["reward"].float()
        t = action_input["reward.current_frame_idx"].long()
        episode_lengths = action_input["reward.episode_lengths"].long()

        assert action_input.state.shape[1] == self.config.state_history_length
        state = action_input.state.reshape(action_input.state.shape[0], -1)

        empirical_return = compute_normalised_returns(
            success=reward >= 0,
            episode_lengths=episode_lengths,
            t=t,
            max_episode_length=self.config.recap_max_episode_length,
            c_fail=self.config.recap_c_fail,
        )
        value_loss = self.value_head.compute_loss(vl_embeds, state, empirical_return)

        return {
            "loss": value_loss,
            "value_loss": value_loss.detach(),
        }

    def forward_action_head_recap(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        """RECAP Stage 2 forward (Eq. 3): advantage-conditioned policy training.

            L = || v_theta(a_t | o, l, NULL)   - v ||^2         (unconditional term)
              + alpha * || v_theta(a_t | o, l, I_t) - v ||^2     (advantage-conditioned term)

        Both terms share the same noised trajectory / velocity target and CFG
        dropout (config.advantage_cfg_dropout_prob) randomly maps I_t -> NULL in
        the second term so a single model can be sampled from unconditionally or
        conditionally at inference time (RECAP App. F).
        """
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device
        embodiment_id = action_input.embodiment_id

        assert action_input.state.shape[1] == self.config.state_history_length
        action_input.state = action_input.state.view(action_input.state.shape[0], 1, -1)
        state_features = self.state_encoder(action_input.state, embodiment_id)

        if self.training and self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device)
                < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout)

        assert "reward" in action_input, (
            "RECAP policy training requires 'reward' in action_input; got "
            f"keys={list(action_input.keys())}. Ensure the dataset has a 'next.done' column."
        )
        reward = action_input["reward"].float()
        with torch.no_grad():
            flat_state = action_input.state.reshape(action_input.state.shape[0], -1)
            adv_labels = self.compute_advantage_labels(
                backbone_feats=vl_embeds,
                state=flat_state,
                reward=reward,
                percentile=self.config.advantage_threshold_percentile,
            )
        advantage_pos_frac = (adv_labels == AdvantageEmbedding.POS_IDX).float().mean()

        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        sa_embs = torch.cat((state_features, action_features), dim=1)
        vl_attn_mask = backbone_output.backbone_attention_mask
        image_mask = backbone_output.image_mask if self.config.use_alternate_vl_dit else None
        action_mask = action_input.action_mask

        def run_model(vl_e: torch.Tensor, mask_e: torch.Tensor, img_e: torch.Tensor | None):
            if self.config.use_alternate_vl_dit:
                model_output, _ = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_e,
                    encoder_attention_mask=mask_e,
                    timestep=t_discretized,
                    return_all_hidden_states=True,
                    image_mask=img_e,
                    backbone_attention_mask=mask_e,
                )
            else:
                model_output, _ = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_e,
                    encoder_attention_mask=mask_e,
                    timestep=t_discretized,
                    return_all_hidden_states=True,
                )
            pred = self.action_decoder(model_output, embodiment_id)
            return pred[:, -actions.shape[1] :]

        # Unconditional term: -log pi(a | o, l)
        vl_null, mask_null, img_null = self._apply_advantage_conditioning(
            vl_embeds, vl_attn_mask, image_mask, advantage_label=None
        )
        pred_null = run_model(vl_null, mask_null, img_null)
        loss_uncond = F.mse_loss(pred_null, velocity, reduction="none") * action_mask
        loss_uncond = loss_uncond.sum() / (action_mask.sum() + 1e-6)

        # Advantage-conditioned term: -alpha * log pi(a | I_t, o, l)
        vl_cond, mask_cond, img_cond = self._apply_advantage_conditioning(
            vl_embeds, vl_attn_mask, image_mask, advantage_label=adv_labels
        )
        pred_cond = run_model(vl_cond, mask_cond, img_cond)
        loss_cond = F.mse_loss(pred_cond, velocity, reduction="none") * action_mask
        loss_cond = loss_cond.sum() / (action_mask.sum() + 1e-6)

        total_loss = loss_uncond + self.config.recap_alpha * loss_cond

        return {
            "loss": total_loss,
            "action_loss_uncond": loss_uncond.detach(),
            "action_loss_cond": loss_cond.detach(),
            "advantage_pos_frac": advantage_pos_frac.detach(),
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        """
        Forward pass through the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - action: [B, action_horizon, action_dim] (during training)
                - embodiment_id: [B] (embodiment IDs)
                - action_mask: [B, action_horizon, action_dim]

        Returns:
            BatchFeature containing:
                - loss: action prediction loss
        """
        if self.recap_enabled:
            if self._phase == "value_head":
                return self.forward_value_head(backbone_output, action_input)
            else:
                return self.forward_action_head_recap(backbone_output, action_input)

        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Handle state history
        assert action_input.state.shape[1] == self.config.state_history_length
        action_input.state = action_input.state.view(action_input.state.shape[0], 1, -1)

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Dropout state features (training only): zero out dropped states.
        if self.training and self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device)
                < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout)

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        sa_embs = torch.cat((state_features, action_features), dim=1)
        vl_attn_mask = backbone_output.backbone_attention_mask

        if self.config.use_alternate_vl_dit:
            image_mask = backbone_output.image_mask
            backbone_attention_mask = backbone_output.backbone_attention_mask
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )
        else:
            model_output, _ = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=vl_attn_mask,
                timestep=t_discretized,
                return_all_hidden_states=True,
            )

        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = action_loss.sum() / (action_mask.sum() + 1e-6)

        return {
            "loss": loss,
            "action_loss": action_loss,
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }

    def _encode_features(
        self, backbone_output: BatchFeature, action_input: BatchFeature
    ) -> BatchFeature:
        """
        Encode features for the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_history_length, max_state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - state_features: [B, 1, input_embedding_dim]
        """
        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Handle state history: if we have fewer timesteps than expected, repeat to fill
        state = action_input.state
        current_T = state.shape[1]
        assert current_T == self.config.state_history_length, "current_T != state_history_length"
        # Reshape state from [B, state_history_length, max_state_dim] to [B, 1, state_history_length * max_state_dim]
        state = state.view(state.shape[0], 1, -1)

        # Embed state.
        state_features = self.state_encoder(state, embodiment_id)

        return BatchFeature(data={"backbone_features": vl_embeds, "state_features": state_features})

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, state_horizon, input_embedding_dim]
            embodiment_id: [B] (embodiment IDs)
            backbone_output: Output from the backbone model
        """
        vl_embeds = backbone_features

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        dt = 1.0 / self.num_inference_timesteps
        vel_strength = torch.ones_like(actions)

        # --- RECAP inference-time advantage conditioning (RECAP App. E) ---
        # Condition on "Advantage: positive" (beta=1 in Eq. 2). If
        # cfg_guidance_weight != 1, additionally run an unconditional pass each
        # denoising step and combine: v_guided = v_null + w * (v_pos - v_null).
        vl_attn_mask_dit = backbone_output.backbone_attention_mask
        image_mask_dit = backbone_output.image_mask if self.config.use_alternate_vl_dit else None
        vl_embeds_uncond = vl_attn_mask_uncond = image_mask_uncond = None
        cfg_guidance_weight = (
            options.get("cfg_guidance_weight", self.config.cfg_guidance_weight)
            if options
            else self.config.cfg_guidance_weight
        )
        use_recap_cfg = self.recap_enabled and cfg_guidance_weight != 1.0
        if self.recap_enabled:
            pos_labels = torch.full(
                (batch_size,), AdvantageEmbedding.POS_IDX, dtype=torch.long, device=device
            )
            vl_embeds, vl_attn_mask_dit, image_mask_dit = self._apply_advantage_conditioning(
                vl_embeds, vl_attn_mask_dit, image_mask_dit, advantage_label=pos_labels
            )
            if use_recap_cfg:
                vl_embeds_uncond, vl_attn_mask_uncond, image_mask_uncond = (
                    self._apply_advantage_conditioning(
                        backbone_features,
                        backbone_output.backbone_attention_mask,
                        backbone_output.image_mask if self.config.use_alternate_vl_dit else None,
                        advantage_label=None,
                    )
                )

        if "action" in action_input:
            # If action in input when doing get action, it means we want to use RTC.
            # action_horizon is the action horizon of the input action.
            # rtc_overlap_steps is the number of steps to overlap with the previous action chunks.
            # rtc_frozen_steps is the number of steps to freeze the action, which is the latency of the policy inference.
            # rtc_ramp_rate is the rate of the ramp of denoising the actions.
            assert options is not None, "options is not None"
            assert "action_horizon" in options, "action_horizon is not in options"
            assert "rtc_overlap_steps" in options, "rtc_overlap_steps is not in options"
            assert "rtc_frozen_steps" in options, "rtc_frozen_steps is not in options"
            assert "rtc_ramp_rate" in options, "rtc_ramp_rate is not in options"

            action_horizon_before_padding = options["action_horizon"]

            # Use previous action instead of pure noise to do inpainting
            actions[:, : options["rtc_overlap_steps"], :] = action_input["action"][
                :,
                action_horizon_before_padding
                - options["rtc_overlap_steps"] : action_horizon_before_padding,
                :,
            ]
            vel_strength[:, : options["rtc_frozen_steps"], :] = 0.0
            # NOTE: use an exponential ramp strength to set the remaining unfrozen rtc_steps
            intermediate_steps = options["rtc_overlap_steps"] - options["rtc_frozen_steps"]
            # Create exponential ramp from 0 to 1 over intermediate steps
            t = torch.linspace(0.0, 1.0, intermediate_steps + 2, device=device)
            ramp = 1 - torch.exp(-options["rtc_ramp_rate"] * t)
            ramp = ramp / ramp[-1].clamp_min(1e-8)  # normalize to [0,1]
            ramp = ramp[
                1:-1
            ]  # we will only take the middle part of the ramp, ignore the 0.0 and 1.0
            # Apply ramp to the intermediate steps [batch, intermediate_steps, action_dim]
            vel_strength[
                :,
                options["rtc_frozen_steps"] : options["rtc_overlap_steps"],
                :,
            ] = ramp[None, :, None].to(device)

        # Run denoising steps.
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            sa_embs = torch.cat((state_features, action_features), dim=1)

            def _run_dit(vl_e, mask_e, img_e):
                if self.config.use_alternate_vl_dit:
                    return self.model(
                        hidden_states=sa_embs,
                        encoder_hidden_states=vl_e,
                        timestep=timesteps_tensor,
                        image_mask=img_e,
                        backbone_attention_mask=mask_e,
                    )
                return self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_e,
                    timestep=timesteps_tensor,
                )

            model_output = _run_dit(vl_embeds, vl_attn_mask_dit, image_mask_dit)
            pred = self.action_decoder(model_output, embodiment_id)

            if use_recap_cfg:
                model_output_uncond = _run_dit(
                    vl_embeds_uncond, vl_attn_mask_uncond, image_mask_uncond
                )
                pred_uncond = self.action_decoder(model_output_uncond, embodiment_id)
                pred = pred_uncond + cfg_guidance_weight * (pred - pred_uncond)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity * vel_strength

        return BatchFeature(
            data={
                "action_pred": actions,
                "backbone_features": backbone_features,
                "state_features": state_features,
            }
        )

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - action_pred: [B, action_horizon, action_dim] predicted actions
        """
        features = self._encode_features(backbone_output, action_input)
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
            action_input=action_input,
            options=options,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare input batch for the action head."""
        return BatchFeature(data=batch)


def get_backbone_cls(config: Gr00tN1d7Config):
    if "nvidia/Cosmos-Reason2" in config.model_name or "Qwen/Qwen3-VL" in config.model_name:
        # We import here as Qwen3Backbone depends on newer transformers versions than the rest of the code.
        from gr00t.model.modules.qwen3_backbone import Qwen3Backbone

        return Qwen3Backbone
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")


class Gr00tN1d7(PreTrainedModel):
    """Gr00tN1d7: VLA model with Cosmos-Reason2-2B (Qwen3-VL) backbone."""

    config_class = Gr00tN1d7Config
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Gr00tN1d7Config,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
    ):
        """
        Initialize Gr00tN1d7 model.

        Args:
            config: Model configuration
            transformers_loading_kwargs: Dict with transformers loading parameters:
                - transformers_trust_remote_code: Whether to trust remote code when loading from HF Hub
                - transformers_local_files_only: Whether to only use local files
                - model_revision: Specific model revision to use
                - transformers_cache_dir: Directory to cache downloaded models
                - transformers_access_token: HuggingFace access token for gated models

        Note: During training, transformers parameters are passed from training config.
              During inference (e.g., from_pretrained), defaults are used.
        """
        super().__init__(config)
        self.config = config

        backbone_cls = get_backbone_cls(config)
        self.backbone = backbone_cls(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

        # Initialize action head
        self.action_head = Gr00tN1d7ActionHead(config)
        from .processing_gr00t_n1d7 import Gr00tN1d7DataCollator

        self.collator = Gr00tN1d7DataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:
        """Prepare inputs for backbone and action head."""

        # NOTE -- currently the eval code doesn't use collator, so we need to add it here
        # this should ideally be fixed upstream
        if "vlm_content" in inputs:
            # Fix for n_envs > 1: Process all environments' VLM content, not just the first
            vlm_content_list = inputs["vlm_content"]
            # Ensure vlm_content_list is always a list for consistent processing
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]

            # Process all VLM contents through the collator
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        # Move to device and dtype
        def to_device_with_dtype(x):
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.dtype)
            else:
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_dtype, action_inputs)

        return backbone_inputs, action_inputs

    def forward(self, inputs: dict) -> BatchFeature:
        """
        Forward pass through the complete model.

        Args:
            inputs: Dictionary containing:
                - Action inputs (state, action, embodiment_id, etc.)

        Returns:
            BatchFeature containing loss and other outputs
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head(backbone_outputs, action_inputs)

        return action_outputs

    def get_action(self, inputs: dict, options: dict[str, Any] | None = None) -> BatchFeature:
        """
        Generate actions using the complete model.
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # Forward through backbone
        backbone_outputs = self.backbone(backbone_inputs)
        action_outputs = self.action_head.get_action(backbone_outputs, action_inputs, options)

        return action_outputs

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# Register the model with HuggingFace
AutoConfig.register("Gr00tN1d7", Gr00tN1d7Config)
AutoModel.register(Gr00tN1d7Config, Gr00tN1d7)
