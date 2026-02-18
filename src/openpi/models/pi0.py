import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import optax
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


class PhaseAuxHead(nnx.Module):
    """Lightweight 2-layer MLP auxiliary head for phase/skill classification.

    Attached to the PaliGemma prefix output (VLM features) to provide
    auxiliary supervision that encourages phase-discriminative representations.

    Architecture:
        mean_pool(prefix_out, mask) → Linear(in_dim, hidden) → SiLU → Linear(hidden, num_classes)
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, num_classes, rngs=rngs)

    def __call__(
        self,
        prefix_out: at.Float[at.Array, "b s emb"],
        prefix_mask: at.Bool[at.Array, "b s"],
    ) -> at.Float[at.Array, "b num_classes"]:
        """Forward pass: mean-pool over valid tokens, then 2-layer MLP.

        Args:
            prefix_out: PaliGemma output features [batch, seq_len, embed_dim].
            prefix_mask: Boolean mask [batch, seq_len] — True for valid tokens.

        Returns:
            Logits [batch, num_classes].
        """
        # Masked mean pooling: average only over valid (non-padding) tokens
        mask_expanded = prefix_mask[:, :, None].astype(prefix_out.dtype)  # [b, s, 1]
        pooled = jnp.sum(prefix_out * mask_expanded, axis=1)  # [b, emb]
        denom = jnp.maximum(jnp.sum(mask_expanded, axis=1), 1.0)  # [b, 1]
        pooled = pooled / denom  # [b, emb]

        # 2-layer MLP
        h = self.fc1(pooled)
        h = nnx.swish(h)
        logits = self.fc2(h)
        return logits


class GroundingAuxHead(nnx.Module):
    """Predicts 2D bounding boxes (cx, cy, w, h) + visibility from pre-LLM image tokens.

    Uses spatial softmax for center (cx, cy) prediction — each object gets a
    learned heatmap over the 16x16 patch grid, and the expected position is
    computed as the weighted sum over a fixed coordinate grid. This preserves
    spatial information that mean-pooling destroys.

    Width/height and visibility are predicted from mean-pooled features via
    small MLPs.

    Architecture:
        tokens [b, 256, emb]
        ├─ heatmap_proj(emb → num_obj) → spatial_softmax → (cx, cy)
        └─ mean_pool → size_fc1 → SiLU → size_fc2 → sigmoid → (w, h)
                      └─ vis_fc → logit → (visibility score per object)
        bbox output: [b, num_obj * 4] = (cx, cy, w, h) per object
        vis output:  [b, num_obj] = visibility logits (pre-sigmoid)
    """

    GRID_SIZE: int = 16  # 224px / 14px per patch

    def __init__(self, in_dim: int, hidden_dim: int, num_objects: int, rngs: nnx.Rngs):
        self.num_objects = num_objects
        self.heatmap_proj = nnx.Linear(in_dim, num_objects, rngs=rngs)
        self.size_fc1 = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.size_fc2 = nnx.Linear(hidden_dim, num_objects * 2, rngs=rngs)
        self.vis_fc = nnx.Linear(in_dim, num_objects, rngs=rngs)

    def __call__(
        self,
        image_tokens: at.Float[at.Array, "b s emb"],
    ) -> tuple[at.Float[at.Array, "b bbox"], at.Float[at.Array, "b vis"]]:
        """Forward pass: spatial softmax + size MLP + visibility classifier.

        Returns:
            bbox_pred: (cx, cy, w, h) per object [batch, num_objects * 4], in [0, 1].
            vis_logits: visibility logits per object [batch, num_objects] (pre-sigmoid).
        """
        b, s, d = image_tokens.shape
        g = self.GRID_SIZE

        # Spatial softmax for center prediction
        logits = self.heatmap_proj(image_tokens)  # [b, 256, num_obj]
        logits_flat = logits.reshape(b, g * g, self.num_objects)  # [b, 256, num_obj]
        weights = jax.nn.softmax(logits_flat, axis=1)  # [b, 256, num_obj]

        # Fixed coordinate grid: normalized positions [0, 1] at patch centers
        coords_y, coords_x = jnp.meshgrid(
            (jnp.arange(g) + 0.5) / g,
            (jnp.arange(g) + 0.5) / g,
            indexing="ij",
        )
        grid_x = coords_x.reshape(g * g)  # [256]
        grid_y = coords_y.reshape(g * g)  # [256]

        # Expected position: weighted sum over grid
        cx = jnp.sum(weights * grid_x[None, :, None], axis=1)  # [b, num_obj]
        cy = jnp.sum(weights * grid_y[None, :, None], axis=1)  # [b, num_obj]

        # Size + visibility from mean-pooled features
        pooled = jnp.mean(image_tokens, axis=1)  # [b, emb]
        size_h = nnx.swish(self.size_fc1(pooled))
        size_out = jax.nn.sigmoid(self.size_fc2(size_h))  # [b, num_obj * 2]
        size_out = size_out.reshape(b, self.num_objects, 2)  # [b, num_obj, 2]

        vis_logits = self.vis_fc(pooled)  # [b, num_obj] — raw logits

        # Interleave: [cx0, cy0, w0, h0, cx1, cy1, w1, h1, ...]
        bbox = jnp.stack([cx, cy, size_out[:, :, 0], size_out[:, :, 1]], axis=-1)
        return bbox.reshape(b, self.num_objects * 4), vis_logits


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # Auxiliary prediction heads (phase/skill classification + grounding).
        self.aux_head_config = config.aux_head
        self.aux_skill_type_head = None
        self.aux_phase_index_head = None
        self.aux_grounding_head = None
        if config.aux_head.enabled:
            vlm_width = paligemma_config.width  # 2048 for gemma_2b
            if config.aux_head.num_skill_type_classes > 0:
                self.aux_skill_type_head = PhaseAuxHead(
                    in_dim=vlm_width,
                    hidden_dim=config.aux_head.hidden_dim,
                    num_classes=config.aux_head.num_skill_type_classes,
                    rngs=rngs,
                )
            if config.aux_head.num_phase_index_classes > 0:
                self.aux_phase_index_head = PhaseAuxHead(
                    in_dim=vlm_width,
                    hidden_dim=config.aux_head.hidden_dim,
                    num_classes=config.aux_head.num_phase_index_classes,
                    rngs=rngs,
                )
            if config.aux_head.grounding_enabled:
                self.aux_grounding_head = GroundingAuxHead(
                    in_dim=vlm_width,
                    hidden_dim=config.aux_head.grounding_hidden_dim,
                    num_objects=config.aux_head.grounding_num_objects,
                    rngs=rngs,
                )

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"], int]:
        input_mask = []
        ar_mask = []
        tokens = []
        num_image_tokens = 0
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]
            num_image_tokens += image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, num_image_tokens

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"] | dict:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask, num_img_tokens = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        action_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)

        # --- Auxiliary head losses ---
        if not self.aux_head_config.enabled or observation.aux_labels is None:
            return action_loss

        # ANTI-CHEAT: Build an image-only mask so aux heads can only pool
        # over image token activations, NOT text tokens that contain the
        # phase label injected by PhaseConditionedPrompt.
        # Also force ALL image tokens to True regardless of camera masking
        # (PhaseAwareCameraMasking sets wrist cams to False during nav,
        # which leaks skill_type through the mask pattern).
        # prefix layout: [img_cam1 | img_cam2 | img_cam3 | text_tokens]
        batch_size = prefix_mask.shape[0]
        img_only_mask = jnp.concatenate([
            jnp.ones((batch_size, num_img_tokens), dtype=jnp.bool_),
            jnp.zeros((batch_size, prefix_mask.shape[1] - num_img_tokens), dtype=jnp.bool_),
        ], axis=1)

        aux_losses = {}
        aux_metrics = {}

        if self.aux_skill_type_head is not None and "skill_type" in observation.aux_labels:
            skill_type_logits = self.aux_skill_type_head(prefix_out, img_only_mask)
            skill_type_labels = observation.aux_labels["skill_type"]  # [b]
            skill_type_ce = jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(skill_type_logits, skill_type_labels)
            )
            skill_type_acc = jnp.mean(jnp.argmax(skill_type_logits, axis=-1) == skill_type_labels)
            aux_losses["aux_skill_type_loss"] = self.aux_head_config.skill_type_loss_weight * skill_type_ce
            aux_metrics["aux_skill_type_acc"] = skill_type_acc

        if self.aux_phase_index_head is not None and "phase_index" in observation.aux_labels:
            phase_index_logits = self.aux_phase_index_head(prefix_out, img_only_mask)
            phase_index_labels = observation.aux_labels["phase_index"]  # [b]
            phase_index_ce = jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(phase_index_logits, phase_index_labels)
            )
            phase_index_acc = jnp.mean(jnp.argmax(phase_index_logits, axis=-1) == phase_index_labels)
            aux_losses["aux_phase_index_loss"] = self.aux_head_config.phase_index_loss_weight * phase_index_ce
            aux_metrics["aux_phase_index_acc"] = phase_index_acc

        if (self.aux_grounding_head is not None
                and "grounding_xywh" in observation.aux_labels
                and "grounding_visible" in observation.aux_labels):
            # Use PRE-LLM head-camera tokens for short gradient path to SigLIP.
            # prefix_tokens layout: [head_cam(256) | left_wrist(256) | right_wrist(256) | text]
            num_head_tokens = 256
            head_cam_tokens = prefix_tokens[:, :num_head_tokens, :]  # [b, 256, 2048]

            bbox_pred, vis_logits = self.aux_grounding_head(head_cam_tokens)
            grounding_gt = observation.aux_labels["grounding_xywh"]  # [b, num_obj*4]
            grounding_vis = observation.aux_labels["grounding_visible"]  # [b, num_obj]

            # --- BBox regression loss (SmoothL1, masked by visibility) ---
            num_obj = self.aux_head_config.grounding_num_objects
            diff = bbox_pred - grounding_gt
            vis_mask = jnp.repeat(grounding_vis, 4, axis=-1)  # [b, num_obj*4]

            abs_diff = jnp.abs(diff)
            smooth_l1 = jnp.where(abs_diff < 1.0, 0.5 * diff ** 2, abs_diff - 0.5)
            masked_loss = smooth_l1 * vis_mask
            denom = jnp.maximum(jnp.sum(vis_mask), 1.0)
            bbox_loss = jnp.sum(masked_loss) / denom

            # --- Visibility classification loss (BCE on ALL objects) ---
            vis_bce = optax.sigmoid_binary_cross_entropy(vis_logits, grounding_vis)
            vis_loss = jnp.mean(vis_bce)

            grounding_loss = bbox_loss + vis_loss
            aux_losses["aux_grounding_loss"] = (
                self.aux_head_config.grounding_loss_weight * grounding_loss
            )
            vis_preds = jax.nn.sigmoid(vis_logits) > 0.5
            aux_metrics["aux_grounding_l1"] = bbox_loss
            aux_metrics["aux_grounding_vis_loss"] = vis_loss
            aux_metrics["aux_grounding_vis_acc"] = jnp.mean(
                vis_preds == (grounding_vis > 0.5)
            )
            aux_metrics["aux_grounding_vis_ratio"] = jnp.mean(grounding_vis)

        return {
            "action_loss": action_loss,
            **aux_losses,
            **aux_metrics,
        }

    def predict_phase(self, observation: _model.Observation) -> dict[str, at.Array]:
        """Run auxiliary heads to predict phase/skill type from the current observation.

        This is a separate method (not inside sample_actions) because
        sample_actions is JIT-compiled and can't have Python-level side effects.

        Runs embed_prefix → PaliGemma LLM forward → aux head MLPs.
        Cost: ~10% of a full sample_actions call (one prefix pass vs 10 denoising steps).

        Args:
            observation: Preprocessed observation (same as passed to sample_actions).

        Returns:
            Dict with predicted class indices and logits:
                "skill_type": int[b] — predicted skill type (0=nav, 1=uncoord, 2=coord)
                "skill_type_logits": float[b, num_classes]
                "phase_index": int[b] — predicted phase index (0-3)
                "phase_index_logits": float[b, num_classes]
            Empty dict if aux heads are not enabled.
        """
        if not self.aux_head_config.enabled:
            return {}

        observation = _model.preprocess_observation(None, observation, train=False)

        # Compute VLM features (same prefix computation as sample_actions)
        prefix_tokens, prefix_mask, prefix_ar_mask, num_img_tokens = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), _ = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )

        # ANTI-CHEAT: pool only over image tokens (all cameras forced active),
        # not text tokens — prevents cheating via prompt text or camera mask patterns.
        batch_size = prefix_mask.shape[0]
        img_only_mask = jnp.concatenate([
            jnp.ones((batch_size, num_img_tokens), dtype=jnp.bool_),
            jnp.zeros((batch_size, prefix_mask.shape[1] - num_img_tokens), dtype=jnp.bool_),
        ], axis=1)

        results = {}

        if self.aux_skill_type_head is not None and prefix_out is not None:
            logits = self.aux_skill_type_head(prefix_out, img_only_mask)
            results["skill_type_logits"] = logits
            results["skill_type"] = jnp.argmax(logits, axis=-1)

        if self.aux_phase_index_head is not None and prefix_out is not None:
            logits = self.aux_phase_index_head(prefix_out, img_only_mask)
            results["phase_index_logits"] = logits
            results["phase_index"] = jnp.argmax(logits, axis=-1)

        return results

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask, _num_img = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
