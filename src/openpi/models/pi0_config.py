import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class AuxHeadConfig:
    """Configuration for auxiliary prediction heads."""

    # Number of skill-type classes (0 = disabled). e.g. 3 for navigation/uncoordinated/coordinated.
    num_skill_type_classes: int = 0
    # Number of phase-index classes (0 = disabled). e.g. 4 for move_to/pick_up/press/place_on.
    num_phase_index_classes: int = 0
    # Hidden dimension of the 2-layer MLP bottleneck.
    hidden_dim: int = 256
    # Loss weight for skill-type classification.
    skill_type_loss_weight: float = 0.1
    # Loss weight for phase-index classification.
    phase_index_loss_weight: float = 0.05

    # --- Visual grounding auxiliary ---
    # Enable grounding head that predicts object 2D positions from SigLIP tokens.
    grounding_enabled: bool = False
    # Number of objects to predict positions for (radio + table = 2).
    grounding_num_objects: int = 2
    # Hidden dimension for grounding MLP.
    grounding_hidden_dim: int = 512
    # Loss weight for grounding regression (SmoothL1).
    grounding_loss_weight: float = 0.05
    # Freeze the grounding head parameters (use after stage 0 to prevent drift).
    grounding_freeze: bool = False

    @property
    def enabled(self) -> bool:
        return (self.num_skill_type_classes > 0
                or self.num_phase_index_classes > 0
                or self.grounding_enabled)


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # Auxiliary prediction head configuration. When enabled, an MLP head is
    # attached to the PaliGemma prefix output to predict phase/skill labels.
    aux_head: AuxHeadConfig = dataclasses.field(default_factory=AuxHeadConfig)

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config.

        The returned filter matches parameters that should be *frozen*.
        TrainConfig.trainable_filter inverts this via nnx.Not().

        When aux_head.grounding_freeze is True, the grounding head parameters
        are added to the freeze set (via nnx.Any) so they are preserved from
        a previous training stage.
        """
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )

        lora_freeze = nnx.All(*filters) if filters else nnx.Nothing

        # Optionally freeze the grounding head to preserve learned detections.
        if self.aux_head.grounding_freeze:
            grounding_freeze = nnx_utils.PathRegex(".*aux_grounding_head.*")
            return nnx.Any(lora_freeze, grounding_freeze)

        return lora_freeze
