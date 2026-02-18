#!/usr/bin/env python3
"""Convert SigLIP weights from a pi0 JAX checkpoint to HuggingFace PaliGemma format.

The pi0 model stores SigLIP (So400m/14) encoder weights in a stacked JAX/Flax
format where all 27 transformer layers are packed along dimension 0.  HuggingFace
PaliGemma stores them as individual per-layer parameters.

This script:
  1. Loads the JAX checkpoint params (via openpi helpers).
  2. Loads the matching HuggingFace PaliGemma model (paligemma2-3b-mix-224).
  3. Copies the 21 transferable SigLIP params (everything except the head/projector)
     into the HF state dict, performing the necessary reshapes and transposes.
  4. Returns or saves the patched HF model.

Usage:
    python scripts/convert_checkpoint_to_hf.py \
        --checkpoint checkpoints/pi0_b1k_turning_on_radio/49999_radio/params \
        [--output /tmp/paligemma_finetuned]

If --output is given, saves the patched model there.  Otherwise, just prints
a summary and validates shapes.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HuggingFace base model to use (must match the image resolution of pi0, 224x224)
# ---------------------------------------------------------------------------
HF_MODEL_ID = "google/paligemma2-3b-mix-224"
NUM_LAYERS = 27
NUM_HEADS = 16
HEAD_DIM = 72  # 1152 / 16

# ---------------------------------------------------------------------------
# JAX -> HF key mapping helpers
# ---------------------------------------------------------------------------

_VT = "model.vision_tower.vision_model"


def _jax_flat(params: dict) -> dict:
    """Flatten nested params dict with '/' separator, keep only img/ keys."""
    import flax.traverse_util
    flat = flax.traverse_util.flatten_dict(params, sep="/")
    return {k: v for k, v in flat.items() if k.startswith("PaliGemma/img/")}


def convert_siglip_weights(jax_params: dict) -> dict[str, torch.Tensor]:
    """Build a dict of {hf_key: tensor} from flat JAX SigLIP params.

    Returns only the SigLIP encoder params (not the head/projector, which has
    a different output dimension in pi0 vs HF PaliGemma).
    """
    flat = _jax_flat(jax_params)
    out: dict[str, torch.Tensor] = {}

    def _t(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.array(arr))

    # --- Non-layer params ---------------------------------------------------

    # Patch embedding: JAX (14,14,3,1152) -> HF (1152,3,14,14)
    k = "PaliGemma/img/embedding/kernel"
    out[f"{_VT}.embeddings.patch_embedding.weight"] = _t(flat[k]).permute(3, 2, 0, 1)

    # Patch embedding bias: direct copy (1152,)
    k = "PaliGemma/img/embedding/bias"
    out[f"{_VT}.embeddings.patch_embedding.bias"] = _t(flat[k])

    # Position embedding: JAX (1,256,1152) -> HF (256,1152)
    k = "PaliGemma/img/pos_embedding"
    out[f"{_VT}.embeddings.position_embedding.weight"] = _t(flat[k]).squeeze(0)

    # Post-layernorm (encoder_norm): scale -> weight, bias -> bias
    k_s = "PaliGemma/img/Transformer/encoder_norm/scale"
    k_b = "PaliGemma/img/Transformer/encoder_norm/bias"
    out[f"{_VT}.post_layernorm.weight"] = _t(flat[k_s])
    out[f"{_VT}.post_layernorm.bias"] = _t(flat[k_b])

    # --- Per-layer params (27 layers, stacked on dim 0) ---------------------
    prefix = "PaliGemma/img/Transformer/encoderblock"
    for i in range(NUM_LAYERS):
        lp = f"{_VT}.encoder.layers.{i}"

        # LayerNorm 0 -> layer_norm1
        out[f"{lp}.layer_norm1.weight"] = _t(flat[f"{prefix}/LayerNorm_0/scale"][i])
        out[f"{lp}.layer_norm1.bias"] = _t(flat[f"{prefix}/LayerNorm_0/bias"][i])

        # LayerNorm 1 -> layer_norm2
        out[f"{lp}.layer_norm2.weight"] = _t(flat[f"{prefix}/LayerNorm_1/scale"][i])
        out[f"{lp}.layer_norm2.bias"] = _t(flat[f"{prefix}/LayerNorm_1/bias"][i])

        # MLP fc1: kernel JAX (1152,4304) -> HF (4304,1152)  [transpose]
        out[f"{lp}.mlp.fc1.weight"] = _t(flat[f"{prefix}/MlpBlock_0/Dense_0/kernel"][i]).T
        out[f"{lp}.mlp.fc1.bias"] = _t(flat[f"{prefix}/MlpBlock_0/Dense_0/bias"][i])

        # MLP fc2: kernel JAX (4304,1152) -> HF (1152,4304)  [transpose]
        out[f"{lp}.mlp.fc2.weight"] = _t(flat[f"{prefix}/MlpBlock_0/Dense_1/kernel"][i]).T
        out[f"{lp}.mlp.fc2.bias"] = _t(flat[f"{prefix}/MlpBlock_0/Dense_1/bias"][i])

        # Attention projections
        attn = f"{prefix}/MultiHeadDotProductAttention_0"
        for proj, hf_name in [("query", "q_proj"), ("key", "k_proj"), ("value", "v_proj")]:
            # kernel: JAX (1152, 16, 72) -> reshape (1152, 1152) -> transpose -> HF (1152, 1152)
            w = flat[f"{attn}/{proj}/kernel"][i]  # (1152, 16, 72)
            w = w.reshape(w.shape[0], NUM_HEADS * HEAD_DIM)  # (1152, 1152)
            out[f"{lp}.self_attn.{hf_name}.weight"] = _t(w).T  # (1152, 1152)

            # bias: JAX (16, 72) -> reshape (1152,)
            b = flat[f"{attn}/{proj}/bias"][i]  # (16, 72)
            out[f"{lp}.self_attn.{hf_name}.bias"] = _t(b.reshape(-1))  # (1152,)

        # Output projection
        # kernel: JAX (16, 72, 1152) -> reshape (1152, 1152) -> transpose -> HF (1152, 1152)
        w = flat[f"{attn}/out/kernel"][i]  # (16, 72, 1152)
        w = w.reshape(NUM_HEADS * HEAD_DIM, w.shape[-1])  # (1152, 1152)
        out[f"{lp}.self_attn.out_proj.weight"] = _t(w).T  # (1152, 1152)

        # Output projection bias: direct copy (1152,)
        out[f"{lp}.self_attn.out_proj.bias"] = _t(flat[f"{attn}/out/bias"][i])

    return out


def load_patched_model(checkpoint_path: str, device: str = "cuda"):
    """Load HF PaliGemma, patch SigLIP with JAX checkpoint weights, return model."""
    import openpi.models.model as _model
    from transformers import PaliGemmaForConditionalGeneration

    log.info("Loading JAX checkpoint from %s ...", checkpoint_path)
    params = _model.restore_params(checkpoint_path, restore_type=np.ndarray)

    log.info("Converting SigLIP weights ...")
    converted = convert_siglip_weights(params)
    log.info("Converted %d HF params from JAX SigLIP", len(converted))

    log.info("Loading HF model %s ...", HF_MODEL_ID)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        HF_MODEL_ID, torch_dtype=torch.bfloat16
    )

    # Patch the state dict
    sd = model.state_dict()
    patched = 0
    mismatched = []
    for hf_key, tensor in converted.items():
        if hf_key not in sd:
            log.warning("Key %s not found in HF model!", hf_key)
            continue
        if sd[hf_key].shape != tensor.shape:
            mismatched.append((hf_key, sd[hf_key].shape, tensor.shape))
            continue
        sd[hf_key] = tensor.to(sd[hf_key].dtype)
        patched += 1

    if mismatched:
        for k, expected, got in mismatched:
            log.error("Shape mismatch: %s  expected %s  got %s", k, expected, got)
        raise ValueError(f"{len(mismatched)} shape mismatches!")

    model.load_state_dict(sd)
    log.info("Patched %d / %d converted params into HF model", patched, len(converted))

    model = model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True,
                        help="Path to JAX checkpoint params directory")
    parser.add_argument("--output", default=None,
                        help="Optional: save patched HF model to this directory")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate shape mapping, don't load full model")
    args = parser.parse_args()

    if args.validate_only:
        import openpi.models.model as _model
        log.info("Loading JAX checkpoint from %s ...", args.checkpoint)
        params = _model.restore_params(args.checkpoint, restore_type=np.ndarray)
        converted = convert_siglip_weights(params)
        log.info("Converted %d HF params", len(converted))
        for k, v in sorted(converted.items()):
            log.info("  %s: %s", k, tuple(v.shape))
        log.info("Validation OK")
        return

    model = load_patched_model(args.checkpoint, device="cpu")

    if args.output:
        out_path = Path(args.output)
        out_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_path)
        log.info("Saved patched model to %s", out_path)
        # Also save processor for convenience
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(HF_MODEL_ID)
        processor.save_pretrained(out_path)
        log.info("Saved processor to %s", out_path)

    log.info("Done!")


if __name__ == "__main__":
    main()
