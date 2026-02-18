# Grounding Auxiliary Head — Implementation Plan

> **Goal**: Add a visual grounding auxiliary head to pi0 that predicts object 2D
> positions from SigLIP image tokens, supervised by privileged 3D-to-2D projected
> coordinates from `task_info` / `cam_rel_poses`.

## Background

- The off-the-shelf PaliGemma VLM detects the radio in only **~2%** of frames.
- After 50k steps of pi0 fine-tuning (`pi0_b1k_turning_on_radio/49999_radio`),
  detection drops to **0%** — SigLIP learned nothing about the radio's identity.
- Root cause: SigLIP gradients from the action loss must travel through
  Projector → Gemma 2B LLM → Action Expert — too diluted to teach object detection.
- **Solution**: Add a grounding head that branches off SigLIP's output *before*
  the LLM, providing a short, direct gradient path.

## Architecture Diagram

```
                          ┌──────────────────┐
   RGB (224×224)  ──────▶ │   SigLIP So400m  │
                          │   27 layers       │
                          │   256 × 1152      │
                          └────────┬──────────┘
                                   │
                          ┌────────▼──────────┐
                          │  Projector 1152→2048│
                          └────────┬──────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
              ▼                    ▼                     ▼
   ┌─────────────────┐   ┌─────────────┐   ┌───────────────────┐
   │ ★ GROUNDING HEAD│   │ Text tokens │   │ Gemma 2B LLM      │
   │                 │   │  + concat   │──▶│  (LoRA)            │
   │ AvgPool 256→1   │   └─────────────┘   └────────┬──────────┘
   │ MLP 2048→512    │                               │
   │ MLP 512→128     │                      ┌────────▼──────────┐
   │ Linear 128→4    │                      │   Action Expert    │
   │ (radio_xy,      │                      │   Gemma 300M       │
   │  table_xy)      │                      └────────┬──────────┘
   └────────┬────────┘                               │
            │                                ┌───────▼──────────┐
            ▼                                │  L_action        │
   ┌─────────────────┐                       │  (flow-matching) │
   │  L_grounding     │                       └──────────────────┘
   │  SmoothL1 on     │
   │  predicted vs GT │
   │  (visibility-    │
   │   masked)        │
   └─────────────────┘

   Total Loss = L_action + λ · L_grounding    (λ = 0.05 default)
```

## Files to Modify

### 1. Data pipeline — get privileged data into `aux_labels`

| File | Change |
|------|--------|
| `src/openpi/policies/b1k_phase_transforms.py` | Add `ComputeGroundingLabels` transform: reads `task_info` + `cam_rel_poses`, projects 3D → 2D, stores `aux_labels["grounding_xy"]` (float32 shape `(4,)` = `[radio_x, radio_y, table_x, table_y]` normalized 0–1) and `aux_labels["grounding_visible"]` (float32 shape `(2,)` = `[radio_vis, table_vis]`) |
| `src/openpi/training/config.py` | In `LeRobotB1KPhaseDataConfig` and `LeRobotB1KCurriculumDataConfig`: add `observation/task_info` + `observation/cam_rel_poses` to `RepackTransform`; insert `ComputeGroundingLabels()` in the data_transforms pipeline before `EncodePhaseLabels` |
| `src/openpi/policies/b1k_policy.py` | In `B1kInputs.__call__`, pass through `observation/task_info` and `observation/cam_rel_poses` keys |
| `src/openpi/models/model.py` | Relax `aux_labels` type hint from `Int` to accept both int and float arrays |

### 2. Model — grounding head + loss

| File | Change |
|------|--------|
| `src/openpi/models/pi0_config.py` | Add to `AuxHeadConfig`: `grounding_enabled`, `grounding_loss_weight` (0.05), `grounding_num_objects` (2), `grounding_hidden_dim` (512) |
| `src/openpi/models/pi0.py` | Add `GroundingAuxHead(nnx.Module)`: pools 256 head-camera tokens (2048-dim), MLP → `num_objects * 2` outputs. In `Pi0.__init__`, instantiate if enabled. In `compute_loss`, extract pre-LLM head tokens (`prefix_tokens[:, :256, :]`), run head, compute SmoothL1 masked by visibility. |

## Key Design Choices

1. **Pre-LLM tokens**: Uses `prefix_tokens[:, :256, :]` from `embed_prefix`
   (after SigLIP projector, 2048-dim), NOT the LLM output. Gradient path:
   `L_grounding → MLP → projected_tokens → projector → SigLIP` — bypasses
   the full Gemma LLM.

2. **Head camera only**: First 256 tokens = head camera (`base_0_rgb`).
   Grounding GT is computed using head camera intrinsics.

3. **Lambda = 0.05**: Small weight to avoid overwhelming the action loss.
   Configurable via `AuxHeadConfig.grounding_loss_weight`.

4. **Visibility masking**: Objects behind the camera or outside the frame get
   zero loss weight via `grounding_visible` mask.

5. **3D → 2D projection**: Reuses the exact math from `audit_vlm_grounding.py`
   (world → robot base → camera → pinhole → normalized pixel coords).

## Projection Constants (from audit script)

```python
# task_info layout
AGENT_OFFSET = 0   # real(1) + pos(3) + ori_cos(3) + ori_sin(3) = 10
RADIO_OFFSET = 10  # real(1) + pos(3) + ori_cos(3) + ori_sin(3) + grip(2) = 12
TABLE_OFFSET = 22  # same layout as radio

# cam_rel_poses: head camera at index 14:21
CAM_HEAD_OFFSET = 14

# Camera intrinsics (R1Pro head camera, 720×720 native)
HEAD_K = [[306, 0, 360], [0, 306, 360], [0, 0, 1]]
HEAD_NATIVE_RES = (720, 720)
```

## Risk Mitigation

- **Lambda too high** → action loss spikes. Start at 0.05, monitor both losses.
- **Gradient magnitude imbalance** → grounding has short path, action has long path.
  The small lambda compensates.
- **Feature conflict** → grounding and action objectives are *aligned* (both need
  the robot to "see" the radio). Unlikely to hurt.
- **Inference overhead** → grounding head is discarded at inference. Zero cost.

## Validation

After training with grounding auxiliary:
1. Re-run `audit_vlm_grounding.py --checkpoint <new_ckpt>` to measure detection rate
2. Compare against 0% (fine-tuned without grounding) and 2% (off-the-shelf)
3. Evaluate action performance (success rate on turning_on_radio task)
