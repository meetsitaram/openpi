# Commands Reference

All commands for serving, evaluating, training, and auditing the pi0.5 model
on the BEHAVIOR-1K `turning_on_radio` task.

---

## Environments

| Env | Purpose | Activation |
|-----|---------|------------|
| **openpi venv** | Training, serving policy, audits | `cd b1k-baselines/baselines/openpi && source .venv/bin/activate` |
| **env_isaaclab** (conda) | Simulation eval (Isaac Sim 5.1, RTX 5090 compatible) | `conda activate env_isaaclab` |

> **Do NOT use `behavior` conda env for eval** — it has Isaac Sim 4.5 which
> renders broken/noisy images on RTX 5090 (Blackwell sm_120).

---

## 1. Policy Server (Terminal 1)

Serves a trained checkpoint via websocket for the simulation eval client.

```bash
cd /home/stickbot/projects/behavior/b1k-baselines/baselines/openpi
source .venv/bin/activate

# XLA_PYTHON_CLIENT_MEM_FRACTION limits JAX GPU memory so Isaac Sim can render.
# 0.65 = ~21GB for JAX, leaving ~11GB for Isaac Sim. Adjust if needed.

# Grounding v2 full task (all 4 phases, base unlocked)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.65 python scripts/serve_b1k.py \
  --phase-conditioning \
  policy:checkpoint \
  --policy.config pi05_b1k_phase_grounding \
  --policy.dir outputs/checkpoints/curriculum_stage2_full_task/grounding_v2_full_stage2_full_task/19999

# Grounding v1 stage 1 (nav + pickup)
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.65 python scripts/serve_b1k.py \
#   --phase-conditioning \
#   policy:checkpoint \
#   --policy.config pi05_b1k_phase_grounding \
#   --policy.dir outputs/checkpoints/curriculum_stage1_nav_pickup/grounding_v1_stage1_nav_pickup/14999

# Non-grounding phase checkpoint (e.g., pi05_b1k_phase)
# python scripts/serve_b1k.py \
#   --phase-conditioning \
#   policy:checkpoint \
#   --policy.config pi05_b1k_phase \
#   --policy.dir outputs/checkpoints/pi05_b1k_phase/pi05_phase_v2/40000
```

Wait until you see `Serving on 0.0.0.0:8000` before starting eval.

---

## 2. Simulation Eval (Terminal 2)

Connects to the policy server and runs rollouts in OmniGibson/Isaac Sim.

```bash
conda activate env_isaaclab
export OMNI_KIT_ACCEPT_EULA=YES
cd /home/stickbot/projects/behavior/BEHAVIOR-1K

# Grounding v2 full task eval
python OmniGibson/omnigibson/learning/eval.py \
  policy=websocket \
  task.name=turning_on_radio \
  headless=false \
  log_path=/home/stickbot/projects/behavior/eval_logs/grounding_v2_full \
  eval_on_train_instances=true \
  eval_instance_ids=[1]
```

Key options:
- `eval_instance_ids=[1]` — specific training instance (change number as needed)
- `eval_on_train_instances=true` — use training instances (omit for test instances)
- `headless=true` — run without GUI (faster, for batch eval)
- `write_video=true` — record rollout video (default: true)

---

## 3. Curriculum Training

```bash
cd /home/stickbot/projects/behavior/b1k-baselines/baselines/openpi
source .venv/bin/activate

# Full curriculum (stage 0 + 1)
python -u scripts/train_curriculum.py \
  --run_name grounding_v1 \
  --start_stage 0 \
  --end_stage 1 \
  2>&1 | tee outputs/grounding_v1_train.log

# Resume from stage 1 only (using stage 0 checkpoint)
python -u scripts/train_curriculum.py \
  --run_name grounding_v1 \
  --start_stage 1 \
  --end_stage 1 \
  --prev_checkpoint outputs/checkpoints/curriculum_stage0_nav/grounding_v1_stage0_nav/14999/params \
  2>&1 | tee outputs/grounding_v1_stage1_train.log

# Stage 2: full task (all 4 phases, base unlocked during manipulation)
python -u scripts/train_curriculum.py \
  --run_name grounding_v2_full \
  --start_stage 2 \
  --end_stage 2 \
  --prev_checkpoint outputs/checkpoints/curriculum_stage1_nav_pickup/grounding_v1_stage1_nav_pickup/14999/params \
  --base_checkpoint pi0_b1k_turning_on_radio/49999_radio \
  2>&1 | tee outputs/grounding_v2_full_stage2_train.log

# Stage 3: grasp-focused (300fr before → 100fr after R_close, 18.6% of data)
python -u scripts/train_curriculum.py \
  --run_name grounding_v2_full \
  --start_stage 3 \
  --end_stage 3 \
  --prev_checkpoint outputs/checkpoints/curriculum_stage2_full_task/grounding_v2_full_stage2_full_task/19999/params \
  2>&1 | tee outputs/grounding_v2_full_stage3_grasp_train.log
```

---

## 4. VLM Audit (text-generation detection)

Tests the VLM's ability to detect objects via text generation. Uses GT overlay
for comparison. Reports per-object detection rates and GT-match percentages.

```bash
cd /home/stickbot/projects/behavior/b1k-baselines/baselines/openpi
source .venv/bin/activate

# Off-the-shelf baseline (224px)
python scripts/audit_vlm_grounding.py \
  --episode 2790 \
  --model google/paligemma2-3b-mix-224 \
  --sample_every 5

# Fine-tuned checkpoint (patches SigLIP into off-the-shelf PaliGemma)
python scripts/audit_vlm_grounding.py \
  --episode 2790 \
  --checkpoint outputs/checkpoints/curriculum_stage0_nav/grounding_v1_stage0_nav/14999/params \
  --sample_every 5
```

Output: `outputs/vlm_audit/audit_head_episode_XXXXXXXX_{baseline|finetuned}.mp4`

---

## 5. Grounding Probe Evaluation

Tests the grounding auxiliary head directly (SigLIP + MLP, no text generation).
Shows whether the visual backbone has learned to spatially localize objects.

```bash
cd /home/stickbot/projects/behavior/b1k-baselines/baselines/openpi
source .venv/bin/activate

python -u scripts/eval_grounding_probe.py \
  --episode 2790 \
  --checkpoint outputs/checkpoints/curriculum_stage0_nav/grounding_v1_stage0_nav/14999/params \
  --sample_every 5
```

Output: `outputs/grounding_probe/probe_head_episode_XXXXXXXX_NNNNN.mp4`

---

## Checkpoints Reference

| Checkpoint | Description |
|------------|-------------|
| `grounding_v1_stage0_nav/14999` | Stage 0 (nav only), 15k steps, grounding aux enabled |
| `grounding_v1_stage1_nav_pickup/14999` | Stage 1 (nav+pickup), 15k steps |
| `grounding_v2_full_stage2_full_task/19999` | Stage 2 (all 4 phases), 20k steps, base unlocked |
| `grounding_v2_full_stage3_grasp/19999` | Stage 3 (grasp-focused), 20k steps, 300fr before→100fr after R_close |
| `pi05_phase_v2/40000` | Phase-aware pi0.5, 40k steps, no grounding |
| `pi0_b1k_turning_on_radio/49999_radio` | Original pi0 baseline, 50k steps |

## Config Names (for serve_b1k.py --policy.config)

| Config | Description |
|--------|-------------|
| `pi05_b1k_phase_grounding` | Phase + grounding aux heads (for grounding_v1 checkpoints) |
| `pi05_b1k_phase` | Phase aux heads only (no grounding) |
| `pi0_b1k` | Original pi0 config |
