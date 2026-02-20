"""Curriculum training for BEHAVIOR-1K turning_on_radio task.

Runs multiple training stages back-to-back, automatically wiring
checkpoint paths between stages. No manual path management needed.

Usage:
    cd /home/stickbot/projects/behavior/b1k-baselines/baselines/openpi
    source .venv/bin/activate
    uv run scripts/train_curriculum.py [--run_name my_run] [--start_stage 0]

Stages (v4 — 2-stage curriculum):
    0: Navigation only         (skill_indices={0})           — 15k steps
       Grounding head trains alongside the policy to learn object detection.
    1: Full task               (skill_indices={0,1,2,3})     — 20k steps
       All video frames including navigation. Grounding head is FROZEN to
       preserve object detection learned in stage 0.
"""

import argparse
import dataclasses
import logging
import pathlib
import sys

# ── Curriculum stage definitions ─────────────────────────────────────────
# Each stage is a dict with the fields that differ per stage.
# Everything else (model config, aux heads, dataset root, etc.) is shared.

CURRICULUM_STAGES = [
    {
        "name": "stage0_nav",
        "skill_indices": (0,),
        "num_train_steps": 15_000,
        "early_transition_frames": 0,
        "freeze_grounding": False,
        "description": "Navigation only — grounding head trains",
    },
    {
        "name": "stage1_full",
        "skill_indices": (0, 1, 2, 3),
        "num_train_steps": 20_000,
        "early_transition_frames": 60,
        "freeze_grounding": True,
        "description": "Full task (all phases incl. nav) — grounding head frozen",
    },
]


def build_stage_config(
    stage: dict,
    run_name: str,
    prev_checkpoint: str | None,
    base_checkpoint: str,
    dataset_root: str,
    assets_dir: str,
    checkpoint_base_dir: str,
    batch_size: int,
):
    """Build a TrainConfig for one curriculum stage.

    Imports are deferred so this module can be inspected without loading JAX.
    """
    import openpi.models.pi0_config as pi0_config
    import openpi.training.config as _config
    import openpi.training.weight_loaders as weight_loaders

    model_config = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=50,
        paligemma_variant="gemma_2b_lora",
        aux_head=pi0_config.AuxHeadConfig(
            num_skill_type_classes=3,
            num_phase_index_classes=4,
            hidden_dim=256,
            skill_type_loss_weight=0.1,
            phase_index_loss_weight=0.05,
            grounding_enabled=True,
            grounding_loss_weight=0.05,
            grounding_num_objects=2,
            grounding_hidden_dim=512,
        ),
    )

    data_config = _config.LeRobotB1KCurriculumDataConfig(
        repo_id="behavior-1k/2025-challenge-demos",
        skill_indices=stage["skill_indices"],
        action_low_weight=0.1,
        base_nav_weight=2.0,
        torso_nav_weight=0.3,
        early_transition_frames=stage.get("early_transition_frames", 0),
        grasp_window=stage.get("grasp_window", None),
        assets=_config.AssetsConfig(assets_dir=assets_dir),
        base_config=_config.DataConfig(
            prompt_from_task=True,
            episodes_index=list(range(190)),
            behavior_dataset_root=dataset_root,
        ),
    )

    # Weight loading: first stage loads from base pi0.5, subsequent stages
    # load from the previous stage's final checkpoint.
    if prev_checkpoint is not None:
        loader = weight_loaders.CheckpointWithAuxHeadWeightLoader(prev_checkpoint)
    else:
        loader = weight_loaders.CheckpointWithAuxHeadWeightLoader(base_checkpoint)

    freeze_grounding = stage.get("freeze_grounding", False)
    freeze_filter_config = pi0_config.Pi0Config(
        pi05=True,
        action_horizon=50,
        paligemma_variant="gemma_2b_lora",
        aux_head=pi0_config.AuxHeadConfig(
            num_skill_type_classes=3,
            num_phase_index_classes=4,
            grounding_enabled=True,
            grounding_freeze=freeze_grounding,
        ),
    )

    # Config name is fixed per stage, exp_name includes the run_name for uniqueness
    config_name = f"curriculum_{stage['name']}"
    exp_name = f"{run_name}_{stage['name']}"

    return _config.TrainConfig(
        name=config_name,
        exp_name=exp_name,
        project_name="B1K",
        model=model_config,
        data=data_config,
        weight_loader=loader,
        num_train_steps=stage["num_train_steps"],
        batch_size=batch_size,
        freeze_filter=freeze_filter_config.get_freeze_filter(),
        ema_decay=None,
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir=checkpoint_base_dir,
        num_workers=1,
        # NEVER overwrite — always resume if dir exists, start fresh if new.
        # overwrite is intentionally omitted (defaults to False).
        resume=True,
    )


def get_final_checkpoint_path(config) -> str:
    """Return the path to the final checkpoint's params dir for a given config."""
    ckpt_dir = config.checkpoint_dir
    final_step = config.num_train_steps - 1
    return str(ckpt_dir / str(final_step) / "params")


def main():
    parser = argparse.ArgumentParser(description="Curriculum training for B1K")
    parser.add_argument(
        "--run_name", type=str, default="curriculum_v1",
        help="Unique name for this curriculum run (used in checkpoint paths)",
    )
    parser.add_argument(
        "--start_stage", type=int, default=0,
        help="Stage index to start from (0-based). Use to resume after a crash.",
    )
    parser.add_argument(
        "--end_stage", type=int, default=None,
        help="Stage index to end at (inclusive). Default: run all stages.",
    )
    parser.add_argument(
        "--base_checkpoint", type=str,
        default="gs://openpi-assets/checkpoints/pi05_base/params",
        help="Base pi0.5 checkpoint for Stage 0",
    )
    parser.add_argument(
        "--dataset_root", type=str,
        default="/home/stickbot/projects/behavior/behavior_data",
        help="Path to behavior_data root",
    )
    parser.add_argument(
        "--assets_dir", type=str,
        default="./outputs/assets/pi05_b1k_phase",
        help="Path to norm stats assets",
    )
    parser.add_argument(
        "--checkpoint_base_dir", type=str,
        default="./outputs/checkpoints",
        help="Base directory for all checkpoints",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Training batch size",
    )
    parser.add_argument(
        "--steps_override", type=int, default=None,
        help="Override num_train_steps for ALL stages (for quick testing, e.g. --steps_override 50)",
    )
    parser.add_argument(
        "--prev_checkpoint", type=str, default=None,
        help="Explicit path to previous stage checkpoint (overrides auto-detection). "
             "Use when resuming from a non-final checkpoint, e.g. step 13000 instead of 14999.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    end_stage = args.end_stage if args.end_stage is not None else len(CURRICULUM_STAGES) - 1
    stages_to_run = CURRICULUM_STAGES[args.start_stage : end_stage + 1]

    # Apply step override if provided
    if args.steps_override is not None:
        for stage in stages_to_run:
            stage["num_train_steps"] = args.steps_override

    logging.info("=" * 60)
    logging.info("CURRICULUM TRAINING: %s", args.run_name)
    if args.steps_override:
        logging.info("  *** STEPS OVERRIDE: %d steps per stage (testing mode) ***", args.steps_override)
    logging.info("=" * 60)
    for i, stage in enumerate(CURRICULUM_STAGES):
        marker = " <-- START" if i == args.start_stage else ""
        marker += " <-- END" if i == end_stage else ""
        steps = stage["num_train_steps"]
        logging.info(
            "  Stage %d: %-30s  skills=%s  steps=%d%s",
            i, stage["description"], stage["skill_indices"],
            steps, marker,
        )
    logging.info("=" * 60)

    # Import training main AFTER parsing args (loads JAX which is slow)
    import importlib.util
    import wandb

    # Import train.main() from the sibling script without relying on
    # scripts/ being a proper Python package.
    train_script = pathlib.Path(__file__).parent / "train.py"
    spec = importlib.util.spec_from_file_location("train_module", train_script)
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    train_main = train_module.main

    prev_checkpoint = None

    # If starting from a stage > 0, we need the previous stage's checkpoint.
    if args.start_stage > 0:
        if args.prev_checkpoint:
            # Explicit checkpoint path provided — use it directly
            prev_checkpoint = args.prev_checkpoint
            logging.info("Using explicit prev checkpoint: %s", prev_checkpoint)
        else:
            # Auto-detect from previous stage's final step
            prev_stage = CURRICULUM_STAGES[args.start_stage - 1]
            prev_config = build_stage_config(
                stage=prev_stage,
                run_name=args.run_name,
                prev_checkpoint=None,  # doesn't matter, just need the path
                base_checkpoint=args.base_checkpoint,
                dataset_root=args.dataset_root,
                assets_dir=args.assets_dir,
                checkpoint_base_dir=args.checkpoint_base_dir,
                batch_size=args.batch_size,
            )
            prev_checkpoint = get_final_checkpoint_path(prev_config)
            logging.info("Resuming from stage %d, expecting checkpoint: %s", args.start_stage, prev_checkpoint)

        if not pathlib.Path(prev_checkpoint).exists():
            logging.error("Previous stage checkpoint not found: %s", prev_checkpoint)
            logging.error("Provide --prev_checkpoint or train stage %d first", args.start_stage - 1)
            sys.exit(1)

    for i, stage in enumerate(stages_to_run):
        stage_idx = args.start_stage + i
        logging.info("")
        logging.info("=" * 60)
        logging.info("STAGE %d/%d: %s", stage_idx, end_stage, stage["description"])
        logging.info("  Skills: %s", stage["skill_indices"])
        logging.info("  Steps: %d", stage["num_train_steps"])
        if prev_checkpoint:
            logging.info("  Loading from: %s", prev_checkpoint)
        else:
            logging.info("  Loading from: %s (base)", args.base_checkpoint)
        logging.info("=" * 60)

        config = build_stage_config(
            stage=stage,
            run_name=args.run_name,
            prev_checkpoint=prev_checkpoint,
            base_checkpoint=args.base_checkpoint,
            dataset_root=args.dataset_root,
            assets_dir=args.assets_dir,
            checkpoint_base_dir=args.checkpoint_base_dir,
            batch_size=args.batch_size,
        )

        logging.info("Checkpoint dir: %s", config.checkpoint_dir)

        # Run training
        train_main(config)

        # Finish wandb run so next stage can start a fresh one
        wandb.finish()

        # Set up checkpoint path for next stage
        prev_checkpoint = get_final_checkpoint_path(config)
        logging.info("Stage %d complete. Checkpoint: %s", stage_idx, prev_checkpoint)

    logging.info("")
    logging.info("=" * 60)
    logging.info("CURRICULUM TRAINING COMPLETE")
    logging.info("Final checkpoint: %s", prev_checkpoint)
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
