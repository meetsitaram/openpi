#!/usr/bin/env python3
"""Phase 2: Grounding overlays + metrics aggregation for eval sweep.

Run in the openpi venv after Phase 1 (run_eval_sweep.sh) completes.

Usage:
    cd /home/stickbot/projects/behavior/b1k-baselines/baselines/openpi
    source .venv/bin/activate
    python scripts/aggregate_eval_sweep.py

    # Skip overlay generation (metrics only):
    python scripts/aggregate_eval_sweep.py --no-overlays

    # Process a single stage:
    python scripts/aggregate_eval_sweep.py --stage stage3_full_task
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

PROJ_ROOT = Path("/home/stickbot/projects/behavior")
OPENPI_DIR = PROJ_ROOT / "b1k-baselines" / "baselines" / "openpi"
EVAL_LOG_BASE = PROJ_ROOT / "eval_logs" / "v3_sweep"
OVERLAY_OUT_BASE = OPENPI_DIR / "outputs" / "v3_sweep_overlays"
CKPT_BASE = OPENPI_DIR / "outputs" / "checkpoints"
COMPARISON_CSV = OPENPI_DIR / "outputs" / "v3_sweep_comparison.csv"

STAGES = {
    "stage0_nav": "curriculum_stage0_nav/grounding_v3_spatial_stage0_nav/14999",
    "stage1_nav_pickup": "curriculum_stage1_nav_pickup/grounding_v3_spatial_stage1_nav_pickup/14999",
    "stage2_grasp": "curriculum_stage2_grasp/grounding_v3_spatial_stage2_grasp/19999",
    "stage3_full_task": "curriculum_stage3_full_task/grounding_v3_spatial_stage3_full_task/19999",
}

RADIO_TASK_KEYS = [
    "total_steps",
    "success",
    "base_to_radio_dist_final",
    "right_gripper_to_radio_dist_final",
    "base_to_radio_dist_at_transition",
    "transition_step",
    "base_angle_to_radio_at_transition",
    "base_angle_to_table_at_transition",
    "radio_head_cam_mean_offset",
    "radio_head_cam_samples",
    "table_collision_frames",
    "table_collision_occurred",
]


def generate_overlays(stage_name: str, ckpt_rel: str) -> None:
    """Generate grounding overlay videos for all eval videos of a stage."""
    video_dir = EVAL_LOG_BASE / stage_name / "videos"
    if not video_dir.exists():
        log.warning("No video dir for %s: %s", stage_name, video_dir)
        return

    ckpt_path = CKPT_BASE / ckpt_rel / "params"
    if not ckpt_path.exists():
        log.warning("Checkpoint not found: %s", ckpt_path)
        return

    out_dir = OVERLAY_OUT_BASE / stage_name
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(video_dir.glob("*.mp4"))
    if not videos:
        log.warning("No videos found in %s", video_dir)
        return

    log.info("Generating %d overlays for %s", len(videos), stage_name)
    ckpt_stem = Path(ckpt_rel).parent.name  # e.g. "19999"
    for video_path in videos:
        expected_out = out_dir / f"probe_eval_{video_path.stem}_{ckpt_stem}.mp4"
        if expected_out.exists():
            log.info("  Skip (exists): %s", expected_out.name)
            continue

        log.info("  Processing: %s", video_path.name)
        cmd = [
            sys.executable, str(OPENPI_DIR / "scripts" / "eval_grounding_probe.py"),
            "--video", str(video_path),
            "--checkpoint", str(ckpt_path),
            "--output_dir", str(out_dir),
        ]
        try:
            subprocess.run(cmd, check=True, timeout=300)
        except subprocess.TimeoutExpired:
            log.error("  Timeout generating overlay for %s", video_path.name)
        except subprocess.CalledProcessError as e:
            log.error("  Failed overlay for %s: %s", video_path.name, e)


def collect_metrics(stage_name: str) -> list[dict]:
    """Read all metrics JSON files for a stage. Returns list of per-instance dicts."""
    metrics_dir = EVAL_LOG_BASE / stage_name / "metrics"
    if not metrics_dir.exists():
        log.warning("No metrics dir for %s", stage_name)
        return []

    results = []
    for json_path in sorted(metrics_dir.glob("*.json")):
        try:
            with open(json_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            log.error("Failed to read %s: %s", json_path, e)
            continue

        row = {"stage": stage_name, "instance": json_path.stem}

        # Extract radio_task metrics
        rt = data.get("radio_task", {})
        for key in RADIO_TASK_KEYS:
            row[key] = rt.get(key)

        # Also grab q_score if available
        qs = data.get("q_score", {})
        row["q_score_final"] = qs.get("final")

        # Agent distance
        ad = data.get("agent_distance", {})
        row["agent_dist_base"] = ad.get("base")
        row["agent_dist_right"] = ad.get("right")

        results.append(row)

    return results


def compute_stage_summary(rows: list[dict]) -> dict:
    """Compute mean/count summary across instances for a stage."""
    if not rows:
        return {}

    summary = {"stage": rows[0]["stage"], "n_instances": len(rows)}

    numeric_keys = [
        "total_steps", "base_to_radio_dist_final",
        "right_gripper_to_radio_dist_final",
        "base_to_radio_dist_at_transition",
        "transition_step",
        "base_angle_to_radio_at_transition",
        "base_angle_to_table_at_transition",
        "radio_head_cam_mean_offset",
        "table_collision_frames",
        "agent_dist_base",
        "q_score_final",
    ]
    for key in numeric_keys:
        vals = [r[key] for r in rows if r.get(key) is not None]
        if vals:
            summary[f"mean_{key}"] = sum(vals) / len(vals)
            summary[f"count_{key}"] = len(vals)

    # Boolean aggregations
    successes = sum(1 for r in rows if r.get("success"))
    summary["success_rate"] = successes / len(rows)
    summary["success_count"] = successes

    collisions = sum(1 for r in rows if r.get("table_collision_occurred"))
    summary["collision_rate"] = collisions / len(rows)

    return summary


def write_comparison_csv(all_rows: list[dict], summaries: list[dict]) -> None:
    """Write per-instance and summary CSV."""
    if not all_rows:
        log.warning("No data to write")
        return

    COMPARISON_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Per-instance CSV
    fieldnames = list(all_rows[0].keys())
    with open(COMPARISON_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    log.info("Wrote per-instance CSV: %s (%d rows)", COMPARISON_CSV, len(all_rows))

    # Summary CSV
    summary_path = COMPARISON_CSV.with_name("v3_sweep_summary.csv")
    if summaries:
        fieldnames = list(summaries[0].keys())
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summaries)
        log.info("Wrote summary CSV: %s", summary_path)


def print_summary_table(summaries: list[dict]) -> None:
    """Print formatted comparison table to stdout."""
    if not summaries:
        return

    print("\n" + "=" * 90)
    print("EVAL SWEEP SUMMARY (v3 spatial grounding, per-stage averages)")
    print("=" * 90)

    header = (
        f"{'Stage':<22} {'N':>3} {'Success':>8} "
        f"{'Base→Radio':>11} {'EEF→Radio':>10} "
        f"{'Trans Dist':>11} {'Angle→R':>8} "
        f"{'Cam Off':>8} {'Collisions':>11} "
        f"{'Steps':>7}"
    )
    print(header)
    print("-" * 90)

    for s in summaries:
        def _fmt(key, fmt=".2f"):
            v = s.get(key)
            return f"{v:{fmt}}" if v is not None else "  --"

        stage = s.get("stage", "?")
        n = s.get("n_instances", 0)
        sr = s.get("success_rate", 0)
        print(
            f"{stage:<22} {n:>3} {sr:>7.0%} "
            f"{_fmt('mean_base_to_radio_dist_final'):>11} "
            f"{_fmt('mean_right_gripper_to_radio_dist_final'):>10} "
            f"{_fmt('mean_base_to_radio_dist_at_transition'):>11} "
            f"{_fmt('mean_base_angle_to_radio_at_transition'):>8} "
            f"{_fmt('mean_radio_head_cam_mean_offset'):>8} "
            f"{s.get('collision_rate', 0):>10.0%} "
            f"{_fmt('mean_total_steps', '.0f'):>7}"
        )

    print("=" * 90)
    print()


def main():
    parser = argparse.ArgumentParser(description="Aggregate eval sweep results")
    parser.add_argument("--stage", type=str, default=None,
                        help="Process only this stage (e.g. stage3_full_task)")
    parser.add_argument("--no-overlays", action="store_true",
                        help="Skip grounding overlay generation")
    args = parser.parse_args()

    stages_to_process = (
        {args.stage: STAGES[args.stage]} if args.stage else STAGES
    )

    # Phase 2A: Generate grounding overlays
    if not args.no_overlays:
        log.info("Phase 2A: Generating grounding overlays...")
        for stage_name, ckpt_rel in stages_to_process.items():
            generate_overlays(stage_name, ckpt_rel)

    # Phase 2B: Aggregate metrics
    log.info("Phase 2B: Aggregating metrics...")
    all_rows = []
    summaries = []
    for stage_name in stages_to_process:
        rows = collect_metrics(stage_name)
        if rows:
            all_rows.extend(rows)
            summary = compute_stage_summary(rows)
            summaries.append(summary)
            log.info("  %s: %d instances", stage_name, len(rows))
        else:
            log.warning("  %s: no metrics found", stage_name)

    write_comparison_csv(all_rows, summaries)
    print_summary_table(summaries)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
