# Eval Metrics Sweep with Grounding Overlays

## 1. Create `RadioTaskMetric` (sim-side metric class)

New file: `BEHAVIOR-1K/OmniGibson/omnigibson/metrics/radio_task_metric.py`

Extends `MetricBase` (same pattern as `agent_metric.py` and `task_metric.py`).

**Object discovery** (in `start_callback`):

- Robot: `env.robots[0]` (already available)
- Radio: iterate `env.task.object_scope` for keys containing `radio_receiver`, fall back to `env.scene.object_registry("category", "radio")`
- Table: iterate `env.task.object_scope` for keys containing `table`, or search by common table categories

**Per-step data** (in `step_callback`):

- `robot_base_pos, robot_base_orn` via `robot.get_position_orientation()`
- `robot_base_yaw` via `T.quat2euler(orn)[2]`
- `radio_pos, radio_orn` via `radio_obj.get_position_orientation()`
- `table_pos, table_orn` via `table_obj.get_position_orientation()`
- `right_eef_pos` via `robot.get_eef_position("right")`
- `head_cam_pose` via `robot.sensors["robot_r1:zed_link:Camera:0"].get_position_orientation()`
- **Table collision**: check if robot's `ContactBodies` state includes any links from the table object

**Metrics computed** (in `gather_results`):

- `base_to_radio_dist_final` — Euclidean XY distance at last step
- `base_to_radio_dist_at_phase_transition` — Distance when phase switches nav→manip
- `base_orientation_to_radio` — Angle between robot forward vector and vector to radio at phase transition
- `base_orientation_to_table` — Angle between robot forward vector and table center at phase transition
- `right_gripper_to_radio_dist_final` — Euclidean distance at last step
- `radio_in_head_cam_center` — Mean normalized offset from image center during manipulation steps
- `table_collision_count` — Total frames where robot base was in contact with the table
- `table_collision_occurred` — Boolean: did any table collision happen during navigation
- `total_steps` — Number of steps in the episode
- `success` — Whether task succeeded

**Phase transition detection**: Track when the robot switches from base movement to arm movement
by monitoring joint velocities (base magnitude drops, arm magnitude rises).

## 2. Wire metric into eval.py

Modify `load_metrics()` in `eval.py` to include `RadioTaskMetric`.

## 3. Phase 1: Bash orchestration script

New file: `b1k-baselines/baselines/openpi/scripts/run_eval_sweep.sh`

Loops over 4 checkpoints (final checkpoint per stage):

- `curriculum_stage0_nav/grounding_v3_spatial_stage0_nav/14999`
- `curriculum_stage1_nav_pickup/grounding_v3_spatial_stage1_nav_pickup/14999`
- `curriculum_stage2_grasp/grounding_v3_spatial_stage2_grasp/19999`
- `curriculum_stage3_full_task/grounding_v3_spatial_stage3_full_task/19999`

For each checkpoint:

1. Start policy server in background (openpi venv, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.65`)
2. Wait for "Serving on" in server output (poll log file)
3. Run eval.py with: `max_steps=1500`, `eval_on_train_instances=true`, all 10 instances, `headless=true`
4. Kill policy server
5. Log output path: `eval_logs/v3_sweep/<stage_name>/`

**Note**: Server runs in openpi venv, eval runs in `env_isaaclab` conda env.

## 4. Phase 2: Grounding overlays + metrics aggregation

New file: `b1k-baselines/baselines/openpi/scripts/aggregate_eval_sweep.py`

**A. Generate grounding overlay videos** (runs in openpi venv):
- For each stage's eval videos, run `eval_grounding_probe.py --video <path> --checkpoint <ckpt>`
- Output to `outputs/v3_sweep_overlays/<stage_name>/`

**B. Aggregate metrics into comparison CSV/table**:
- Read all `*.json` metric files from `eval_logs/v3_sweep/<stage_name>/metrics/`
- Compute per-stage averages and per-instance breakdowns
- Output: `outputs/v3_sweep_comparison.csv`

## 5. Single test run (validate first)

Before the full sweep, run one instance with stage 3 checkpoint (already loaded in the server):
1. Create the `RadioTaskMetric` class
2. Wire it into eval.py
3. Run eval for 1 instance, 1500 steps
4. Verify metrics JSON and grounding overlay

## Checkpoint paths reference

- Stage 0: `outputs/checkpoints/curriculum_stage0_nav/grounding_v3_spatial_stage0_nav/14999`
- Stage 1: `outputs/checkpoints/curriculum_stage1_nav_pickup/grounding_v3_spatial_stage1_nav_pickup/14999`
- Stage 2: `outputs/checkpoints/curriculum_stage2_grasp/grounding_v3_spatial_stage2_grasp/19999`
- Stage 3: `outputs/checkpoints/curriculum_stage3_full_task/grounding_v3_spatial_stage3_full_task/19999`

## File changes summary

- **New**: `BEHAVIOR-1K/OmniGibson/omnigibson/metrics/radio_task_metric.py` (~150 lines)
- **Edit**: `BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval.py` (1 line in `load_metrics()`)
- **New**: `b1k-baselines/baselines/openpi/scripts/run_eval_sweep.sh` (~80 lines)
- **New**: `b1k-baselines/baselines/openpi/scripts/aggregate_eval_sweep.py` (~200 lines)
- **Edit**: `b1k-baselines/baselines/openpi/docs/COMMANDS.md` (add sweep commands)
