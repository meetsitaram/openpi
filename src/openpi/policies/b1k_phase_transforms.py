"""Phase-conditioned transforms for BEHAVIOR-1K dataset.

These transforms inject skill/phase information from dataset annotations
into the data pipeline, enabling phase-aware prompts, camera masking,
proprioception selection, and action weighting.

Usage:
    These transforms are designed to be inserted into the LeRobotB1KDataConfig
    pipeline in config.py.
"""

import dataclasses
import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

from openpi.transforms import DataTransformFn, DataDict

logger = logging.getLogger(__name__)


# Skill type constants
SKILL_TYPE_NAVIGATION = "navigation"
SKILL_TYPE_UNCOORDINATED = "uncoordinated"
SKILL_TYPE_COORDINATED = "coordinated"

# Indices into the 23-dim state vector (output of extract_state_from_proprio)
# state = [base_qvel(3), trunk_qpos(4), arm_left_qpos(7), arm_right_qpos(7),
#          gripper_left_width(1), gripper_right_width(1)]
STATE_BASE_QVEL = slice(0, 3)
STATE_TRUNK_QPOS = slice(3, 7)
STATE_ARM_LEFT_QPOS = slice(7, 14)
STATE_ARM_RIGHT_QPOS = slice(14, 21)
STATE_GRIPPER_LEFT = slice(21, 22)
STATE_GRIPPER_RIGHT = slice(22, 23)

# Indices into the 23-dim action vector
# action = [base(3), torso(4), left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)]
ACTION_BASE = slice(0, 3)
ACTION_TORSO = slice(3, 7)
ACTION_LEFT_ARM = slice(7, 14)
ACTION_LEFT_GRIPPER = slice(14, 15)
ACTION_RIGHT_ARM = slice(15, 22)
ACTION_RIGHT_GRIPPER = slice(22, 23)


@lru_cache(maxsize=512)
def _load_annotation(annotation_path: str) -> dict:
    """Load and cache a skill annotation JSON file."""
    with open(annotation_path, "r") as f:
        return json.load(f)


def get_navigation_frame_ranges(
    dataset_root: str, task_id: int = 0
) -> dict[int, tuple[int, int]]:
    """Return {episode_idx: (nav_start_frame, nav_end_frame)} for all episodes.

    Convenience wrapper around get_skill_frame_ranges for navigation-only.
    """
    return get_skill_frame_ranges(dataset_root, task_id=task_id, skill_indices={0})


def get_skill_frame_ranges(
    dataset_root: str,
    task_id: int = 0,
    skill_indices: set[int] | None = None,
) -> dict[int, list[tuple[int, int]]]:
    """Return {episode_idx: [(start, end), ...]} for selected skill phases.

    Reads skill annotations and extracts frame ranges for the specified skill
    indices. Used by curriculum training to filter the dataset to specific
    phase combinations.

    Args:
        dataset_root: Path to behavior_data root.
        task_id: Task ID (0 for turning_on_radio).
        skill_indices: Set of skill_idx values to include (e.g. {0} for nav,
            {0, 1} for nav + pick-up). If None, includes all skills.

    Returns:
        Dict mapping episode index to list of (start_frame, end_frame) tuples
        for the selected skills.
    """
    ann_dir = Path(dataset_root) / "annotations" / f"task-{task_id:04d}"
    if not ann_dir.exists():
        logger.warning("Annotation dir %s does not exist", ann_dir)
        return {}

    ranges = {}
    for ann_file in sorted(ann_dir.glob("episode_*.json")):
        ep_str = ann_file.stem.replace("episode_", "")
        ep_idx = int(ep_str)
        try:
            annotation = json.load(open(ann_file))
            ep_ranges = []
            for skill in annotation.get("skill_annotation", []):
                if skill_indices is not None and skill["skill_idx"] not in skill_indices:
                    continue
                start, end = skill["frame_duration"]
                ep_ranges.append((start, end))
            if ep_ranges:
                ranges[ep_idx] = ep_ranges
        except (json.JSONDecodeError, KeyError) as e:
            logger.debug("Skipping %s: %s", ann_file, e)

    total_frames = sum(e - s for segs in ranges.values() for s, e in segs)
    num_segments = sum(len(segs) for segs in ranges.values())
    skill_str = str(skill_indices) if skill_indices else "all"
    logger.info(
        "Found %d segments across %d episodes for skills %s (%d total frames, avg %.0f/ep)",
        num_segments, len(ranges), skill_str, total_frames,
        total_frames / max(len(ranges), 1),
    )
    return ranges


def get_grasp_frame_ranges(
    dataset_root: str,
    task_id: int = 0,
    before_close: int = 300,
    after_close: int = 100,
) -> dict[int, list[tuple[int, int]]]:
    """Return frame ranges centered on the right-gripper close event.

    Scans each episode's action data to find the first frame where the right
    gripper transitions from open (>0) to closed (<0). Returns a window of
    [close_frame - before_close, close_frame + after_close] for each episode.

    Args:
        dataset_root: Path to behavior_data root.
        task_id: Task ID (0 for turning_on_radio).
        before_close: Frames to include before the right-gripper close.
        after_close: Frames to include after the right-gripper close.

    Returns:
        Dict mapping episode index to [(start, end)] window.
    """
    import pyarrow.parquet as pq

    data_dir = Path(dataset_root) / "data" / f"task-{task_id:04d}"
    if not data_dir.exists():
        logger.warning("Data dir %s does not exist", data_dir)
        return {}

    ranges = {}
    for parquet_file in sorted(data_dir.glob("episode_*.parquet")):
        ep_str = parquet_file.stem.replace("episode_", "")
        ep_idx = int(ep_str)
        try:
            table = pq.read_table(parquet_file, columns=["action"])
            actions = np.array(table["action"].to_pylist())
            n = len(actions)

            # Find first right-gripper close: action[i, 22] < 0 and action[i-1, 22] > 0
            r_close = None
            for i in range(1, n):
                if actions[i, 22] < 0 and actions[i - 1, 22] > 0:
                    r_close = i
                    break

            if r_close is None:
                continue

            w_start = max(0, r_close - before_close)
            w_end = min(n, r_close + after_close)
            ranges[ep_idx] = [(w_start, w_end)]
        except Exception as e:
            logger.debug("Skipping %s: %s", parquet_file, e)

    total_frames = sum(e - s for segs in ranges.values() for s, e in segs)
    logger.info(
        "Grasp windows (%d before, %d after R_close): %d episodes, %d total frames (avg %.0f/ep)",
        before_close, after_close, len(ranges), total_frames,
        total_frames / max(len(ranges), 1),
    )
    return ranges


def get_skill_for_frame(
    annotation: dict, frame_idx: int, early_transition_frames: int = 0
) -> dict:
    """Look up the skill annotation for a given frame index.

    Args:
        annotation: Parsed annotation JSON with 'skill_annotation' list.
        frame_idx: The frame index within the episode.
        early_transition_frames: Number of frames before each phase boundary
            to relabel as the NEXT phase. E.g., if nav ends at frame 500 and
            early_transition_frames=60 (2 sec at 30fps), frames 440-499 get
            labeled as "pick up" instead of "move to". This teaches the model
            to anticipate phase transitions. 0 = use exact boundaries.

    Returns:
        Dict with keys: skill_phase (int), skill_description (str),
        skill_type (str), skill_objects (list[str]).
        Returns defaults (phase=0, navigation) if frame is before first skill
        or after last skill.
    """
    skills = annotation.get("skill_annotation", [])
    if not skills:
        return {
            "skill_phase": 0,
            "skill_description": "unknown",
            "skill_type": SKILL_TYPE_NAVIGATION,
            "skill_objects": [],
        }

    # Check if frame falls in an "early transition" zone — the last N frames
    # of a phase, where we relabel it as the NEXT phase.
    if early_transition_frames > 0:
        for i, skill in enumerate(skills[:-1]):  # skip last phase (no next phase)
            start, end = skill["frame_duration"]
            next_skill = skills[i + 1]
            # If frame is in the tail of this phase, label it as next phase
            transition_start = max(start, end - early_transition_frames)
            if transition_start <= frame_idx < end:
                return {
                    "skill_phase": next_skill["skill_idx"],
                    "skill_description": next_skill["skill_description"][0],
                    "skill_type": next_skill["skill_type"][0],
                    "skill_objects": [
                        obj for obj_list in next_skill["object_id"] for obj in obj_list
                    ],
                }

    for skill in skills:
        start, end = skill["frame_duration"]
        if start <= frame_idx < end:
            return {
                "skill_phase": skill["skill_idx"],
                "skill_description": skill["skill_description"][0],
                "skill_type": skill["skill_type"][0],
                "skill_objects": [
                    obj for obj_list in skill["object_id"] for obj in obj_list
                ],
            }

    # Frame is outside all annotated ranges (gap between skills or after last skill)
    # Find the closest skill
    for i, skill in enumerate(skills):
        start, end = skill["frame_duration"]
        if frame_idx < start:
            # Before this skill starts — use this skill (upcoming)
            return {
                "skill_phase": skill["skill_idx"],
                "skill_description": skill["skill_description"][0],
                "skill_type": skill["skill_type"][0],
                "skill_objects": [
                    obj for obj_list in skill["object_id"] for obj in obj_list
                ],
            }

    # After all skills — use the last one
    last = skills[-1]
    return {
        "skill_phase": last["skill_idx"],
        "skill_description": last["skill_description"][0],
        "skill_type": last["skill_type"][0],
        "skill_objects": [
            obj for obj_list in last["object_id"] for obj in obj_list
        ],
    }


@dataclasses.dataclass(frozen=True)
class InjectSkillAnnotation(DataTransformFn):
    """Loads skill annotation for the current frame and injects phase info.

    Must run BEFORE RepackTransform so it can access episode_index, task_index,
    and timestamp from the raw dataset sample.

    Adds to the data dict:
        - skill_phase (int): skill index (0=nav, 1=pick, 2=press, 3=place for turning_on_radio)
        - skill_description (str): e.g. "move to", "pick up from", "press", "place on"
        - skill_type (str): "navigation", "uncoordinated", or "coordinated"
        - skill_objects (list[str]): target object IDs e.g. ["radio_89"]
    """

    dataset_root: str  # Path to behavior_data root

    # Early transition: relabel the last N frames of each phase as the NEXT
    # phase. At 30fps, 60 frames = 2 seconds. This teaches the model to
    # anticipate transitions (e.g., start "pick up" behavior while still
    # finishing navigation approach). 0 = use exact annotation boundaries.
    early_transition_frames: int = 60

    def __call__(self, data: DataDict) -> DataDict:
        # Get episode info from raw dataset fields
        ep_idx = int(data.get("episode_index", 0))
        task_idx = int(data.get("task_index", 0))
        timestamp = float(data.get("timestamp", 0.0))

        # Convert timestamp (seconds) to frame index (30 FPS)
        frame_idx = int(round(timestamp * 30.0))

        # Build annotation path
        ann_path = (
            Path(self.dataset_root)
            / "annotations"
            / f"task-{task_idx:04d}"
            / f"episode_{ep_idx:08d}.json"
        )

        try:
            annotation = _load_annotation(str(ann_path))
            skill_info = get_skill_for_frame(
                annotation, frame_idx,
                early_transition_frames=self.early_transition_frames,
            )
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.debug(f"Could not load annotation {ann_path}: {e}")
            skill_info = {
                "skill_phase": 0,
                "skill_description": "unknown",
                "skill_type": SKILL_TYPE_NAVIGATION,
                "skill_objects": [],
            }

        return {**data, **skill_info}


@dataclasses.dataclass(frozen=True)
class PhaseConditionedPrompt(DataTransformFn):
    """Enriches the task prompt with the current skill phase description.

    Transforms (when phase is included):
        "Turn on the radio receiver that's on the table in the living room."
    Into:
        "Turn on the radio ..., Phase: move to radio"

    Phase dropout: during training, the phase string is randomly omitted
    with probability `dropout_rate`.  This ensures the model learns to
    leverage phase cues when available but can still operate without them
    at inference time (where ground-truth phases are unavailable).

    Must run AFTER InjectSkillAnnotation and AFTER RepackTransform (which
    sets the "prompt" key), but BEFORE TokenizePrompt.
    """

    # Probability of OMITTING the phase string (0.0 = always include, 1.0 = never include)
    dropout_rate: float = 0.5

    def __call__(self, data: DataDict) -> DataDict:
        prompt = data.get("prompt", "")
        if not isinstance(prompt, str):
            prompt = str(prompt)

        skill_desc = data.get("skill_description", "unknown")
        if skill_desc == "unknown":
            # No phase info available (e.g. baseline eval without phase injection)
            logger.debug("[PhaseConditionedPrompt] No skill_description — skipping")
            return data

        # Phase dropout: randomly skip phase injection during TRAINING only.
        # At inference, "actions" key is absent from the data dict, so we
        # always inject the phase when it's available.
        is_training = "actions" in data
        if is_training and np.random.random() < self.dropout_rate:
            # Leave the prompt unchanged — model must rely on visual/proprio cues
            logger.debug("[PhaseConditionedPrompt] Dropout — prompt unchanged")
            return data

        skill_objects = data.get("skill_objects", [])

        # Build a concise phase description
        if skill_objects:
            # e.g. "move to radio_89" -> "move to radio"
            obj_name = skill_objects[0].split("_")[0]  # "radio_89" -> "radio"
            phase_str = f"{skill_desc} {obj_name}"
        else:
            phase_str = skill_desc

        data["prompt"] = f"{prompt}, Phase: {phase_str}"
        logger.debug("[PhaseConditionedPrompt] prompt='%s'", data["prompt"])

        return data


@dataclasses.dataclass(frozen=True)
class PhaseAwareCameraMasking(DataTransformFn):
    """Masks out wrist cameras during navigation phase.

    The head camera is ALWAYS on — it provides global context that is useful
    in every phase (and is the only fallback when wrist cameras are occluded
    during grasping).

    Wrist cameras are turned off during navigation only, since they see
    nothing useful while the robot is driving toward the target. During all
    manipulation phases (pick, press, place), wrist cameras are kept on.

    Dropout: with probability `dropout_rate`, skip masking entirely so the
    model also learns from full unmasked observations. This prevents a hard
    dependency on phase-correct masking at inference.

    Must run AFTER B1kInputs (which creates the image/image_mask dict).
    """

    dropout_rate: float = 0.3

    def __call__(self, data: DataDict) -> DataDict:
        skill_type = data.get("skill_type", SKILL_TYPE_NAVIGATION)
        image_mask = data.get("image_mask", {})

        if not image_mask:
            return data

        # Dropout: skip masking so model sees full observations sometimes.
        # At inference ("actions" absent), always apply masking when phase info is available.
        is_training = "actions" in data
        if is_training and np.random.random() < self.dropout_rate:
            # Leave all cameras on (default from B1kInputs is all True)
            logger.debug("[CameraMask] DROPOUT — all cameras ON")
            return data

        # Head camera is ALWAYS on
        image_mask["base_0_rgb"] = np.True_

        if skill_type == SKILL_TYPE_NAVIGATION:
            # Wrist cameras off during navigation (they see floor/walls)
            image_mask["left_wrist_0_rgb"] = np.False_
            image_mask["right_wrist_0_rgb"] = np.False_
        else:
            # All manipulation phases: wrist cameras on
            image_mask["left_wrist_0_rgb"] = np.True_
            image_mask["right_wrist_0_rgb"] = np.True_

        logger.debug(
            "[CameraMask] phase=%s | head=ON | left_wrist=%s | right_wrist=%s",
            skill_type,
            "ON" if image_mask.get("left_wrist_0_rgb") else "OFF",
            "ON" if image_mask.get("right_wrist_0_rgb") else "OFF",
        )

        data["image_mask"] = image_mask
        return data


@dataclasses.dataclass(frozen=True)
class PhaseAwareProprioception(DataTransformFn):
    """Masks proprioception state dimensions based on current skill phase.

    Strategy:
        - navigation: keep base_qvel + trunk, zero out arm/gripper
        - manipulation: keep arm/gripper + trunk, zero out base_qvel

    Dropout: with probability `dropout_rate`, skip masking entirely so the
    model also learns from full unmasked proprioception. This prevents a
    hard dependency on phase-correct masking at inference.

    Must run AFTER B1kInputs (which creates the "state" key with 23 dims).
    """

    dropout_rate: float = 0.3

    def __call__(self, data: DataDict) -> DataDict:
        skill_type = data.get("skill_type", SKILL_TYPE_NAVIGATION)
        state = data.get("state", None)

        if state is None:
            return data

        # Dropout: skip masking so model sees full state sometimes.
        # At inference ("actions" absent), always apply masking when phase info is available.
        is_training = "actions" in data
        if is_training and np.random.random() < self.dropout_rate:
            logger.debug("[Proprio] DROPOUT — full state preserved")
            return data

        state = np.array(state, copy=True)

        if skill_type == SKILL_TYPE_NAVIGATION:
            # Keep base_qvel (0:3) and trunk_qpos (3:7), zero arm/gripper
            state[STATE_ARM_LEFT_QPOS] = 0.0
            state[STATE_ARM_RIGHT_QPOS] = 0.0
            state[STATE_GRIPPER_LEFT] = 0.0
            state[STATE_GRIPPER_RIGHT] = 0.0
            logger.debug(
                "[Proprio] phase=navigation | KEEP base=%.3f,%.3f,%.3f trunk | ZERO arms+grippers",
                state[0], state[1], state[2],
            )
        elif skill_type in (SKILL_TYPE_UNCOORDINATED, SKILL_TYPE_COORDINATED):
            # Keep full state including base — allows small base corrections
            # during manipulation (base actions are clamped at eval time)
            logger.debug(
                "[Proprio] phase=%s | KEEP all (base=%.3f,%.3f,%.3f) arms L=[%.2f..%.2f] R=[%.2f..%.2f] grip=%.2f/%.2f",
                skill_type,
                state[0], state[1], state[2],
                state[7], state[13], state[14], state[20],
                state[21], state[22],
            )

        data["state"] = state
        return data


@dataclasses.dataclass(frozen=True)
class PhaseAwareActionWeighting(DataTransformFn):
    """Applies per-dimension action weighting based on current skill phase.

    Strategy:
        - navigation: boost base actions, dampen torso, scale down arm/gripper
        - manipulation: scale down base actions (keep arm/gripper/torso actions)

    This modifies the actions directly by scaling dimensions, so the model
    learns to predict near-zero for irrelevant dims and accurate values for
    important dims. This is simpler than modifying the loss function.

    Per-component weights during navigation:
        - base (0-2):       `base_nav_weight` (default 2.0 — boost wheel movement)
        - torso (3-6):      `torso_nav_weight` (default 0.3 — dampen leaning)
        - arms/grippers:    `low_weight` (default 0.1 — suppress)

    Dropout: with probability `dropout_rate`, skip weighting entirely so the
    model also learns from unmodified actions.

    Must run AFTER B1kInputs but BEFORE Normalize.
    """

    # Scale factor for de-emphasized action dimensions (0.0 = zero, 0.1 = 10%)
    low_weight: float = 0.1
    # Boost factor for base velocity during navigation (> 1.0 amplifies learning signal)
    base_nav_weight: float = 2.0
    # Dampen factor for torso during navigation (< 1.0 discourages torso movement)
    torso_nav_weight: float = 0.3
    dropout_rate: float = 0.3

    def __call__(self, data: DataDict) -> DataDict:
        skill_type = data.get("skill_type", SKILL_TYPE_NAVIGATION)
        actions = data.get("actions", None)

        if actions is None:
            return data

        # Dropout: skip action weighting sometimes
        if np.random.random() < self.dropout_rate:
            return data

        actions = np.array(actions, copy=True)

        if skill_type == SKILL_TYPE_NAVIGATION:
            # Boost base velocity — forces model to learn wheel movement
            actions[..., ACTION_BASE] *= self.base_nav_weight
            # Dampen torso — discourages using torso rocking for locomotion
            actions[..., ACTION_TORSO] *= self.torso_nav_weight
            # Suppress arm/gripper — irrelevant during navigation
            actions[..., ACTION_LEFT_ARM] *= self.low_weight
            actions[..., ACTION_LEFT_GRIPPER] *= self.low_weight
            actions[..., ACTION_RIGHT_ARM] *= self.low_weight
            actions[..., ACTION_RIGHT_GRIPPER] *= self.low_weight
        elif skill_type in (SKILL_TYPE_UNCOORDINATED, SKILL_TYPE_COORDINATED):
            # During manipulation, moderately scale down base (allows corrections)
            actions[..., ACTION_BASE] *= 0.3

        data["actions"] = actions
        return data


@dataclasses.dataclass(frozen=True)
class EncodePhaseLabels(DataTransformFn):
    """Encodes skill_type and skill_phase as integer arrays for aux head training.

    Converts string labels to integer class indices that can be collated by
    np.stack in the data loader. Stores them in ``aux_labels`` dict which
    gets carried through to the Observation struct.

    Mapping:
        skill_type  -> {navigation: 0, uncoordinated: 1, coordinated: 2}
        skill_phase -> kept as-is (already int from InjectSkillAnnotation)

    Must run AFTER all phase-aware transforms (prompt, camera, proprio, action)
    have consumed the string fields, but BEFORE CleanupSkillFields.
    """

    def __call__(self, data: DataDict) -> DataDict:
        skill_type_str = data.get("skill_type", SKILL_TYPE_NAVIGATION)
        skill_phase_int = int(data.get("skill_phase", 0))

        type_map = {
            SKILL_TYPE_NAVIGATION: 0,
            SKILL_TYPE_UNCOORDINATED: 1,
            SKILL_TYPE_COORDINATED: 2,
        }
        type_idx = type_map.get(skill_type_str, 0)

        data["aux_labels"] = {
            "skill_type": np.array(type_idx, dtype=np.int32),
            "phase_index": np.array(skill_phase_int, dtype=np.int32),
        }
        return data


@dataclasses.dataclass(frozen=True)
class CleanupSkillFields(DataTransformFn):
    """Removes skill string fields from the data dict to prevent collation errors.

    Must run AFTER EncodePhaseLabels (which preserves them as numeric arrays
    inside ``aux_labels``). Python strings and lists cannot be np.stacked by
    the collate function.

    Note: ``aux_labels`` is kept — it contains only numpy int32 arrays.
    """

    def __call__(self, data: DataDict) -> DataDict:
        return {
            k: v for k, v in data.items()
            if k not in ("skill_phase", "skill_description", "skill_type", "skill_objects",
                         "observation/task_info", "observation/cam_rel_poses")
        }


# ---------------------------------------------------------------------------
# Visual grounding labels (3D → 2D projection from privileged sim data)
# ---------------------------------------------------------------------------

# task_info layout (per object: real(1) + pos(3) + ori_cos(3) + ori_sin(3))
_AGENT_OFFSET = 0   # 10 values
_RADIO_OFFSET = 10  # 12 values (includes grip(2))
_TABLE_OFFSET = 22  # 12 values

# cam_rel_poses layout: [pos(3)+quat_xyzw(4)] × 3 cameras
# order: left_wrist(0:7), right_wrist(7:14), head(14:21)
_CAM_HEAD_OFFSET = 14

# R1Pro head camera intrinsics at native 720×720
_HEAD_K = np.array([[306.0, 0.0, 360.0],
                    [0.0, 306.0, 360.0],
                    [0.0,   0.0,   1.0]], dtype=np.float64)
_HEAD_NATIVE_W = 720
_HEAD_NATIVE_H = 720

# 180-degree X-axis rotation (camera coordinate system correction)
_RX_180 = np.diag([1.0, -1.0, -1.0])

# Approximate 3D half-extents per object (meters).
# Can be refined by querying obj.aabb_extent in the simulator.
_OBJ_HALF_EXTENTS = {
    "radio": np.array([0.08, 0.06, 0.05], dtype=np.float64),
    "table": np.array([0.40, 0.30, 0.02], dtype=np.float64),
}
_OBJ_NAMES = ["radio", "table"]


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """(x, y, z, w) quaternion → 3×3 rotation matrix."""
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _euler_to_rotmat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Euler (roll, pitch, yaw) → rotation matrix (ZYX intrinsic convention)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def _world_to_cam(
    world_pos: np.ndarray,
    agent_pos: np.ndarray,
    R_world_to_base: np.ndarray,
    R_base_to_cam: np.ndarray,
    cam_pos: np.ndarray,
) -> np.ndarray:
    """Transform a world-space point to camera-space [x_cam, y_cam, z_cam]."""
    p_base = R_world_to_base @ (world_pos - agent_pos)
    return R_base_to_cam @ (p_base - cam_pos)


def _build_cam_transforms(
    agent_ori_cos: np.ndarray,
    agent_ori_sin: np.ndarray,
    cam_rel_pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-compute reusable camera transforms.

    Returns (R_world_to_base, R_base_to_cam, cam_pos).
    """
    euler = np.arctan2(agent_ori_sin, agent_ori_cos)
    R_base_to_world = _euler_to_rotmat(euler[0], euler[1], euler[2])
    R_world_to_base = R_base_to_world.T
    cam_pos = cam_rel_pose[:3].astype(np.float64)
    cam_quat = cam_rel_pose[3:7].astype(np.float64)
    R_cam_to_base = _quat_to_rotmat(cam_quat) @ _RX_180
    R_base_to_cam = R_cam_to_base.T
    return R_world_to_base, R_base_to_cam, cam_pos


def _project_to_normalized(
    world_pos: np.ndarray,
    agent_pos: np.ndarray,
    agent_ori_cos: np.ndarray,
    agent_ori_sin: np.ndarray,
    cam_rel_pose: np.ndarray,
) -> tuple[float, float, bool]:
    """Project a 3D world point to normalized (0–1) pixel coordinates.

    Returns (norm_x, norm_y, visible). Coordinates are in [0, 1] range where
    (0, 0) is top-left. visible=False if behind camera or outside frame.
    """
    R_w2b, R_b2c, cam_pos = _build_cam_transforms(
        agent_ori_cos, agent_ori_sin, cam_rel_pose,
    )
    p_cam = _world_to_cam(world_pos, agent_pos, R_w2b, R_b2c, cam_pos)

    if p_cam[2] <= 0.05:
        return 0.0, 0.0, False

    p_proj = _HEAD_K @ p_cam
    norm_x = (p_proj[0] / p_proj[2]) / _HEAD_NATIVE_W
    norm_y = (p_proj[1] / p_proj[2]) / _HEAD_NATIVE_H

    visible = 0.0 <= norm_x <= 1.0 and 0.0 <= norm_y <= 1.0
    return float(np.clip(norm_x, 0, 1)), float(np.clip(norm_y, 0, 1)), visible


def _project_bbox_to_normalized(
    obj_pos: np.ndarray,
    obj_ori_cos: np.ndarray,
    obj_ori_sin: np.ndarray,
    half_extents: np.ndarray,
    agent_pos: np.ndarray,
    agent_ori_cos: np.ndarray,
    agent_ori_sin: np.ndarray,
    cam_rel_pose: np.ndarray,
) -> tuple[float, float, float, float, bool]:
    """Project an object's 3D AABB to a 2D bounding box in normalized coords.

    Constructs 8 OBB corners from the object center + orientation + half-extents,
    projects each to 2D, and returns the tightest enclosing axis-aligned 2D box.

    Returns (cx, cy, w, h, visible), all in [0, 1]. visible=False if center
    is behind camera or the 2D box is entirely outside the frame.
    """
    # Build object rotation from its orientation
    obj_euler = np.arctan2(obj_ori_sin, obj_ori_cos)
    R_obj = _euler_to_rotmat(obj_euler[0], obj_euler[1], obj_euler[2])

    # 8 OBB corners in world space
    signs = np.array([
        [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
        [1, -1, -1],  [1, -1, 1],  [1, 1, -1],  [1, 1, 1],
    ], dtype=np.float64)
    corners_local = signs * half_extents[None, :]
    corners_world = (R_obj @ corners_local.T).T + obj_pos[None, :]

    # Camera transforms (computed once)
    R_w2b, R_b2c, cam_pos = _build_cam_transforms(
        agent_ori_cos, agent_ori_sin, cam_rel_pose,
    )

    # Project each corner
    nxs, nys = [], []
    for corner in corners_world:
        p_cam = _world_to_cam(corner, agent_pos, R_w2b, R_b2c, cam_pos)
        if p_cam[2] <= 0.05:
            continue
        p_proj = _HEAD_K @ p_cam
        nx = (p_proj[0] / p_proj[2]) / _HEAD_NATIVE_W
        ny = (p_proj[1] / p_proj[2]) / _HEAD_NATIVE_H
        nxs.append(nx)
        nys.append(ny)

    if len(nxs) < 2:
        return 0.0, 0.0, 0.0, 0.0, False

    x_min, x_max = float(np.clip(min(nxs), 0, 1)), float(np.clip(max(nxs), 0, 1))
    y_min, y_max = float(np.clip(min(nys), 0, 1)), float(np.clip(max(nys), 0, 1))

    w = x_max - x_min
    h = y_max - y_min
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0

    visible = w > 0.005 and h > 0.005 and 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
    return cx, cy, w, h, visible


@dataclasses.dataclass(frozen=True)
class ComputeGroundingLabels(DataTransformFn):
    """Computes 2D grounding labels from privileged 3D sim data.

    Projects radio and table 3D OBBs to normalized head-camera 2D bounding
    boxes using task_info (positions + orientations) and cam_rel_poses.

    Stores in aux_labels:
        grounding_xywh: float32 (8,) = [radio_cx, radio_cy, radio_w, radio_h,
                                         table_cx, table_cy, table_w, table_h]
        grounding_visible: float32 (2,) = [radio_vis, table_vis]

    Must run AFTER B1kInputs (which passes through observation keys) and
    AFTER EncodePhaseLabels (which initializes aux_labels dict).
    """

    def __call__(self, data: DataDict) -> DataDict:
        task_info = data.get("observation/task_info")
        cam_rel_poses = data.get("observation/cam_rel_poses")

        num_obj = len(_OBJ_NAMES)
        grounding_xywh = np.array([0.5, 0.5, 0.0, 0.0] * num_obj, dtype=np.float32)
        grounding_visible = np.zeros(num_obj, dtype=np.float32)

        if task_info is not None and cam_rel_poses is not None:
            task_info = np.asarray(task_info, dtype=np.float64)
            cam_rel_poses = np.asarray(cam_rel_poses, dtype=np.float64)

            agent_pos = task_info[_AGENT_OFFSET + 1 : _AGENT_OFFSET + 4]
            agent_ori_cos = task_info[_AGENT_OFFSET + 4 : _AGENT_OFFSET + 7]
            agent_ori_sin = task_info[_AGENT_OFFSET + 7 : _AGENT_OFFSET + 10]
            head_cam = cam_rel_poses[_CAM_HEAD_OFFSET : _CAM_HEAD_OFFSET + 7]

            objects = [
                (_RADIO_OFFSET, 0, "radio"),
                (_TABLE_OFFSET, 1, "table"),
            ]
            for obj_offset, idx, obj_name in objects:
                obj_pos = task_info[obj_offset + 1 : obj_offset + 4]
                obj_ori_cos = task_info[obj_offset + 4 : obj_offset + 7]
                obj_ori_sin = task_info[obj_offset + 7 : obj_offset + 10]
                half_ext = _OBJ_HALF_EXTENTS[obj_name]

                cx, cy, w, h, vis = _project_bbox_to_normalized(
                    obj_pos, obj_ori_cos, obj_ori_sin, half_ext,
                    agent_pos, agent_ori_cos, agent_ori_sin, head_cam,
                )
                base = idx * 4
                grounding_xywh[base] = cx
                grounding_xywh[base + 1] = cy
                grounding_xywh[base + 2] = w
                grounding_xywh[base + 3] = h
                grounding_visible[idx] = 1.0 if vis else 0.0

        aux_labels = data.get("aux_labels", {})
        if aux_labels is None:
            aux_labels = {}
        aux_labels["grounding_xywh"] = grounding_xywh
        aux_labels["grounding_visible"] = grounding_visible
        data["aux_labels"] = aux_labels
        return data
