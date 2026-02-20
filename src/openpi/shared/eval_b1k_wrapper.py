import numpy as np
import torch
from openpi_client.base_policy import BasePolicy
from openpi_client.image_tools import resize_with_pad
from collections import deque
import copy
import json
import logging

logger = logging.getLogger(__name__)

RESIZE_SIZE = 224

# Warmup: no masking for first N steps (model sees full observations)
_WARMUP_STEPS = 30

# After this many steps, force wrist cameras ON regardless of phase.
# This helps the model see the target object as it approaches, enabling
# a smoother transition from navigation to manipulation.
_UNMASK_WRIST_AFTER = 100

# ── Predicted-action indices (23-dim output) ────────────────────────────
# action = [base(3), torso(4), left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)]
_ACT_BASE = slice(0, 3)
_ACT_TORSO = slice(3, 7)
_ACT_LEFT_ARM = slice(7, 14)
_ACT_LEFT_GRIPPER = slice(14, 15)
_ACT_RIGHT_ARM = slice(15, 22)
_ACT_RIGHT_GRIPPER = slice(22, 23)

# Phase detection thresholds (applied to PREDICTED actions as fallback)
_ACTION_ARM_THRESHOLD = 0.05    # arm action magnitude above this → manipulating
_ACTION_BASE_THRESHOLD = 0.10   # base action magnitude below this → decelerating/stopped
_ACTION_WAS_NAVIGATING = 0.20   # base must have been above this recently to count as "arrived"
_PHASE_SWITCH_PATIENCE = 10     # consecutive decelerated frames (~1s at 10Hz) before transition
_MIN_NAV_STEPS = 50             # minimum steps in navigation before action fallback can trigger
_BASE_HISTORY_WINDOW = 30       # look-back window for "was actively navigating" check
_MAX_BASE_CORRECTION = 0.05     # clamp base actions during manipulation (~10% of nav speed)

# Grounding visibility gate for phase transitions
_RADIO_VIS_THRESHOLD = 0.50     # radio must be >= this confidence before nav→manip transition
_RADIO_BBOX_MIN_SIZE = 0.08     # radio bbox (max of w,h) must be >= this fraction of image to transition
_NAV_SLOWDOWN_VIS = 0.30        # start slowing down when radio visibility exceeds this
_NAV_MIN_SPEED_SCALE = 0.3      # minimum speed scale when radio is fully visible and close

# Gripper hysteresis filter — prevents flapping by requiring sustained intent
_GRIPPER_OPEN_VALUE = 1.0       # action value for open gripper
_GRIPPER_CLOSE_THRESHOLD = 0.0  # below this → model wants to close
_GRIPPER_OPEN_THRESHOLD = 0.0   # above this → model wants to open
_GRIPPER_PATIENCE = 5           # consecutive frames of intent before switching

# Skill type class indices (must match EncodePhaseLabels in b1k_phase_transforms.py)
_SKILL_TYPE_NAVIGATION = 0
_SKILL_TYPE_UNCOORDINATED = 1
_SKILL_TYPE_COORDINATED = 2

# Phase index mapping for turning_on_radio
_PHASE_INDEX_TO_DESC = {0: "move to", 1: "pick up from", 2: "press", 3: "place on"}
_PHASE_INDEX_TO_TYPE = {0: "navigation", 1: "uncoordinated", 2: "coordinated", 3: "uncoordinated"}


# ═══════════════════════════════════════════════════════════════════════════
# Task Phase Sequencer — driven by MODEL OUTPUT ACTIONS
# ═══════════════════════════════════════════════════════════════════════════

# Simplified 2-phase system: navigation → manipulation.
# All manipulation sub-phases (pick up, press, place on) are treated as one.
TURNING_ON_RADIO_PHASES = [
    {
        "skill_description": "move to",
        "skill_type": "navigation",
        "skill_objects": ["radio"],
    },
    {
        "skill_description": "manipulate",
        "skill_type": "uncoordinated",
        "skill_objects": ["radio"],
    },
]


class ModelDrivenPhaseDetector:
    """Detects phase transitions using the model's own outputs.

    Two signals, in priority order:

    1. **Auxiliary head predictions** (primary, when available):
       The trained aux head classifies the current phase directly from VLM
       features. This is the model's learned understanding of "what phase am I in?"
       based on what it SEES (images + text). Most reliable after the aux head
       is trained. For sub-phase transitions (1→2→3) where all involve arm
       movement, the phase_index head can distinguish pick_up vs press vs place_on.

    2. **Predicted action analysis** (fallback):
       If aux heads are not available (model not yet trained with them), analyze
       the model's predicted action vector. If it predicts arm movement, it
       intends to manipulate; if base movement, it intends to navigate.

    The phase SEQUENCE is known from training annotations. Only the TIMING
    of transitions is driven by model output.
    """

    def __init__(self, phases: list[dict], total_budget: int = 8000):
        self.phases = phases
        self.total_budget = total_budget
        self.current_idx = 0
        self.steps_in_phase = 0
        self._total_steps = 0
        self._aux_candidate_count = 0
        self._action_candidate_count = 0
        self._aux_available = False  # set to True when first aux prediction arrives
        self._base_history: list[float] = []  # recent base magnitudes for arrival detection

        # No timeouts — phase transitions are driven purely by aux heads
        # and action analysis. This avoids forced transitions that mask
        # whether the model actually learned phase-discriminative features.
        self.phase_durations = [None] * len(phases)

        logger.info("=== Model-Driven Phase Detector ===")
        logger.info("  Primary: aux head predictions (learned phase classifier)")
        logger.info("  Fallback: predicted action magnitudes (NO timeouts)")
        for i, p in enumerate(phases):
            logger.info("  Phase %d: %-15s (%s) — no timeout",
                        i, p["skill_description"], p["skill_type"])

    @property
    def current_phase(self) -> dict:
        return self.phases[min(self.current_idx, len(self.phases) - 1)]

    @property
    def is_navigation(self) -> bool:
        return self.current_phase["skill_type"] == "navigation"

    def step(
        self,
        predicted_actions: np.ndarray | None = None,
        aux_predictions: dict | None = None,
    ) -> dict:
        """Advance by one step using model outputs for phase transition detection.

        Args:
            predicted_actions: Model's predicted action chunk [horizon, 23] or [23].
            aux_predictions: Dict from model.predict_phase() with keys like
                "skill_type" (int), "phase_index" (int), and their logits.
                When available, this is the PRIMARY signal.

        Returns:
            Phase metadata dict with skill_description, skill_type, skill_objects.
        """
        self.steps_in_phase += 1
        self._total_steps += 1

        if predicted_actions is None and aux_predictions is None:
            return self.current_phase

        # Track whether aux heads are producing predictions
        if aux_predictions and "phase_index" in aux_predictions:
            self._aux_available = True

        should_advance = False
        trigger_reason = ""

        # ── Primary: Aux head predictions ──
        if self._aux_available and aux_predictions and "phase_index" in aux_predictions:
            aux_phase_idx = int(aux_predictions["phase_index"])

            # The aux head predicts the phase index directly (0-3).
            # If it consistently predicts a LATER phase than current, advance.
            if aux_phase_idx > self.current_idx:
                self._aux_candidate_count += 1
            else:
                self._aux_candidate_count = max(0, self._aux_candidate_count - 1)

            if self._aux_candidate_count >= _PHASE_SWITCH_PATIENCE:
                # Jump to whatever phase the aux head predicts
                # (can skip phases if the model is confident)
                should_advance = True
                trigger_reason = (
                    f"AUX HEAD predicts phase {aux_phase_idx} "
                    f"('{_PHASE_INDEX_TO_DESC.get(aux_phase_idx, '?')}')"
                )

        # ── Fallback: predicted action analysis (no timeouts) ──
        # Detect "arrival": robot was actively navigating, then came to a
        # sustained stop with arm intent.  Three conditions must ALL hold:
        #   1. Minimum navigation steps elapsed
        #   2. Robot WAS navigating recently (high base in look-back window)
        #   3. Robot has NOW stopped (low base) and predicts arm actions
        if not should_advance:
            if (predicted_actions is not None
                    and self.current_idx == 0
                    and self.steps_in_phase >= _MIN_NAV_STEPS):
                act = predicted_actions[0] if predicted_actions.ndim == 2 else predicted_actions
                arm_action = (
                    np.linalg.norm(act[_ACT_LEFT_ARM])
                    + np.linalg.norm(act[_ACT_RIGHT_ARM])
                    + np.abs(act[_ACT_LEFT_GRIPPER]).sum()
                    + np.abs(act[_ACT_RIGHT_GRIPPER]).sum()
                )
                base_action = np.linalg.norm(act[_ACT_BASE])

                # Track recent base magnitudes
                self._base_history.append(float(base_action))
                if len(self._base_history) > _BASE_HISTORY_WINDOW:
                    self._base_history = self._base_history[-_BASE_HISTORY_WINDOW:]

                # Was the robot actively navigating in the recent window?
                was_navigating = max(self._base_history) >= _ACTION_WAS_NAVIGATING

                # Is it now stopped with arm intent?
                is_stopped_with_arm = (
                    base_action < _ACTION_BASE_THRESHOLD
                    and arm_action > _ACTION_ARM_THRESHOLD
                )

                if was_navigating and is_stopped_with_arm:
                    self._action_candidate_count += 1
                else:
                    self._action_candidate_count = max(0, self._action_candidate_count - 1)

                if self._action_candidate_count >= _PHASE_SWITCH_PATIENCE:
                    should_advance = True
                    recent_max_base = max(self._base_history)
                    trigger_reason = (
                        f"ARRIVAL: arm={arm_action:.4f} base={base_action:.4f} "
                        f"(was navigating: max_base={recent_max_base:.4f})"
                    )

        # ── Grounding visibility gate ──
        # Block nav→manip transition unless the radio is actually visible
        # AND close enough (bbox large enough to manipulate).
        if should_advance and self.current_idx == 0:
            radio_vis = 0.0
            radio_size = 0.0
            if aux_predictions and "grounding_vis" in aux_predictions:
                radio_vis = float(aux_predictions["grounding_vis"][0])
            if aux_predictions and "grounding_bbox" in aux_predictions:
                bbox = aux_predictions["grounding_bbox"]
                radio_size = max(float(bbox[2]), float(bbox[3]))  # max(w, h)
            blocked = False
            if radio_vis < _RADIO_VIS_THRESHOLD:
                blocked = True
                reason = f"radio_vis={radio_vis:.0%} < {_RADIO_VIS_THRESHOLD:.0%}"
            elif radio_size < _RADIO_BBOX_MIN_SIZE:
                blocked = True
                reason = f"radio_size={radio_size:.3f} < {_RADIO_BBOX_MIN_SIZE} (too far)"
            if blocked:
                should_advance = False
                if self._total_steps % 50 == 0:
                    logger.info("  BLOCKED nav→manip: %s", reason)

        # ── Execute transition ──
        if should_advance and self.current_idx < len(self.phases) - 1:
            old_idx = self.current_idx

            # If aux head triggered the advance AND predicts a later phase, jump to it.
            # Otherwise (timeout or action-based trigger), always go to next phase.
            if "AUX HEAD" in trigger_reason and aux_predictions and "phase_index" in aux_predictions:
                target = int(aux_predictions["phase_index"])
                self.current_idx = min(max(target, self.current_idx + 1), len(self.phases) - 1)
            else:
                self.current_idx += 1

            old_phase = self.phases[old_idx]
            self.steps_in_phase = 0
            self._aux_candidate_count = 0
            self._action_candidate_count = 0
            self._base_history.clear()
            new_phase = self.current_phase

            logger.info(
                "Phase %d→%d: '%s' (%s) → '%s' (%s) [%s]",
                old_idx, self.current_idx,
                old_phase["skill_description"], old_phase["skill_type"],
                new_phase["skill_description"], new_phase["skill_type"],
                trigger_reason,
            )

        return self.current_phase

    def get_log_dict(self) -> dict:
        budget = self.phase_durations[min(self.current_idx, len(self.phase_durations) - 1)]
        return {
            "phase_idx": self.current_idx,
            "phase_desc": self.current_phase["skill_description"],
            "phase_type": self.current_phase["skill_type"],
            "steps_in_phase": self.steps_in_phase,
            "budget": budget if budget is not None else "inf",
            "aux_available": self._aux_available,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Policy Wrapper
# ═══════════════════════════════════════════════════════════════════════════

class B1KPolicyWrapper():
    def __init__(
        self, 
        policy: BasePolicy,
        text_prompt : str = "Turn on the radio receiver that's on the table in the living room.",
        control_mode : str = "temporal_ensemble",
        action_horizon : int = 10,
        enable_phase_conditioning: bool = True,
        eval_budget: int = 8000,
    ) -> None:
        self.policy = policy
        self.text_prompt = text_prompt
        self.control_mode = control_mode
        self.action_queue = deque([], maxlen=action_horizon)
        self.last_action = {"actions": np.zeros((action_horizon, 23), dtype=np.float64)}
        self.action_horizon = action_horizon
        self.enable_phase_conditioning = enable_phase_conditioning
        
        self.replan_interval = action_horizon
        self.max_len = 50
        self.temporal_ensemble_max = 5
        self.step_counter = 0

        # Model-driven phase detector: uses aux head predictions (primary)
        # or predicted action magnitudes (fallback) for phase transitions.
        if enable_phase_conditioning:
            self.phase_detector = ModelDrivenPhaseDetector(
                TURNING_ON_RADIO_PHASES, total_budget=eval_budget
            )
        else:
            self.phase_detector = None

        # Track last predicted actions for phase detection on the next step
        self._last_predicted_actions = None
        # Track last aux head predictions
        self._last_aux_predictions = None

        self._log_interval = 50
        self._phase_history = []

        # Gripper hysteresis state: grippers start open
        self._gripper_state = [_GRIPPER_OPEN_VALUE, _GRIPPER_OPEN_VALUE]  # [left, right]
        self._gripper_close_count = [0, 0]  # consecutive "close" predictions
        self._gripper_open_count = [0, 0]   # consecutive "open" predictions
    
    def reset(self):
        self.action_queue = deque([],maxlen=self.action_horizon)
        self.last_action = {"actions": np.zeros((self.action_horizon, 23), dtype=np.float64)}
        self.step_counter = 0
        self._last_predicted_actions = None
        self._last_aux_predictions = None
        self._phase_history = []
        self._gripper_state = [_GRIPPER_OPEN_VALUE, _GRIPPER_OPEN_VALUE]
        self._gripper_close_count = [0, 0]
        self._gripper_open_count = [0, 0]
        if self.phase_detector:
            self.phase_detector = ModelDrivenPhaseDetector(
                TURNING_ON_RADIO_PHASES, total_budget=self.phase_detector.total_budget
            )

    def _grounding_log_dict(self) -> dict:
        """Extract grounding predictions from last aux output for logging."""
        aux = self._last_aux_predictions
        if not aux or "grounding_vis" not in aux:
            return {}
        vis = aux["grounding_vis"]
        bbox = aux["grounding_bbox"]
        return {
            "radio_vis": float(vis[0]),
            "radio_cx": float(bbox[0]), "radio_cy": float(bbox[1]),
            "radio_w": float(bbox[2]), "radio_h": float(bbox[3]),
            "table_vis": float(vis[1]),
            "table_cx": float(bbox[4]), "table_cy": float(bbox[5]),
            "table_w": float(bbox[6]), "table_h": float(bbox[7]),
        }

    def _inject_phase_metadata(self, batch: dict) -> dict:
        """Inject phase metadata into the batch dict using model's own outputs.

        Signal priority:
            1. Aux head predictions (learned phase classifier on VLM features)
            2. Predicted action magnitudes (arm vs base activity)

        The phase detection loop is:
            Step N: model predicts actions + aux heads classify phase
            Step N+1: we inject that phase into the batch → transforms apply masking
            Step N+1: model predicts with phase-appropriate inputs → repeat

        This 1-step delay is harmless because phases last hundreds of steps.

        During warmup (first N steps), no metadata is injected — the model sees
        full unmasked observations; its outputs bootstrap the detector.
        """
        if not self.enable_phase_conditioning or self.phase_detector is None:
            return batch

        # Warmup: let the model see full observations and start producing actions.
        # We still feed actions/aux to the detector for tracking, but don't inject phase.
        if self.step_counter < _WARMUP_STEPS:
            phase_info = self.phase_detector.step(
                self._last_predicted_actions, self._last_aux_predictions
            )
            if self.step_counter % self._log_interval == 0:
                aux_str = ""
                if self._last_aux_predictions:
                    aux_str = f" | aux_phase={self._last_aux_predictions.get('phase_index', '?')}"
                logger.info(
                    "[Step %4d] WARMUP (no masking) | detector at '%s' (%s)%s",
                    self.step_counter,
                    phase_info["skill_description"],
                    phase_info["skill_type"],
                    aux_str,
                )
            log_entry = {
                "step": self.step_counter,
                "warmup": True,
                **self.phase_detector.get_log_dict(),
            }
            log_entry.update(self._grounding_log_dict())
            self._phase_history.append(log_entry)
            return batch

        # Normal operation: feed both signals to detector, inject phase
        phase_info = self.phase_detector.step(
            self._last_predicted_actions, self._last_aux_predictions
        )

        batch["skill_type"] = phase_info["skill_type"]
        batch["skill_description"] = phase_info["skill_description"]
        batch["skill_objects"] = phase_info["skill_objects"]

        # After N steps, force wrist cameras ON even during navigation.
        # This lets the model see the target object as it approaches,
        # giving aux heads and action predictions visual context for transition.
        if self.step_counter >= _UNMASK_WRIST_AFTER and phase_info["skill_type"] == "navigation":
            batch["skill_type"] = "uncoordinated"  # wrist cams ON, but keep nav prompt

        # Periodic log
        if self.step_counter % self._log_interval == 0:
            effective_type = batch.get("skill_type", phase_info["skill_type"])
            wrist = "OFF" if effective_type == "navigation" else "ON"
            proprio_zeroed = "arms+grip" if phase_info["skill_type"] == "navigation" else "base"
            log_d = self.phase_detector.get_log_dict()
            budget_str = str(log_d["budget"]) if log_d["budget"] != "inf" else "inf"

            # Log predicted actions info
            act_info = ""
            if self._last_predicted_actions is not None:
                act = self._last_predicted_actions
                if act.ndim == 2:
                    act = act[0]
                base_mag = np.linalg.norm(act[_ACT_BASE])
                arm_mag = np.linalg.norm(act[_ACT_LEFT_ARM]) + np.linalg.norm(act[_ACT_RIGHT_ARM])
                grip_l = float(act[14])
                grip_r = float(act[22])
                act_info = (f" | pred_action: base={base_mag:.4f} arm={arm_mag:.4f}"
                            f" grip_L={grip_l:.3f} grip_R={grip_r:.3f}")

            # Log aux head predictions if available
            aux_info = ""
            if self._last_aux_predictions and "phase_index" in self._last_aux_predictions:
                aux_phase = int(self._last_aux_predictions["phase_index"])
                aux_type = int(self._last_aux_predictions.get("skill_type", -1))
                aux_info = (
                    f" | AUX: phase={aux_phase}"
                    f"('{_PHASE_INDEX_TO_DESC.get(aux_phase, '?')}') "
                    f"type={aux_type}('{_PHASE_INDEX_TO_TYPE.get(aux_phase, '?')}')"
                )

            logger.info(
                "[Step %4d] phase=%d:'%-15s' (%s) | step_in_phase=%d/%s | "
                "wrist=%s | proprio_zeroed=%s%s%s",
                self.step_counter,
                log_d["phase_idx"], log_d["phase_desc"], log_d["phase_type"],
                log_d["steps_in_phase"], budget_str,
                wrist, proprio_zeroed, act_info, aux_info,
            )

        log_entry = {
            "step": self.step_counter,
            "warmup": False,
            **self.phase_detector.get_log_dict(),
        }
        log_entry.update(self._grounding_log_dict())
        self._phase_history.append(log_entry)
        return batch

    def process_obs(self, obs: dict) -> dict:
        """
        Process the observation dictionary to match the expected input format for the model.
        """
        prop_state = obs["robot_r1::proprio"][None]
        img_obs = np.stack(
            [
                resize_with_pad(
                    obs["robot_r1::robot_r1:zed_link:Camera:0::rgb"][None, ..., :3],
                    RESIZE_SIZE,
                    RESIZE_SIZE
                ),
                resize_with_pad(
                    obs["robot_r1::robot_r1:left_realsense_link:Camera:0::rgb"][None, ..., :3], 
                    RESIZE_SIZE,
                    RESIZE_SIZE
                ),
                resize_with_pad(
                    obs["robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"][None, ..., :3],
                    RESIZE_SIZE,
                    RESIZE_SIZE
                ),
            ],
            axis=1,
        )
        processed_obs = {
            "observation": img_obs,  # Shape: (1, 3, H, W, C)
            "proprio": prop_state,
            "prompt": self.text_prompt,
        }
        return processed_obs
    
    def _filter_grippers(self, action: np.ndarray) -> np.ndarray:
        """Apply hysteresis filter to gripper actions to prevent flapping.

        During navigation: force grippers open.
        During manipulation: require N consecutive frames of sustained
        close/open intent before switching gripper state.
        """
        action = np.array(action, copy=True)
        gripper_indices = [14, 22]  # left_gripper, right_gripper

        # During navigation, force grippers open
        if (self.phase_detector is not None
                and self.phase_detector.is_navigation):
            for idx in gripper_indices:
                action[..., idx] = _GRIPPER_OPEN_VALUE
            return action

        # During manipulation, apply hysteresis
        raw_vals = []
        for i, idx in enumerate(gripper_indices):
            raw_value = float(action.flat[idx] if action.ndim == 1
                              else action[..., idx].flat[0])
            raw_vals.append(raw_value)

            if self._gripper_state[i] == _GRIPPER_OPEN_VALUE:
                # Currently open — count consecutive close predictions
                if raw_value < _GRIPPER_CLOSE_THRESHOLD:
                    self._gripper_close_count[i] += 1
                else:
                    self._gripper_close_count[i] = 0

                if self._gripper_close_count[i] >= _GRIPPER_PATIENCE:
                    side = "LEFT" if i == 0 else "RIGHT"
                    print(f"[gripper] {side} CLOSE at step {self.step_counter}")
                    self._gripper_state[i] = -1.0  # close
                    self._gripper_close_count[i] = 0
                    self._gripper_open_count[i] = 0
            else:
                # Currently closed — count consecutive open predictions
                if raw_value > _GRIPPER_OPEN_THRESHOLD:
                    self._gripper_open_count[i] += 1
                else:
                    self._gripper_open_count[i] = 0

                if self._gripper_open_count[i] >= _GRIPPER_PATIENCE:
                    side = "LEFT" if i == 0 else "RIGHT"
                    print(f"[gripper] {side} OPEN at step {self.step_counter}")
                    self._gripper_state[i] = _GRIPPER_OPEN_VALUE  # open
                    self._gripper_open_count[i] = 0
                    self._gripper_close_count[i] = 0

            action[..., idx] = self._gripper_state[i]

        if self.step_counter % 50 == 0:
            print(f"[gripper-debug] step={self.step_counter} "
                  f"raw_L={raw_vals[0]:.4f} raw_R={raw_vals[1]:.4f} "
                  f"state_L={self._gripper_state[0]:.1f} state_R={self._gripper_state[1]:.1f} "
                  f"close_cnt=[{self._gripper_close_count[0]},{self._gripper_close_count[1]}]")

        return action

    def _clamp_base_if_manipulating(self, action: np.ndarray) -> np.ndarray:
        """Clamp base action dims during manipulation to small corrections only.

        Reads raw predicted actions for phase detection (unclamped), but the
        executed actions are clamped so the robot can only nudge its base
        during manipulation, not navigate.
        """
        if (self.phase_detector is not None
                and not self.phase_detector.is_navigation):
            action = np.array(action, copy=True)
            action[..., :3] = np.clip(action[..., :3],
                                      -_MAX_BASE_CORRECTION, _MAX_BASE_CORRECTION)
        return action

    def _scale_nav_speed(self, action: np.ndarray) -> np.ndarray:
        """Slow down base movement when the radio is visible and close.

        Uses radio visibility as a proxy for proximity. Linearly ramps
        speed from 100% (vis <= _NAV_SLOWDOWN_VIS) to _NAV_MIN_SPEED_SCALE
        (vis = 1.0). Only active during navigation phase.
        """
        if self.phase_detector is None or not self.phase_detector.is_navigation:
            return action
        aux = self._last_aux_predictions
        if not aux or "grounding_vis" not in aux:
            return action
        radio_vis = float(aux["grounding_vis"][0])
        if radio_vis <= _NAV_SLOWDOWN_VIS:
            return action
        t = (radio_vis - _NAV_SLOWDOWN_VIS) / (1.0 - _NAV_SLOWDOWN_VIS)
        scale = 1.0 - t * (1.0 - _NAV_MIN_SPEED_SCALE)
        action = np.array(action, copy=True)
        action[..., :3] *= scale
        return action

    def _store_predicted_actions(self, action_dict: dict, obs_batch: dict | None = None):
        """Store the model's raw predicted actions AND run aux heads for phase detection.

        After the main inference call (sample_actions), we separately run the
        lightweight aux heads (predict_phase) on the same observation.  This is
        done outside of JIT and costs ~10% of a full inference step.

        Args:
            action_dict: Output of policy.infer() — contains "actions".
            obs_batch: The same observation dict that was passed to policy.infer().
                Used to run predict_phase(). If None, aux predictions are skipped.
        """
        actions = action_dict.get("actions", None)
        if actions is not None:
            self._last_predicted_actions = np.asarray(actions)

        # Run aux heads for phase prediction (primary signal for phase detector)
        # ANTI-CHEAT: Strip the phase suffix from the prompt so aux heads can't
        # trivially read the answer from the text tokens. Forces them to rely on
        # visual features only.
        if obs_batch is not None and hasattr(self.policy, "predict_phase"):
            try:
                # Build a clean obs copy without phase info in the prompt
                aux_batch = copy.deepcopy(obs_batch)
                if "prompt" in aux_batch:
                    prompt = aux_batch["prompt"]
                    if isinstance(prompt, str) and ", Phase:" in prompt:
                        aux_batch["prompt"] = prompt.split(", Phase:")[0]
                # Remove skill fields so PhaseConditionedPrompt transform
                # (which runs inside predict_phase → _input_transform) can't
                # re-inject them.
                for k in ("skill_type", "skill_description", "skill_objects",
                           "skill_phase", "skill_annotation"):
                    aux_batch.pop(k, None)

                aux_preds = self.policy.predict_phase(aux_batch)
                if aux_preds:
                    self._last_aux_predictions = aux_preds
                    if self.step_counter % self._log_interval == 0:
                        phase_idx = int(aux_preds.get("phase_index", -1))
                        skill_type = int(aux_preds.get("skill_type", -1))
                        grounding_str = ""
                        if "grounding_vis" in aux_preds and "grounding_bbox" in aux_preds:
                            vis = aux_preds["grounding_vis"]
                            bbox = aux_preds["grounding_bbox"]
                            obj_names = ["radio", "table"]
                            parts = []
                            for i, name in enumerate(obj_names):
                                v = float(vis[i])
                                cx, cy = float(bbox[i * 4]), float(bbox[i * 4 + 1])
                                parts.append(f"{name}={v:.0%}@({cx:.2f},{cy:.2f})")
                            grounding_str = " | " + " ".join(parts)
                        logger.info(
                            "[Step %4d] Aux head: phase=%d skill=%d%s",
                            self.step_counter, phase_idx, skill_type, grounding_str,
                        )
            except Exception as e:
                logger.debug("Aux head predict_phase failed (non-fatal): %s", e)

    def act_receeding_temporal(self, input_obs):
        # Step 1: check if we should re-run policy
        if self.step_counter % self.replan_interval == 0:
            # Run policy every K steps
            nbatch = copy.deepcopy(input_obs)
            nbatch["observation"] = nbatch["observation"][:, -1]
            if nbatch["observation"].shape[-1] != 3:
                nbatch["observation"] = np.transpose(nbatch["observation"], (0, 1, 3, 4, 2))

            joint_positions = nbatch["proprio"][0, -1]
            batch = {
                "observation/egocentric_camera": nbatch["observation"][0, 0],
                "observation/wrist_image_left": nbatch["observation"][0, 1],
                "observation/wrist_image_right": nbatch["observation"][0, 2],
                "observation/state": joint_positions,
                "prompt": self.text_prompt,
            }

            # Inject phase metadata based on LAST predicted actions
            batch = self._inject_phase_metadata(batch)

            try:
                action = self.policy.infer(batch)
                self.last_action = action
                # Store predicted actions + run aux heads for phase detection
                self._store_predicted_actions(action, obs_batch=batch)
            except Exception as e:
                action = self.last_action
                print(f"Error in action prediction, using last action: {e}")

            target_joint_positions = action["actions"].copy()

            # Add this sequence to action queue
            new_seq = deque([a for a in target_joint_positions[:self.max_len]])
            self.action_queue.append(new_seq)

            # Optional: limit memory
            while len(self.action_queue) > self.temporal_ensemble_max:
                self.action_queue.popleft()

        # Step 2: Smooth across current step from all stored sequences
        if len(self.action_queue) == 0:
            raise ValueError("Action queue empty in receeding_temporal mode.")

        actions_current_timestep = np.empty((len(self.action_queue), self.action_queue[0][0].shape[0]))

        for i in range(len(self.action_queue)):
            actions_current_timestep[i] = self.action_queue[i].popleft()

        # Drop exhausted sequences
        self.action_queue = deque([q for q in self.action_queue if len(q) > 0])

        # Apply temporal ensemble
        k = 0.005
        exp_weights = np.exp(k * np.arange(actions_current_timestep.shape[0]))
        exp_weights = exp_weights / exp_weights.sum()

        final_action = (actions_current_timestep * exp_weights[:, None]).sum(axis=0)

        # Preserve grippers from most recent rollout
        final_action[-9] = actions_current_timestep[0, -9]
        final_action[-1] = actions_current_timestep[0, -1]
        final_action = final_action[None]

        final_action = self._clamp_base_if_manipulating(final_action)
        final_action = self._scale_nav_speed(final_action)
        final_action = self._filter_grippers(final_action)

        self.step_counter += 1

        return final_action


    def act(self, input_obs):
        """
        Model input expected: 
            📌 Key: observation/exterior_image_1_left
            Type: ndarray, Dtype: uint8, Shape: (224, 224, 3)
            📌 Key: observation/joint_position
            Type: ndarray, Dtype: float64, Shape: (16,)
            📌 Key: prompt
            Type: str
        Model will output:
            📌 Key: actions
            Type: ndarray, Dtype: float64, Shape: (10, 16)
        """
        input_obs = self.process_obs(input_obs)
        if self.control_mode == 'receeding_temporal':
            return self.act_receeding_temporal(input_obs)
        
        if self.control_mode == 'receeding_horizon':
            if len(self.action_queue) > 0:
                # pop the first action in the queue
                final_action = self.action_queue.popleft()[None]
                final_action = self._clamp_base_if_manipulating(final_action)
                final_action = self._scale_nav_speed(final_action)
                final_action = self._filter_grippers(final_action)
                return torch.from_numpy(final_action)
        
        nbatch = copy.deepcopy(input_obs)
        if nbatch["observation"].shape[-1] != 3: 
            nbatch["observation"] = np.transpose(nbatch["observation"], (0, 1, 3, 4, 2))

        joint_positions = nbatch["proprio"][0]
        batch = {
            "observation/egocentric_camera": nbatch["observation"][0, 0],
            "observation/wrist_image_left": nbatch["observation"][0, 1],
            "observation/wrist_image_right": nbatch["observation"][0, 2],
            "observation/state": joint_positions,
            "prompt": self.text_prompt,
        }

        # Inject phase metadata based on LAST predicted actions
        batch = self._inject_phase_metadata(batch)

        try:
            action = self.policy.infer(batch) 
            self.last_action = action
            # Store predicted actions + run aux heads for phase detection
            self._store_predicted_actions(action, obs_batch=batch)
        except Exception as e:
            action = self.last_action
            raise e

        target_joint_positions = action["actions"].copy() 
        if self.control_mode == 'receeding_horizon':
            self.action_queue = deque([a for a in target_joint_positions[:self.max_len]])
            final_action = self.action_queue.popleft()[None]

        elif self.control_mode == 'temporal_ensemble':
            new_actions = deque(target_joint_positions)
            self.action_queue.append(new_actions)
            actions_current_timestep = np.empty((len(self.action_queue), target_joint_positions.shape[1]))
            
            k = 0.005
            for i, q in enumerate(self.action_queue):
                actions_current_timestep[i] = q.popleft()

            exp_weights = np.exp(k * np.arange(actions_current_timestep.shape[0]))
            exp_weights = exp_weights / exp_weights.sum()

            final_action = (actions_current_timestep * exp_weights[:, None]).sum(axis=0)
            final_action[-9] = target_joint_positions[0, -9]
            final_action[-1] = target_joint_positions[0, -1]
            final_action = final_action[None]
        else:
            final_action = target_joint_positions

        final_action = self._clamp_base_if_manipulating(final_action)
        final_action = self._scale_nav_speed(final_action)
        final_action = self._filter_grippers(final_action)

        self.step_counter += 1
        return torch.from_numpy(final_action)

    def get_phase_history(self) -> list:
        """Return the phase detection history for post-processing/visualization."""
        return self._phase_history.copy()

    def save_phase_log(self, path: str) -> None:
        """Save the phase history to a JSON file for post-processing."""
        log = {
            "text_prompt": self.text_prompt,
            "phase_conditioning_enabled": self.enable_phase_conditioning,
            "detection_method": "model_driven (aux_head + action_fallback)",
            "aux_head_available": self.phase_detector._aux_available if self.phase_detector else False,
            "eval_budget": self.phase_detector.total_budget if self.phase_detector else None,
            "phase_sequence": [
                {
                    "idx": i,
                    "desc": p["skill_description"],
                    "type": p["skill_type"],
                    "budget": self.phase_detector.phase_durations[i] if self.phase_detector else None,
                }
                for i, p in enumerate(TURNING_ON_RADIO_PHASES)
            ],
            "history": self._phase_history,
        }
        with open(path, "w") as f:
            json.dump(log, f, indent=2)
        logger.info(f"Phase log saved to {path} ({len(self._phase_history)} entries)")
