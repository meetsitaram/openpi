import dataclasses
import enum
import logging
import socket

import tyro

from omnigibson.learning.utils.network_utils import WebsocketPolicyServer
from omnigibson.learning.datas import BehaviorLerobotDatasetMetadata

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Dataset root, used to retrieve the prompt of the task if taskname is not None.
    dataset_root: str | None = "/home/stickbot/projects/behavior/behavior_data"
    # If provided, will be used to retrieve the prompt of the task, otherwise use turning_on_radio as default.
    task_name: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Enable phase-aware conditioning at inference.
    # Detects navigation vs manipulation from robot velocity and injects
    # skill_type into the batch so server-side transforms (camera masking,
    # proprio masking, prompt conditioning) apply correctly.
    phase_conditioning: bool = False

    # Path to save phase detection log (JSON) after serving ends.
    # Useful for overlaying phase annotations on eval videos.
    phase_log_path: str | None = None

    # Total eval step budget. Phase durations are scaled proportionally.
    # Training episodes average ~2000 frames; set higher to give the model more time.
    eval_budget: int = 8000

    # Enable verbose (DEBUG-level) logging for phase transforms.
    # Shows per-step camera mask, proprio mask, and prompt decisions.
    verbose: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    return _policy_config.create_trained_policy(
        _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
    )


def main(args: Args) -> None:
    metadata = BehaviorLerobotDatasetMetadata(
        repo_id="behavior-1k/2025-challenge-demos",
        root=args.dataset_root,
        tasks=[args.task_name] if args.task_name else ["turning_on_radio"],
        modalities=[],
        cameras=[],
    )
    prompt = list(metadata.tasks.values())[0]
    # log the prompt used
    logging.info(f"Using prompt: {prompt}")

    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    logging.info(f"Phase conditioning: {'ENABLED' if args.phase_conditioning else 'DISABLED'}")
    logging.info(f"Eval budget: {args.eval_budget} steps")
    wrapper = B1KPolicyWrapper(
        policy,
        text_prompt=prompt,
        enable_phase_conditioning=args.phase_conditioning,
        eval_budget=args.eval_budget,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = WebsocketPolicyServer(
        policy=wrapper,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )

    try:
        server.serve_forever()
    finally:
        # Save phase log on shutdown (Ctrl+C or eval completion)
        if args.phase_log_path:
            wrapper.save_phase_log(args.phase_log_path)
        elif wrapper.enable_phase_conditioning and wrapper._phase_history:
            # Auto-save to a default location if phases were tracked
            default_path = "phase_detection_log.json"
            wrapper.save_phase_log(default_path)


if __name__ == "__main__":
    args = tyro.cli(Args)
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, force=True)
    # Even in non-verbose mode, ensure our modules log at INFO
    logging.getLogger("openpi.shared.eval_b1k_wrapper").setLevel(logging.INFO)
    logging.getLogger("openpi.policies.b1k_phase_transforms").setLevel(
        logging.DEBUG if args.verbose else logging.WARNING
    )
    main(args)
