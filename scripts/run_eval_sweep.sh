#!/usr/bin/env bash
#
# Phase 1: Eval sweep across curriculum stage checkpoints.
#
# For each checkpoint: starts the policy server (openpi venv), runs
# OmniGibson eval (env_isaaclab conda), then stops the server.
#
# Usage:
#   # Run all 4 stages:
#   bash scripts/run_eval_sweep.sh
#
#   # Run a single stage (0-indexed):
#   bash scripts/run_eval_sweep.sh --stage 3
#
#   # Custom max steps and instances:
#   bash scripts/run_eval_sweep.sh --max_steps 800 --instances "0,1,2"
#
# Requirements:
#   - Terminal must NOT have openpi .venv activated (script manages envs)
#   - conda env `env_isaaclab` must exist with Isaac Sim + OmniGibson
#   - openpi .venv must exist at OPENPI_DIR/.venv

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────
PROJ_ROOT="/home/stickbot/projects/behavior"
OPENPI_DIR="$PROJ_ROOT/b1k-baselines/baselines/openpi"
BEHAVIOR_DIR="$PROJ_ROOT/BEHAVIOR-1K"
JOYLO_DIR="$BEHAVIOR_DIR/joylo"
CKPT_BASE="$OPENPI_DIR/outputs/checkpoints"
EVAL_LOG_BASE="$PROJ_ROOT/eval_logs/v3_sweep"
CONDA_SH="/home/stickbot/miniconda3/etc/profile.d/conda.sh"

# ── Defaults ──────────────────────────────────────────────────────────
MAX_STEPS=1500
INSTANCES="0,1,2,3,4,5,6,7,8,9"
SINGLE_STAGE=""
SERVER_PORT=8000
EVAL_BUDGET=8000

# ── Parse args ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)      SINGLE_STAGE="$2"; shift 2 ;;
        --max_steps)  MAX_STEPS="$2"; shift 2 ;;
        --instances)  INSTANCES="$2"; shift 2 ;;
        --port)       SERVER_PORT="$2"; shift 2 ;;
        *)            echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Checkpoint definitions ────────────────────────────────────────────
declare -a STAGE_NAMES=(
    "stage0_nav"
    "stage1_nav_pickup"
    "stage2_grasp"
    "stage3_full_task"
)
declare -a STAGE_CKPTS=(
    "curriculum_stage0_nav/grounding_v3_spatial_stage0_nav/14999"
    "curriculum_stage1_nav_pickup/grounding_v3_spatial_stage1_nav_pickup/14999"
    "curriculum_stage2_grasp/grounding_v3_spatial_stage2_grasp/19999"
    "curriculum_stage3_full_task/grounding_v3_spatial_stage3_full_task/19999"
)

# Format instance list for hydra (e.g. "0,1,2" -> "[0,1,2]")
INSTANCE_LIST="[${INSTANCES}]"

# ── Helper: clean PATH of .venv entries ───────────────────────────────
clean_path() {
    echo "$PATH" | tr ':' '\n' | grep -v '.venv' | tr '\n' ':'
}

# ── Helper: start policy server ───────────────────────────────────────
start_server() {
    local ckpt_dir="$1"
    local log_file="$2"

    echo "  Starting policy server: $ckpt_dir"
    (
        cd "$OPENPI_DIR"
        source .venv/bin/activate
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.65 python scripts/serve_b1k.py \
            --phase-conditioning \
            --port "$SERVER_PORT" \
            --eval-budget "$EVAL_BUDGET" \
            policy:checkpoint \
            --policy.config pi05_b1k_phase_grounding \
            --policy.dir "$ckpt_dir" \
            > "$log_file" 2>&1
    ) &
    SERVER_PID=$!
    echo "  Server PID: $SERVER_PID"

    # Wait for server to be ready
    local timeout=180
    local elapsed=0
    while ! grep -q "server listening on" "$log_file" 2>/dev/null; do
        sleep 3
        elapsed=$((elapsed + 3))
        if [[ $elapsed -ge $timeout ]]; then
            echo "  ERROR: Server did not start within ${timeout}s"
            kill "$SERVER_PID" 2>/dev/null || true
            return 1
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "  ERROR: Server process died. Check $log_file"
            return 1
        fi
    done
    echo "  Server ready (${elapsed}s)"
}

# ── Helper: stop policy server ────────────────────────────────────────
stop_server() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "  Stopping server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    # Force-kill anything still holding port 8000
    local port_pids
    port_pids=$(lsof -ti :"$SERVER_PORT" 2>/dev/null || true)
    if [[ -n "$port_pids" ]]; then
        echo "  Killing leftover processes on port $SERVER_PORT: $port_pids"
        echo "$port_pids" | xargs kill -9 2>/dev/null || true
    fi
    sleep 3
    echo "  Server stopped."
}

# ── Helper: run eval ──────────────────────────────────────────────────
run_eval() {
    local log_path="$1"
    local eval_log="$2"
    local max_steps="$3"

    echo "  Running eval → $log_path  (max_steps=$max_steps)"
    (
        unset VIRTUAL_ENV
        export PATH="$(clean_path)"
        source "$CONDA_SH"
        conda activate env_isaaclab
        export OMNI_KIT_ACCEPT_EULA=YES
        export PYTHONPATH="${JOYLO_DIR}:${PYTHONPATH:-}"
        cd "$BEHAVIOR_DIR"
        python OmniGibson/omnigibson/learning/eval.py \
            policy=websocket \
            task.name=turning_on_radio \
            headless=true \
            log_path="$log_path" \
            eval_on_train_instances=true \
            "eval_instance_ids=${INSTANCE_LIST}" \
            max_steps="$max_steps"
    ) > "$eval_log" 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "  WARNING: eval exited with code $rc. Check $eval_log"
    fi
    return $rc
}

# ── Main loop ─────────────────────────────────────────────────────────
trap 'stop_server; echo "Interrupted."; exit 1' INT TERM

for i in "${!STAGE_NAMES[@]}"; do
    # Skip if --stage specified and doesn't match
    if [[ -n "$SINGLE_STAGE" && "$i" != "$SINGLE_STAGE" ]]; then
        continue
    fi

    stage_name="${STAGE_NAMES[$i]}"
    ckpt_dir="$CKPT_BASE/${STAGE_CKPTS[$i]}"
    log_path="$EVAL_LOG_BASE/$stage_name"
    server_log="$EVAL_LOG_BASE/${stage_name}_server.log"
    eval_log="$EVAL_LOG_BASE/${stage_name}_eval.log"

    echo ""
    echo "================================================================"
    echo "Stage $i: $stage_name"
    echo "  Checkpoint: $ckpt_dir"
    echo "  Output:     $log_path"
    echo "================================================================"

    # Verify checkpoint exists
    if [[ ! -d "$ckpt_dir" ]]; then
        echo "  SKIP: checkpoint not found at $ckpt_dir"
        continue
    fi

    mkdir -p "$log_path" "$EVAL_LOG_BASE"

    # Start server, run eval, stop server
    start_server "$ckpt_dir" "$server_log"
    run_eval "$log_path" "$eval_log" "$MAX_STEPS" || true
    stop_server

    echo "  Done with $stage_name"
done

echo ""
echo "================================================================"
echo "Sweep complete. Results in: $EVAL_LOG_BASE"
echo "================================================================"
echo ""
echo "Next: run Phase 2 (grounding overlays + metrics aggregation):"
echo "  cd $OPENPI_DIR"
echo "  source .venv/bin/activate"
echo "  python scripts/aggregate_eval_sweep.py"
