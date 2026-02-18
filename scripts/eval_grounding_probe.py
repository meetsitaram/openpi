#!/usr/bin/env python3
"""
Grounding Probe Evaluation.

Loads the SigLIP encoder + grounding MLP directly from a JAX checkpoint
and evaluates how well the grounding head predicts radio/table positions.

This bypasses the LLM text-generation pathway entirely — it tests whether
the SigLIP visual backbone has learned to encode object spatial information
through the grounding auxiliary training signal.

Generates:
- Annotated MP4 showing predicted positions (solid circles) vs GT (crosshairs)
- Per-frame CSV with predicted/GT coordinates and L1 errors
- Console summary with mean L1, median L1, and accuracy within thresholds

Usage
-----
    python scripts/eval_grounding_probe.py --episode 2790 \
        --checkpoint outputs/checkpoints/curriculum_stage0_nav/grounding_v1_stage0_nav/14999/params
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

import cv2
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import pyarrow.parquet as pq

from openpi.models import siglip as _siglip
from openpi.models import model as _model

log = logging.getLogger(__name__)

# ── Dataset paths ──────────────────────────────────────────────────────
DATA_ROOT = Path("/home/stickbot/projects/behavior/behavior_data")
TASK = "task-0000"

# ── task_info offsets ──────────────────────────────────────────────────
AGENT_OFFSET = 0
RADIO_OFFSET = 10
TABLE_OFFSET = 22

# ── Camera ─────────────────────────────────────────────────────────────
CAM_HEAD_OFFSET = 14
HEAD_K = np.array([[306.0, 0.0, 360.0],
                   [0.0, 306.0, 360.0],
                   [0.0,   0.0,   1.0]], dtype=np.float64)
HEAD_NATIVE_W, HEAD_NATIVE_H = 720, 720

# ── 3D projection helpers ─────────────────────────────────────────────
_RX_180 = np.diag([1.0, -1.0, -1.0])

OBJ_DIMENSIONS = {
    "radio": (0.32, 0.24),
    "table": (1.66, 0.41),
}

PRED_COLOURS = {
    "radio": (0, 255, 0),
    "table": (255, 180, 0),
}
GT_COLOURS = {
    "radio": (255, 0, 255),
    "table": (0, 255, 255),
}
OBJ_NAMES = ["radio", "table"]


def _quat_to_rotmat(q):
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _euler_to_rotmat(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


def project_to_normalized(world_pos, agent_pos, agent_ori_cos, agent_ori_sin, cam_rel_pose):
    """Project a 3D world point to normalized [0,1] pixel coordinates."""
    euler = np.arctan2(agent_ori_sin, agent_ori_cos)
    R_base_to_world = _euler_to_rotmat(euler[0], euler[1], euler[2])
    R_world_to_base = R_base_to_world.T
    p_base = R_world_to_base @ (world_pos - agent_pos)

    cam_pos = cam_rel_pose[:3].astype(np.float64)
    cam_quat = cam_rel_pose[3:7].astype(np.float64)
    R_cam_to_base = _quat_to_rotmat(cam_quat) @ _RX_180
    R_base_to_cam = R_cam_to_base.T
    p_cam = R_base_to_cam @ (p_base - cam_pos)

    if p_cam[2] <= 0.05:
        return None

    p_proj = HEAD_K @ p_cam
    nx = (p_proj[0] / p_proj[2]) / HEAD_NATIVE_W
    ny = (p_proj[1] / p_proj[2]) / HEAD_NATIVE_H

    visible = 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0
    if not visible:
        return None
    return float(nx), float(ny)


def compute_gt_normalized(task_info, cam_rel_pose):
    """Compute GT normalized (x,y) for radio and table. Returns dict."""
    agent_pos = task_info[AGENT_OFFSET + 1: AGENT_OFFSET + 4]
    agent_ori_cos = task_info[AGENT_OFFSET + 4: AGENT_OFFSET + 7]
    agent_ori_sin = task_info[AGENT_OFFSET + 7: AGENT_OFFSET + 10]

    results = {}
    for name, offset in [("radio", RADIO_OFFSET), ("table", TABLE_OFFSET)]:
        obj_pos = task_info[offset + 1: offset + 4]
        proj = project_to_normalized(obj_pos, agent_pos, agent_ori_cos,
                                     agent_ori_sin, cam_rel_pose)
        if proj is not None:
            results[name] = {"nx": proj[0], "ny": proj[1], "visible": True}
        else:
            results[name] = {"nx": 0.5, "ny": 0.5, "visible": False}
    return results


# ── SigLIP forward pass ───────────────────────────────────────────────

def create_siglip_module():
    """Create SigLIP linen module matching pi0's configuration."""
    return _siglip.Module(
        num_classes=2048,
        variant="So400m/14",
        pool_type="none",
        scan=True,
        dtype_mm="bfloat16",
    )


def extract_camera_crops(frame, video_w, video_h):
    """Extract individual camera crops from a (possibly composite) eval video frame.

    Composite layout (672x448):
        [left_wrist 224x224 | head_cam 448x448]
        [right_wrist 224x224|                  ]

    Returns list of (name, crop, x_offset, y_offset, crop_w, crop_h).
    If the frame is already square (single camera), returns just one entry.
    """
    aspect = video_w / video_h
    if 1.45 < aspect < 1.55 and video_w > video_h:
        wrist_w = video_w - video_h  # 224
        head_size = video_h           # 448
        wrist_h = head_size // 2      # 224
        return [
            ("head",        frame[:head_size, wrist_w:wrist_w + head_size],
             wrist_w, 0, head_size, head_size),
            ("left_wrist",  frame[:wrist_h, :wrist_w],
             0, 0, wrist_w, wrist_h),
            ("right_wrist", frame[wrist_h:wrist_h + wrist_h, :wrist_w],
             0, wrist_h, wrist_w, wrist_h),
        ]
    return [("head", frame, 0, 0, video_w, video_h)]


def preprocess_image(frame_bgr, target_size=224):
    """BGR uint8 frame → JAX float32 array in [-1, 1], shape [1, H, W, 3]."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (target_size, target_size), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32) / 255.0 * 2.0 - 1.0
    return jnp.array(arr[None])  # [1, 224, 224, 3]


GRID_SIZE = 16  # 224px / 14px per patch
NUM_OBJECTS = 2

def run_grounding_head(image_tokens, grounding_params):
    """Spatial softmax for (cx,cy) + MLP for (w,h) + visibility classifier.

    Returns:
        bbox: [b, num_obj*4] = [cx0, cy0, w0, h0, cx1, cy1, w1, h1]
        vis_prob: [b, num_obj] = visibility probabilities in [0, 1]
    """
    b, s, d = image_tokens.shape
    g = GRID_SIZE

    # Spatial softmax → (cx, cy)
    logits = (image_tokens @ grounding_params["heatmap_proj"]["kernel"]
              + grounding_params["heatmap_proj"]["bias"])  # [b, 256, num_obj]
    logits_flat = logits.reshape(b, g * g, NUM_OBJECTS)
    weights = jax.nn.softmax(logits_flat, axis=1)  # [b, 256, num_obj]

    coords_y, coords_x = jnp.meshgrid(
        (jnp.arange(g) + 0.5) / g,
        (jnp.arange(g) + 0.5) / g,
        indexing="ij",
    )
    grid_x = coords_x.reshape(g * g)
    grid_y = coords_y.reshape(g * g)

    cx = jnp.sum(weights * grid_x[None, :, None], axis=1)  # [b, num_obj]
    cy = jnp.sum(weights * grid_y[None, :, None], axis=1)

    # Size MLP → (w, h)
    pooled = jnp.mean(image_tokens, axis=1)  # [b, emb]
    h = jax.nn.swish(pooled @ grounding_params["size_fc1"]["kernel"]
                     + grounding_params["size_fc1"]["bias"])
    size_out = jax.nn.sigmoid(h @ grounding_params["size_fc2"]["kernel"]
                              + grounding_params["size_fc2"]["bias"])  # [b, num_obj*2]
    size_out = size_out.reshape(b, NUM_OBJECTS, 2)

    # Visibility → probability
    vis_logits = (pooled @ grounding_params["vis_fc"]["kernel"]
                  + grounding_params["vis_fc"]["bias"])  # [b, num_obj]
    vis_prob = jax.nn.sigmoid(vis_logits)

    bbox = jnp.stack([cx, cy, size_out[:, :, 0], size_out[:, :, 1]], axis=-1)
    return bbox.reshape(b, NUM_OBJECTS * 4), vis_prob


# ── Drawing helpers ───────────────────────────────────────────────────

_MIN_BOX_PX = 10  # minimum box half-size in pixels for visibility
_VIS_THRESHOLD = 0.5  # visibility confidence threshold for drawing

def draw_predictions(frame, preds, vis_probs, gt, head_w, head_h,
                     head_x_off=0, head_y_off=0):
    """Draw predicted bounding boxes + labels on the full composite frame.

    preds layout: [cx0, cy0, w0, h0, cx1, cy1, w1, h1] (normalized [0,1]).
    vis_probs: [num_obj] visibility probabilities.
    Only draws boxes for objects with vis_prob > _VIS_THRESHOLD.
    """
    out = frame.copy()

    for idx, name in enumerate(OBJ_NAMES):
        vis_p = float(vis_probs[idx])
        pred_col = PRED_COLOURS[name]

        base = idx * 4
        pred_cx = float(preds[base])
        pred_cy = float(preds[base + 1])
        pred_w = float(preds[base + 2])
        pred_h = float(preds[base + 3])

        cx_px = int(pred_cx * head_w) + head_x_off
        cy_px = int(pred_cy * head_h) + head_y_off

        if vis_p >= _VIS_THRESHOLD:
            half_w = max(int(pred_w * head_w / 2), _MIN_BOX_PX)
            half_h = max(int(pred_h * head_h / 2), _MIN_BOX_PX)

            x1 = max(cx_px - half_w, 0)
            y1 = max(cy_px - half_h, 0)
            x2 = min(cx_px + half_w, frame.shape[1] - 1)
            y2 = min(cy_px + half_h, frame.shape[0] - 1)
            cv2.rectangle(out, (x1, y1), (x2, y2), pred_col, 2)
            cv2.circle(out, (cx_px, cy_px), 4, pred_col, -1)

            label = f"{name.upper()} {vis_p:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
            label_y = max(y1 - 6, th + 2)
            cv2.putText(out, label, (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, pred_col, 2)
        else:
            # Dim marker for low-confidence predictions
            cv2.circle(out, (cx_px, cy_px), 3, pred_col, 1)
            cv2.putText(out, f"{name.upper()} {vis_p:.0%}",
                        (cx_px + 6, cy_px + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, pred_col, 1)

        # GT crosshair (only when GT is available and visible)
        gt_info = gt[name]
        if gt_info["visible"]:
            gt_col = GT_COLOURS[name]
            gt_px = int(gt_info["nx"] * head_w) + head_x_off
            gt_py = int(gt_info["ny"] * head_h) + head_y_off
            arm = 14
            cv2.line(out, (gt_px - arm, gt_py), (gt_px + arm, gt_py), gt_col, 2)
            cv2.line(out, (gt_px, gt_py - arm), (gt_px, gt_py + arm), gt_col, 2)
            cv2.circle(out, (gt_px, gt_py), 4, gt_col, -1)
            cv2.putText(out, f"GT:{name}", (gt_px + 12, gt_py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, gt_col, 2)

    return out


def draw_stats_bar(frame, frame_idx, total_frames, running_stats, has_gt=False):
    """Draw semi-transparent info bar at top."""
    out = frame
    h, w = out.shape[:2]
    bar_h = 40
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)

    cv2.putText(out, f"Grounding Probe | Frame {frame_idx}/{total_frames}",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    return out


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Grounding probe evaluation")
    parser.add_argument("--episode", type=int, default=None,
                        help="Training episode ID (uses dataset video + GT)")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to an arbitrary video (eval rollout, etc). Skips GT.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to pi0 JAX checkpoint params dir")
    parser.add_argument("--sample_every", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default="outputs/grounding_probe")
    parser.add_argument("--camera", type=str, default="head")
    args = parser.parse_args()

    if args.episode is None and args.video is None:
        parser.error("Provide either --episode (training data) or --video (arbitrary mp4)")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S", force=True)
    log.setLevel(logging.INFO)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve video path and GT availability ────────────────────────
    has_gt = False
    task_infos = None
    cam_rel_poses = None

    if args.video is not None:
        video_path = Path(args.video)
        if not video_path.exists():
            log.error("Video not found: %s", video_path)
            return
        ep_id = video_path.stem
        cam = "eval"
        log.info("Using arbitrary video (no GT): %s", video_path)
    else:
        ep_id = f"episode_{args.episode:08d}"
        cam = args.camera
        parquet_path = DATA_ROOT / "data" / TASK / f"{ep_id}.parquet"
        video_path = DATA_ROOT / "videos" / TASK / f"observation.images.rgb.{cam}" / f"{ep_id}.mp4"

        for p in (parquet_path, video_path):
            if not p.exists():
                log.error("Missing: %s", p)
                return

        log.info("Loading ground truth from %s", parquet_path)
        table = pq.read_table(str(parquet_path))
        task_infos = [np.array(row.as_py()) for row in table.column("observation.task_info")]
        cam_rel_poses = [np.array(row.as_py(), dtype=np.float64)
                         for row in table.column("observation.cam_rel_poses")]
        has_gt = True

    total_frames_from_gt = len(task_infos) if has_gt else None

    # ── Load checkpoint ───────────────────────────────────────────────
    log.info("Loading checkpoint: %s", args.checkpoint)
    params = _model.restore_params(args.checkpoint, restore_type=np.ndarray)

    # Extract SigLIP params (under PaliGemma/img/)
    siglip_params = params["PaliGemma"]["img"]
    log.info("SigLIP params loaded (%d leaves)",
             len(jax.tree.leaves(siglip_params)))

    # Extract grounding head params (spatial softmax + size MLP + visibility)
    gh = params["aux_grounding_head"]
    grounding_params = {
        "heatmap_proj": {
            "kernel": jnp.array(gh["heatmap_proj"]["kernel"]),
            "bias": jnp.array(gh["heatmap_proj"]["bias"]),
        },
        "size_fc1": {
            "kernel": jnp.array(gh["size_fc1"]["kernel"]),
            "bias": jnp.array(gh["size_fc1"]["bias"]),
        },
        "size_fc2": {
            "kernel": jnp.array(gh["size_fc2"]["kernel"]),
            "bias": jnp.array(gh["size_fc2"]["bias"]),
        },
        "vis_fc": {
            "kernel": jnp.array(gh["vis_fc"]["kernel"]),
            "bias": jnp.array(gh["vis_fc"]["bias"]),
        },
    }
    log.info("Grounding head: heatmap_proj=%s, size_fc1=%s → size_fc2=%s, vis_fc=%s",
             grounding_params["heatmap_proj"]["kernel"].shape,
             grounding_params["size_fc1"]["kernel"].shape,
             grounding_params["size_fc2"]["kernel"].shape,
             grounding_params["vis_fc"]["kernel"].shape)

    # Create SigLIP module and convert params for apply
    siglip_module = create_siglip_module()
    siglip_variables = {"params": siglip_params}

    # JIT-compile the forward pass
    @jax.jit
    def forward(image, siglip_vars, g_params):
        tokens, _ = siglip_module.apply(siglip_vars, image, train=False)
        bbox, vis_prob = run_grounding_head(tokens, g_params)
        return bbox, vis_prob

    # Warmup
    log.info("Warming up JIT...")
    dummy_img = jnp.zeros((1, 224, 224, 3), dtype=jnp.float32)
    _ = forward(dummy_img, siglip_variables, grounding_params)
    log.info("JIT warmup complete")

    # ── Open video ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    log.info("Video: %dx%d @ %.0f fps", img_w, img_h, fps)

    # Detect composite layout
    _dummy_frame = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    _crops = extract_camera_crops(_dummy_frame, img_w, img_h)
    is_composite = len(_crops) > 1
    if is_composite:
        log.info("Detected composite video with %d camera views:", len(_crops))
        for cname, _, cx, cy, cw, ch in _crops:
            log.info("  %-12s offset=(%d,%d) size=%dx%d", cname, cx, cy, cw, ch)

    ckpt_name = Path(args.checkpoint).parent.name
    vid_out_path = out_dir / f"probe_{cam}_{ep_id}_{ckpt_name}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_fps = fps / args.sample_every
    writer = cv2.VideoWriter(str(vid_out_path), fourcc, out_fps, (img_w, img_h))

    csv_path = out_dir / f"probe_{cam}_{ep_id}_{ckpt_name}.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    header = ["frame_idx"]
    for name in OBJ_NAMES:
        header.extend([f"{name}_pred_cx", f"{name}_pred_cy",
                        f"{name}_pred_w", f"{name}_pred_h",
                        f"{name}_pred_vis",
                        f"{name}_gt_cx", f"{name}_gt_cy",
                        f"{name}_gt_visible", f"{name}_center_l1"])
    csv_writer.writerow(header)

    # ── Process frames ────────────────────────────────────────────────
    total_frames = total_frames_from_gt if has_gt else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log.info("Episode %s: %d frames", ep_id, total_frames)
    frame_indices = list(range(0, total_frames, args.sample_every))
    n_frames = len(frame_indices)
    log.info("Processing %d frames (every %d-th)…", n_frames, args.sample_every)

    running_stats = {}
    for name in OBJ_NAMES:
        running_stats[name] = {
            "n_visible": 0, "sum_l1": 0.0,
            "within_10pct": 0, "within_20pct": 0,
            "all_l1": [],
        }

    t_start = time.time()

    for i, fidx in enumerate(frame_indices):
        if args.sample_every == 1:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if not ret:
                continue

        # Extract camera crops and run grounding on each
        crops = extract_camera_crops(frame, img_w, img_h)
        # Run on head camera (first entry) for scoring
        head_crop = crops[0][1]
        img_jax = preprocess_image(head_crop)
        bbox_out, vis_out = forward(img_jax, siglip_variables, grounding_params)
        preds_np = np.array(bbox_out[0])   # [8] = [cx0,cy0,w0,h0, cx1,cy1,w1,h1]
        vis_np = np.array(vis_out[0])      # [2] = [radio_vis, table_vis]

        # Compute GT if available
        gt = None
        if has_gt and fidx < len(task_infos):
            head_cam_rel = cam_rel_poses[fidx][CAM_HEAD_OFFSET: CAM_HEAD_OFFSET + 7]
            gt = compute_gt_normalized(task_infos[fidx], head_cam_rel)

        # Score per object (center L1 only — no GT for w/h in old datasets)
        csv_row = [fidx]
        for idx, name in enumerate(OBJ_NAMES):
            base = idx * 4
            pred_cx = float(preds_np[base])
            pred_cy = float(preds_np[base + 1])
            pred_w = float(preds_np[base + 2])
            pred_h = float(preds_np[base + 3])

            if gt is not None:
                gt_x = gt[name]["nx"]
                gt_y = gt[name]["ny"]
                visible = gt[name]["visible"]
                l1 = abs(pred_cx - gt_x) + abs(pred_cy - gt_y) if visible else float("nan")

                if visible:
                    s = running_stats[name]
                    s["n_visible"] += 1
                    s["sum_l1"] += l1
                    s["all_l1"].append(l1)
                    if l1 < 0.10:
                        s["within_10pct"] += 1
                    if l1 < 0.20:
                        s["within_20pct"] += 1

                csv_row.extend([
                    f"{pred_cx:.4f}", f"{pred_cy:.4f}",
                    f"{pred_w:.4f}", f"{pred_h:.4f}",
                    f"{float(vis_np[idx]):.4f}",
                    f"{gt_x:.4f}", f"{gt_y:.4f}",
                    visible, f"{l1:.4f}" if visible else "",
                ])
            else:
                csv_row.extend([
                    f"{pred_cx:.4f}", f"{pred_cy:.4f}",
                    f"{pred_w:.4f}", f"{pred_h:.4f}",
                    f"{float(vis_np[idx]):.4f}",
                    "", "", "", "",
                ])

        csv_writer.writerow(csv_row)

        # Annotate — pass dummy GT when unavailable
        if gt is None:
            gt = {name: {"nx": 0.5, "ny": 0.5, "visible": False} for name in OBJ_NAMES}

        # Draw predictions on all camera views
        annotated = frame.copy()
        for cam_name, cam_crop, cx, cy, cw, ch in crops:
            if cam_name == "head":
                cam_preds, cam_vis = preds_np, vis_np
            else:
                cam_jax = preprocess_image(cam_crop)
                cam_bbox, cam_v = forward(cam_jax, siglip_variables, grounding_params)
                cam_preds = np.array(cam_bbox[0])
                cam_vis = np.array(cam_v[0])
            annotated = draw_predictions(annotated, cam_preds, cam_vis, gt,
                                         cw, ch, cx, cy)
        draw_stats_bar(annotated, fidx, total_frames, running_stats, has_gt=has_gt)
        writer.write(annotated)

        # Progress
        if i % 100 == 0 or i == n_frames - 1:
            elapsed = time.time() - t_start
            fps_actual = (i + 1) / max(elapsed, 0.001)
            eta = (n_frames - i - 1) / max(fps_actual, 0.001)
            if has_gt:
                r = running_stats["radio"]
                r_l1 = r["sum_l1"] / max(r["n_visible"], 1)
                log.info(
                    "frame %4d/%d  radio_L1=%.3f  within_10%%=%d/%d  "
                    "speed=%.1f fps  ETA=%.0fs",
                    fidx, total_frames, r_l1,
                    r["within_10pct"], r["n_visible"],
                    fps_actual, eta,
                )
            else:
                log.info(
                    "frame %4d/%d  speed=%.1f fps  ETA=%.0fs",
                    fidx, total_frames, fps_actual, eta,
                )

    cap.release()
    writer.release()
    csv_file.close()

    # ── Summary ───────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    log.info("=" * 60)
    log.info("GROUNDING PROBE COMPLETE — %s camera, %s", cam, ep_id)
    log.info("  Checkpoint : %s", args.checkpoint)
    log.info("  GT mode    : %s", "yes" if has_gt else "no (predictions only)")
    log.info("=" * 60)

    if has_gt:
        for name in OBJ_NAMES:
            s = running_stats[name]
            n_vis = s["n_visible"]
            if n_vis == 0:
                log.info("  --- %s --- (never visible)", name.upper())
                continue
            mean_l1 = s["sum_l1"] / n_vis
            median_l1 = float(np.median(s["all_l1"]))
            pct10 = 100 * s["within_10pct"] / n_vis
            pct20 = 100 * s["within_20pct"] / n_vis
            log.info("  --- %s ---", name.upper())
            log.info("    Visible frames : %d / %d", n_vis, n_frames)
            log.info("    Mean L1        : %.4f", mean_l1)
            log.info("    Median L1      : %.4f", median_l1)
            log.info("    Within 10%%     : %d / %d (%.1f%%)",
                     s["within_10pct"], n_vis, pct10)
            log.info("    Within 20%%     : %d / %d (%.1f%%)",
                     s["within_20pct"], n_vis, pct20)
    else:
        log.info("  No ground truth — overlay video shows predictions only.")

    log.info("  Time  : %.1fs (%.1f fps)", elapsed, n_frames / max(elapsed, 0.001))
    log.info("  Video : %s", vid_out_path)
    log.info("  CSV   : %s", csv_path)


if __name__ == "__main__":
    main()
