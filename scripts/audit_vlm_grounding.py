#!/usr/bin/env python3
"""
VLM Visual-Grounding Audit for BEHAVIOR-1K turn-on-radio episodes.

Loads standalone PaliGemma (same SigLIP + Gemma-2B backbone used in pi0.5)
and runs "detect <object>" on every frame, overlaying bounding boxes,
ground-truth distance, and a running detection-rate counter.

Outputs
-------
- annotated MP4 at 30fps (original frame + bbox overlay + stats HUD)
- per-frame CSV  (frame, distance, detected, bbox, raw_output)
- console summary

Usage
-----
    # Off-the-shelf VLM (baseline)
    python scripts/audit_vlm_grounding.py --episode 2790
    python scripts/audit_vlm_grounding.py --episode 2790 --sample_every 5

    # Fine-tuned pi0 checkpoint (patched SigLIP weights)
    python scripts/audit_vlm_grounding.py --episode 2790 \
        --checkpoint checkpoints/pi0_b1k_turning_on_radio/49999_radio/params
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Suppress the noisy PaliGemma processor warnings
warnings.filterwarnings("ignore", message=".*image tokens.*")
warnings.filterwarnings("ignore", message=".*passing both.*text.*images.*")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/home/stickbot/projects/behavior/behavior_data")
TASK = "task-0000"

_LOC_RE = re.compile(r"<loc(\d{4})>")

DIST_CLOSE = 1.0
DIST_MEDIUM = 3.0

# task_info layout
AGENT_OFFSET = 0   # real(1) + pos(3) + ori_cos(3) + ori_sin(3) = 10
RADIO_OFFSET = 10  # real(1) + pos(3) + ori_cos(3) + ori_sin(3) + grip(2) = 12
TABLE_OFFSET = 22  # same layout as radio

# cam_rel_poses layout: [pos(3)+quat(4)] x3 cameras
# order: left_wrist(0:7), right_wrist(7:14), head(14:21)
CAM_HEAD_OFFSET = 14  # start index for head camera in cam_rel_poses

# Camera intrinsics for R1Pro head camera at native 720x720
# From eval_utils.py: [[306, 0, 360], [0, 306, 360], [0, 0, 1]]
HEAD_K = np.array([[306.0, 0.0, 360.0],
                   [0.0, 306.0, 360.0],
                   [0.0,   0.0,   1.0]], dtype=np.float64)
HEAD_NATIVE_RES = (720, 720)  # (width, height)

# Physical bounding-box dimensions from OmniGibson asset metadata.json
# bbox_size is (x, y, z) in the object's local frame; we project the two
# largest extents as (width, height) since orientation varies.
# Radio wxnicr:  bbox_size = [0.138, 0.321, 0.235]  → max_horiz=0.321, vert=0.235
# Table koagbh:  bbox_size = [0.800, 1.656, 0.406]  → max_horiz=1.656, vert=0.406
OBJ_DIMENSIONS = {
    "radio": (0.32, 0.24),   # (width_m, height_m)
    "table": (1.66, 0.41),
}

# Objects to detect
DETECT_OBJECTS = ["radio", "table"]

# Colours (BGR)
COLOURS = {
    "radio": (0, 255, 0),      # green  (VLM detection)
    "table": (255, 180, 0),    # cyan-ish (VLM detection)
    "default": (0, 200, 255),  # orange
}
GT_COLOURS = {
    "radio": (255, 0, 255),    # magenta (ground truth)
    "table": (0, 255, 255),    # yellow  (ground truth)
    "default": (128, 0, 255),  # purple
}


def parse_episode_id(episode: int) -> str:
    return f"episode_{episode:08d}"


# ------------------------------------------------------------------
# Bounding-box parsing
# ------------------------------------------------------------------

def parse_detect_output(text: str, img_w: int, img_h: int):
    """Parse PaliGemma detect output -> list of (label, [x1,y1,x2,y2])."""
    detections = []
    for segment in text.split(";"):
        locs = _LOC_RE.findall(segment)
        if len(locs) < 4:
            continue
        y1, x1, y2, x2 = [int(v) for v in locs[:4]]
        px1 = int(x1 / 1024 * img_w)
        py1 = int(y1 / 1024 * img_h)
        px2 = int(x2 / 1024 * img_w)
        py2 = int(y2 / 1024 * img_h)
        label_match = re.search(r"<loc\d{4}>\s*(.+)", segment)
        label = label_match.group(1).strip().rstrip("<").strip() if label_match else "unknown"
        # Clean up trailing eos tokens
        label = re.sub(r"<[^>]+>", "", label).strip()
        detections.append((label, [px1, py1, px2, py2]))
    return detections


def get_positions(task_info: np.ndarray):
    robot_pos = task_info[AGENT_OFFSET + 1: AGENT_OFFSET + 4]
    radio_pos = task_info[RADIO_OFFSET + 1: RADIO_OFFSET + 4]
    table_pos = task_info[TABLE_OFFSET + 1: TABLE_OFFSET + 4]
    return robot_pos, radio_pos, table_pos


def distance_bucket(dist: float) -> str:
    if dist < DIST_CLOSE:
        return "close"
    elif dist < DIST_MEDIUM:
        return "medium"
    return "far"


def frame_to_pil(frame_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


# ------------------------------------------------------------------
# 3D → 2D projection helpers (pure numpy, matches OmniGibson conventions)
# ------------------------------------------------------------------

def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert (x, y, z, w) quaternion to 3x3 rotation matrix (OmniGibson convention)."""
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _euler_to_rotmat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Euler (roll, pitch, yaw) → rotation matrix.

    Matches OmniGibson convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


# 180-degree rotation about the X axis (camera coord-sys correction).
_RX_180 = np.diag([1.0, -1.0, -1.0])


def project_world_to_pixel(
    world_pos: np.ndarray,       # (3,) object position in world frame
    agent_pos: np.ndarray,       # (3,) robot base position in world frame
    agent_ori_cos: np.ndarray,   # (3,) cos(euler) of robot base
    agent_ori_sin: np.ndarray,   # (3,) sin(euler) of robot base
    cam_rel_pose: np.ndarray,    # (7,) camera pose relative to base [pos(3), quat_xyzw(4)]
    K: np.ndarray,               # (3,3) camera intrinsics at native resolution
    native_res: tuple[int, int], # (native_w, native_h)
    video_res: tuple[int, int],  # (video_w, video_h)
) -> tuple[float, float, float] | None:
    """Project a 3D world point to 2D pixel coordinates in the video frame.

    Returns (px_x, px_y, depth) or None if the point is behind the camera or
    outside the image bounds.
    """
    # --- World → Robot base ---
    euler = np.arctan2(agent_ori_sin, agent_ori_cos)
    R_base_to_world = _euler_to_rotmat(euler[0], euler[1], euler[2])
    R_world_to_base = R_base_to_world.T
    p_base = R_world_to_base @ (world_pos - agent_pos)

    # --- Robot base → Camera ---
    cam_pos = cam_rel_pose[:3].astype(np.float64)
    cam_quat = cam_rel_pose[3:7].astype(np.float64)
    R_cam_basis = _quat_to_rotmat(cam_quat)
    R_cam_to_base = R_cam_basis @ _RX_180       # camera_to_base rotation
    R_base_to_cam = R_cam_to_base.T
    p_cam = R_base_to_cam @ (p_base - cam_pos)

    # Depth check — must be in front of camera
    depth = p_cam[2]
    if depth <= 0.05:
        return None

    # --- Camera → Pixel (pinhole model) ---
    p_proj = K @ p_cam
    u_native = p_proj[0] / p_proj[2]
    v_native = p_proj[1] / p_proj[2]

    # Scale from native to video resolution
    native_w, native_h = native_res
    video_w, video_h = video_res
    u = u_native * video_w / native_w
    v = v_native * video_h / native_h

    # Bounds check (generous margin)
    margin = 50
    if u < -margin or u > video_w + margin or v < -margin or v > video_h + margin:
        return None

    return (float(u), float(v), float(depth))


def compute_gt_projections(
    task_info: np.ndarray,
    cam_rel_pose: np.ndarray,
    video_w: int,
    video_h: int,
) -> list[tuple[str, float, float, float]]:
    """Compute projected ground-truth positions for radio and table.

    Returns list of (label, px_x, px_y, depth_m) for objects that are visible.
    """
    agent_pos = task_info[AGENT_OFFSET + 1 : AGENT_OFFSET + 4]
    agent_ori_cos = task_info[AGENT_OFFSET + 4 : AGENT_OFFSET + 7]
    agent_ori_sin = task_info[AGENT_OFFSET + 7 : AGENT_OFFSET + 10]

    objects = [
        ("radio", task_info[RADIO_OFFSET + 1 : RADIO_OFFSET + 4]),
        ("table", task_info[TABLE_OFFSET + 1 : TABLE_OFFSET + 4]),
    ]

    results = []
    for label, obj_pos in objects:
        proj = project_world_to_pixel(
            obj_pos, agent_pos, agent_ori_cos, agent_ori_sin,
            cam_rel_pose, HEAD_K, HEAD_NATIVE_RES, (video_w, video_h),
        )
        if proj is not None:
            results.append((label, proj[0], proj[1], proj[2]))
    return results


# ------------------------------------------------------------------
# Annotation drawing
# ------------------------------------------------------------------

def compute_iou(box_a, box_b):
    """Compute Intersection-over-Union between two pixel-space boxes [x1,y1,x2,y2]."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def gt_proj_to_bbox(label, cx, cy, depth, video_w, video_h):
    """Convert a GT projection (center + depth) into a pixel-space bbox [x1,y1,x2,y2]."""
    focal_px = HEAD_K[0, 0]  # 306 at 720px
    scale_x = video_w / HEAD_NATIVE_RES[0]
    scale_y = video_h / HEAD_NATIVE_RES[1]
    obj_w_m, obj_h_m = OBJ_DIMENSIONS.get(label, (0.15, 0.10))
    half_w = max(int(focal_px * obj_w_m / depth * scale_x / 2), 6)
    half_h = max(int(focal_px * obj_h_m / depth * scale_y / 2), 6)
    return [int(cx) - half_w, int(cy) - half_h, int(cx) + half_w, int(cy) + half_h]


def match_detections_to_gt(detections, gt_projections, video_w, video_h, iou_threshold=0.05):
    """Match VLM detections to GT objects using IoU.

    Returns a dict like {"radio": {"detected": bool, "gt_visible": bool, "iou": float, "matched": bool},
                         "table": {"detected": bool, "gt_visible": bool, "iou": float, "matched": bool}}.

    A detection "matches" a GT object if:
      1) The VLM detection label contains the GT label (text match), AND
      2) IoU between VLM bbox and GT bbox >= iou_threshold (spatial match).

    Alternatively, if no text match but the VLM bbox center falls inside the GT
    bbox, we count it as a "spatial-only" match (the VLM found the object but
    mislabelled it).
    """
    gt_objects = {}
    for label, cx, cy, depth in gt_projections:
        gt_bbox = gt_proj_to_bbox(label, cx, cy, depth, video_w, video_h)
        gt_objects[label] = {"bbox": gt_bbox, "cx": cx, "cy": cy, "depth": depth}

    results = {}
    for obj_name in DETECT_OBJECTS:
        gt_visible = obj_name in gt_objects
        best_iou = 0.0
        text_matched = False
        spatial_matched = False

        if gt_visible:
            gt_box = gt_objects[obj_name]["bbox"]
            for det_label, det_bbox in detections:
                iou = compute_iou(det_bbox, gt_box)
                # Text match: VLM label contains the GT object name
                if obj_name in det_label.lower():
                    text_matched = True
                    best_iou = max(best_iou, iou)
                # Spatial match: VLM detection center inside GT box
                det_cx = (det_bbox[0] + det_bbox[2]) / 2
                det_cy = (det_bbox[1] + det_bbox[3]) / 2
                if (gt_box[0] <= det_cx <= gt_box[2] and
                        gt_box[1] <= det_cy <= gt_box[3]):
                    spatial_matched = True
                    best_iou = max(best_iou, iou)

        matched = (text_matched and best_iou >= iou_threshold) or spatial_matched
        results[obj_name] = {
            "detected": text_matched,       # VLM said the label
            "gt_visible": gt_visible,       # GT says object is in frame
            "matched": matched,             # label + spatial overlap
            "iou": best_iou,
        }

    return results


def draw_overlay(
    frame: np.ndarray,
    frame_idx: int,
    total_frames: int,
    dist: float,
    detections: list,
    per_obj_stats: dict,
    total_processed: int,
    frame_match: dict,
    raw_vlm_output: str = "",
    model_label: str = "",
) -> np.ndarray:
    """Draw annotation overlay with per-object detection + GT-match stats.

    Args:
        per_obj_stats: running stats dict, e.g.
            {"radio": {"gt_visible": 50, "detected": 5, "matched": 3},
             "table": {"gt_visible": 60, "detected": 40, "matched": 35}}.
        total_processed: total frames processed so far.
        frame_match: this frame's match result from match_detections_to_gt.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # --- Semi-transparent top bar ---
    bar_h = 155
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)

    # --- Top bar text ---
    bucket = distance_bucket(dist)
    line1 = f"Frame {frame_idx}/{total_frames}  |  Robot-to-Radio: {dist:.2f}m ({bucket})"
    cv2.putText(out, line1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Per-object stats: two lines each — detected/total, matched/gt_visible
    y_pos = 44
    for obj_name in DETECT_OBJECTS:
        s = per_obj_stats.get(obj_name, {})
        gt_n = s.get("gt_visible", 0)
        det_n = s.get("detected", 0)
        match_n = s.get("matched", 0)
        det_pct = 100 * det_n / max(total_processed, 1)
        match_pct = 100 * match_n / max(gt_n, 1)
        obj_colour = COLOURS.get(obj_name, COLOURS["default"])
        stat_line = (f"{obj_name.capitalize()}: "
                     f"{det_n}/{total_processed} frames ({det_pct:.0f}%) detected, "
                     f"{match_n}/{gt_n} GT ({match_pct:.0f}%) matched")
        cv2.putText(out, stat_line, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, obj_colour, 2)
        y_pos += 22

    # Detected objects list
    obj_names = [lbl for lbl, _ in detections]
    line_det = f"Detected: {', '.join(obj_names)}" if obj_names else "Detected: (none)"
    cv2.putText(out, line_det, (10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 1)
    y_pos += 22
    if model_label:
        cv2.putText(out, f"Model: {model_label}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 255), 1)

    # --- Detection status indicators (top-right) ---
    ind_y = 30
    for obj_name in DETECT_OBJECTS:
        fm = frame_match.get(obj_name, {})
        if not fm.get("gt_visible", False):
            continue
        matched = fm.get("matched", False)
        detected = fm.get("detected", False)
        iou = fm.get("iou", 0.0)
        if matched:
            txt = f"{obj_name.upper()} (IoU={iou:.2f})"
            clr = (0, 255, 0)
        elif detected:
            txt = f"{obj_name.upper()} (no spatial match)"
            clr = (0, 180, 255)
        else:
            txt = f"NO {obj_name.upper()}"
            clr = (0, 0, 255)
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(out, txt, (w - tw - 15, ind_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, clr, 2)
        ind_y += 25

    # --- Draw bounding boxes with labels and dimensions ---
    for label, (x1, y1, x2, y2) in detections:
        key = label.lower().split()[0] if label else "default"
        colour = COLOURS.get(key, COLOURS["default"])
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        bw, bh = x2 - x1, y2 - y1
        lbl_text = f"{label} ({bw}x{bh}px)"
        (tw, th), _ = cv2.getTextSize(lbl_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(out, (x1, max(y1 - th - 8, 0)), (x1 + tw + 6, y1), colour, -1)
        cv2.putText(out, lbl_text, (x1 + 3, max(y1 - 4, th + 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # --- Bottom bar: raw VLM output ---
    if raw_vlm_output:
        display_text = raw_vlm_output.replace("\n", " ").strip()
        if len(display_text) > 120:
            display_text = display_text[:117] + "..."
        bar_bot_h = 30
        overlay2 = out.copy()
        cv2.rectangle(overlay2, (0, h - bar_bot_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.6, out, 0.4, 0, out)
        cv2.putText(out, f"VLM: {display_text}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 255), 1)

    return out


def _draw_dashed_rect(img, pt1, pt2, colour, thickness=2, dash_len=8):
    """Draw a dashed rectangle on *img* (in-place)."""
    x1, y1 = pt1
    x2, y2 = pt2
    # Edges: top, right, bottom, left
    edges = [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]
    for (sx, sy), (ex, ey) in edges:
        dx = ex - sx
        dy = ey - sy
        length = max(abs(dx), abs(dy))
        if length == 0:
            continue
        num_dashes = max(1, int(length / dash_len))
        for d in range(0, num_dashes, 2):
            t0 = d / num_dashes
            t1 = min((d + 1) / num_dashes, 1.0)
            p0 = (int(sx + dx * t0), int(sy + dy * t0))
            p1 = (int(sx + dx * t1), int(sy + dy * t1))
            cv2.line(img, p0, p1, colour, thickness)


def draw_gt_overlay(
    frame: np.ndarray,
    gt_projections: list[tuple[str, float, float, float]],
    video_w: int,
    video_h: int,
) -> np.ndarray:
    """Draw ground-truth crosshairs + approximate bounding boxes on *frame*.

    Args:
        frame: BGR image (modified in-place and returned).
        gt_projections: list of (label, px_x, px_y, depth_m) from compute_gt_projections.
        video_w, video_h: video resolution for bbox size scaling.
    """
    out = frame  # modify in place (caller already has a copy)
    focal_px = HEAD_K[0, 0]  # 306 at 720px
    scale_x = video_w / HEAD_NATIVE_RES[0]
    scale_y = video_h / HEAD_NATIVE_RES[1]

    for label, cx, cy, depth in gt_projections:
        colour = GT_COLOURS.get(label, GT_COLOURS["default"])
        icx, icy = int(round(cx)), int(round(cy))

        # --- Crosshair ---
        arm = 12
        cv2.line(out, (icx - arm, icy), (icx + arm, icy), colour, 2)
        cv2.line(out, (icx, icy - arm), (icx, icy + arm), colour, 2)
        cv2.circle(out, (icx, icy), 4, colour, -1)

        # --- Approximate bounding box ---
        obj_w_m, obj_h_m = OBJ_DIMENSIONS.get(label, (0.15, 0.10))
        half_w_px = int(focal_px * obj_w_m / depth * scale_x / 2)
        half_h_px = int(focal_px * obj_h_m / depth * scale_y / 2)
        # Clamp minimum visible size
        half_w_px = max(half_w_px, 6)
        half_h_px = max(half_h_px, 6)

        x1 = icx - half_w_px
        y1 = icy - half_h_px
        x2 = icx + half_w_px
        y2 = icy + half_h_px
        _draw_dashed_rect(out, (x1, y1), (x2, y2), colour, thickness=2, dash_len=6)

        # --- Label with distance ---
        lbl_text = f"GT:{label} {depth:.1f}m"
        (tw, th_txt), _ = cv2.getTextSize(lbl_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        # Place label below the box (or above if too close to bottom)
        lbl_y = y2 + th_txt + 6
        if lbl_y > video_h - 40:
            lbl_y = y1 - 6
        cv2.rectangle(out, (x1, lbl_y - th_txt - 4), (x1 + tw + 6, lbl_y + 2), colour, -1)
        cv2.putText(out, lbl_text, (x1 + 3, lbl_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return out


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VLM grounding audit video")
    parser.add_argument("--episode", type=int, default=2790)
    parser.add_argument("--sample_every", type=int, default=1,
                        help="Process every N-th frame (1=all frames)")
    parser.add_argument("--model", type=str, default="google/paligemma2-3b-mix-448",
                        help="HF model id (ignored when --checkpoint is set)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to pi0 JAX checkpoint params dir. When set, "
                             "loads paligemma2-3b-mix-224 and patches its SigLIP "
                             "encoder with the fine-tuned weights from this checkpoint.")
    parser.add_argument("--output_dir", type=str, default="outputs/vlm_audit")
    parser.add_argument("--detect", nargs="+", default=DETECT_OBJECTS,
                        help="Objects to detect (default: radio table)")
    parser.add_argument("--camera", type=str, default="head",
                        help="Camera to use (head, left_wrist, right_wrist)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ep_id = parse_episode_id(args.episode)

    # ---- Load ground-truth --------------------------------------------------
    parquet_path = DATA_ROOT / "data" / TASK / f"{ep_id}.parquet"
    log.info("Loading ground truth from %s", parquet_path)
    table = pq.read_table(str(parquet_path))
    task_infos = [np.array(row.as_py()) for row in table.column("observation.task_info")]
    cam_rel_poses = [np.array(row.as_py(), dtype=np.float64)
                     for row in table.column("observation.cam_rel_poses")]
    total_frames = len(task_infos)
    log.info("Episode has %d frames (task_info=%d, cam_rel_poses=%d)",
             total_frames, len(task_infos), len(cam_rel_poses))

    # ---- Load VLM -----------------------------------------------------------
    from transformers import PaliGemmaForConditionalGeneration, AutoProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.checkpoint:
        # Load HF base model + patch SigLIP with fine-tuned JAX checkpoint
        from convert_checkpoint_to_hf import load_patched_model, HF_MODEL_ID
        log.info("Loading fine-tuned checkpoint: %s", args.checkpoint)
        log.info("Base HF model for patching: %s", HF_MODEL_ID)
        model = load_patched_model(args.checkpoint, device=str(device))
        processor = AutoProcessor.from_pretrained(HF_MODEL_ID)
        model_label = f"finetuned:{Path(args.checkpoint).parent.name}"
    else:
        log.info("Loading off-the-shelf PaliGemma model: %s", args.model)
        processor = AutoProcessor.from_pretrained(args.model)
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
        ).to(device)
        model.eval()
        model_label = args.model.split("/")[-1]

    log.info("Model loaded on %s (%s)", device,
             torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")

    # Build the detect prompt: "detect radio ; table"
    detect_prompt = "detect " + " ; ".join(args.detect)
    log.info("Detect prompt: %r", detect_prompt)

    # ---- Open video ---------------------------------------------------------
    cam = args.camera
    video_path = DATA_ROOT / "videos" / TASK / f"observation.images.rgb.{cam}" / f"{ep_id}.mp4"
    if not video_path.exists():
        log.error("Video not found: %s", video_path)
        return
    cap = cv2.VideoCapture(str(video_path))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    log.info("Video: %dx%d @ %.0f fps, %d frames", img_w, img_h, fps, total_frames)

    # ---- Prepare outputs ----------------------------------------------------
    suffix = "finetuned" if args.checkpoint else "baseline"
    vid_out_path = out_dir / f"audit_{cam}_{ep_id}_{suffix}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_fps = fps / args.sample_every  # keep real-time if sample_every=1
    writer = cv2.VideoWriter(str(vid_out_path), fourcc, out_fps, (img_w, img_h))

    csv_path = out_dir / f"audit_{cam}_{ep_id}_{suffix}.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame_idx", "distance_m", "bucket",
        # Per-object columns
        "radio_gt_visible", "radio_detected", "radio_matched", "radio_iou",
        "table_gt_visible", "table_detected", "table_matched", "table_iou",
        "num_detections", "bbox", "raw_output",
    ])

    # ---- Process frames -----------------------------------------------------
    frame_indices = list(range(0, total_frames, args.sample_every))
    n_frames = len(frame_indices)
    log.info("Processing %d frames (every %d-th)…", n_frames, args.sample_every)

    # Per-object running stats
    per_obj_stats = {}
    for obj in DETECT_OBJECTS:
        per_obj_stats[obj] = {"gt_visible": 0, "detected": 0, "matched": 0}
    # Distance-bucketed stats (for radio backward compat)
    bucket_stats = {b: {"total": 0, "det": 0, "match": 0}
                    for b in ("close", "medium", "far")}
    total_frames_processed = 0
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

        pil_img = frame_to_pil(frame)
        robot_pos, radio_pos, table_pos = get_positions(task_infos[fidx])
        dist = float(np.linalg.norm(robot_pos - radio_pos))
        bucket = distance_bucket(dist)

        # --- Detect ---
        inputs = processor(images=pil_img, text=detect_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        raw_output = processor.decode(out_ids[0], skip_special_tokens=False)
        if detect_prompt in raw_output:
            raw_output = raw_output.split(detect_prompt, 1)[-1].strip()
        detections = parse_detect_output(raw_output, img_w, img_h)

        # --- Ground-truth projection ---
        head_cam_rel = cam_rel_poses[fidx][CAM_HEAD_OFFSET : CAM_HEAD_OFFSET + 7]
        gt_projs = compute_gt_projections(task_infos[fidx], head_cam_rel, img_w, img_h)

        # --- Match detections against GT ---
        frame_match = match_detections_to_gt(detections, gt_projs, img_w, img_h)

        # --- Update per-object running stats ---
        total_frames_processed += 1
        bucket_stats[bucket]["total"] += 1
        for obj in DETECT_OBJECTS:
            fm = frame_match[obj]
            if fm["gt_visible"]:
                per_obj_stats[obj]["gt_visible"] += 1
            if fm["detected"]:
                per_obj_stats[obj]["detected"] += 1
            if fm["matched"]:
                per_obj_stats[obj]["matched"] += 1
        # Radio-specific bucket stats for backward compat
        if frame_match["radio"]["detected"]:
            bucket_stats[bucket]["det"] += 1
        if frame_match["radio"]["matched"]:
            bucket_stats[bucket]["match"] += 1

        # --- CSV ---
        bbox_str = "; ".join(
            f"{lbl}:[{x1},{y1},{x2},{y2}]" for lbl, (x1, y1, x2, y2) in detections
        )
        csv_writer.writerow([
            fidx, f"{dist:.3f}", bucket,
            frame_match["radio"]["gt_visible"],
            frame_match["radio"]["detected"],
            frame_match["radio"]["matched"],
            f"{frame_match['radio']['iou']:.3f}",
            frame_match["table"]["gt_visible"],
            frame_match["table"]["detected"],
            frame_match["table"]["matched"],
            f"{frame_match['table']['iou']:.3f}",
            len(detections), bbox_str,
            raw_output.replace("\n", " "),
        ])

        # --- Annotate & write video ---
        annotated = draw_overlay(
            frame, fidx, total_frames, dist, detections,
            per_obj_stats, total_frames_processed, frame_match,
            raw_vlm_output=raw_output,
            model_label=model_label,
        )
        draw_gt_overlay(annotated, gt_projs, img_w, img_h)
        writer.write(annotated)

        # --- Progress log (every 100 frames) ---
        if i % 100 == 0 or i == n_frames - 1:
            elapsed = time.time() - t_start
            fps_actual = (i + 1) / max(elapsed, 0.001)
            eta = (n_frames - i - 1) / max(fps_actual, 0.001)
            r = per_obj_stats["radio"]
            r_pct = 100 * r["matched"] / max(r["gt_visible"], 1)
            log.info(
                "frame %4d/%d  dist=%.2fm  radio_match=%s  running=%.0f%%  "
                "speed=%.1f fps  ETA=%.0fs",
                fidx, total_frames, dist, frame_match["radio"]["matched"],
                r_pct, fps_actual, eta,
            )

    cap.release()
    writer.release()
    csv_file.close()

    # ---- Summary ------------------------------------------------------------
    elapsed = time.time() - t_start
    n = total_frames_processed
    log.info("=" * 60)
    log.info("AUDIT COMPLETE — %s camera, episode %s", cam, ep_id)
    log.info("  Model   : %s", model_label)
    log.info("=" * 60)
    pct = lambda num, den: f"{100*num/den:.1f}%" if den > 0 else "N/A"
    for obj in DETECT_OBJECTS:
        s = per_obj_stats[obj]
        gt_n = s["gt_visible"]
        log.info("  --- %s ---", obj.upper())
        log.info("    Detected      : %d / %d frames (%s)",
                 s["detected"], n, pct(s["detected"], n))
        log.info("    GT-matched    : %d / %d GT-visible (%s)",
                 s["matched"], gt_n, pct(s["matched"], gt_n))
    log.info("  --- RADIO by distance ---")
    for b in ("close", "medium", "far"):
        bs = bucket_stats[b]
        log.info("    %-6s: detected %s (%d/%d)  matched %s (%d/%d)", b,
                 pct(bs["det"], bs["total"]), bs["det"], bs["total"],
                 pct(bs["match"], bs["total"]), bs["match"], bs["total"])
    log.info("  Time    : %.1fs (%.1f fps)", elapsed, n / max(elapsed, 0.001))
    log.info("  Video   : %s", vid_out_path)
    log.info("  CSV     : %s", csv_path)


if __name__ == "__main__":
    main()
