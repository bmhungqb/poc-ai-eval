"""Automatic ROI estimation — no manual configuration required.

The needle zone is localized from video statistics: it is the most active
region (feed motion, needle bar, hand adjustments) and sits under the machine
lamp, so a combined motion + brightness map peaks there. The remaining ROIs
are derived geometrically from the needle position, since the layout of a
lockstitch machine is fixed (machine head above the needle, work areas to the
left/right, fabric moving around the needle plate).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SAMPLE_FPS = 4.0        # analysis rate; pairs of consecutive frames at this rate
MAX_SAMPLES = 120
ANALYSIS_WIDTH = 240


def _activity_maps(video_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (motion_map, brightness_map), both normalized to [0, 1]."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(fps / SAMPLE_FPS)))
    idxs = list(range(0, max(n_frames - 1, 1), step))[:MAX_SAMPLES]

    motion = None
    brightness = None
    for fi in idxs:
        # diff ADJACENT frames: fast needle-bar / thread take-up motion dominates,
        # while slow fabric drift contributes little per single frame interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok1, f1 = cap.read()
        ok2, f2 = cap.read()
        if not (ok1 and ok2):
            break
        h = int(f1.shape[0] * ANALYSIS_WIDTH / f1.shape[1])
        g1 = cv2.cvtColor(cv2.resize(f1, (ANALYSIS_WIDTH, h)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        g2 = cv2.cvtColor(cv2.resize(f2, (ANALYSIS_WIDTH, h)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if brightness is None:
            brightness = np.zeros_like(g1)
            motion = np.zeros_like(g1)
        brightness += g1
        motion += np.abs(g2 - g1)
    cap.release()
    if motion is None or motion.max() <= 0:
        raise ValueError(f"Could not compute activity maps for {video_path}")

    motion = cv2.GaussianBlur(motion, (0, 0), sigmaX=ANALYSIS_WIDTH * 0.03)
    brightness = cv2.GaussianBlur(brightness, (0, 0), sigmaX=ANALYSIS_WIDTH * 0.03)
    motion /= motion.max()
    brightness /= brightness.max()
    return motion, brightness


def estimate_rois(video_path: str | Path) -> dict[str, list[float]]:
    """Estimate the ROI dictionary (normalized [x1, y1, x2, y2]) for a video."""
    motion, brightness = _activity_maps(video_path)
    h, w = motion.shape

    # needle zone: high sustained motion AND under the machine lamp (product
    # requires both, so bright static table and dark moving fabric are rejected);
    # ignore frame borders where bystanders / camera shake dominate
    score = motion * brightness
    border_x, border_y = int(0.15 * w), int(0.15 * h)
    inner = np.zeros_like(score)
    inner[border_y: h - border_y, border_x: w - border_x] = \
        score[border_y: h - border_y, border_x: w - border_x]
    cy, cx = np.unravel_index(np.argmax(inner), inner.shape)
    ncx, ncy = float(cx) / w, float(cy) / h

    def box(x1, y1, x2, y2):
        return [round(max(0.0, x1), 3), round(max(0.0, y1), 3),
                round(min(1.0, x2), 3), round(min(1.0, y2), 3)]

    return {
        "needle": box(ncx - 0.07, ncy - 0.09, ncx + 0.07, ncy + 0.09),
        "fabric_area": box(ncx - 0.28, ncy - 0.10, ncx + 0.28, ncy + 0.35),
        "left_work_area": box(0.0, ncy - 0.30, ncx - 0.03, 1.0),
        "right_work_area": box(ncx + 0.10, ncy - 0.15, 1.0, 1.0),
        "machine_button": box(ncx - 0.02, ncy - 0.35, ncx + 0.16, ncy - 0.12),
        "lever": box(ncx + 0.02, ncy - 0.22, ncx + 0.18, ncy - 0.02),
    }
