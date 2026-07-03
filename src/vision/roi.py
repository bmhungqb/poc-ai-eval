"""ROI config helpers. ROIs are stored normalized [x1,y1,x2,y2] per video role."""
from __future__ import annotations

import json
from pathlib import Path

ROI_NAMES = ["needle", "fabric_area", "left_work_area", "right_work_area", "machine_button", "lever"]


def load_rois(roi_path: str | Path, role: str) -> dict[str, list[float]]:
    cfg = json.loads(Path(roi_path).read_text())
    section = cfg.get(role) or cfg.get("default")
    if section is None:
        raise ValueError(f"No ROI section for role '{role}' and no 'default' in {roi_path}")
    return {k: v for k, v in section.items() if not k.startswith("_")}


def roi_to_pixels(roi: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    return (
        max(0, int(x1 * width)),
        max(0, int(y1 * height)),
        min(width, int(x2 * width)),
        min(height, int(y2 * height)),
    )


def point_in_roi(x: float, y: float, roi: list[float]) -> bool:
    """x, y normalized (0-1)."""
    return roi[0] <= x <= roi[2] and roi[1] <= y <= roi[3]


def roi_center(roi: list[float]) -> tuple[float, float]:
    return (roi[0] + roi[2]) / 2.0, (roi[1] + roi[3]) / 2.0
