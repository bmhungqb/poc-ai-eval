"""Fixed per-video work-area ROI (proposal §3.3).

The work area is the region around the needle / presser foot / both hands.
It is computed ONCE per video as the (outlier-trimmed) union of every hand
bounding box detected across the video, padded — then reused by:

  * feature extraction: the frame is cropped to the ROI and upscaled before
    WiLoR runs, so hands cover more pixels (the worker video is 480x368 and
    loses 37-53% of hand detections at native resolution);
  * every frame sent to the VLM (tier 1 scene classification and tier 2
    aux-operation checks): less background noise, cheaper per call.

A legacy `roi_auto.json` (named zones: needle / fabric_area / machine_button
/ lever / ...) is also accepted; its zones stay available for the cheap
button/lever activity detector in src/vlm/aux_check.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROI_FILENAME = "work_area_roi.json"
ROI_PAD_FRAC = 0.15       # pad the trimmed union bbox by this fraction per side
ROI_TRIM_PERCENTILE = 1.0  # trim outlier hand boxes (spurious detections) by
                           # taking the 1st/99th percentile of box edges
MIN_ROI_SIZE = 0.25        # ROI must cover at least this fraction of each axis
DEFAULT_UPSCALE_WIDTH = 960

# legacy roi_auto.json zones whose union approximates the work area
_LEGACY_WORK_ZONES = ("needle", "fabric_area")


def roi_from_hand_boxes(boxes: np.ndarray,
                        pad_frac: float = ROI_PAD_FRAC,
                        trim_percentile: float = ROI_TRIM_PERCENTILE) -> list[float]:
    """Normalized [x1, y1, x2, y2] work area from (N, 4) normalized hand
    boxes collected over the whole video. Percentile-trimmed union (a few
    spurious detections in a corner must not blow the ROI up to the full
    frame), padded and clipped to [0, 1]."""
    boxes = np.asarray(boxes, dtype=float)
    if len(boxes) == 0:
        return [0.0, 0.0, 1.0, 1.0]
    x1 = float(np.percentile(boxes[:, 0], trim_percentile))
    y1 = float(np.percentile(boxes[:, 1], trim_percentile))
    x2 = float(np.percentile(boxes[:, 2], 100.0 - trim_percentile))
    y2 = float(np.percentile(boxes[:, 3], 100.0 - trim_percentile))
    pad_x, pad_y = (x2 - x1) * pad_frac, (y2 - y1) * pad_frac
    x1, y1 = max(0.0, x1 - pad_x), max(0.0, y1 - pad_y)
    x2, y2 = min(1.0, x2 + pad_x), min(1.0, y2 + pad_y)
    # never collapse below a sane minimum size (grow around the center)
    for lo, hi, idx in ((x1, x2, 0), (y1, y2, 1)):
        if hi - lo < MIN_ROI_SIZE:
            c = (lo + hi) / 2
            lo, hi = max(0.0, c - MIN_ROI_SIZE / 2), min(1.0, c + MIN_ROI_SIZE / 2)
            if idx == 0:
                x1, x2 = lo, hi
            else:
                y1, y2 = lo, hi
    return [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]


def roi_from_keypoints_jsonl(path: str | Path, **kwargs) -> list[float]:
    """Work area from an existing extraction's keypoints.jsonl (normalized
    hand bboxes are already stored per frame)."""
    boxes = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            for side in ("left_hand", "right_hand"):
                hi = rec.get(side)
                if hi and hi.get("bbox"):
                    boxes.append(hi["bbox"])
    return roi_from_hand_boxes(np.array(boxes) if boxes else np.empty((0, 4)), **kwargs)


def estimate_roi_from_video(video_path: str | Path, hand_model=None,
                            sample_stride_s: float = 1.0,
                            max_samples: int = 120, **kwargs) -> list[float]:
    """Pass-1 ROI estimation: sample frames at full resolution, detect hands,
    union their boxes. `hand_model` is a WiLorHand-compatible callable (built
    lazily when omitted, so tests can inject a fake)."""
    if hand_model is None:
        from src.vision.hand_model import WiLorHand
        hand_model = WiLorHand()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, int(round(sample_stride_s * fps)))
    sample_idxs = list(range(0, n_frames, stride))[:max_samples]

    boxes = []
    for fi in sample_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            break
        for h in hand_model(frame):
            kps = h["keypoints"]
            boxes.append([kps[:, 0].min() / width, kps[:, 1].min() / height,
                          kps[:, 0].max() / width, kps[:, 1].max() / height])
    cap.release()
    return roi_from_hand_boxes(np.array(boxes) if boxes else np.empty((0, 4)), **kwargs)


def load_roi(path: str | Path) -> dict:
    """Load a ROI json. Accepts both the current format
    ({"work_area": [...], "zones": {...}}) and the legacy roi_auto.json
    (flat dict of named zones) — for the latter the work area is the union
    of the needle + fabric zones, and all zones are kept for aux_check."""
    data = json.loads(Path(path).read_text())
    if "work_area" in data:
        data.setdefault("zones", {})
        return data
    # legacy: flat {zone_name: [x1, y1, x2, y2]}
    zones = {k: v for k, v in data.items() if isinstance(v, list) and len(v) == 4}
    work = [z for name, z in zones.items() if name in _LEGACY_WORK_ZONES]
    if work:
        arr = np.array(work)
        wa = [float(arr[:, 0].min()), float(arr[:, 1].min()),
              float(arr[:, 2].max()), float(arr[:, 3].max())]
    else:
        wa = [0.0, 0.0, 1.0, 1.0]
    return {"work_area": wa, "zones": zones, "source": "legacy"}


def save_roi(path: str | Path, work_area: list[float], zones: dict | None = None,
             source: str = "hand_union") -> dict:
    data = {"work_area": [round(float(v), 4) for v in work_area],
            "zones": zones or {}, "source": source}
    Path(path).write_text(json.dumps(data, indent=1))
    return data


@dataclass
class RoiMapper:
    """Maps between full-frame coordinates and the cropped/upscaled ROI frame.

    crop pixel = (full pixel - offset) * scale, per axis.
    """
    work_area: list[float]   # normalized [x1, y1, x2, y2] in the full frame
    frame_width: int
    frame_height: int
    upscale_width: int = DEFAULT_UPSCALE_WIDTH

    def __post_init__(self):
        x1, y1, x2, y2 = self.work_area
        self.x0 = int(x1 * self.frame_width)
        self.y0 = int(y1 * self.frame_height)
        self.x1 = max(self.x0 + 2, int(x2 * self.frame_width))
        self.y1 = max(self.y0 + 2, int(y2 * self.frame_height))
        self.crop_w = self.x1 - self.x0
        self.crop_h = self.y1 - self.y0
        # never downscale: on high-res expert video the crop may already be big
        self.scale = max(1.0, self.upscale_width / self.crop_w)
        self.out_w = int(round(self.crop_w * self.scale))
        self.out_h = int(round(self.crop_h * self.scale))

    def crop(self, frame: np.ndarray) -> np.ndarray:
        """Crop the work area out of a full frame and upscale it."""
        patch = frame[self.y0:self.y1, self.x0:self.x1]
        if self.scale == 1.0:
            return patch
        return cv2.resize(patch, (self.out_w, self.out_h), interpolation=cv2.INTER_CUBIC)

    def to_full_px(self, pts_crop_px: np.ndarray) -> np.ndarray:
        """(N, 2) pixel points in the cropped/upscaled frame -> full-frame pixels."""
        pts = np.asarray(pts_crop_px, dtype=float)
        return pts / self.scale + np.array([self.x0, self.y0])

    def full_norm_bbox_to_crop_px(self, bbox_norm: list[float]) -> tuple[int, int, int, int]:
        """Full-frame-normalized [x1, y1, x2, y2] -> pixel bbox in the crop
        (clipped), e.g. for computing flow stats on the cropped frame."""
        x1 = (bbox_norm[0] * self.frame_width - self.x0) * self.scale
        y1 = (bbox_norm[1] * self.frame_height - self.y0) * self.scale
        x2 = (bbox_norm[2] * self.frame_width - self.x0) * self.scale
        y2 = (bbox_norm[3] * self.frame_height - self.y0) * self.scale
        return (max(0, int(x1)), max(0, int(y1)),
                min(self.out_w, int(x2)), min(self.out_h, int(y2)))
