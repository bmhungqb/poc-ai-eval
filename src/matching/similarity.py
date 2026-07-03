"""Feature preparation and similarity scoring between expert scene templates
and worker candidate windows.

score = 0.50*keypoint + 0.30*flow + 0.20*duration
(image embedding term from the plan is deferred; its weight is folded
into duration for version 1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WEIGHTS = {"keypoint": 0.50, "flow": 0.30, "duration": 0.20}

KEYPOINT_CHANNELS = [
    "left_hand_cx", "left_hand_cy", "right_hand_cx", "right_hand_cy",
    "left_hand_speed", "right_hand_speed", "hands_distance",
]
FLOW_CHANNELS = [
    "left_hand_flow_mean_mag", "right_hand_flow_mean_mag",
]
DTW_SAMPLES = 24  # every sequence is resampled to this length before DTW
# NOTE: long occlusion gaps get straight-line interpolated across, which can
# fabricate a fake hand trajectory for DTW. A gap-length cap (hold last known
# position instead) was tried and measurably changes matching results on the
# reference video (17->13 segments, 0->4 missing) with no ground truth
# available to say which is more correct, so it was reverted rather than
# risk silently regressing this already-tuned pipeline. Revisit with labeled
# data before changing this again.


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate gaps, normalize signals to comparable ranges."""
    out = df.copy()
    for col in KEYPOINT_CHANNELS:
        out[col] = out[col].interpolate(limit_direction="both").fillna(0.0)
    # robust-scale speeds and flow so expert/worker fps+resolution differences cancel
    for col in ["left_hand_speed", "right_hand_speed"] + FLOW_CHANNELS:
        scale = np.percentile(out[col].abs(), 95)
        out[col] = out[col] / max(scale, 1e-6)
    return out


def resample(seq: np.ndarray, n: int = DTW_SAMPLES) -> np.ndarray:
    """Linear-resample (T, D) sequence to (n, D)."""
    if len(seq) == 0:
        return np.zeros((n, seq.shape[1] if seq.ndim > 1 else 1))
    t_old = np.linspace(0, 1, len(seq))
    t_new = np.linspace(0, 1, n)
    if seq.ndim == 1:
        seq = seq[:, None]
    return np.stack([np.interp(t_new, t_old, seq[:, d]) for d in range(seq.shape[1])], axis=1)


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """DTW with euclidean local cost; a, b are (T, D). Returns path-normalized cost.

    Both inputs are fixed-length (DTW_SAMPLES) after `resample`, so this runs
    on tiny matrices (~24x24) but is called once per (step, window size,
    scene) combination in the matching loop — vectorizing the inner loop over
    j (instead of a pure Python double loop) keeps that call cheap.
    """
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        row_prev = acc[i - 1]
        row = acc[i]
        ci = cost[i - 1]
        # acc[i, j] = cost[i-1,j-1] + min(acc[i-1,j], acc[i,j-1], acc[i-1,j-1])
        # the acc[i,j-1] term still has a strict left-to-right dependency
        # within the row, so only the other two terms can be vectorized.
        diag_up = np.minimum(row_prev[1:], row_prev[:-1])
        for j in range(1, m + 1):
            row[j] = ci[j - 1] + min(diag_up[j - 1], row[j - 1])
    return float(acc[n, m] / (n + m))


def dtw_score(a: np.ndarray, b: np.ndarray, temperature: float) -> float:
    return float(np.exp(-dtw_distance(a, b) / temperature))


def duration_score(worker_dur: float, expert_dur: float) -> float:
    ratio = max(worker_dur, 1e-3) / max(expert_dur, 1e-3)
    return float(np.exp(-abs(np.log(ratio))))


class SceneTemplate:
    """Pre-resampled expert scene signals."""

    def __init__(self, scene_index: int, label: str, operations: list[str],
                 start: float, end: float, seg: pd.DataFrame):
        self.scene_index = scene_index
        self.label = label
        self.operations = operations
        self.start = start
        self.end = end
        self.duration = end - start
        self.kp = resample(seg[KEYPOINT_CHANNELS].to_numpy())
        self.flow = resample(seg[FLOW_CHANNELS].to_numpy())

    def to_json(self) -> dict:
        return {
            "scene_index": self.scene_index, "label": self.label,
            "operations": self.operations, "start": self.start, "end": self.end,
            "duration": self.duration,
            "keypoint_template": self.kp.tolist(), "flow_template": self.flow.tolist(),
        }


def build_templates(expert_df: pd.DataFrame, scenes, fps: float) -> list[SceneTemplate]:
    templates = []
    for sc in scenes:
        f0, f1 = int(sc.start * fps), max(int(sc.end * fps), int(sc.start * fps) + 2)
        seg = expert_df.iloc[f0:f1]
        templates.append(SceneTemplate(sc.scene_index, sc.label, sc.operations, sc.start, sc.end, seg))
    return templates


def window_scores(win: pd.DataFrame, win_duration: float, tpl: SceneTemplate,
                  kp_temp: float = 1.0, flow_temp: float = 1.0) -> dict[str, float]:
    kp = resample(win[KEYPOINT_CHANNELS].to_numpy())
    fl = resample(win[FLOW_CHANNELS].to_numpy())
    s_kp = dtw_score(kp, tpl.kp, kp_temp)
    s_fl = dtw_score(fl, tpl.flow, flow_temp)
    s_du = duration_score(win_duration, tpl.duration)
    total = (WEIGHTS["keypoint"] * s_kp + WEIGHTS["flow"] * s_fl
             + WEIGHTS["duration"] * s_du)
    return {"total": total, "keypoint": s_kp, "flow": s_fl, "duration": s_du}
