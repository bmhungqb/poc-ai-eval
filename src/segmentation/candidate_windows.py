"""Worker candidate segment generation.

Dense multi-scale sliding windows on a fixed time-step grid. Each grid step is
one Viterbi observation; its emission uses the best-matching window size
centered on that step. Change-point proposals are added as extra candidates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WINDOW_SIZES_S = [0.3, 0.5, 0.8, 1.3, 2.0]  # supports ~0.3s actions up to long seams
STRIDE_S = 0.2


@dataclass
class WorkerWindow:
    window_id: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration: float
    source: str  # "sliding" | "changepoint"


def grid_steps(n_frames: int, fps: float, stride_s: float = STRIDE_S) -> list[int]:
    """Frame indices of observation steps."""
    stride_f = max(1, int(round(stride_s * fps)))
    return list(range(0, n_frames, stride_f))


def sliding_windows(n_frames: int, fps: float) -> list[WorkerWindow]:
    windows = []
    for step_i, center in enumerate(grid_steps(n_frames, fps)):
        for size_s in WINDOW_SIZES_S:
            half = int(size_s * fps / 2)
            f0, f1 = max(0, center - half), min(n_frames, center + half)
            if f1 - f0 < 3:
                continue
            windows.append(WorkerWindow(
                window_id=f"s{step_i}_w{size_s}",
                start_frame=f0, end_frame=f1,
                start_time=f0 / fps, end_time=f1 / fps,
                duration=(f1 - f0) / fps, source="sliding"))
    return windows


def changepoint_proposals(df: pd.DataFrame, fps: float) -> list[int]:
    """Simple change-point detection: peaks in the derivative of smoothed motion curves."""
    from scipy.signal import find_peaks

    signals = []
    for col in ["needle_flow_mean_mag", "fabric_area_flow_mean_mag",
                "left_hand_speed", "right_hand_speed",
                "left_hand_dist_needle", "right_hand_dist_needle"]:
        s = df[col].interpolate(limit_direction="both").fillna(0.0).to_numpy()
        scale = np.percentile(np.abs(s), 95)
        signals.append(s / max(scale, 1e-6))
    sig = np.stack(signals)
    win = max(3, int(0.3 * fps))
    kernel = np.ones(win) / win
    smooth = np.stack([np.convolve(s, kernel, mode="same") for s in sig])
    change = np.abs(np.diff(smooth, axis=1)).sum(axis=0)
    peaks, _ = find_peaks(change, distance=max(3, int(0.25 * fps)),
                          height=np.percentile(change, 75))
    return peaks.tolist()


def changepoint_windows(df: pd.DataFrame, fps: float) -> list[WorkerWindow]:
    cps = [0] + changepoint_proposals(df, fps) + [len(df)]
    windows = []
    for i in range(len(cps) - 1):
        f0, f1 = cps[i], cps[i + 1]
        if f1 - f0 < 3:
            continue
        windows.append(WorkerWindow(
            window_id=f"cp{i}", start_frame=f0, end_frame=f1,
            start_time=f0 / fps, end_time=f1 / fps,
            duration=(f1 - f0) / fps, source="changepoint"))
    return windows


def all_candidates(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    wins = sliding_windows(len(df), fps) + changepoint_windows(df, fps)
    return pd.DataFrame([w.__dict__ for w in wins])
