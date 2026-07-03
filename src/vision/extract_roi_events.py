"""Convert continuous frame features into binary ROI event channels.

Events are computed on the full feature table (post-extraction) so thresholds
can be set adaptively from each video's own signal statistics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EVENT_COLUMNS = [
    "ev_left_hand_in_needle",
    "ev_right_hand_in_needle",
    "ev_needle_flow_active",
    "ev_fabric_motion_active",
    "ev_button_interaction",
    "ev_lever_interaction",
    "ev_high_motion_peak",
    "ev_pause",
]


def _active(series: pd.Series, rel_thresh: float = 0.35) -> np.ndarray:
    """Binary 'activity' from a flow-magnitude curve using an adaptive threshold."""
    s = series.fillna(0.0).to_numpy()
    lo, hi = np.percentile(s, 10), np.percentile(s, 95)
    thresh = lo + rel_thresh * max(hi - lo, 1e-6)
    return (s > thresh).astype(int)


def compute_events(df: pd.DataFrame) -> pd.DataFrame:
    ev = pd.DataFrame(index=df.index)
    ev["ev_left_hand_in_needle"] = df["left_hand_in_needle_roi"].fillna(0).astype(int)
    ev["ev_right_hand_in_needle"] = df["right_hand_in_needle_roi"].fillna(0).astype(int)
    ev["ev_needle_flow_active"] = _active(df["needle_flow_mean_mag"])
    ev["ev_fabric_motion_active"] = _active(df["fabric_area_flow_mean_mag"])

    # Button/lever interaction: motion in that ROI while a hand is near it (proxy: hand present)
    hand_present = ((df["left_hand_present"] + df["right_hand_present"]) > 0).astype(int)
    ev["ev_button_interaction"] = _active(df["machine_button_flow_mean_mag"], 0.5) & hand_present
    ev["ev_lever_interaction"] = _active(df["lever_flow_mean_mag"], 0.5) & hand_present

    total_motion = (
        df["needle_flow_mean_mag"].fillna(0)
        + df["fabric_area_flow_mean_mag"].fillna(0)
        + df["left_hand_flow_mean_mag"].fillna(0)
        + df["right_hand_flow_mean_mag"].fillna(0)
    )
    hi = np.percentile(total_motion, 90)
    lo = np.percentile(total_motion, 25)
    ev["ev_high_motion_peak"] = (total_motion > hi).astype(int)
    ev["ev_pause"] = (total_motion < max(lo, 1e-6)).astype(int)
    return ev


def transitions(ev: pd.DataFrame) -> pd.DataFrame:
    """Rising/falling edges, e.g. enter/leave events for each channel."""
    out = pd.DataFrame(index=ev.index)
    for col in ev.columns:
        d = ev[col].diff().fillna(0)
        out[col + "_start"] = (d > 0).astype(int)
        out[col + "_stop"] = (d < 0).astype(int)
    return out
