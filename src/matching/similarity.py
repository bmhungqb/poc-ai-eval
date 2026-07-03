"""Feature preparation and similarity scoring between expert scene templates
and worker candidate windows.

score = 0.25*keypoint + 0.15*flow + 0.10*duration + 0.20*frame_nn + 0.30*image_embed
(weights renormalized over whichever terms are actually available -- see
window_scores -- so this still works on frame_features.csv/embeddings.npy
produced by an older version of the extraction pipeline)

keypoint/flow are DTW distances between the window and the scene, both
resampled to a fixed length (DTW_SAMPLES) — this assumes a roughly linear
time correspondence between window and scene. frame_nn and image_embed both
complement that with a Chamfer-style per-frame nearest-neighbor query (no
resampling, no linear-warp assumption): frame_nn over the raw hand
keypoint/flow features, image_embed over a DINOv2 visual embedding of the
hands' crop (src/vision/embed_model.py) -- so image_embed can pick up on
visual state (fabric/tool position) that pose/flow features don't encode.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

WEIGHTS = {"keypoint": 0.25, "flow": 0.15, "duration": 0.10, "frame_nn": 0.20, "image_embed": 0.30}

KEYPOINT_CHANNELS = [
    "left_hand_cx", "left_hand_cy", "right_hand_cx", "right_hand_cy",
    "left_hand_speed", "right_hand_speed", "hands_distance",
]
FLOW_CHANNELS = [
    "left_hand_flow_mean_mag", "right_hand_flow_mean_mag",
]
FEATURE_CHANNELS = KEYPOINT_CHANNELS + FLOW_CHANNELS  # raw (unresampled) per-frame vector, for frame_nn
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
    """Pre-resampled expert scene signals, plus KD-trees over the scene's raw
    (unresampled) per-frame features -- pose/flow (frame_nn_score) and
    DINOv2 image embedding (embed_nn_score) -- for frame-level
    nearest-neighbor queries."""

    def __init__(self, scene_index: int, label: str, operations: list[str],
                 start: float, end: float, seg: pd.DataFrame,
                 embed_seg: np.ndarray | None = None):
        self.scene_index = scene_index
        self.label = label
        self.operations = operations
        self.start = start
        self.end = end
        self.duration = end - start
        self.kp = resample(seg[KEYPOINT_CHANNELS].to_numpy())
        self.flow = resample(seg[FLOW_CHANNELS].to_numpy())
        raw = seg[FEATURE_CHANNELS].to_numpy()
        self.kdtree = cKDTree(raw) if len(raw) > 0 else None
        self.embed_kdtree = (cKDTree(embed_seg) if embed_seg is not None and len(embed_seg) > 0
                             else None)

    def to_json(self) -> dict:
        return {
            "scene_index": self.scene_index, "label": self.label,
            "operations": self.operations, "start": self.start, "end": self.end,
            "duration": self.duration,
            "keypoint_template": self.kp.tolist(), "flow_template": self.flow.tolist(),
        }


def build_templates(expert_df: pd.DataFrame, scenes, fps: float,
                    expert_embeddings: np.ndarray | None = None) -> list[SceneTemplate]:
    templates = []
    for sc in scenes:
        f0, f1 = int(sc.start * fps), max(int(sc.end * fps), int(sc.start * fps) + 2)
        seg = expert_df.iloc[f0:f1]
        embed_seg = expert_embeddings[f0:f1] if expert_embeddings is not None else None
        templates.append(SceneTemplate(sc.scene_index, sc.label, sc.operations, sc.start, sc.end,
                                       seg, embed_seg))
    return templates


def _chamfer_score(query: np.ndarray, tree: cKDTree | None, temperature: float) -> float | None:
    """Mean nearest-neighbor distance from each row of `query` to `tree`,
    converted to a similarity score. None if the query or tree is empty."""
    if tree is None or len(query) == 0:
        return None
    dists, _ = tree.query(query)
    return float(np.exp(-dists.mean() / temperature))


def frame_nn_score(win: pd.DataFrame, tpl: SceneTemplate, temperature: float = 1.0) -> float:
    """Chamfer-style per-frame nearest-neighbor score over raw hand
    keypoint/flow features: for every raw worker frame in the window, query
    its nearest raw expert frame within the scene (native resolution --
    each worker frame effectively "asks" the expert scene "which of your
    frames do I look most like"), then average those distances into a
    single score. Unlike the DTW score this needs no resampling and makes
    no linear-time-correspondence assumption, so it can catch a real match
    even when the window's internal pacing doesn't line up with the scene's.
    """
    if len(win) == 0:
        return 0.0
    return _chamfer_score(win[FEATURE_CHANNELS].to_numpy(), tpl.kdtree, temperature) or 0.0


def embed_nn_score(win_embed: np.ndarray | None, tpl: SceneTemplate, temperature: float = 0.5) -> float | None:
    """Same Chamfer-style nearest-neighbor query as frame_nn_score, but over
    DINOv2 image embeddings of the hands' crop instead of pose/flow
    features -- see src/vision/embed_model.py. Returns None (not 0.0) when
    embeddings aren't available, so window_scores can drop this term from
    the weighted average entirely rather than penalizing every window
    equally for missing data."""
    if win_embed is None or len(win_embed) == 0:
        return None
    return _chamfer_score(win_embed, tpl.embed_kdtree, temperature)


def window_scores(win: pd.DataFrame, win_duration: float, tpl: SceneTemplate,
                  kp_temp: float = 1.0, flow_temp: float = 1.0, nn_temp: float = 1.0,
                  win_embed: np.ndarray | None = None, embed_temp: float = 0.5) -> dict[str, float]:
    kp = resample(win[KEYPOINT_CHANNELS].to_numpy())
    fl = resample(win[FLOW_CHANNELS].to_numpy())
    scores = {
        "keypoint": dtw_score(kp, tpl.kp, kp_temp),
        "flow": dtw_score(fl, tpl.flow, flow_temp),
        "duration": duration_score(win_duration, tpl.duration),
        "frame_nn": frame_nn_score(win, tpl, nn_temp),
    }
    s_embed = embed_nn_score(win_embed, tpl, embed_temp)
    if s_embed is not None:
        scores["image_embed"] = s_embed
    weight_sum = sum(WEIGHTS[k] for k in scores)
    total = sum(WEIGHTS[k] * v for k, v in scores.items()) / weight_sum
    return {"total": total, **scores}
