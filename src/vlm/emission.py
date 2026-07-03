"""Tier 1 (proposal §3.1): VLM emission scores for the Viterbi decoder.

Change-point-first sampling: the worker video is cut into ~15-30 candidate
segments at motion change-points, each segment gets 3-5 evenly spaced
keyframes (ROI crops), and one VLM call classifies the segment against the
full scene catalog (optionally with expert reference crops as few-shot).
The per-segment score distributions are broadcast onto the observation-step
grid and blended with the pose/flow emissions (VLM weight ~0.6).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from src.segmentation.candidate_windows import changepoint_proposals
from src.vlm.cache import VlmCache
from src.vlm.frames import FrameSampler
from src.vlm.openrouter_client import OpenRouterClient, VlmError
from src.vlm.scene_prompts import PROMPT_VERSION, build_classify_messages

VLM_BLEND_WEIGHT = 0.6   # weight of the VLM term in the blended emission
KEYFRAMES_PER_SEGMENT = 4
MIN_SEGMENT_S = 1.0      # merge change-point segments shorter than this
MAX_SEGMENTS = 30        # hard cap on VLM calls per video
REFS_PER_SCENE = 1       # few-shot expert reference frames per scene


@dataclass
class VlmSegmentResult:
    segment_id: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    keyframes: list[int]
    scores: dict[int, float]      # scene_index -> 0..1
    evidence: str = ""
    cached: bool = False
    error: str = ""

    def to_json(self) -> dict:
        d = asdict(self)
        d["scores"] = {str(k): round(v, 4) for k, v in self.scores.items()}
        return d


def candidate_segments(worker_df: pd.DataFrame, fps: float,
                       min_dur_s: float = MIN_SEGMENT_S,
                       max_segments: int = MAX_SEGMENTS) -> list[tuple[int, int]]:
    """Cut the video into [f0, f1) segments at motion change-points, merging
    segments shorter than `min_dur_s`, and (if still too many) merging the
    shortest neighbors until under `max_segments`."""
    cps = [0] + changepoint_proposals(worker_df, fps) + [len(worker_df)]
    cps = sorted(set(cps))
    segs: list[list[int]] = []
    for f0, f1 in zip(cps[:-1], cps[1:]):
        if segs and (f1 - f0) < min_dur_s * fps:
            segs[-1][1] = f1          # merge short segment into the previous one
        else:
            segs.append([f0, f1])
    while len(segs) > max_segments:
        i = min(range(len(segs)), key=lambda k: segs[k][1] - segs[k][0])
        j = i - 1 if i > 0 else i + 1  # merge shortest into its shorter neighbor
        a, b = sorted((i, j))
        segs[a][1] = segs[b][1]
        del segs[b]
    return [(f0, f1) for f0, f1 in segs if f1 - f0 >= 3]


def sample_keyframes(f0: int, f1: int, n: int = KEYFRAMES_PER_SEGMENT) -> list[int]:
    """Evenly spaced interior keyframes of [f0, f1)."""
    n = max(1, min(n, f1 - f0))
    pos = np.linspace(f0, f1 - 1, n + 2)[1:-1] if n + 2 <= f1 - f0 else np.linspace(f0, f1 - 1, n)
    return sorted({int(round(p)) for p in pos})


def expert_reference_frames(expert_sampler: FrameSampler, scenes, expert_fps: float,
                            refs_per_scene: int = REFS_PER_SCENE) -> list[tuple[int, str]]:
    """Few-shot [(scene_index, jpeg_b64)] crops from the expert video: the
    temporal midpoint(s) of each scene."""
    refs: list[tuple[int, str]] = []
    for sc in scenes:
        pos = np.linspace(sc.start, sc.end, refs_per_scene + 2)[1:-1]
        for t in pos:
            b64 = expert_sampler.jpeg_b64(int(t * expert_fps))
            if b64:
                refs.append((sc.scene_index, b64))
    return refs


def _normalize_scores(raw_scores: dict, n_scenes: int) -> dict[int, float]:
    scores = {i: 0.0 for i in range(n_scenes)}
    for k, v in (raw_scores or {}).items():
        try:
            i = int(k)
        except (TypeError, ValueError):
            continue
        if 0 <= i < n_scenes:
            scores[i] = float(np.clip(float(v), 0.0, 1.0))
    return scores


class VlmEmissionScorer:
    """Classifies worker candidate segments against the expert scene catalog
    via one VLM call per segment, with disk caching."""

    def __init__(self, client: OpenRouterClient, cache: VlmCache,
                 sampler: FrameSampler, scenes, descriptions: dict[str, str],
                 video_fp: str, ref_frames: list[tuple[int, str]] | None = None,
                 keyframes_per_segment: int = KEYFRAMES_PER_SEGMENT,
                 log=lambda *_a: None):
        self.client = client
        self.cache = cache
        self.sampler = sampler
        self.scenes = scenes
        self.descriptions = descriptions
        self.video_fp = video_fp
        self.ref_frames = ref_frames or []
        self.keyframes_per_segment = keyframes_per_segment
        self.log = log

    def score_segment(self, seg_id: str, f0: int, f1: int, fps: float) -> VlmSegmentResult:
        keyframes = sample_keyframes(f0, f1, self.keyframes_per_segment)
        result = VlmSegmentResult(
            segment_id=seg_id, start_frame=f0, end_frame=f1,
            start_time=f0 / fps, end_time=f1 / fps,
            keyframes=keyframes, scores={i: 0.0 for i in range(len(self.scenes))})

        key = self.cache.key(
            kind="classify", video=self.video_fp, frames=keyframes,
            prompt_version=PROMPT_VERSION, model=self.client.model,
            n_scenes=len(self.scenes),
            refs=[i for i, _ in self.ref_frames])
        cached = self.cache.get(key)
        if cached is not None:
            result.scores = _normalize_scores(cached.get("scores"), len(self.scenes))
            result.evidence = cached.get("evidence", "")
            result.cached = True
            return result

        frames_b64 = self.sampler.jpeg_b64_many(keyframes)
        if not frames_b64:
            result.error = "no frames decoded"
            return result
        messages = build_classify_messages(self.scenes, self.descriptions,
                                           seg_id, frames_b64, self.ref_frames)
        try:
            reply = self.client.chat_json(messages)
        except VlmError as e:
            result.error = str(e)
            self.log(f"VLM classify failed for {seg_id}: {e}")
            return result
        result.scores = _normalize_scores(reply.get("scores"), len(self.scenes))
        result.evidence = str(reply.get("evidence", ""))
        self.cache.put(key, {"scores": {str(k): v for k, v in result.scores.items()},
                             "evidence": result.evidence})
        return result

    def score_all(self, worker_df: pd.DataFrame, fps: float) -> list[VlmSegmentResult]:
        segs = candidate_segments(worker_df, fps)
        self.log(f"VLM classifying {len(segs)} candidate segments "
                 f"({self.keyframes_per_segment} keyframes each, model={self.client.model})")
        results = []
        for i, (f0, f1) in enumerate(segs):
            r = self.score_segment(f"seg{i:02d}", f0, f1, fps)
            self.log(f"  {r.segment_id} {r.start_time:.1f}-{r.end_time:.1f}s "
                     f"top={max(r.scores, key=r.scores.get)} "
                     f"({max(r.scores.values()):.2f}){' [cache]' if r.cached else ''}"
                     f"{' ERROR: ' + r.error if r.error else ''}")
            results.append(r)
        return results


def vlm_emission_matrix(results: list[VlmSegmentResult], steps: list[int],
                        n_scenes: int) -> np.ndarray:
    """(T, N) probability-like emission from per-segment score distributions.

    Each observation step takes the (row-normalized) distribution of the
    segment covering its center frame. Steps not covered by any scored
    segment — or covered by a segment whose call failed (all-zero scores) —
    get a uniform row, i.e. the VLM stays neutral there and the pose/flow
    term decides."""
    T = len(steps)
    mat = np.full((T, n_scenes), 1.0 / n_scenes)
    for r in results:
        total = sum(r.scores.values())
        if total <= 0:
            continue
        row = np.array([r.scores[i] for i in range(n_scenes)])
        row = (row + 1e-3) / (row + 1e-3).sum()  # floor so log-space never sees a hard 0
        for si, center in enumerate(steps):
            if r.start_frame <= center < r.end_frame:
                mat[si] = row
    return mat


def blend_emissions(base: np.ndarray, vlm: np.ndarray,
                    weight: float = VLM_BLEND_WEIGHT) -> np.ndarray:
    """Convex row-wise blend of the sharpened pose/flow emissions with the
    VLM emissions; rows renormalized. `weight` is the VLM share (~0.6 per
    the proposal — the VLM is the primary term, pose/flow the fallback)."""
    out = (1.0 - weight) * base + weight * vlm
    return out / out.sum(axis=1, keepdims=True)
