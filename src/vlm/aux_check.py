"""Tier 2 (proposal §3.2): presence check of auxiliary operations inside
each aligned segment.

A composite expert scene holds one main operation plus auxiliary operations
that can be sub-second (backtack lever flick, button press, thread cut...).
These cannot be segmented at the observation-step stride — instead, for each
matched scene that has aux operations we ask the VLM a directed yes/no
question over dense frame clusters sampled around the segment's motion
spikes (a short aux action always produces a flow spike), corroborated by a
cheap activity detector on the machine-button/lever zone when the ROI file
provides one. The output is a present / absent / uncertain checklist — the
POC never claims an exact timestamp for an aux operation.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.vlm.cache import VlmCache
from src.vlm.frames import FrameSampler
from src.vlm.openrouter_client import OpenRouterClient, VlmError
from src.vlm.scene_prompts import PROMPT_VERSION, build_aux_check_messages, describe_op

PRESENT, ABSENT, UNCERTAIN = "present", "absent", "uncertain"

SPIKE_CLUSTERS_PER_SEGMENT = 2   # dense frame clusters sampled around motion peaks
CLUSTER_LEN_FRAMES = 6           # consecutive full-fps frames per cluster
MIN_VLM_CONFIDENCE = 0.6         # below this, a yes/no collapses to UNCERTAIN
ZONE_ACTIVITY_THRESH = 2.0       # zone spike ratio vs segment median => "active"

DEFAULT_AUX_CONFIG_PATH = Path("configs/aux_operations.json")

# op-name keyword (lowercase) -> named ROI zone used by the cheap detector
ZONE_FOR_OP_KEYWORD = {
    "cần gạt": "lever",
    "nút nhấn": "machine_button",
    "cắt chỉ": "machine_button",
}


@dataclass
class AuxCheckResult:
    scene_index: int
    scene_label: str
    operation: str
    verdict: str                  # present | absent | uncertain
    confidence: float
    worker_time: list[float]      # [start, end] of the checked segment
    evidence: str = ""
    zone_activity: float | None = None   # spike ratio in the op's machine zone
    frames_checked: list[int] | None = None
    cached: bool = False

    def to_json(self) -> dict:
        return asdict(self)


def load_aux_config(path: str | Path | None = None) -> dict:
    """Optional config: {"always_aux": [op names], "never_aux": [op names]}.
    Used to override the TMU-based main/aux split."""
    p = Path(path) if path else DEFAULT_AUX_CONFIG_PATH
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def split_main_aux(scene, task_spec, aux_config: dict | None = None) -> tuple[str, list[str]]:
    """(main_op, aux_ops) for a scene. In a composite scene the main
    operation is the one with the largest TMU time budget; the rest are
    auxiliary. `always_aux`/`never_aux` in the config override the heuristic."""
    ops = [op for op in scene.operations if op != "UNKNOWN"]
    if len(ops) <= 1:
        return (ops[0] if ops else "UNKNOWN"), []
    cfg = aux_config or {}
    always = {o.lower() for o in cfg.get("always_aux", [])}
    never = {o.lower() for o in cfg.get("never_aux", [])}

    def tmu(op_name: str) -> float:
        for o in task_spec.operations:
            if o.name.lower() == op_name.lower():
                return o.tmu_manual + o.tmu_machine
        return 0.0

    forced_main = [op for op in ops if op.lower() in never]
    candidates = forced_main or [op for op in ops if op.lower() not in always] or ops
    main = max(candidates, key=tmu)
    return main, [op for op in ops if op != main]


def motion_spike_clusters(worker_df: pd.DataFrame, f0: int, f1: int, fps: float,
                          n_clusters: int = SPIKE_CLUSTERS_PER_SEGMENT,
                          cluster_len: int = CLUSTER_LEN_FRAMES) -> list[list[int]]:
    """Clusters of consecutive full-fps frame indices centered on the
    segment's optical-flow peaks (short aux actions always spike the flow)."""
    from scipy.signal import find_peaks

    f0, f1 = max(0, f0), min(len(worker_df), f1)
    if f1 - f0 < 3:
        return []
    seg = worker_df.iloc[f0:f1]
    motion = (seg["left_hand_flow_mean_mag"].fillna(0.0)
              + seg["right_hand_flow_mean_mag"].fillna(0.0)).to_numpy()
    peaks, props = find_peaks(motion, distance=max(3, int(0.3 * fps)))
    if len(peaks) == 0:
        peaks = np.array([int(np.argmax(motion))])
        props = {}
    order = np.argsort(motion[peaks])[::-1][:n_clusters]
    clusters = []
    for pk in sorted(peaks[order]):
        center = f0 + int(pk)
        half = cluster_len // 2
        c0 = max(f0, center - half)
        frames = list(range(c0, min(f1, c0 + cluster_len)))
        if frames:
            clusters.append(frames)
    return clusters


def zone_activity(video_path: str | Path, zone_norm: list[float],
                  frames: list[int], baseline_frames: list[int]) -> float | None:
    """Cheap corroborating signal for button/lever ops: mean frame-difference
    energy inside the zone over `frames`, as a ratio to the same energy over
    `baseline_frames`. > ZONE_ACTIVITY_THRESH means something moved there."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x1, y1 = int(zone_norm[0] * w), int(zone_norm[1] * h)
    x2, y2 = max(x1 + 2, int(zone_norm[2] * w)), max(y1 + 2, int(zone_norm[3] * h))

    def diff_energy(idxs: list[int]) -> float | None:
        prev, vals = None, []
        for fi in sorted(idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            patch = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            if prev is not None and prev.shape == patch.shape:
                vals.append(float(np.abs(patch.astype(np.int16) - prev).mean()))
            prev = patch
        return float(np.mean(vals)) if vals else None

    act = diff_energy(frames)
    base = diff_energy(baseline_frames)
    cap.release()
    if act is None or base is None:
        return None
    return act / max(base, 1e-3)


def zone_for_op(op_name: str, zones: dict) -> list[float] | None:
    low = op_name.lower()
    for kw, zone_name in ZONE_FOR_OP_KEYWORD.items():
        if kw in low and zone_name in zones:
            return zones[zone_name]
    return None


class AuxChecker:
    """Runs the per-segment aux-operation checks. Works in three modes:
    VLM + zone detector (best), zone detector only (no API key), or
    neither (verdicts all UNCERTAIN, checklist still emitted)."""

    def __init__(self, client: OpenRouterClient | None, cache: VlmCache | None,
                 sampler: FrameSampler | None, video_path: str | Path | None,
                 zones: dict | None, descriptions: dict[str, str],
                 video_fp: str = "", log=lambda *_a: None):
        self.client = client if client and client.available else None
        self.cache = cache
        self.sampler = sampler
        self.video_path = video_path
        self.zones = zones or {}
        self.descriptions = descriptions
        self.video_fp = video_fp
        self.log = log

    def _ask_vlm(self, op: str, scene_label: str, frames: list[int]) -> dict | None:
        if self.client is None or self.sampler is None:
            return None
        key = VlmCache.key(kind="aux", video=self.video_fp, frames=frames, op=op,
                           prompt_version=PROMPT_VERSION, model=self.client.model)
        cached = self.cache.get(key) if self.cache else None
        if cached is not None:
            return {**cached, "cached": True}
        b64s = self.sampler.jpeg_b64_many(frames)
        if not b64s:
            return None
        messages = build_aux_check_messages(op, describe_op(op, self.descriptions),
                                            b64s, scene_label)
        try:
            reply = self.client.chat_json(messages)
        except VlmError as e:
            self.log(f"VLM aux check failed for '{op}': {e}")
            return None
        out = {"answer": str(reply.get("answer", "uncertain")).lower(),
               "confidence": float(np.clip(float(reply.get("confidence", 0.0)), 0.0, 1.0)),
               "evidence": str(reply.get("evidence", ""))}
        if self.cache:
            self.cache.put(key, out)
        return out

    def check_segment(self, scene, aux_ops: list[str], seg_start: float, seg_end: float,
                      worker_df: pd.DataFrame, fps: float) -> list[AuxCheckResult]:
        f0, f1 = int(seg_start * fps), int(seg_end * fps)
        clusters = motion_spike_clusters(worker_df, f0, f1, fps)
        all_frames = [fi for cl in clusters for fi in cl]
        # zone baseline: evenly spread frames over the whole segment
        baseline = sorted({int(v) for v in np.linspace(f0, max(f0 + 1, f1 - 1), 8)})

        results = []
        for op in aux_ops:
            zone = zone_for_op(op, self.zones)
            z_act = (zone_activity(self.video_path, zone, all_frames, baseline)
                     if zone is not None and self.video_path and all_frames else None)

            # VLM votes per spike cluster; strongest positive wins ("present
            # somewhere inside the segment" is the claim, not a timestamp)
            votes = []
            for cl in clusters:
                ans = self._ask_vlm(op, scene.label, cl)
                if ans is not None:
                    votes.append(ans)

            verdict, conf, evidence = self._combine(votes, z_act)
            results.append(AuxCheckResult(
                scene_index=scene.scene_index, scene_label=scene.label, operation=op,
                verdict=verdict, confidence=round(conf, 3),
                worker_time=[round(seg_start, 2), round(seg_end, 2)],
                evidence=evidence, zone_activity=round(z_act, 2) if z_act is not None else None,
                frames_checked=all_frames or None,
                cached=all(v.get("cached") for v in votes) if votes else False))
        return results

    @staticmethod
    def _combine(votes: list[dict], z_act: float | None) -> tuple[str, float, str]:
        """Combine per-cluster VLM answers and the zone-activity signal into
        one verdict. Any confident 'yes' => present; all confident 'no' =>
        absent; everything else => uncertain. Zone activity only breaks ties
        (VLM unavailable/uncertain)."""
        yes = [v for v in votes if v["answer"] == "yes" and v["confidence"] >= MIN_VLM_CONFIDENCE]
        no = [v for v in votes if v["answer"] == "no" and v["confidence"] >= MIN_VLM_CONFIDENCE]
        if yes:
            best = max(yes, key=lambda v: v["confidence"])
            return PRESENT, best["confidence"], best["evidence"]
        if votes and len(no) == len(votes):
            best = max(no, key=lambda v: v["confidence"])
            return ABSENT, best["confidence"], best["evidence"]
        if not votes and z_act is not None:
            # zone detector alone: weak evidence either way
            if z_act >= ZONE_ACTIVITY_THRESH:
                return PRESENT, 0.5, f"zone activity spike (x{z_act:.1f} vs baseline), no VLM"
            return UNCERTAIN, 0.3, f"no zone activity spike (x{z_act:.1f}), no VLM"
        if votes:
            evid = "; ".join(v["evidence"] for v in votes if v.get("evidence"))[:300]
            return UNCERTAIN, max(v["confidence"] for v in votes), evid
        return UNCERTAIN, 0.0, "no VLM available and no zone mapped for this operation"


def run_aux_checks(segments, scenes, task_spec, worker_df: pd.DataFrame, fps: float,
                   checker: AuxChecker, aux_config: dict | None = None,
                   log=lambda *_a: None) -> list[dict]:
    """Checklist over all decoded segments matched to a scene with aux ops.
    Returns JSON-ready rows (one per scene occurrence x aux op). Scenes with
    aux ops that were never matched are reported as uncertain (nowhere to
    look)."""
    checklist: list[dict] = []
    checked_scenes: set[int] = set()
    for seg in segments:
        idx = seg.matched_expert_scene_index
        if idx is None:
            continue
        scene = scenes[idx]
        _, aux_ops = split_main_aux(scene, task_spec, aux_config)
        if not aux_ops:
            continue
        checked_scenes.add(idx)
        log(f"aux check: scene {idx} ({scene.label}) "
            f"@ {seg.start_time:.1f}-{seg.end_time:.1f}s, aux={aux_ops}")
        for r in checker.check_segment(scene, aux_ops, seg.start_time, seg.end_time,
                                       worker_df, fps):
            checklist.append(r.to_json())
    for scene in scenes:
        _, aux_ops = split_main_aux(scene, task_spec, aux_config)
        if aux_ops and scene.scene_index not in checked_scenes:
            for op in aux_ops:
                checklist.append(AuxCheckResult(
                    scene_index=scene.scene_index, scene_label=scene.label, operation=op,
                    verdict=UNCERTAIN, confidence=0.0, worker_time=[],
                    evidence="scene has no matched worker segment").to_json())
    return checklist
