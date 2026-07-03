"""Error detection over the decoded path and report file generation."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.matching.viterbi_decoder import DecodedSegment

TIMING = {"too_fast": 0.6, "too_slow": 1.6}
MIN_EXTRA_DURATION = 0.6      # ignore blips shorter than this
LOW_CONFIDENCE = 0.35


def detect_errors(segments: list[DecodedSegment], scenes, task_spec) -> list[dict]:
    errors: list[dict] = []
    scene_runs: dict[int, list[DecodedSegment]] = {}
    for seg in segments:
        if seg.matched_expert_scene_index is not None:
            scene_runs.setdefault(seg.matched_expert_scene_index, []).append(seg)

    # MISSING: expert scenes never visited
    for sc in scenes:
        if sc.scene_index not in scene_runs:
            errors.append({
                "type": "MISSING",
                "expert_scene_index": sc.scene_index,
                "operations": sc.operations,
                "expert_time": [sc.start, sc.end],
                "message": f"Expert scene {sc.scene_index} ({sc.label}) has no confident match in the worker video.",
            })

    # WRONG_ORDER: backward transition between matched scenes
    prev_idx = None
    for seg in segments:
        idx = seg.matched_expert_scene_index
        if idx is None:
            continue
        if prev_idx is not None and idx < prev_idx:
            errors.append({
                "type": "WRONG_ORDER",
                "expert_scene_index": idx,
                "operations": seg.operations,
                "worker_time": [seg.start_time, seg.end_time],
                "message": f"Worker performed scene {idx} ({' + '.join(seg.operations)}) after scene {prev_idx} — out of expert order.",
            })
        prev_idx = idx

    # EXTRA_ACTION
    for seg in segments:
        if seg.assigned_state == "EXTRA" and seg.end_time - seg.start_time >= MIN_EXTRA_DURATION:
            errors.append({
                "type": "EXTRA_ACTION",
                "worker_time": [seg.start_time, seg.end_time],
                "message": f"Worker performed an action at {seg.start_time:.1f}-{seg.end_time:.1f}s that matches no expert scene.",
            })

    # DUPLICATED_ACTION: scene appears in >1 disjoint runs beyond expected count.
    for idx, runs in scene_runs.items():
        if len(runs) <= 1:
            continue
        ops = scenes[idx].operations
        expected = max(1, round(min(task_spec.operation_frequency(op) for op in ops if op != "UNKNOWN") or 1))
        # frequency counts operation occurrences across ALL scenes; discount scenes sharing ops
        scenes_with_same_ops = sum(1 for sc in scenes if set(sc.operations) & set(ops) - {"UNKNOWN"})
        allowed = max(1, expected - max(0, scenes_with_same_ops - 1))
        if len(runs) > allowed:
            errors.append({
                "type": "DUPLICATED_ACTION",
                "expert_scene_index": idx,
                "operations": ops,
                "worker_time": [[r.start_time, r.end_time] for r in runs],
                "message": f"Scene {idx} ({scenes[idx].label}) was matched {len(runs)} separate times (expected {allowed}).",
            })
    return errors


def timing_status(worker_dur: float, expert_dur: float) -> str:
    ratio = worker_dur / max(expert_dur, 1e-6)
    if ratio < TIMING["too_fast"]:
        return "TOO_FAST"
    if ratio > TIMING["too_slow"]:
        return "TOO_SLOW"
    return "NORMAL"


def build_report(task_spec, segments: list[DecodedSegment], errors: list[dict],
                 expert_video: str, worker_video: str, out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = task_spec.expert_scenes

    seg_rows = []
    for seg in segments:
        idx = seg.matched_expert_scene_index
        if idx is None:
            status, tstat = "EXTRA", ""
        else:
            status = "LOW_CONFIDENCE" if seg.confidence < LOW_CONFIDENCE else "MATCHED"
            tstat = timing_status(seg.end_time - seg.start_time, scenes[idx].duration)
        seg_rows.append({
            "worker_start": round(seg.start_time, 2),
            "worker_end": round(seg.end_time, 2),
            "matched_expert_scene_index": idx,
            "operations": seg.operations,
            "confidence": seg.confidence,
            "status": status,
            "timing_status": tstat,
        })

    counts = {
        "num_expected_scenes": len(scenes),
        "num_detected_segments": len(segments),
        "missing_count": sum(e["type"] == "MISSING" for e in errors),
        "extra_count": sum(e["type"] == "EXTRA_ACTION" for e in errors),
        "wrong_order_count": sum(e["type"] == "WRONG_ORDER" for e in errors),
        "duplicated_count": sum(e["type"] == "DUPLICATED_ACTION" for e in errors),
    }
    matched = [r for r in seg_rows if r["status"] == "MATCHED"]
    penalty = (1.5 * counts["missing_count"] + 0.8 * counts["extra_count"]
               + 3.0 * counts["wrong_order_count"] + 1.0 * counts["duplicated_count"])
    avg_conf = sum(r["confidence"] for r in matched) / len(matched) if matched else 0.0
    overall = max(0.0, round(10 * avg_conf - penalty, 2))

    report = {
        "task_name": task_spec.task_name,
        "expert_video": expert_video,
        "worker_video": worker_video,
        "summary": {**counts, "avg_matched_confidence": round(avg_conf, 3),
                    "overall_score": overall},
        "segments": seg_rows,
        "errors": errors,
    }
    (out_dir / "alignment_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_rows = [{**r, "operations": " + ".join(r["operations"])} for r in seg_rows]
    pd.DataFrame(csv_rows).to_csv(out_dir / "alignment_report.csv", index=False)
    return report
