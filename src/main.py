"""CLI for the sewing operation alignment POC.

  python -m src.main extract-expert --course-json ... --video ... --roi ... --out outputs/expert
  python -m src.main extract-worker --video ... --roi ... --out outputs/worker
  python -m src.main match --expert-dir ... --worker-dir ... --course-json ... --out outputs/reports
  python -m src.main run-all --course-json ... --expert-video ... --worker-video ... --roi ... --out outputs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.io.load_course_data import load_task_spec
from src.vision.roi import load_rois


def cmd_extract(video: str, roi: str | None, out: str, role: str):
    from src.vision.extract_frame_features import extract_video_features
    if roi:
        rois = load_rois(roi, role)
    else:
        from src.vision.auto_roi import estimate_rois
        print(f"Estimating ROIs automatically for {video} ...")
        rois = estimate_rois(video)
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "roi_auto.json").write_text(json.dumps(rois, indent=2))
        print(f"  needle={rois['needle']} (saved to {out}/roi_auto.json)")
    extract_video_features(video, rois, out)


def cmd_match(expert_dir: str, worker_dir: str, course_json: str, out: str):
    from src.matching import viterbi_decoder
    from src.matching.similarity import build_templates, prepare_features, window_scores
    from src.reporting.build_report import build_report, detect_errors
    from src.reporting.visualize_alignment import save_score_matrix, save_timeline_html
    from src.segmentation.candidate_windows import (
        STRIDE_S, WINDOW_SIZES_S, all_candidates, grid_steps)

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = load_task_spec(course_json)
    scenes = spec.expert_scenes

    expert_meta = json.loads((Path(expert_dir) / "meta.json").read_text())
    worker_meta = json.loads((Path(worker_dir) / "meta.json").read_text())
    expert_df = prepare_features(pd.read_csv(Path(expert_dir) / "frame_features.csv"))
    worker_df = prepare_features(pd.read_csv(Path(worker_dir) / "frame_features.csv"))
    e_fps, w_fps = expert_meta["fps"], worker_meta["fps"]

    # expert templates
    templates = build_templates(expert_df, scenes, e_fps)
    (Path(expert_dir) / "templates.json").write_text(
        json.dumps([t.to_json() for t in templates], ensure_ascii=False, indent=1), encoding="utf-8")

    # worker candidates
    candidates = all_candidates(worker_df, w_fps)
    candidates.to_csv(Path(worker_dir) / "candidate_windows.csv", index=False)

    # emission matrix: per observation step, best window size per scene
    steps = grid_steps(len(worker_df), w_fps)
    n_steps, n_scenes = len(steps), len(scenes)
    emissions = np.zeros((n_steps, n_scenes))
    motion = (worker_df["needle_flow_mean_mag"] + worker_df["fabric_area_flow_mean_mag"]
              + worker_df["left_hand_flow_mean_mag"] + worker_df["right_hand_flow_mean_mag"]).to_numpy()
    motion = np.clip(motion / max(np.percentile(motion, 90), 1e-6), 0, 1)

    for si, center in enumerate(tqdm(steps, desc="score steps")):
        for size_s in WINDOW_SIZES_S:
            half = int(size_s * w_fps / 2)
            f0, f1 = max(0, center - half), min(len(worker_df), center + half)
            if f1 - f0 < 3:
                continue
            win = worker_df.iloc[f0:f1]
            dur = (f1 - f0) / w_fps
            for ti, tpl in enumerate(templates):
                s = window_scores(win, dur, tpl)["total"]
                if s > emissions[si, ti]:
                    emissions[si, ti] = s

    save_score_matrix(emissions, out_dir, [sc.label for sc in scenes])

    # sharpen emissions (remove per-scene bias, make steps discriminative)
    sharp = viterbi_decoder.sharpen_emissions(emissions)
    # EXTRA competes at ~uniform level, gated by motion in the worker video
    step_motion = np.array([motion[min(c, len(motion) - 1)] for c in steps])
    extra_emission = np.clip(step_motion * (1.2 / n_scenes) * (1.0 - sharp.max(axis=1)), 1e-3, 1.0)

    # minimum dwell per scene: half its expert duration, scaled by video-length ratio
    expert_total = max(sc.end for sc in scenes)
    worker_total = len(worker_df) / w_fps
    ratio = worker_total / expert_total
    dwell_steps = [max(1, int(round(0.5 * sc.duration / STRIDE_S * ratio))) for sc in scenes]

    # decode + report
    path = viterbi_decoder.decode(sharp, extra_emission, dwell_steps)
    step_times = [c / w_fps for c in steps]
    segments = viterbi_decoder.merge_path(path, step_times, STRIDE_S, emissions, extra_emission, scenes)
    errors = detect_errors(segments, scenes, spec)
    report = build_report(spec, segments, errors,
                          expert_meta.get("video", ""), worker_meta.get("video", ""), out_dir)
    save_timeline_html(report, scenes, out_dir)

    s = report["summary"]
    print(f"\nTask: {report['task_name']}")
    print(f"Segments: {s['num_detected_segments']}  missing={s['missing_count']} "
          f"extra={s['extra_count']} wrong_order={s['wrong_order_count']} "
          f"duplicated={s['duplicated_count']}  overall={s['overall_score']}")
    print(f"Reports written to {out_dir}")


def main():
    ap = argparse.ArgumentParser(prog="src.main")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("extract-expert")
    p.add_argument("--course-json", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--roi", default=None, help="optional ROI config; auto-estimated when omitted")
    p.add_argument("--out", required=True)

    p = sub.add_parser("extract-worker")
    p.add_argument("--video", required=True)
    p.add_argument("--roi", default=None, help="optional ROI config; auto-estimated when omitted")
    p.add_argument("--out", required=True)

    p = sub.add_parser("match")
    p.add_argument("--expert-dir", required=True)
    p.add_argument("--worker-dir", required=True)
    p.add_argument("--course-json", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("run-all")
    p.add_argument("--course-json", required=True)
    p.add_argument("--expert-video", required=True)
    p.add_argument("--worker-video", required=True)
    p.add_argument("--roi", default=None, help="optional ROI config; auto-estimated when omitted")
    p.add_argument("--out", required=True)

    args = ap.parse_args()
    if args.cmd == "extract-expert":
        cmd_extract(args.video, args.roi, args.out, "expert")
    elif args.cmd == "extract-worker":
        cmd_extract(args.video, args.roi, args.out, "worker")
    elif args.cmd == "match":
        cmd_match(args.expert_dir, args.worker_dir, args.course_json, args.out)
    elif args.cmd == "run-all":
        out = Path(args.out)
        cmd_extract(args.expert_video, args.roi, str(out / "expert"), "expert")
        cmd_extract(args.worker_video, args.roi, str(out / "worker"), "worker")
        cmd_match(str(out / "expert"), str(out / "worker"), args.course_json, str(out / "reports"))


if __name__ == "__main__":
    main()
