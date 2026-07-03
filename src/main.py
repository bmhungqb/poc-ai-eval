"""CLI for the sewing operation alignment POC.

  python -m src.main extract-expert --course-json ... --video ... --out outputs/expert
  python -m src.main extract-worker --video ... --out outputs/worker
  python -m src.main match --expert-dir ... --worker-dir ... --course-json ... --out outputs/reports
  python -m src.main run-all --course-json ... --expert-video ... --worker-video ... --out outputs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.io.load_course_data import load_task_spec


def cmd_extract(video: str, out: str, roi: str | None = "auto", roi_upscale_width: int = 960,
                flow_overlay: bool = False):
    from src.vision.extract_frame_features import extract_video_features
    extract_video_features(video, out, roi=roi, roi_upscale_width=roi_upscale_width,
                            flow_overlay=flow_overlay)


def _load_video_roi(feature_dir: Path, meta: dict) -> dict:
    """Work-area ROI + named zones for a video, from the extraction output
    (work_area_roi.json), a legacy roi_auto.json, or meta.json. Falls back
    to the full frame."""
    from src.vision.work_area_roi import ROI_FILENAME, load_roi
    for name in (ROI_FILENAME, "roi_auto.json"):
        p = feature_dir / name
        if p.exists():
            return load_roi(p)
    wa = meta.get("work_area_roi")
    return {"work_area": wa or [0.0, 0.0, 1.0, 1.0], "zones": {}, "source": "meta"}


def cmd_match(expert_dir: str, worker_dir: str, course_json: str, out: str, debug: bool = False,
              no_vlm: bool = False, vlm_model: str | None = None,
              vlm_weight: float | None = None, refs_per_scene: int | None = None,
              vlm_cache_dir: str | None = None):
    from src.matching import viterbi_decoder
    from src.matching.similarity import build_templates, prepare_features, window_scores
    from src.reporting.build_report import build_report, detect_errors, format_report_text
    from src.reporting.visualize_alignment import (
        build_frame_labels, save_debug_path, save_debug_signal, save_score_matrix,
        save_timeline_html, save_worker_frame_labels)
    from src.segmentation.candidate_windows import (
        STRIDE_S, WINDOW_SIZES_S, all_candidates, grid_steps)

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "debug"
    if debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str):
        if debug:
            print(f"[debug] {msg}")

    spec = load_task_spec(course_json)
    scenes = spec.expert_scenes

    expert_meta = json.loads((Path(expert_dir) / "meta.json").read_text())
    worker_meta = json.loads((Path(worker_dir) / "meta.json").read_text())
    expert_df = prepare_features(pd.read_csv(Path(expert_dir) / "frame_features.csv"))
    worker_df = prepare_features(pd.read_csv(Path(worker_dir) / "frame_features.csv"))
    e_fps, w_fps = expert_meta["fps"], worker_meta["fps"]

    # image embeddings (optional -- absent if features were extracted with an
    # older version of the pipeline that didn't produce embeddings.npy)
    expert_embed_path = Path(expert_dir) / "embeddings.npy"
    worker_embed_path = Path(worker_dir) / "embeddings.npy"
    expert_embeddings = np.load(expert_embed_path) if expert_embed_path.exists() else None
    worker_embeddings = np.load(worker_embed_path) if worker_embed_path.exists() else None
    if expert_embeddings is None or worker_embeddings is None:
        print("No embeddings.npy found for expert/worker -- image_embed score term "
              "will be skipped (re-run extract-expert/extract-worker to enable it).")

    # expert templates
    log(f"building {len(scenes)} expert scene templates")
    templates = build_templates(expert_df, scenes, e_fps, expert_embeddings)
    (Path(expert_dir) / "templates.json").write_text(
        json.dumps([t.to_json() for t in templates], ensure_ascii=False, indent=1), encoding="utf-8")

    # worker candidates
    candidates = all_candidates(worker_df, w_fps)
    candidates.to_csv(Path(worker_dir) / "candidate_windows.csv", index=False)
    log(f"generated {len(candidates)} worker candidate windows")

    # emission matrix: per observation step, best window size per scene
    steps = grid_steps(len(worker_df), w_fps)
    n_steps, n_scenes = len(steps), len(scenes)
    emissions = np.zeros((n_steps, n_scenes))
    motion = (worker_df["left_hand_flow_mean_mag"] + worker_df["right_hand_flow_mean_mag"]).to_numpy()
    motion = np.clip(motion / max(np.percentile(motion, 90), 1e-6), 0, 1)
    log(f"scoring {n_steps} observation steps x {n_scenes} scenes")

    # per-component score matrices (keypoint/flow/duration/frame_nn/image_embed), for the
    # winning window size at each (step, scene) -- debug-only, to inspect which term drives a match
    component_names = ["keypoint", "flow", "duration", "frame_nn", "image_embed"]
    score_components = {name: np.zeros((n_steps, n_scenes)) for name in component_names} if debug else {}

    for si, center in enumerate(tqdm(steps, desc="score steps")):
        for size_s in WINDOW_SIZES_S:
            half = int(size_s * w_fps / 2)
            f0, f1 = max(0, center - half), min(len(worker_df), center + half)
            if f1 - f0 < 3:
                continue
            win = worker_df.iloc[f0:f1]
            win_embed = worker_embeddings[f0:f1] if worker_embeddings is not None else None
            dur = (f1 - f0) / w_fps
            for ti, tpl in enumerate(templates):
                scores = window_scores(win, dur, tpl, win_embed=win_embed)
                s = scores["total"]
                if s > emissions[si, ti]:
                    emissions[si, ti] = s
                    for name, mat in score_components.items():
                        if name in scores:
                            mat[si, ti] = scores[name]

    save_score_matrix(emissions, out_dir, [sc.label for sc in scenes])
    step_times = [c / w_fps for c in steps]
    if debug:
        save_score_matrix(emissions, debug_dir, [sc.label for sc in scenes],
                          name="01_emissions_raw", title="Raw emissions (before sharpening)")
        save_debug_signal(motion, debug_dir, "00_motion", "motion (0-1)")
        for name, mat in score_components.items():
            if mat.any():
                save_score_matrix(mat, debug_dir, [sc.label for sc in scenes],
                                  name=f"01a_score_{name}", title=f"'{name}' score component (winning window)")

    # sharpen emissions (remove per-scene bias, make steps discriminative)
    sharp = viterbi_decoder.sharpen_emissions(emissions)
    if debug:
        save_score_matrix(sharp, debug_dir, [sc.label for sc in scenes],
                          name="02_emissions_sharp", title="Sharpened emissions (softmax'd)")

    # --- tier 1: VLM emission (proposal §3.1) -- primary term when available ---
    emission_source = "pose_flow"
    vlm_client = vlm_cache = worker_sampler = None
    worker_roi = _load_video_roi(Path(worker_dir), worker_meta)
    worker_video = worker_meta.get("video", "")
    if not no_vlm:
        from src.vlm import emission as vlm_emission_mod
        from src.vlm.cache import VlmCache, video_fingerprint
        from src.vlm.frames import FrameSampler
        from src.vlm.openrouter_client import OpenRouterClient
        from src.vlm.scene_prompts import load_op_descriptions

        vlm_client = OpenRouterClient(**({"model": vlm_model} if vlm_model else {}))
        if not vlm_client.available:
            print("OPENROUTER_API_KEY not set -- VLM emission + aux checks skipped "
                  "(pose/flow fallback).")
        elif not worker_video or not Path(worker_video).exists():
            print(f"worker video not found ({worker_video!r}) -- VLM stages skipped.")
        else:
            vlm_cache = VlmCache(vlm_cache_dir or (out_dir / "vlm_cache"))
            worker_fp = video_fingerprint(worker_video)
            worker_sampler = FrameSampler(worker_video, worker_roi["work_area"])
            descriptions = load_op_descriptions()

            # few-shot reference crops from the expert video (if reachable)
            ref_frames = []
            n_refs = vlm_emission_mod.REFS_PER_SCENE if refs_per_scene is None else refs_per_scene
            expert_video = expert_meta.get("video", "")
            if n_refs > 0 and expert_video and Path(expert_video).exists():
                expert_roi = _load_video_roi(Path(expert_dir), expert_meta)
                expert_sampler = FrameSampler(expert_video, expert_roi["work_area"])
                ref_frames = vlm_emission_mod.expert_reference_frames(
                    expert_sampler, scenes, e_fps, refs_per_scene=n_refs)
                expert_sampler.close()

            scorer = vlm_emission_mod.VlmEmissionScorer(
                vlm_client, vlm_cache, worker_sampler, scenes, descriptions,
                worker_fp, ref_frames=ref_frames, log=print)
            vlm_results = scorer.score_all(worker_df, w_fps)
            (out_dir / "vlm_segments.json").write_text(
                json.dumps([r.to_json() for r in vlm_results], ensure_ascii=False, indent=1),
                encoding="utf-8")
            vlm_mat = vlm_emission_mod.vlm_emission_matrix(vlm_results, steps, n_scenes)
            weight = vlm_emission_mod.VLM_BLEND_WEIGHT if vlm_weight is None else vlm_weight
            sharp = vlm_emission_mod.blend_emissions(sharp, vlm_mat, weight)
            emission_source = f"vlm({vlm_client.model}, w={weight})+pose_flow"
            if debug:
                save_score_matrix(vlm_mat, debug_dir, [sc.label for sc in scenes],
                                  name="02a_emissions_vlm", title="VLM emissions")
                save_score_matrix(sharp, debug_dir, [sc.label for sc in scenes],
                                  name="02b_emissions_blended", title="Blended emissions")

    # EXTRA competes at ~uniform level, gated by motion in the worker video
    step_motion = np.array([motion[min(c, len(motion) - 1)] for c in steps])
    extra_emission = np.clip(step_motion * (1.2 / n_scenes) * (1.0 - sharp.max(axis=1)), 1e-3, 1.0)
    if debug:
        save_debug_signal(extra_emission, debug_dir, "03_extra_emission", "extra emission (0-1)",
                          step_times=step_times)

    # minimum dwell per scene: half its expert duration, scaled by video-length ratio
    expert_total = max(sc.end for sc in scenes)
    worker_total = len(worker_df) / w_fps
    ratio = worker_total / expert_total
    dwell_steps = [max(1, int(round(0.5 * sc.duration / STRIDE_S * ratio))) for sc in scenes]
    log(f"dwell_steps={dwell_steps} (worker/expert duration ratio={ratio:.3f})")

    # decode + report; penalties are retuned when VLM emissions are blended in
    # (per-step log-prob differences are much larger -- proposal §3.4)
    penalties = (viterbi_decoder.PENALTIES_VLM if emission_source != "pose_flow"
                 else viterbi_decoder.PENALTIES)
    path = viterbi_decoder.decode(sharp, extra_emission, dwell_steps, penalties=penalties)
    if debug:
        save_debug_path(path, step_times, debug_dir, name="04_viterbi_path")
    # confidence comes from the matrix that was actually decoded (probability
    # scale, comparable against uniform 1/n_scenes -> UNMATCHED threshold)
    segments = viterbi_decoder.merge_path(path, step_times, STRIDE_S, sharp, extra_emission, scenes)
    log(f"decoded {len(segments)} segments")
    if debug:
        (debug_dir / "05_segments.json").write_text(
            json.dumps([s.__dict__ for s in segments], ensure_ascii=False, indent=1, default=str),
            encoding="utf-8")

        # per-frame nearest-neighbor correspondence for each decoded (non-EXTRA) segment:
        # which exact expert frame each worker frame matched against, for both the
        # pose/flow and image-embedding queries
        from src.matching.similarity import frame_correspondence
        from src.reporting.visualize_alignment import save_frame_correspondence_plot, save_frame_overlays

        corr_rows, overlay_pairs = [], []
        for seg in segments:
            if seg.matched_expert_scene_index is None:
                continue
            tpl = templates[seg.matched_expert_scene_index]
            f0 = int(seg.start_time * w_fps)
            f1 = min(len(worker_df), max(f0 + 1, int(seg.end_time * w_fps)))
            win = worker_df.iloc[f0:f1]
            win_embed = worker_embeddings[f0:f1] if worker_embeddings is not None else None
            corr = frame_correspondence(win, tpl, w_fps, win_embed=win_embed)
            corr["scene"] = seg.matched_expert_scene_index
            corr["expert_time_nn"] = np.where(corr["expert_frame_nn"] >= 0,
                                              corr["expert_frame_nn"] / e_fps, np.nan)
            corr["expert_time_nn_embed"] = np.where(corr["expert_frame_nn_embed"] >= 0,
                                                    corr["expert_frame_nn_embed"] / e_fps, np.nan)
            corr_rows.append(corr)

            valid = corr[corr["expert_frame_nn"] >= 0]
            if len(valid):
                sample = valid.iloc[np.linspace(0, len(valid) - 1, min(6, len(valid))).astype(int)]
                for _, row in sample.iterrows():
                    overlay_pairs.append({
                        "worker_frame": int(row["worker_frame"]), "worker_time": float(row["worker_time"]),
                        "expert_frame": int(row["expert_frame_nn"]), "expert_time": float(row["expert_time_nn"]),
                        "scene": seg.matched_expert_scene_index,
                    })

        if corr_rows:
            all_corr = pd.concat(corr_rows, ignore_index=True)
            all_corr.to_csv(debug_dir / "06_frame_correspondence.csv", index=False)
            save_frame_correspondence_plot(all_corr, debug_dir, name="06_frame_correspondence")
            log(f"wrote per-frame correspondence for {len(all_corr)} worker frames")

        if overlay_pairs:
            save_frame_overlays(overlay_pairs, expert_meta.get("video", ""), worker_meta.get("video", ""),
                                debug_dir / "frame_overlays", log=log)

    # --- tier 2: aux-operation checklist (proposal §3.2) ---
    aux_checklist = None
    if not no_vlm and worker_video and Path(worker_video).exists():
        from src.vlm.aux_check import AuxChecker, load_aux_config, run_aux_checks
        from src.vlm.cache import VlmCache, video_fingerprint
        from src.vlm.frames import FrameSampler
        from src.vlm.scene_prompts import load_op_descriptions

        if worker_sampler is None:
            worker_sampler = FrameSampler(worker_video, worker_roi["work_area"])
        if vlm_cache is None:
            vlm_cache = VlmCache(vlm_cache_dir or (out_dir / "vlm_cache"))
        checker = AuxChecker(
            vlm_client, vlm_cache, worker_sampler, worker_video,
            zones=worker_roi.get("zones"), descriptions=load_op_descriptions(),
            video_fp=video_fingerprint(worker_video), log=print)
        aux_checklist = run_aux_checks(segments, scenes, spec, worker_df, w_fps,
                                       checker, load_aux_config(), log=log)
    if worker_sampler is not None:
        worker_sampler.close()

    errors = detect_errors(segments, scenes, spec)
    report = build_report(spec, segments, errors,
                          expert_meta.get("video", ""), worker_meta.get("video", ""), out_dir,
                          aux_checklist=aux_checklist, emission_source=emission_source)
    save_timeline_html(report, scenes, out_dir)

    # per-frame "what operation is the worker doing right now" labels + annotated video
    labels_df = build_frame_labels(worker_df, report)
    save_worker_frame_labels(labels_df, worker_meta.get("video", ""), out_dir, log=print)

    print("\n" + format_report_text(report))
    print(f"\nReports written to {out_dir}")
    if debug:
        print(f"Debug artifacts written to {debug_dir}")


def _add_roi_args(p):
    p.add_argument("--roi", default="auto",
                   help="work-area ROI: 'auto' (estimate from hand detections, default), "
                        "path to a work_area_roi.json / legacy roi_auto.json, or 'none'")
    p.add_argument("--roi-upscale-width", type=int, default=960,
                   help="upscale the ROI crop to this width before hand detection")
    p.add_argument("--flow-overlay", action="store_true",
                   help="write <out>/overlay_optical_flow.mp4, an HSV-encoded visualization "
                        "of the dense Farneback flow field (hue=direction, value=magnitude)")


def _add_vlm_args(p):
    p.add_argument("--no-vlm", action="store_true",
                   help="disable the VLM emission + aux checklist stages (pose/flow only)")
    p.add_argument("--vlm-model", default=None,
                   help="OpenRouter model id (default: %s)" % "google/gemini-2.5-flash")
    p.add_argument("--vlm-weight", type=float, default=None,
                   help="weight of the VLM term in the blended emission (default 0.6)")
    p.add_argument("--refs-per-scene", type=int, default=None,
                   help="few-shot expert reference frames per scene in the classify prompt "
                        "(default 1; 0 disables)")
    p.add_argument("--vlm-cache-dir", default=None,
                   help="VLM response cache directory (default <out>/vlm_cache)")


def _roi_arg(value: str):
    return None if value == "none" else value


def main():
    from src.io.env import load_dotenv
    load_dotenv()  # OPENROUTER_API_KEY etc. from ./.env (shell env wins)

    ap = argparse.ArgumentParser(prog="src.main")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("extract-expert")
    p.add_argument("--course-json", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    _add_roi_args(p)

    p = sub.add_parser("extract-worker")
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    _add_roi_args(p)

    p = sub.add_parser("estimate-roi",
                       help="estimate the work-area ROI of a video and write it to a json")
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True, help="output json path")

    p = sub.add_parser("match")
    p.add_argument("--expert-dir", required=True)
    p.add_argument("--worker-dir", required=True)
    p.add_argument("--course-json", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--debug", action="store_true",
                   help="save intermediate per-phase artifacts/visualizations to <out>/debug/")
    _add_vlm_args(p)

    p = sub.add_parser("run-all")
    p.add_argument("--course-json", required=True)
    p.add_argument("--expert-video", required=True)
    p.add_argument("--worker-video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--debug", action="store_true",
                   help="save intermediate per-phase artifacts/visualizations to <out>/reports/debug/")
    _add_roi_args(p)
    _add_vlm_args(p)

    args = ap.parse_args()
    if args.cmd in ("extract-expert", "extract-worker"):
        cmd_extract(args.video, args.out, roi=_roi_arg(args.roi),
                    roi_upscale_width=args.roi_upscale_width, flow_overlay=args.flow_overlay)
    elif args.cmd == "estimate-roi":
        from src.vision.work_area_roi import estimate_roi_from_video, save_roi
        work_area = estimate_roi_from_video(args.video)
        save_roi(args.out, work_area)
        print(f"work area {work_area} -> {args.out}")
    elif args.cmd == "match":
        cmd_match(args.expert_dir, args.worker_dir, args.course_json, args.out, debug=args.debug,
                  no_vlm=args.no_vlm, vlm_model=args.vlm_model, vlm_weight=args.vlm_weight,
                  refs_per_scene=args.refs_per_scene, vlm_cache_dir=args.vlm_cache_dir)
    elif args.cmd == "run-all":
        out = Path(args.out)
        cmd_extract(args.expert_video, str(out / "expert"), roi=_roi_arg(args.roi),
                    roi_upscale_width=args.roi_upscale_width, flow_overlay=args.flow_overlay)
        cmd_extract(args.worker_video, str(out / "worker"), roi=_roi_arg(args.roi),
                    roi_upscale_width=args.roi_upscale_width, flow_overlay=args.flow_overlay)
        cmd_match(str(out / "expert"), str(out / "worker"), args.course_json, str(out / "reports"),
                  debug=args.debug,
                  no_vlm=args.no_vlm, vlm_model=args.vlm_model, vlm_weight=args.vlm_weight,
                  refs_per_scene=args.refs_per_scene, vlm_cache_dir=args.vlm_cache_dir)


if __name__ == "__main__":
    main()
