# Sewing Operation Alignment POC

Compares a worker sewing video against an expert reference video and reports
missing / wrong-order / extra / duplicated operations, timing deviations, and
a present/absent/uncertain checklist of sub-second auxiliary operations.

Implements the pipeline described in `plan.md`, extended per
`proposal-vlm-alignment.md` (work-area ROI, VLM emission, aux checklist):

```
expert video + expert JSON      worker video
        │                            │
        ▼                            ▼
  work-area ROI (union of hand boxes; crop + upscale before detection)
        │                            │
  per-frame signals (WiLoR hand keypoints, Farneback optical flow per hand,
                      DINOv2 embedding of the hands' crop)
        │                            │
  expert scene templates      candidate windows (multi-scale sliding + change-points)
        └────────────┬───────────────┘
                     ▼
      similarity matrix (keypoint DTW + flow DTW + duration prior +
                          per-frame nearest-neighbor over pose/flow + DINOv2 embedding)
                     ▼
      tier 1: VLM emission (change-point segments → ROI keyframes → OpenRouter
              VLM classify vs the scene catalog; blended in at weight ~0.6;
              cached; silently skipped when OPENROUTER_API_KEY is unset)
                     ▼
      Viterbi decoding constrained by expert scene order (with EXTRA states,
              no-confident-match → UNMATCHED instead of forced matches)
                     ▼
      tier 2: aux-operation checklist (motion-spike frame clusters → VLM
              yes/no + button/lever zone activity detector)
                     ▼
      alignment report v2 (JSON / CSV / HTML timeline / aux checklist)
```

## Setup

Hand keypoints come from [WiLoR](https://github.com/rolpotamias/WiLoR) (a YOLO
hand detector + a transformer model that regresses full 3D hand pose/mesh,
via the [`wilor-mini`](https://github.com/warmshao/WiLoR-mini) packaging —
CPU or CUDA, no mmcv/mmdet build chain). It also classifies handedness
directly, which we rely on for left/right hand assignment (see Notes below).
Model weights (~200 MB, PyTorch) are downloaded automatically from Hugging
Face on first run. Requires PyTorch — install a CUDA build first if a GPU is
available, otherwise the default CPU wheel from `requirements.txt` is used.

Frame embeddings come from [DINOv2](https://github.com/facebookresearch/dinov2)
(`facebook/dinov2-small`, ~22M params, via `transformers`), used for the
`image_embed` nearest-neighbor score term (see Configuration below). Weights
(~90 MB) are also downloaded automatically from Hugging Face on first run.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

(or with plain pip: `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`)

Place the input videos at `data/expert.mp4` and `data/worker.mp4` (they are
not committed to git).

## Run

```bash
.venv/bin/python -m src.main run-all \
  --course-json data/course-builder-data.json \
  --expert-video data/expert.mp4 \
  --worker-video data/worker.mp4 \
  --out outputs
```

Or step by step: `extract-expert`, `extract-worker`, `match` (see `python -m src.main -h`).

### VLM stages (OpenRouter)

Set `OPENROUTER_API_KEY` to enable the tier-1 VLM emission and the tier-2 aux
checklist; without it the pipeline runs on the pose/flow terms only (and the
aux checklist falls back to the button/lever zone detector when a legacy
`roi_auto.json` with named zones is present). Flags on `match` / `run-all`:
`--no-vlm`, `--vlm-model` (default `google/gemini-2.5-flash`), `--vlm-weight`
(default 0.6), `--refs-per-scene` (few-shot expert crops, default 1),
`--vlm-cache-dir` (default `<out>/vlm_cache` — responses are cached by
(video fingerprint, frames, prompt version, model), so re-runs are free).

Scene descriptions used in the prompts live in
`configs/scene_descriptions.json` (the proposal's one-time manual
confirmation); the main/aux split for composite scenes can be overridden in
`configs/aux_operations.json`.

### Work-area ROI

Extraction estimates a fixed work-area ROI per video (union of detected hand
boxes over a sampling pass) and runs hand detection on the cropped + upscaled
region, which recovers detections that fail at 480×368 native resolution.
Control with `--roi auto|none|<path>` and `--roi-upscale-width` on the
extract commands / `run-all`; `estimate-roi --video ... --out roi.json`
computes one standalone. The ROI is saved to `<out>/work_area_roi.json` and
reused for every frame sent to the VLM.

## Outputs

```
outputs/expert/frame_features.csv    per-frame signals
outputs/expert/keypoints.jsonl       raw hand/pose landmarks
outputs/expert/embeddings.npy        per-frame DINOv2 embedding (hands' crop)
outputs/expert/templates.json        per-scene templates
outputs/expert/overlay.mp4           debug overlay video
outputs/worker/...                   same for worker + candidate_windows.csv
outputs/expert/work_area_roi.json    fixed work-area ROI used for crop+upscale
outputs/reports/alignment_report.json / .csv
outputs/reports/alignment_timeline.html   visual timeline + error + aux tables
outputs/reports/aux_checklist.csv    aux-operation present/absent/uncertain
outputs/reports/vlm_segments.json    per-segment VLM scores + evidence
outputs/reports/vlm_cache/           cached VLM responses (safe to keep)
outputs/reports/score_matrix.npy / .png
```

## Configuration

- **Viterbi penalties** — `src/matching/viterbi_decoder.py` `PENALTIES`
  (skip / extra / backward…). The decoder is duration-aware: each scene expands
  into chained sub-states enforcing a minimum dwell (half the expert scene
  duration, scaled by the video-length ratio), and the raw similarity matrix is
  column-z-scored + row-softmaxed (`sharpen_emissions`, temperature 4.0) to
  remove per-scene bias before decoding.
- **Score weights** — `src/matching/similarity.py` `WEIGHTS`
  (keypoint 0.25, flow 0.15, duration 0.10, frame_nn 0.20, image_embed 0.30).
  `frame_nn` and `image_embed` are both Chamfer-style per-frame
  nearest-neighbor scores (no resampling, no linear-time-warp assumption,
  unlike the DTW terms) — `frame_nn` over raw pose/flow features,
  `image_embed` over the DINOv2 embedding. Weights are renormalized over
  whichever terms are actually available, so this still works without
  `embeddings.npy` (older extraction runs).
- **Timing thresholds** — `src/reporting/build_report.py` `TIMING`
  (TOO_FAST < 0.6, TOO_SLOW > 1.6).
- **No-confident-match** — `src/reporting/build_report.py` `UNMATCHED_REL` /
  `LOW_CONF_REL`: segment confidence (mean decoded emission probability) is
  compared against the uniform level 1/n_scenes; segments below the
  thresholds are reported UNMATCHED / LOW_CONFIDENCE instead of being forced
  into a match, and UNMATCHED segments do not satisfy a scene (it surfaces
  as MISSING). With pose/flow-only emissions (near-uniform, see the
  proposal's diagnosis) most segments are honestly UNMATCHED — the VLM
  emission is what lifts real matches above the thresholds.
- **VLM emission/aux** — `src/vlm/emission.py` (`VLM_BLEND_WEIGHT`,
  `KEYFRAMES_PER_SEGMENT`, `MAX_SEGMENTS`) and `src/vlm/aux_check.py`
  (`MIN_VLM_CONFIDENCE`, `ZONE_ACTIVITY_THRESH`, spike-cluster sizes);
  Viterbi penalties used with VLM emissions: `PENALTIES_VLM` in
  `src/matching/viterbi_decoder.py`.
- **Window sizes / stride** — `src/segmentation/candidate_windows.py`
  (0.3–2.0 s windows, 0.2 s stride, expressed in seconds so any fps works).

## Notes / deviations from plan.md

- Course JSON exposes the video under `input_version.videos_export`, not
  `videos`; the loader accepts both.
- Expert video is 25 fps / 1080p, worker 60 fps / 480×368 — signals are
  robust-scaled per video and windows are defined in seconds, so the fps
  mismatch cancels out.
- Image-embedding similarity, deferred in earlier versions of this POC, is now
  implemented as the `image_embed` term (DINOv2 embedding of the hands' crop,
  Chamfer-style nearest-neighbor against each expert scene) -- see Configuration.
- Hand detector backend moved from MediaPipe → RTMPose-Hand → WiLoR. WiLoR's
  YOLO hand detector classifies handedness (anatomical left/right) directly,
  so `left_hand`/`right_hand` are now assigned from that instead of the
  x-position heuristic used previously (still correct for a seated operator
  filmed from above/front, where anatomical and screen side coincide).
- The plan's MediaPipe Pose wrist fallback was dropped with the move off
  MediaPipe; short detection gaps are covered by interpolation in
  `prepare_features` instead.
- ROI-based features (needle/fabric/button/lever zones, ROI events, manual
  `configs/roi.json`) were removed entirely. Flow and motion signals are now
  computed only around the detected hand bounding boxes; the similarity score
  dropped the event term and redistributed its weight across keypoint/flow/
  duration.
