# Sewing Operation Alignment POC

Compares a worker sewing video against an expert reference video and reports
missing / wrong-order / extra / duplicated operations and timing deviations.

Implements the pipeline described in `plan.md`:

```
expert video + expert JSON      worker video
        │                            │
        ▼                            ▼
  per-frame signals (WiLoR hand keypoints, Farneback optical flow, ROI events)
        │                            │
  expert scene templates      candidate windows (multi-scale sliding + change-points)
        └────────────┬───────────────┘
                     ▼
      similarity matrix (keypoint DTW + flow DTW + ROI-event overlap + duration prior)
                     ▼
      Viterbi decoding constrained by expert scene order (with EXTRA states)
                     ▼
      alignment report (JSON / CSV / HTML timeline / score-matrix heatmap)
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

ROIs are estimated automatically per video (no manual input needed): the
needle zone is localized from adjacent-frame motion × lamp brightness
statistics, and the other ROIs are derived geometrically from it. The
estimate used is saved to `outputs/<role>/roi_auto.json` and drawn in the
overlay video. To override, pass `--roi configs/roi.json`.

## Outputs

```
outputs/expert/frame_features.csv    per-frame signals
outputs/expert/keypoints.jsonl       raw hand/pose landmarks
outputs/expert/templates.json        per-scene templates
outputs/expert/overlay.mp4           debug overlay video
outputs/worker/...                   same for worker + candidate_windows.csv
outputs/reports/alignment_report.json / .csv
outputs/reports/alignment_timeline.html   visual timeline + error table
outputs/reports/score_matrix.npy / .png
```

## Configuration

- **ROIs** — auto-estimated by default (`src/vision/auto_roi.py`). Optional
  manual override: `configs/roi.json`, normalized `[x1, y1, x2, y2]` (0–1),
  separate `expert` / `worker` sections because camera framing differs
  (use `outputs/*/overlay.mp4` to verify placement).
- **Viterbi penalties** — `src/matching/viterbi_decoder.py` `PENALTIES`
  (skip / extra / backward…). The decoder is duration-aware: each scene expands
  into chained sub-states enforcing a minimum dwell (half the expert scene
  duration, scaled by the video-length ratio), and the raw similarity matrix is
  column-z-scored + row-softmaxed (`sharpen_emissions`, temperature 4.0) to
  remove per-scene bias before decoding.
- **Score weights** — `src/matching/similarity.py` `WEIGHTS`
  (keypoint 0.40, flow 0.25, event 0.20, duration 0.15; image embedding
  deferred per plan §11).
- **Timing thresholds** — `src/reporting/build_report.py` `TIMING`
  (TOO_FAST < 0.6, TOO_SLOW > 1.6).
- **Window sizes / stride** — `src/segmentation/candidate_windows.py`
  (0.3–2.0 s windows, 0.2 s stride, expressed in seconds so any fps works).

## Notes / deviations from plan.md

- Course JSON exposes the video under `input_version.videos_export`, not
  `videos`; the loader accepts both.
- Expert video is 25 fps / 1080p, worker 60 fps / 480×368 — signals are
  robust-scaled per video and windows are defined in seconds, so the fps
  mismatch cancels out.
- Image-embedding similarity (0.05 weight) is deferred, as the plan allows for
  version 1; its weight was folded into the duration term.
- Hand detector backend moved from MediaPipe → RTMPose-Hand → WiLoR. WiLoR's
  YOLO hand detector classifies handedness (anatomical left/right) directly,
  so `left_hand`/`right_hand` are now assigned from that instead of the
  x-position heuristic used previously (still correct for a seated operator
  filmed from above/front, where anatomical and screen side coincide).
- The plan's MediaPipe Pose wrist fallback was dropped with the move off
  MediaPipe; short detection gaps are covered by interpolation in
  `prepare_features` instead.
