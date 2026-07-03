# Sewing Operation Alignment POC

Compares a worker sewing video against an expert reference video and reports
missing / wrong-order / extra / duplicated operations and timing deviations.

Implements the pipeline described in `plan.md`:

```
expert video + expert JSON      worker video
        │                            │
        ▼                            ▼
  per-frame signals (WiLoR hand keypoints, Farneback optical flow per hand,
                      DINOv2 embedding of the hands' crop)
        │                            │
  expert scene templates      candidate windows (multi-scale sliding + change-points)
        └────────────┬───────────────┘
                     ▼
      similarity matrix (keypoint DTW + flow DTW + duration prior +
                          per-frame nearest-neighbor over pose/flow + DINOv2 embedding)
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

## Outputs

```
outputs/expert/frame_features.csv    per-frame signals
outputs/expert/keypoints.jsonl       raw hand/pose landmarks
outputs/expert/embeddings.npy        per-frame DINOv2 embedding (hands' crop)
outputs/expert/templates.json        per-scene templates
outputs/expert/overlay.mp4           debug overlay video
outputs/worker/...                   same for worker + candidate_windows.csv
outputs/reports/alignment_report.json / .csv
outputs/reports/alignment_timeline.html   visual timeline + error table
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
