# POC Proposal: Expert-to-Worker Sewing Operation Alignment

## 1. Goal

Implement a POC pipeline to compare a worker sewing video against an expert reference video.

The expert video already has scene-level timestamps and operations from `course-builder-data.json`. The goal is to:

1. Extract visual-motion signals from the expert video.
2. Extract the same signals from the worker video.
3. Split the worker video into candidate action segments.
4. Match worker segments to expert operations using raw image, keypoint, ROI, and optical-flow similarity.
5. Use sequence constraints from the expert operation order to detect:

   * missing operation
   * wrong-order operation
   * extra operation
   * repeated / duplicated operation
   * timing deviation

This is a POC. Do not train or fine-tune any model. Use off-the-shelf models and rule-based / dynamic-programming logic.

---

## 2. Input Data

### 2.1 Expert metadata JSON

Input file:

```text
course-builder-data.json
```

Relevant fields:

```text
input_version.task_name
input_version.operations
input_version.videos[0].filename
input_version.videos[0].scenes
```

The current task is:

```text
Diễu 7 li nẹp đỡ
```

The JSON contains:

```text
15 distinct operation definitions
17 expert scenes
scene-level timestamp_start / timestamp_end
operation names attached to each scene
operation frequency and TMU metadata
```

Important: some scenes contain multiple operations, for example:

```text
["Điều chỉnh mép", "Diễu cạnh dài"]
```

Some scenes may contain an empty operations list. These should be handled as `UNKNOWN` or `IDLE`.

---

## 3. POC Constraints

### Must-have

Implement a working end-to-end pipeline:

```text
expert video + expert JSON
worker video
        ↓
extract signals
        ↓
generate worker candidate segments
        ↓
match worker segments to expert operation sequence
        ↓
produce report
```

### No retraining

Do not train action recognition models for this POC.

Use only:

```text
MediaPipe / MMPose / OpenCV
image embeddings from pretrained model if easy
optical flow
DTW / Viterbi / dynamic programming
rule-based ROI events
```

### Performance target

POC can run offline, but should be practical:

```text
Input video length: ~1 minute
Target runtime: a few minutes per video on CPU/GPU
Support sub-second operations around 0.5s
```

Because some actions may be very short, avoid aggressive frame skipping. Use at least 30 FPS input if available.

---

## 4. Recommended Tech Stack

Use Python.

Core libraries:

```text
opencv-python
mediapipe
numpy
pandas
scipy
scikit-learn
tqdm
ruptures
fastdtw or custom DTW
torch optional
timm optional
```

Models / tools:

```text
MediaPipe Hands: hand keypoints
MediaPipe Pose: wrist, elbow, shoulder context
OpenCV Farneback Optical Flow: fast dense optical flow
Optional: CLIP / DINOv2 / MobileNet image embedding for crop similarity
```

For POC, start with:

```text
MediaPipe Hands + Pose
OpenCV Optical Flow
ROI event features
DTW / Viterbi sequence matching
```

Only add raw image embeddings after the first version works.

---

## 5. Expected Project Structure

```text
sewing_action_poc/
├── data/
│   ├── course-builder-data.json
│   ├── expert.mp4
│   └── worker.mp4
├── configs/
│   └── roi.json
├── outputs/
│   ├── expert/
│   ├── worker/
│   └── reports/
├── src/
│   ├── io/
│   │   ├── load_course_data.py
│   │   └── schemas.py
│   ├── vision/
│   │   ├── extract_keypoints.py
│   │   ├── extract_flow.py
│   │   ├── extract_roi_events.py
│   │   └── extract_frame_features.py
│   ├── segmentation/
│   │   ├── candidate_windows.py
│   │   └── changepoint.py
│   ├── matching/
│   │   ├── similarity.py
│   │   ├── dtw_matcher.py
│   │   └── viterbi_decoder.py
│   ├── reporting/
│   │   ├── build_report.py
│   │   └── visualize_alignment.py
│   └── main.py
├── scripts/
│   ├── run_extract_expert.py
│   ├── run_extract_worker.py
│   └── run_match.py
└── README.md
```

---

## 6. Data Normalization

Implement a loader for `course-builder-data.json`.

The loader should output:

```python
TaskSpec:
    task_id: str
    task_name: str
    operations: list[OperationSpec]
    expert_video: VideoSpec
    expert_scenes: list[SceneSpec]
```

Where:

```python
OperationSpec:
    name: str
    frequency: float
    tmu_manual: float
    tmu_machine: float
    tmu_bundle: float

SceneSpec:
    scene_index: int
    start: float
    end: float
    operations: list[str]
```

Rules:

1. Sort expert scenes by `timestamp_start`.
2. If a scene has no operations, set operations to `["UNKNOWN"]`.
3. If a scene has multiple operations, keep them as multi-label scene.
4. Build an expert ordered operation sequence from scenes.
5. Preserve repeated operations.
6. Use `frequency` as an expected count prior, not as hard ground truth.

Example expert order can look like:

```text
Lấy nẹp đỡ đặt vào chân vịt
Lại mũi bằng nút nhấn
Diễu đầu nẹp đỡ + Điều chỉnh mép
Lại mũi bằng cần gạt
Xoay quay kim (xoay góc)
Đẩy cữ vào nẹp đỡ
Điều chỉnh mép + Diễu cạnh dài
Điều chỉnh mép + Diễu cạnh dài
...
Cắt chỉ + Đưa ra sau may
Kiểm tra
Đưa chi tiết ra ngoài
```

For POC, treat multi-operation scenes as `COMPOSITE` states. Later, we can split composite scenes into atomic operations.

---

## 7. ROI Configuration

Create `configs/roi.json`.

Format:

```json
{
  "needle": [x1, y1, x2, y2],
  "fabric_area": [x1, y1, x2, y2],
  "left_work_area": [x1, y1, x2, y2],
  "right_work_area": [x1, y1, x2, y2],
  "machine_button": [x1, y1, x2, y2],
  "lever": [x1, y1, x2, y2]
}
```

The POC can require the user to manually define ROI once per camera setup.

Important ROIs:

```text
needle
fabric_area
button
lever
presser_foot_area
left_hand_work_area
right_hand_work_area
```

---

## 8. Feature Extraction

For every frame, extract:

### 8.1 Hand features

Use MediaPipe Hands.

Per frame:

```text
left_hand_present
right_hand_present
left_hand_bbox
right_hand_bbox
left_hand_center_x/y
right_hand_center_x/y
left_hand_21_keypoints
right_hand_21_keypoints
left_hand_speed
right_hand_speed
left_hand_acceleration
right_hand_acceleration
distance_left_hand_to_needle
distance_right_hand_to_needle
distance_between_hands
left_hand_inside_needle_roi
right_hand_inside_needle_roi
left_hand_inside_fabric_roi
right_hand_inside_fabric_roi
```

### 8.2 Pose features

Use MediaPipe Pose.

Extract:

```text
left_wrist
right_wrist
left_elbow
right_elbow
left_shoulder
right_shoulder
```

Use pose as fallback when hand detector fails.

### 8.3 Optical flow features

Use OpenCV Farneback Optical Flow.

Compute flow inside:

```text
left hand bbox
right hand bbox
needle ROI
fabric ROI
button ROI
lever ROI
```

Per ROI:

```text
flow_mean_magnitude
flow_max_magnitude
flow_p90_magnitude
flow_mean_dx
flow_mean_dy
```

### 8.4 ROI event features

Convert continuous features into events:

```text
left_hand_enter_needle_roi
right_hand_enter_needle_roi
left_hand_leave_needle_roi
right_hand_leave_needle_roi
needle_flow_start
needle_flow_stop
fabric_motion_start
fabric_motion_stop
button_interaction_candidate
lever_interaction_candidate
high_motion_peak
pause_candidate
```

These events are important for detecting short actions.

---

## 9. Expert Template Building

For each expert scene, build a template.

```python
ExpertTemplate:
    scene_index: int
    start: float
    end: float
    operations: list[str]
    duration: float
    frame_range: tuple[int, int]
    feature_sequence: np.ndarray
    keypoint_sequence: np.ndarray
    flow_sequence: np.ndarray
    event_sequence: np.ndarray
    aggregate_features: dict
```

For each expert scene, store:

```text
raw frame indices
normalized hand keypoint trajectory
hand velocity curve
needle ROI flow curve
fabric ROI flow curve
ROI event timeline
duration prior
```

Normalization rules:

```text
Normalize coordinates relative to needle ROI center or fabric ROI size.
Normalize hand speed by frame size.
Normalize duration by expert scene duration.
```

---

## 10. Worker Candidate Segment Generation

Do not use scene-cut detection.

Generate dense candidate windows from worker video.

For sub-second actions, use multiple window sizes:

```text
At 30 FPS:
  8 frames  ≈ 0.27s
  12 frames ≈ 0.40s
  16 frames ≈ 0.53s
  24 frames ≈ 0.80s
  32 frames ≈ 1.07s

Stride:
  2-4 frames
```

For each window:

```python
WorkerWindow:
    window_id: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration: float
    feature_sequence: np.ndarray
    aggregate_features: dict
```

Also implement optional change-point proposals using signal curves:

```text
hand speed
needle ROI flow
fabric ROI flow
hand distance to needle
ROI enter/leave events
```

Merge sliding-window proposals and change-point proposals.

---

## 11. Similarity Scoring

For each worker candidate window `Wj` and expert scene template `Ei`, compute:

```text
score(Ei, Wj)
```

The final score should combine:

```text
keypoint similarity
flow similarity
ROI event similarity
duration similarity
optional image crop similarity
```

Recommended formula:

```text
score = 
    0.40 * keypoint_score
  + 0.25 * flow_score
  + 0.20 * roi_event_score
  + 0.10 * duration_score
  + 0.05 * image_score
```

For version 1, skip `image_score` if it slows implementation.

### 11.1 Keypoint similarity

Use DTW distance between normalized hand trajectory sequences.

Compare:

```text
left hand center
right hand center
left wrist
right wrist
index fingertip
thumb fingertip
distance_between_hands
distance_to_needle
```

Convert DTW distance to score:

```text
score = exp(-distance / temperature)
```

### 11.2 Flow similarity

Compare flow magnitude curves:

```text
needle_roi_flow_mean_mag
fabric_roi_flow_mean_mag
left_hand_flow_mean_mag
right_hand_flow_mean_mag
```

Use DTW or cosine similarity.

### 11.3 ROI event similarity

Convert events to binary sequence.

Example:

```text
[hand_enter_needle, hand_leave_needle, needle_flow_peak, pause]
```

Score by event overlap with temporal tolerance:

```text
event_score = matched_events / total_expected_events
```

### 11.4 Duration score

```text
duration_ratio = worker_duration / expert_duration
duration_score = exp(-abs(log(duration_ratio)))
```

This allows worker to be faster or slower, but penalizes extreme mismatch.

---

## 12. Sequence Decoding

The key requirement is not just local matching. The worker video must be decoded as a sequence constrained by expert order.

Implement a Viterbi-style decoder.

### 12.1 States

States are expert scenes plus special states:

```text
START
E0
E1
E2
...
EN
EXTRA
END
```

Each expert scene may map to one or more operations.

### 12.2 Observations

Observations are worker candidate windows ordered by time:

```text
W0, W1, W2, ..., WT
```

### 12.3 Emission score

Emission score is the similarity score:

```text
emission(Ei, Wj) = score(Ei, Wj)
emission(EXTRA, Wj) = extra_score(Wj)
```

`extra_score(Wj)` should be high when:

```text
worker window has strong motion
but low similarity to all expert scenes
```

### 12.4 Transition rules

Allowed transitions:

```text
Ei -> Ei        same operation continues
Ei -> Ei+1      normal next operation
Ei -> Ei+k      skip operation, penalty = missing
Ei -> EXTRA     possible extra operation
EXTRA -> Ei     return to normal sequence
Ei -> Ei-1      wrong-order, high penalty
```

Penalties:

```text
same_state_penalty = 0.0
next_state_penalty = 0.0
skip_penalty = 1.5
extra_penalty = 0.8
backward_penalty = 3.0
unknown_penalty = 0.5
```

Do not completely forbid backward transition. Keep it possible with high penalty so the system can detect wrong-order actions.

---

## 13. Error Detection Logic

After decoding, merge consecutive windows assigned to the same state.

Output:

```python
DecodedSegment:
    assigned_state: str
    operations: list[str]
    start_time: float
    end_time: float
    confidence: float
    matched_expert_scene_index: int | None
```

### 13.1 Missing operation

If expert scene `Ei` is skipped in decoded path:

```text
Ei-1 -> Ei+1
```

Flag:

```text
MISSING
```

### 13.2 Wrong order

If decoded path contains backward transition:

```text
Ei -> Ei-1
```

or if an expert state appears earlier than expected, flag:

```text
WRONG_ORDER
```

### 13.3 Extra operation

If a worker segment is assigned to `EXTRA`, flag:

```text
EXTRA_ACTION
```

### 13.4 Duplicate / repeated operation

If the same expert scene appears in multiple separated decoded segments beyond expected frequency, flag:

```text
DUPLICATED_ACTION
```

### 13.5 Timing deviation

Compare worker duration with expert duration:

```text
duration_ratio = worker_duration / expert_duration
```

Flag:

```text
TOO_FAST  if duration_ratio < 0.6
TOO_SLOW  if duration_ratio > 1.6
NORMAL    otherwise
```

Make thresholds configurable.

---

## 14. Required Output Files

After running the pipeline, generate:

```text
outputs/expert/frame_features.csv
outputs/expert/keypoints.jsonl
outputs/expert/templates.json
outputs/expert/overlay.mp4

outputs/worker/frame_features.csv
outputs/worker/keypoints.jsonl
outputs/worker/candidate_windows.csv
outputs/worker/overlay.mp4

outputs/reports/alignment_report.json
outputs/reports/alignment_report.csv
outputs/reports/alignment_timeline.html
outputs/reports/score_matrix.npy
outputs/reports/score_matrix.png
```

### 14.1 `alignment_report.json`

Example:

```json
{
  "task_name": "Diễu 7 li nẹp đỡ",
  "expert_video": "expert.mp4",
  "worker_video": "worker.mp4",
  "summary": {
    "num_expected_scenes": 17,
    "num_detected_segments": 16,
    "missing_count": 1,
    "extra_count": 2,
    "wrong_order_count": 0,
    "duplicated_count": 1
  },
  "segments": [
    {
      "worker_start": 0.2,
      "worker_end": 4.5,
      "matched_expert_scene_index": 0,
      "operations": ["Lấy nẹp đỡ đặt vào chân vịt"],
      "confidence": 0.86,
      "status": "MATCHED",
      "timing_status": "NORMAL"
    }
  ],
  "errors": [
    {
      "type": "MISSING",
      "expert_scene_index": 10,
      "operations": ["UNKNOWN"],
      "message": "Worker skipped this expert scene or no confident match was found."
    }
  ]
}
```

### 14.2 Timeline visualization

Generate an HTML timeline with:

```text
Expert scenes on top
Worker decoded segments below
Color by status:
  matched
  missing
  extra
  wrong-order
  low-confidence
```

Also generate score matrix heatmap:

```text
x-axis: expert scene index
y-axis: worker window index
value: similarity score
```

---

## 15. CLI Requirements

Implement CLI commands.

### Extract expert

```bash
python -m src.main extract-expert \
  --course-json data/course-builder-data.json \
  --video data/expert.mp4 \
  --roi configs/roi.json \
  --out outputs/expert
```

### Extract worker

```bash
python -m src.main extract-worker \
  --video data/worker.mp4 \
  --roi configs/roi.json \
  --out outputs/worker
```

### Match

```bash
python -m src.main match \
  --expert-dir outputs/expert \
  --worker-dir outputs/worker \
  --course-json data/course-builder-data.json \
  --out outputs/reports
```

### Full pipeline

```bash
python -m src.main run-all \
  --course-json data/course-builder-data.json \
  --expert-video data/expert.mp4 \
  --worker-video data/worker.mp4 \
  --roi configs/roi.json \
  --out outputs
```

---

## 16. Implementation Phases

### Phase 1: Data loader and expert template

Implement:

```text
load_course_data.py
schemas.py
expert template builder
```

Acceptance:

```text
Can load course-builder-data.json
Can print ordered expert scenes
Can handle multi-operation scenes
Can handle empty operation scenes as UNKNOWN
```

### Phase 2: Feature extraction

Implement:

```text
MediaPipe hand + pose extraction
OpenCV optical flow
ROI event extraction
overlay debug video
```

Acceptance:

```text
Can produce frame_features.csv
Can produce keypoints.jsonl
Can produce overlay.mp4
```

### Phase 3: Worker candidate windows

Implement:

```text
sliding windows with multiple sizes
optional change-point proposals
candidate_windows.csv
```

Acceptance:

```text
Windows cover the full video
Short windows support 0.5s actions
No large gaps between windows
```

### Phase 4: Similarity matrix

Implement:

```text
expert scene template vs worker candidate window score
keypoint DTW
flow DTW
ROI event overlap
duration prior
```

Acceptance:

```text
Can produce score_matrix.npy
Can produce score_matrix.png
High similarity appears around visually matching segments
```

### Phase 5: Sequence decoder

Implement:

```text
Viterbi decoder with expert order constraints
EXTRA state
skip / missing penalty
backward / wrong-order penalty
```

Acceptance:

```text
Can decode worker video into ordered segments
Can detect missing / extra / wrong-order / duplicate
```

### Phase 6: Report

Implement:

```text
alignment_report.json
alignment_report.csv
alignment_timeline.html
```

Acceptance:

```text
Report is readable by non-technical user
Each error links to worker timestamp
Each matched segment links to expert scene
```

---

## 17. Important Design Decisions

### 17.1 Use scene-level matching first

Because the expert JSON has scene timestamps, use scenes as the first unit of matching.

Do not try to split all composite scenes into atomic operations in the first POC.

For example:

```text
Scene: ["Điều chỉnh mép", "Diễu cạnh dài"]
```

Treat this as one composite state.

Later, we can split it into atomic sub-actions.

### 17.2 Use operation frequency as a soft prior

The `frequency` field should help detect repeated operations, but should not be a hard rule.

Example:

```text
Điều chỉnh mép frequency = 10
Diễu cạnh dài frequency = 5
```

These operations may appear in multiple scenes and should not be flagged as duplicated too aggressively.

### 17.3 Use ROI event heavily

For sewing operations, raw image similarity may be unreliable because:

```text
fabric color may change
lighting may change
worker hands look different
camera exposure may differ
```

Keypoint, ROI event, and flow are more robust for the POC.

### 17.4 Handle 0.5s actions

Use short sliding windows and small stride.

Recommended:

```text
window sizes: 8, 12, 16, 24, 32 frames
stride: 2-4 frames
```

Do not downsample below 15 FPS.

Prefer 30 FPS or 60 FPS.

---

## 18. Scoring Summary

For each decoded expert scene:

```text
scene_score =
  0.4 * keypoint_score
+ 0.25 * flow_score
+ 0.2 * event_score
+ 0.1 * duration_score
+ 0.05 * image_score
```

For final worker score:

```text
overall_score =
  average matched scene score
  - missing_penalty * num_missing
  - extra_penalty * num_extra
  - wrong_order_penalty * num_wrong_order
  - duplicate_penalty * num_duplicate
```

Return both:

```text
technical score
human-readable explanation
```

---

## 19. Minimal MVP Acceptance Criteria

The MVP is successful if it can:

1. Load the provided JSON and expert video.
2. Extract expert hand keypoints, pose, flow, and ROI events.
3. Extract the same signals from worker video.
4. Build expert scene templates.
5. Generate worker candidate windows.
6. Compute similarity matrix.
7. Decode worker sequence using expert order.
8. Output a report with:

   * matched scenes
   * missing scenes
   * extra segments
   * wrong-order transitions
   * timing deviation
9. Generate an overlay video for debugging.
10. Work without model retraining.

---

## 20. Out of Scope for POC

Do not implement yet:

```text
custom model training
fine-tuning VideoMAE
full action recognition classifier
multi-camera calibration
automatic ROI detection
LLM-based video reasoning
real-time UI
production database
```

Focus on a reliable offline POC first.

---

## 21. Final Deliverable

Deliver a Python project that can be run with:

```bash
python -m src.main run-all \
  --course-json data/course-builder-data.json \
  --expert-video data/expert.mp4 \
  --worker-video data/worker.mp4 \
  --roi configs/roi.json \
  --out outputs
```

The final output should include:

```text
outputs/reports/alignment_report.json
outputs/reports/alignment_report.csv
outputs/reports/alignment_timeline.html
outputs/reports/score_matrix.png
outputs/expert/overlay.mp4
outputs/worker/overlay.mp4
```

The report must clearly answer:

```text
Worker did which operations correctly?
Which operations are missing?
Which operations are in wrong order?
Which operations are extra?
Which operations are too slow or too fast?
At what timestamps do these problems happen?
```
