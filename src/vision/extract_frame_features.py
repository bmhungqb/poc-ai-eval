"""Per-frame signal extraction: MediaPipe Hands + Pose, Farneback optical flow per ROI.

Produces, per video:
  <out>/frame_features.csv   one row per frame
  <out>/keypoints.jsonl      raw hand/pose landmarks per frame
  <out>/overlay.mp4          debug overlay video
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.vision.roi import roi_center, roi_to_pixels

FLOW_ROIS = ["needle", "fabric_area", "machine_button", "lever"]
FLOW_SCALE_WIDTH = 320  # dense flow computed on a downscaled gray frame


def _hand_summary(landmarks) -> dict:
    xs = np.array([lm.x for lm in landmarks])
    ys = np.array([lm.y for lm in landmarks])
    return {
        "cx": float(xs.mean()),
        "cy": float(ys.mean()),
        "bbox": [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
        "index_tip": [float(landmarks[8].x), float(landmarks[8].y)],
        "thumb_tip": [float(landmarks[4].x), float(landmarks[4].y)],
        "keypoints": [[float(lm.x), float(lm.y)] for lm in landmarks],
    }


def _flow_stats(flow: np.ndarray, roi_px: tuple[int, int, int, int], scale: float) -> dict:
    x1, y1, x2, y2 = (int(v * scale) for v in roi_px)
    patch = flow[y1:y2, x1:x2]
    if patch.size == 0:
        return {"mean_mag": 0.0, "max_mag": 0.0, "p90_mag": 0.0, "mean_dx": 0.0, "mean_dy": 0.0}
    mag = np.linalg.norm(patch, axis=2)
    return {
        "mean_mag": float(mag.mean()),
        "max_mag": float(mag.max()),
        "p90_mag": float(np.percentile(mag, 90)),
        "mean_dx": float(patch[..., 0].mean()),
        "mean_dy": float(patch[..., 1].mean()),
    }


def extract_video_features(
    video_path: str | Path,
    rois: dict[str, list[float]],
    out_dir: str | Path,
    overlay: bool = True,
    max_frames: int | None = None,
) -> pd.DataFrame:
    import mediapipe as mp

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        n_frames = min(n_frames, max_frames)

    needle_c = roi_center(rois["needle"])
    flow_scale = FLOW_SCALE_WIDTH / width
    flow_size = (FLOW_SCALE_WIDTH, int(height * flow_scale))

    writer = None
    if overlay:
        ow = 960 if width > 960 else width
        oh = int(height * ow / width)
        writer = cv2.VideoWriter(
            str(out_dir / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh)
        )

    hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.4, min_tracking_confidence=0.4,
    )
    pose = mp.solutions.pose.Pose(
        static_image_mode=False, model_complexity=0,
        min_detection_confidence=0.4, min_tracking_confidence=0.4,
    )
    POSE_IDX = {"left_wrist": 15, "right_wrist": 16, "left_elbow": 13, "right_elbow": 14,
                "left_shoulder": 11, "right_shoulder": 12}

    rows: list[dict] = []
    prev_gray = None
    prev_hand_pos = {"left": None, "right": None}
    prev_hand_speed = {"left": 0.0, "right": 0.0}
    kp_file = open(out_dir / "keypoints.jsonl", "w")

    for fi in tqdm(range(n_frames), desc=f"extract {Path(video_path).name}"):
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small_gray = cv2.cvtColor(cv2.resize(frame, flow_size), cv2.COLOR_BGR2GRAY)

        # --- hands ---
        hres = hands.process(rgb)
        hand_info: dict[str, dict | None] = {"left": None, "right": None}
        if hres.multi_hand_landmarks:
            for lm, handed in zip(hres.multi_hand_landmarks, hres.multi_handedness):
                # MediaPipe labels assume mirrored webcam view; on third-person video
                # the label is unreliable, so also fall back to x-position ordering.
                label = handed.classification[0].label.lower()
                side = "left" if label == "left" else "right"
                if hand_info[side] is not None:
                    side = "left" if hand_info["left"] is None else "right"
                hand_info[side] = _hand_summary(lm.landmark)
            if hand_info["left"] and hand_info["right"] and hand_info["left"]["cx"] > hand_info["right"]["cx"]:
                hand_info["left"], hand_info["right"] = hand_info["right"], hand_info["left"]

        # --- pose ---
        pres = pose.process(rgb)
        pose_pts = {}
        if pres.pose_landmarks:
            for name, idx in POSE_IDX.items():
                lm = pres.pose_landmarks.landmark[idx]
                if lm.visibility > 0.3:
                    pose_pts[name] = (float(lm.x), float(lm.y))
        # fallback: use wrist as hand center when hand detector failed
        for side in ("left", "right"):
            if hand_info[side] is None and f"{side}_wrist" in pose_pts:
                wx, wy = pose_pts[f"{side}_wrist"]
                hand_info[side] = {"cx": wx, "cy": wy, "bbox": [wx - 0.05, wy - 0.05, wx + 0.05, wy + 0.05],
                                   "index_tip": [wx, wy], "thumb_tip": [wx, wy], "keypoints": [], "from_pose": True}

        # --- optical flow ---
        flow_feats = {}
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, small_gray, None,
                                                0.5, 2, 13, 2, 5, 1.1, 0)
            for rname in FLOW_ROIS:
                st = _flow_stats(flow, roi_to_pixels(rois[rname], width, height), flow_scale)
                for k, v in st.items():
                    flow_feats[f"{rname}_flow_{k}"] = v
            for side in ("left", "right"):
                hi = hand_info[side]
                if hi:
                    st = _flow_stats(flow, roi_to_pixels(hi["bbox"], width, height), flow_scale)
                else:
                    st = {"mean_mag": 0.0, "max_mag": 0.0, "p90_mag": 0.0, "mean_dx": 0.0, "mean_dy": 0.0}
                for k, v in st.items():
                    flow_feats[f"{side}_hand_flow_{k}"] = v
        else:
            for rname in FLOW_ROIS:
                for k in ("mean_mag", "max_mag", "p90_mag", "mean_dx", "mean_dy"):
                    flow_feats[f"{rname}_flow_{k}"] = 0.0
            for side in ("left", "right"):
                for k in ("mean_mag", "max_mag", "p90_mag", "mean_dx", "mean_dy"):
                    flow_feats[f"{side}_hand_flow_{k}"] = 0.0
        prev_gray = small_gray

        # --- per-frame row ---
        row: dict = {"frame": fi, "time": fi / fps}
        for side in ("left", "right"):
            hi = hand_info[side]
            present = hi is not None
            cx = hi["cx"] if present else np.nan
            cy = hi["cy"] if present else np.nan
            if present and prev_hand_pos[side] is not None:
                speed = float(np.hypot(cx - prev_hand_pos[side][0], cy - prev_hand_pos[side][1]) * fps)
            else:
                speed = 0.0
            accel = (speed - prev_hand_speed[side]) * fps
            prev_hand_pos[side] = (cx, cy) if present else prev_hand_pos[side]
            prev_hand_speed[side] = speed
            d_needle = float(np.hypot(cx - needle_c[0], cy - needle_c[1])) if present else np.nan
            row.update({
                f"{side}_hand_present": int(present),
                f"{side}_hand_cx": cx, f"{side}_hand_cy": cy,
                f"{side}_hand_speed": speed, f"{side}_hand_accel": accel,
                f"{side}_hand_dist_needle": d_needle,
                f"{side}_hand_in_needle_roi": int(present and rois["needle"][0] <= cx <= rois["needle"][2] and rois["needle"][1] <= cy <= rois["needle"][3]),
                f"{side}_hand_in_fabric_roi": int(present and rois["fabric_area"][0] <= cx <= rois["fabric_area"][2] and rois["fabric_area"][1] <= cy <= rois["fabric_area"][3]),
            })
        li, ri = hand_info["left"], hand_info["right"]
        row["hands_distance"] = float(np.hypot(li["cx"] - ri["cx"], li["cy"] - ri["cy"])) if li and ri else np.nan
        row.update(flow_feats)
        rows.append(row)

        kp_file.write(json.dumps({
            "frame": fi, "time": row["time"],
            "left_hand": {k: v for k, v in (li or {}).items() if k != "keypoints"} if li else None,
            "right_hand": {k: v for k, v in (ri or {}).items() if k != "keypoints"} if ri else None,
            "left_hand_keypoints": li["keypoints"] if li else None,
            "right_hand_keypoints": ri["keypoints"] if ri else None,
            "pose": pose_pts,
        }) + "\n")

        if writer is not None:
            vis = frame.copy()
            for rname, roi in rois.items():
                x1, y1, x2, y2 = roi_to_pixels(roi, width, height)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 200), 1)
                cv2.putText(vis, rname, (x1 + 2, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 200), 1)
            for side, color in (("left", (0, 255, 0)), ("right", (0, 128, 255))):
                hi = hand_info[side]
                if hi:
                    x1, y1, x2, y2 = roi_to_pixels(hi["bbox"], width, height)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(vis, side, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    for kx, ky in hi.get("keypoints", []):
                        cv2.circle(vis, (int(kx * width), int(ky * height)), 2, color, -1)
            nf = row.get("needle_flow_mean_mag", 0.0)
            cv2.putText(vis, f"f={fi} t={row['time']:.2f}s needle_flow={nf:.2f}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if vis.shape[1] != (writer_w := (960 if width > 960 else width)):
                vis = cv2.resize(vis, (writer_w, int(height * writer_w / width)))
            writer.write(vis)

    cap.release()
    kp_file.close()
    if writer is not None:
        writer.release()
    hands.close()
    pose.close()

    df = pd.DataFrame(rows)
    df.attrs["fps"] = fps
    df.to_csv(out_dir / "frame_features.csv", index=False)
    (out_dir / "meta.json").write_text(json.dumps(
        {"video": str(video_path), "fps": fps, "width": width, "height": height, "n_frames": len(df)}))
    return df
