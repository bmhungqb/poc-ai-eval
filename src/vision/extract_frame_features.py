"""Per-frame signal extraction: WiLoR (YOLO hand detector + WiLoR 3D hand
pose/mesh reconstruction) for hand landmarks, Farneback optical flow per hand.

Produces, per video:
  <out>/frame_features.csv   one row per frame
  <out>/keypoints.jsonl      raw hand landmarks per frame
  <out>/overlay.mp4          debug overlay video
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.vision.embed_model import MODEL_NAME as EMBED_MODEL_NAME
from src.vision.embed_model import DinoEmbedder
from src.vision.hand_model import WiLorHand

FLOW_SCALE_WIDTH = 320   # dense flow computed on a downscaled gray frame
HAND_SCORE_THRESH = 0.3  # min YOLO hand-detector confidence to accept a detected hand
                         # (gloved hands on low-res factory footage score low)
EMBED_PAD_FRAC = 0.3     # pad the hands' union bbox by this fraction before embedding,
                         # so the crop includes a bit of the fabric/tool around the hands


def _bbox_to_pixels(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, int(x1 * width)),
        max(0, int(y1 * height)),
        min(width, int(x2 * width)),
        min(height, int(y2 * height)),
    )


def _hand_summary(kps: np.ndarray, score: float) -> dict:
    """kps: (21, 2) normalized keypoints, WiLoR/MANO hand layout (wrist=0,
    thumb tip=4, index tip=8 — same ordering as MediaPipe)."""
    xs, ys = kps[:, 0], kps[:, 1]
    return {
        "cx": float(xs.mean()),
        "cy": float(ys.mean()),
        "score": float(score),
        "bbox": [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
        "index_tip": [float(kps[8, 0]), float(kps[8, 1])],
        "thumb_tip": [float(kps[4, 0]), float(kps[4, 1])],
        "keypoints": kps.tolist(),
    }


def _embed_crop_bbox(hand_info: dict, width: int, height: int) -> tuple[int, int, int, int]:
    """Padded pixel bbox around whichever hands are present, for the
    embedding crop. Falls back to the full frame if neither hand is
    detected (still gives the embedding model something to work with)."""
    boxes = [_bbox_to_pixels(hi["bbox"], width, height) for hi in hand_info.values() if hi]
    if not boxes:
        return 0, 0, width, height
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    pad_x, pad_y = int((x2 - x1) * EMBED_PAD_FRAC), int((y2 - y1) * EMBED_PAD_FRAC)
    return (max(0, x1 - pad_x), max(0, y1 - pad_y),
            min(width, x2 + pad_x), min(height, y2 + pad_y))


def _flow_stats(flow: np.ndarray, bbox_px: tuple[int, int, int, int], scale: float) -> dict:
    x1, y1, x2, y2 = (int(v * scale) for v in bbox_px)
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
    out_dir: str | Path,
    overlay: bool = True,
    max_frames: int | None = None,
) -> pd.DataFrame:
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

    flow_scale = FLOW_SCALE_WIDTH / width
    flow_size = (FLOW_SCALE_WIDTH, int(height * flow_scale))

    writer = None
    if overlay:
        ow = 960 if width > 960 else width
        oh = int(height * ow / width)
        writer = cv2.VideoWriter(
            str(out_dir / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh)
        )

    hand_model = WiLorHand(conf_thresh=HAND_SCORE_THRESH)  # YOLO hand det + WiLoR 3D pose
    embedder = DinoEmbedder()  # frame-level embedding for image_embed nearest-neighbor score

    rows: list[dict] = []
    embeddings: list[np.ndarray] = []
    prev_gray = None
    prev_hand_pos = {"left": None, "right": None}
    prev_hand_speed = {"left": 0.0, "right": 0.0}
    kp_file = open(out_dir / "keypoints.jsonl", "w")

    for fi in tqdm(range(n_frames), desc=f"extract {Path(video_path).name}"):
        ok, frame = cap.read()
        if not ok:
            break
        small_gray = cv2.cvtColor(cv2.resize(frame, flow_size), cv2.COLOR_BGR2GRAY)

        # --- hands (WiLoR: YOLO hand detector + WiLoR 3D pose/mesh model) ---
        raw_hands = hand_model(frame)
        by_side: dict[str, list[dict]] = {"left": [], "right": []}
        for h in raw_hands:
            kps_norm = h["keypoints"].astype(np.float64) / np.array([width, height])
            side = "right" if h["is_right"] else "left"
            by_side[side].append(_hand_summary(kps_norm, h["score"]))
        # handedness comes straight from the detector; if it (rarely) yields
        # more than one hand on a side, keep the most confident one
        hand_info: dict[str, dict | None] = {"left": None, "right": None}
        for side in ("left", "right"):
            if by_side[side]:
                hand_info[side] = max(by_side[side], key=lambda s: s["score"])

        # --- image embedding (hands' union bbox, padded) ---
        ex1, ey1, ex2, ey2 = _embed_crop_bbox(hand_info, width, height)
        embeddings.append(embedder(frame[ey1:ey2, ex1:ex2]))

        # --- optical flow ---
        flow_feats = {}
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, small_gray, None,
                                                0.5, 2, 13, 2, 5, 1.1, 0)
            for side in ("left", "right"):
                hi = hand_info[side]
                if hi:
                    st = _flow_stats(flow, _bbox_to_pixels(hi["bbox"], width, height), flow_scale)
                else:
                    st = {"mean_mag": 0.0, "max_mag": 0.0, "p90_mag": 0.0, "mean_dx": 0.0, "mean_dy": 0.0}
                for k, v in st.items():
                    flow_feats[f"{side}_hand_flow_{k}"] = v
        else:
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
            row.update({
                f"{side}_hand_present": int(present),
                f"{side}_hand_cx": cx, f"{side}_hand_cy": cy,
                f"{side}_hand_speed": speed, f"{side}_hand_accel": accel,
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
        }) + "\n")

        if writer is not None:
            vis = frame.copy()
            for side, color in (("left", (0, 255, 0)), ("right", (0, 128, 255))):
                hi = hand_info[side]
                if hi:
                    x1, y1, x2, y2 = _bbox_to_pixels(hi["bbox"], width, height)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(vis, side, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    for kx, ky in hi.get("keypoints", []):
                        cv2.circle(vis, (int(kx * width), int(ky * height)), 2, color, -1)
            cv2.putText(vis, f"f={fi} t={row['time']:.2f}s",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if vis.shape[1] != (writer_w := (960 if width > 960 else width)):
                vis = cv2.resize(vis, (writer_w, int(height * writer_w / width)))
            writer.write(vis)

    cap.release()
    kp_file.close()
    if writer is not None:
        writer.release()

    df = pd.DataFrame(rows)
    df.attrs["fps"] = fps
    df.to_csv(out_dir / "frame_features.csv", index=False)
    embed_arr = np.stack(embeddings).astype(np.float32)
    np.save(out_dir / "embeddings.npy", embed_arr)
    (out_dir / "meta.json").write_text(json.dumps(
        {"video": str(video_path), "fps": fps, "width": width, "height": height, "n_frames": len(df),
         "embed_model": EMBED_MODEL_NAME, "embed_dim": embed_arr.shape[1]}))
    return df
