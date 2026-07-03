"""Per-frame signal extraction: WiLoR (YOLO hand detector + WiLoR 3D hand
pose/mesh reconstruction) for hand landmarks, Farneback optical flow per hand.

Detection runs on the fixed work-area ROI (crop + upscale, see
src/vision/work_area_roi.py) so hands cover more pixels on low-res factory
footage; keypoints are mapped back to full-frame coordinates, so everything
downstream (features, overlay, matching) is unchanged.

Produces, per video:
  <out>/frame_features.csv     one row per frame
  <out>/keypoints.jsonl        raw hand landmarks per frame
  <out>/embeddings.npy         per-frame DINOv2 embedding of the hands' crop
  <out>/work_area_roi.json     the ROI used (reused by the VLM stages)
  <out>/overlay.mp4            debug overlay video (hands/keypoints/ROI)
  <out>/overlay_optical_flow.mp4  dense Farneback flow, HSV-encoded (hue=direction,
                                   value=magnitude) -- only with flow_overlay=True
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
from src.vision.work_area_roi import (
    DEFAULT_UPSCALE_WIDTH, ROI_FILENAME, RoiMapper, estimate_roi_from_video, load_roi, save_roi)

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


def _embed_crop_bbox(hand_info: dict, width: int, height: int,
                     mapper: RoiMapper | None = None) -> tuple[int, int, int, int]:
    """Padded pixel bbox around whichever hands are present, for the
    embedding crop. Coordinates are in the processing frame (the ROI crop
    when a mapper is given, else the full frame). Falls back to the whole
    processing frame if neither hand is detected or the boxes degenerate
    (e.g. a hand detected right at / outside the ROI edge clips to an empty
    rectangle — an empty crop crashes the embedding processor)."""
    if mapper is not None:
        pw, ph = mapper.out_w, mapper.out_h
        boxes = [mapper.full_norm_bbox_to_crop_px(hi["bbox"]) for hi in hand_info.values() if hi]
    else:
        pw, ph = width, height
        boxes = [_bbox_to_pixels(hi["bbox"], width, height) for hi in hand_info.values() if hi]
    boxes = [b for b in boxes if b[2] - b[0] >= 2 and b[3] - b[1] >= 2]
    if not boxes:
        return 0, 0, pw, ph
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    pad_x, pad_y = int((x2 - x1) * EMBED_PAD_FRAC), int((y2 - y1) * EMBED_PAD_FRAC)
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(pw, x2 + pad_x), min(ph, y2 + pad_y)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return 0, 0, pw, ph
    return x1, y1, x2, y2


def _flow_to_bgr(flow: np.ndarray) -> np.ndarray:
    """HSV color-wheel encoding of a dense flow field: hue = direction,
    value = magnitude (normalized per-frame so slow segments are still visible)."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ang * (180 / (2 * np.pi))
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


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
    flow_overlay: bool = False,
    max_frames: int | None = None,
    roi: str | dict | None = "auto",
    roi_upscale_width: int = DEFAULT_UPSCALE_WIDTH,
) -> pd.DataFrame:
    """`roi` selects the work-area ROI (proposal §3.3): "auto" estimates it
    with a sampling pass over the video, a path/dict uses a saved
    work_area_roi.json (or legacy roi_auto.json), None disables cropping
    (previous behavior)."""
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

    hand_model = WiLorHand(conf_thresh=HAND_SCORE_THRESH)  # YOLO hand det + WiLoR 3D pose
    embedder = DinoEmbedder()  # frame-level embedding for image_embed nearest-neighbor score

    # --- work-area ROI (crop + upscale before hand detection) ---
    roi_data: dict | None = None
    if roi == "auto":
        print(f"estimating work-area ROI ({Path(video_path).name})...")
        work_area = estimate_roi_from_video(video_path, hand_model=hand_model)
        roi_data = save_roi(out_dir / ROI_FILENAME, work_area)
    elif isinstance(roi, (str, Path)):
        roi_data = load_roi(roi)
        save_roi(out_dir / ROI_FILENAME, roi_data["work_area"], roi_data.get("zones"),
                 source=roi_data.get("source", "file"))
    elif isinstance(roi, dict):
        roi_data = roi if "work_area" in roi else {"work_area": [0, 0, 1, 1], "zones": {}}
        save_roi(out_dir / ROI_FILENAME, roi_data["work_area"], roi_data.get("zones"),
                 source=roi_data.get("source", "dict"))
    mapper = (RoiMapper(roi_data["work_area"], width, height, roi_upscale_width)
              if roi_data is not None else None)
    # dimensions of the frame the detectors/flow actually see
    proc_w, proc_h = (mapper.out_w, mapper.out_h) if mapper else (width, height)

    flow_scale = FLOW_SCALE_WIDTH / proc_w
    flow_size = (FLOW_SCALE_WIDTH, max(2, int(proc_h * flow_scale)))

    writer = None
    if overlay:
        ow = 960 if width > 960 else width
        oh = int(height * ow / width)
        writer = cv2.VideoWriter(
            str(out_dir / "overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh)
        )

    flow_writer = None
    flow_writer_size = None
    if flow_overlay:
        fw = proc_w if proc_w <= 960 else 960
        fh = int(proc_h * fw / proc_w)
        flow_writer_size = (fw, fh)
        flow_writer = cv2.VideoWriter(
            str(out_dir / "overlay_optical_flow.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh)
        )

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
        proc = mapper.crop(frame) if mapper else frame  # what the detectors see
        small_gray = cv2.cvtColor(cv2.resize(proc, flow_size), cv2.COLOR_BGR2GRAY)

        # --- hands (WiLoR: YOLO hand detector + WiLoR 3D pose/mesh model) ---
        # detect on the (cropped + upscaled) processing frame, then map the
        # keypoints back to full-frame normalized coordinates
        raw_hands = hand_model(proc)
        by_side: dict[str, list[dict]] = {"left": [], "right": []}
        for h in raw_hands:
            kps_px = h["keypoints"].astype(np.float64)
            if mapper:
                kps_px = mapper.to_full_px(kps_px)
            kps_norm = kps_px / np.array([width, height])
            side = "right" if h["is_right"] else "left"
            by_side[side].append(_hand_summary(kps_norm, h["score"]))
        # handedness comes straight from the detector; if it (rarely) yields
        # more than one hand on a side, keep the most confident one
        hand_info: dict[str, dict | None] = {"left": None, "right": None}
        for side in ("left", "right"):
            if by_side[side]:
                hand_info[side] = max(by_side[side], key=lambda s: s["score"])

        # --- image embedding (hands' union bbox, padded) ---
        ex1, ey1, ex2, ey2 = _embed_crop_bbox(hand_info, width, height, mapper)
        embeddings.append(embedder(proc[ey1:ey2, ex1:ex2]))

        # --- optical flow (on the processing frame) ---
        flow_feats = {}
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, small_gray, None,
                                                0.5, 2, 13, 2, 5, 1.1, 0)
            for side in ("left", "right"):
                hi = hand_info[side]
                if hi:
                    bbox_px = (mapper.full_norm_bbox_to_crop_px(hi["bbox"]) if mapper
                               else _bbox_to_pixels(hi["bbox"], width, height))
                    st = _flow_stats(flow, bbox_px, flow_scale)
                else:
                    st = {"mean_mag": 0.0, "max_mag": 0.0, "p90_mag": 0.0, "mean_dx": 0.0, "mean_dy": 0.0}
                for k, v in st.items():
                    flow_feats[f"{side}_hand_flow_{k}"] = v
        else:
            for side in ("left", "right"):
                for k in ("mean_mag", "max_mag", "p90_mag", "mean_dx", "mean_dy"):
                    flow_feats[f"{side}_hand_flow_{k}"] = 0.0
            flow = None
        prev_gray = small_gray

        if flow_writer is not None:
            flow_vis = _flow_to_bgr(flow) if flow is not None else np.zeros(
                (flow_size[1], flow_size[0], 3), dtype=np.uint8)
            flow_vis = cv2.resize(flow_vis, flow_writer_size)
            flow_writer.write(flow_vis)

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
            if mapper:
                cv2.rectangle(vis, (mapper.x0, mapper.y0), (mapper.x1, mapper.y1),
                              (255, 255, 0), 1)
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
    if flow_writer is not None:
        flow_writer.release()

    df = pd.DataFrame(rows)
    df.attrs["fps"] = fps
    df.to_csv(out_dir / "frame_features.csv", index=False)
    embed_arr = np.stack(embeddings).astype(np.float32)
    np.save(out_dir / "embeddings.npy", embed_arr)
    (out_dir / "meta.json").write_text(json.dumps(
        {"video": str(video_path), "fps": fps, "width": width, "height": height, "n_frames": len(df),
         "embed_model": EMBED_MODEL_NAME, "embed_dim": embed_arr.shape[1],
         "work_area_roi": roi_data["work_area"] if roi_data else None}))
    return df
