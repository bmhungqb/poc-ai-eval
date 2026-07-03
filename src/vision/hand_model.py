"""Hand pose backend: WiLoR (YOLO hand detector + WiLoR 3D hand pose/mesh
reconstruction, https://github.com/warmshao/WiLoR-mini).

Unlike the previous RTMPose-Hand backend, WiLoR's detector classifies
handedness (anatomical left/right) directly instead of us having to infer
side from x-position, and is more robust on low-res/gloved factory footage.
"""
from __future__ import annotations

import numpy as np
import torch


class WiLorHand:
    """Callable hand detector: frame (BGR, HxWx3) -> list of per-hand dicts
    with 21 MANO-order 2D keypoints (pixel coords, same tip indices as
    MediaPipe: wrist=0, thumb tip=4, index tip=8), detector confidence, and
    handedness."""

    def __init__(self, device: str | None = None, conf_thresh: float = 0.3):
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
            WiLorHandPose3dEstimationPipeline,
        )

        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = torch.float16 if dev.type == "cuda" else torch.float32
        self._pipe = WiLorHandPose3dEstimationPipeline(device=dev, dtype=dtype, verbose=False)
        self._conf_thresh = conf_thresh

    def __call__(self, frame_bgr: np.ndarray) -> list[dict]:
        image_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])

        # detect first (ourselves) so we keep the YOLO confidence, which the
        # WiLoR-mini pipeline's own predict() discards
        detections = self._pipe.hand_detector(image_rgb, conf=self._conf_thresh, verbose=False)[0]
        if len(detections.boxes) == 0:
            return []
        bboxes = detections.boxes.xyxy.cpu().numpy()
        is_rights = detections.boxes.cls.cpu().numpy()
        confs = detections.boxes.conf.cpu().numpy()

        results = self._pipe.predict_with_bboxes(image_rgb, bboxes, is_rights)
        hands = []
        for res, conf in zip(results, confs):
            kps = np.asarray(res["wilor_preds"]["pred_keypoints_2d"][0])  # (21, 2) px
            hands.append({
                "keypoints": kps,
                "is_right": bool(res["is_right"]),
                "score": float(conf),
            })
        return hands
