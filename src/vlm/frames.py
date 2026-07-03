"""Frame sampling for VLM calls: read frames from a video, crop to the
work-area ROI, upscale and JPEG-encode. Every frame sent to the VLM goes
through this (proposal §3.3: less background noise, cheaper per call)."""
from __future__ import annotations

import base64
from pathlib import Path

import cv2

from src.vision.work_area_roi import RoiMapper

VLM_FRAME_WIDTH = 640   # width of frames sent to the VLM (after ROI crop)
JPEG_QUALITY = 85


class FrameSampler:
    """Random-access frame reader with ROI crop + JPEG base64 encoding.

    `work_area` is a normalized [x1, y1, x2, y2] box (None = full frame).
    """

    def __init__(self, video_path: str | Path, work_area: list[float] | None = None,
                 frame_width: int = VLM_FRAME_WIDTH, jpeg_quality: int = JPEG_QUALITY):
        self.video_path = Path(video_path)
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video {video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.mapper = (RoiMapper(work_area, w, h, upscale_width=frame_width)
                       if work_area else None)
        self.frame_width = frame_width
        self.jpeg_quality = jpeg_quality
        self._last_pos = -2  # sequential reads avoid a seek

    def read(self, frame_idx: int):
        """Raw (uncropped) BGR frame at `frame_idx`, or None."""
        frame_idx = max(0, min(frame_idx, self.n_frames - 1))
        if frame_idx != self._last_pos + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        self._last_pos = frame_idx
        return frame if ok else None

    def crop(self, frame):
        if self.mapper is not None:
            return self.mapper.crop(frame)
        if frame.shape[1] > self.frame_width:
            scale = self.frame_width / frame.shape[1]
            frame = cv2.resize(frame, (self.frame_width, int(frame.shape[0] * scale)))
        return frame

    def jpeg_b64(self, frame_idx: int) -> str | None:
        frame = self.read(frame_idx)
        if frame is None:
            return None
        crop = self.crop(frame)
        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        return base64.b64encode(buf).decode() if ok else None

    def jpeg_b64_many(self, frame_idxs: list[int]) -> list[str]:
        out = []
        for fi in frame_idxs:
            b = self.jpeg_b64(fi)
            if b is not None:
                out.append(b)
        return out

    def close(self):
        self.cap.release()
