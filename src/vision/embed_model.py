"""Frame embedding via DINOv2 (facebook/dinov2-small): a self-supervised ViT
that needs no fine-tuning and gives strong generic visual embeddings, used
for image-level nearest-neighbor matching between worker frames and expert
scenes (src/matching/similarity.py: embed_nn_score). ~22M params -- light
enough to run alongside the WiLoR hand model on a single consumer GPU.
"""
from __future__ import annotations

import numpy as np
import torch

MODEL_NAME = "facebook/dinov2-small"


class DinoEmbedder:
    """Callable: BGR crop (HxWx3 uint8) -> L2-normalized embedding (dim,)."""

    def __init__(self, device: str | None = None):
        from transformers import AutoImageProcessor, AutoModel

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device).eval()
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def __call__(self, crop_bgr: np.ndarray) -> np.ndarray:
        if crop_bgr.shape[0] < 2 or crop_bgr.shape[1] < 2:
            # empty/degenerate crop would crash the processor's resize;
            # a zero vector is maximally distant from every real embedding
            return np.zeros(self.dim, dtype=np.float32)
        rgb = np.ascontiguousarray(crop_bgr[:, :, ::-1])
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        emb = out.last_hidden_state[:, 0, :]  # CLS token
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb[0].float().cpu().numpy()
