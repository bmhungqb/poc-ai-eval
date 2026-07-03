"""Disk cache for VLM results, keyed by (video fingerprint, frame indices,
prompt version, model, prompt hash) — re-running the pipeline never re-pays
for a call it already made (proposal §2.4)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

_FP_CHUNK = 1 << 20  # 1 MiB head + tail


def video_fingerprint(path: str | Path) -> str:
    """Cheap content fingerprint: file size + sha1 of the first and last MiB
    (hashing multi-GB videos in full would dominate runtime)."""
    p = Path(path)
    h = hashlib.sha1()
    size = p.stat().st_size
    h.update(str(size).encode())
    with open(p, "rb") as f:
        h.update(f.read(_FP_CHUNK))
        if size > 2 * _FP_CHUNK:
            f.seek(-_FP_CHUNK, 2)
            h.update(f.read(_FP_CHUNK))
    return h.hexdigest()[:16]


class VlmCache:
    def __init__(self, cache_dir: str | Path):
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(**parts) -> str:
        blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:32]

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> dict | None:
        p = self._path(key)
        if not p.exists():
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(p.read_text(encoding="utf-8"))

    def put(self, key: str, value: dict) -> None:
        self._path(key).write_text(
            json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")

    def get_or_call(self, key: str, fn) -> dict:
        cached = self.get(key)
        if cached is not None:
            return cached
        value = fn()
        self.put(key, value)
        return value
