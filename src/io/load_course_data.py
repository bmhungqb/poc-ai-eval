"""Loader for course-builder-data.json -> TaskSpec."""
from __future__ import annotations

import json
from pathlib import Path

from src.io.schemas import OperationSpec, SceneSpec, TaskSpec, VideoSpec

UNKNOWN_OP = "UNKNOWN"


def load_task_spec(json_path: str | Path) -> TaskSpec:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    iv = data["input_version"]

    operations = [
        OperationSpec(
            name=o["name"],
            frequency=float(o.get("frequency", 1.0)),
            tmu_manual=float(o.get("tmu_manual", 0.0)),
            tmu_machine=float(o.get("tmu_machine", 0.0)),
            tmu_bundle=float(o.get("tmu_bundle", 0.0)),
        )
        for o in iv.get("operations", [])
    ]

    # The export uses "videos_export"; keep "videos" as fallback for older dumps.
    videos = iv.get("videos_export") or iv.get("videos") or []
    if not videos:
        raise ValueError("No videos found in course JSON")
    v0 = videos[0]
    video = VideoSpec(filename=v0.get("filename", ""), file_path=v0.get("file_path", ""))

    scenes = []
    for s in sorted(v0.get("scenes", []), key=lambda s: float(s["timestamp_start"])):
        ops = [op for op in (s.get("operations") or []) if op] or [UNKNOWN_OP]
        scenes.append(
            SceneSpec(
                scene_index=int(s["scene_index"]),
                start=float(s["timestamp_start"]),
                end=float(s["timestamp_end"]),
                operations=ops,
            )
        )
    # Re-index by temporal order so scene_index == position in expert sequence.
    for i, sc in enumerate(scenes):
        sc.scene_index = i

    return TaskSpec(
        task_id=str(iv.get("id", "")),
        task_name=iv.get("task_name", ""),
        operations=operations,
        expert_video=video,
        expert_scenes=scenes,
    )


if __name__ == "__main__":
    import sys

    spec = load_task_spec(sys.argv[1] if len(sys.argv) > 1 else "data/course-builder-data.json")
    print(f"Task: {spec.task_name} ({len(spec.expert_scenes)} scenes, {len(spec.operations)} operations)")
    for sc in spec.expert_scenes:
        print(f"  [{sc.scene_index:2d}] {sc.start:6.2f}-{sc.end:6.2f}s  {sc.label}")
