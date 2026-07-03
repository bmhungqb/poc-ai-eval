"""Dataclasses shared across the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OperationSpec:
    name: str
    frequency: float
    tmu_manual: float
    tmu_machine: float
    tmu_bundle: float


@dataclass
class SceneSpec:
    scene_index: int
    start: float
    end: float
    operations: list[str]

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def label(self) -> str:
        return " + ".join(self.operations)


@dataclass
class VideoSpec:
    filename: str
    file_path: str


@dataclass
class TaskSpec:
    task_id: str
    task_name: str
    operations: list[OperationSpec] = field(default_factory=list)
    expert_video: VideoSpec | None = None
    expert_scenes: list[SceneSpec] = field(default_factory=list)

    def operation_frequency(self, name: str) -> float:
        for op in self.operations:
            if op.name == name:
                return op.frequency
        return 1.0
