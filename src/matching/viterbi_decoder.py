"""Viterbi decoding of worker observation steps against expert scene order.

Duration-aware state layout: each scene i expands into K_i chained sub-states
(minimum dwell = K_i observation steps, derived from the expert scene duration),
followed by a self-looping "stay" sub-state. Each scene also gets an
EXTRA-after-scene-i state so out-of-vocabulary actions keep sequence position.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PENALTIES = {
    "next_state": 0.0,
    "skip": 1.5,        # per skipped scene
    "extra": 0.8,       # enter EXTRA
    "extra_stay": 0.05,
    "backward": 4.0,    # first backward step; +0.5 per additional scene back
    "reenter": 4.0,     # re-enter the same scene after leaving (duplicate)
    "start_skip": 1.0,  # per scene skipped at the start
    "end_skip": 1.0,    # per scene left unvisited at the end
}

# With VLM emissions blended in, per-step log-prob differences are much larger
# than with the near-uniform pose/flow emissions the penalties above were tuned
# against (proposal §3.4: backward/reenter were high enough that the cheapest
# path was always "walk every scene in order" regardless of video content).
# Lower penalties let sustained VLM evidence actually create skips/duplicates.
PENALTIES_VLM = {**PENALTIES, "skip": 1.0, "backward": 2.5, "reenter": 2.5,
                 "start_skip": 0.7, "end_skip": 0.7}


@dataclass
class DecodedSegment:
    assigned_state: str          # "E<i>" or "EXTRA"
    operations: list[str]
    start_time: float
    end_time: float
    confidence: float
    matched_expert_scene_index: int | None
    step_range: tuple[int, int] = field(default=(0, 0))


def sharpen_emissions(raw: np.ndarray, temperature: float = 4.0) -> np.ndarray:
    """Raw similarity matrix (T, N) -> per-step probability-like emissions.

    Column z-scoring removes per-scene bias (templates that match everything),
    then a per-row softmax makes each observation step discriminative. The
    temperature keeps per-step log-prob differences small relative to the
    transition penalties, so path jumps need sustained evidence.
    """
    z = (raw - raw.mean(axis=0)) / (raw.std(axis=0) + 1e-6)
    z = z / temperature
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def _build_states(n_scenes: int, dwell_steps: list[int]):
    """Return (scene_of_state, entry_idx, stay_idx, extra_idx, n_states)."""
    scene_of_state: list[int] = []
    entry, stay, extra = [], [], []
    for i in range(n_scenes):
        entry.append(len(scene_of_state))
        k = max(1, dwell_steps[i])
        scene_of_state.extend([i] * k)
        stay.append(len(scene_of_state) - 1)
    extra_base = len(scene_of_state)
    for i in range(n_scenes):
        extra.append(extra_base + i)
    return scene_of_state, entry, stay, extra, extra_base + n_scenes


def decode(emissions: np.ndarray, extra_emission: np.ndarray,
           dwell_steps: list[int], penalties: dict | None = None) -> list[int]:
    """emissions: (T, N) per-step scene emissions; extra_emission: (T,).
    dwell_steps[i]: minimum observation steps scene i must occupy if visited.
    penalties: transition penalty set (defaults to PENALTIES; use
    PENALTIES_VLM when VLM emissions are blended in).

    Returns state path of length T: scene index, or -1 for EXTRA.
    """
    T, N = emissions.shape
    p = penalties or PENALTIES
    scene_of_state, entry, stay, extra, S = _build_states(N, dwell_steps)

    pen = np.full((S, S), np.inf)
    # chain within each scene + stay self-loop
    for i in range(N):
        for s in range(entry[i], stay[i]):
            pen[s, s + 1] = 0.0
        pen[stay[i], stay[i]] = 0.0
        pen[stay[i], extra[i]] = p["extra"]
        pen[extra[i], extra[i]] = p["extra_stay"]
        for j in range(N):
            if j == i + 1:
                d = p["next_state"]
            elif j > i + 1:
                d = p["skip"] * (j - i - 1)
            elif j == i:
                d = p["reenter"]
            else:
                d = p["backward"] + 0.5 * (i - j - 1)
            pen[stay[i], entry[j]] = d
            # leaving EXTRA back into the sequence; resuming scene i is free
            pen[extra[i], entry[j]] = p["next_state"] if j == i else d

    log_scene = np.log(np.clip(emissions, 1e-6, None))
    log_extra = np.log(np.clip(extra_emission, 1e-6, None))
    scene_arr = np.array(scene_of_state)

    def emis_row(t: int) -> np.ndarray:
        row = np.empty(S)
        row[: len(scene_arr)] = log_scene[t][scene_arr]
        row[len(scene_arr):] = log_extra[t]
        return row

    start_pen = np.full(S, np.inf)
    for i in range(N):
        start_pen[entry[i]] = p["start_skip"] * i
        start_pen[extra[i]] = p["start_skip"] * i + p["extra"]
    end_pen = np.empty(S)
    for s in range(len(scene_arr)):
        end_pen[s] = p["end_skip"] * (N - 1 - scene_arr[s])
    for i in range(N):
        end_pen[extra[i]] = p["end_skip"] * (N - 1 - i)

    delta = emis_row(0) - start_pen
    back = np.zeros((T, S), dtype=int)
    for t in range(1, T):
        cand = delta[:, None] - pen
        back[t] = np.argmax(cand, axis=0)
        delta = cand[back[t], np.arange(S)] + emis_row(t)
    last = int(np.argmax(delta - end_pen))

    states = [last]
    for t in range(T - 1, 0, -1):
        states.append(int(back[t][states[-1]]))
    states.reverse()
    return [scene_of_state[s] if s < len(scene_arr) else -1 for s in states]


def merge_path(path: list[int], step_times: list[float], step_duration: float,
               emissions: np.ndarray, extra_emission: np.ndarray,
               scenes) -> list[DecodedSegment]:
    """Merge consecutive equal states into segments; confidence = mean of the
    passed emission matrix over the segment's steps (pass whichever matrix was
    actually decoded — for the VLM pipeline that is the blended, probability-
    like matrix, so confidence is comparable to 1/n_scenes uniform)."""
    segments: list[DecodedSegment] = []
    t0 = 0
    for t in range(1, len(path) + 1):
        if t == len(path) or path[t] != path[t0]:
            state = path[t0]
            conf = float(extra_emission[t0:t].mean()) if state == -1 else float(emissions[t0:t, state].mean())
            if state == -1:
                name, ops, idx = "EXTRA", ["EXTRA"], None
            else:
                sc = scenes[state]
                name, ops, idx = f"E{state}", sc.operations, state
            segments.append(DecodedSegment(
                assigned_state=name, operations=ops,
                start_time=step_times[t0],
                end_time=step_times[t - 1] + step_duration,
                confidence=round(conf, 3),
                matched_expert_scene_index=idx,
                step_range=(t0, t)))
            t0 = t
    return segments
