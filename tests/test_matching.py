"""Minimal regression tests for the matching/report core logic.

Run with: .venv/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import unittest

import numpy as np

from src.io.schemas import OperationSpec, SceneSpec, TaskSpec
from src.matching.similarity import dtw_distance
from src.matching.viterbi_decoder import decode, sharpen_emissions
from src.reporting.build_report import detect_errors
from src.matching.viterbi_decoder import DecodedSegment


class DtwDistanceTest(unittest.TestCase):
    def test_identical_sequences_have_zero_distance(self):
        a = np.random.RandomState(0).randn(10, 3)
        self.assertAlmostEqual(dtw_distance(a, a.copy()), 0.0, places=9)

    def test_matches_naive_reference(self):
        rng = np.random.RandomState(1)
        a, b = rng.randn(6, 2), rng.randn(8, 2)
        cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
        n, m = cost.shape
        acc = np.full((n + 1, m + 1), np.inf)
        acc[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
        expected = acc[n, m] / (n + m)
        self.assertAlmostEqual(dtw_distance(a, b), expected, places=9)


class SharpenEmissionsTest(unittest.TestCase):
    def test_rows_sum_to_one(self):
        raw = np.random.RandomState(2).rand(20, 4)
        sharp = sharpen_emissions(raw)
        np.testing.assert_allclose(sharp.sum(axis=1), np.ones(20), atol=1e-9)


class ViterbiDecodeTest(unittest.TestCase):
    def test_decodes_clean_in_order_sequence(self):
        # 3 scenes, each strongly preferred over a 4-step span in order.
        n_scenes, steps_per_scene = 3, 4
        T = n_scenes * steps_per_scene
        emissions = np.full((T, n_scenes), 0.01)
        for scene in range(n_scenes):
            emissions[scene * steps_per_scene:(scene + 1) * steps_per_scene, scene] = 0.9
        extra_emission = np.full(T, 1e-3)
        dwell_steps = [1] * n_scenes
        path = decode(emissions, extra_emission, dwell_steps)
        # scene indices visited, in order, should be non-decreasing and cover all scenes
        visited = [s for s in path if s != -1]
        self.assertEqual(visited, sorted(visited))
        self.assertEqual(set(visited), set(range(n_scenes)))


class DetectErrorsUnknownOpsTest(unittest.TestCase):
    def test_duplicated_scene_with_only_unknown_ops_does_not_crash(self):
        scenes = [SceneSpec(scene_index=0, start=0.0, end=1.0, operations=["UNKNOWN"])]
        spec = TaskSpec(task_id="t", task_name="test", operations=[], expert_scenes=scenes)
        segments = [
            DecodedSegment("E0", ["UNKNOWN"], 0.0, 0.5, 0.8, 0, (0, 5)),
            DecodedSegment("E0", ["UNKNOWN"], 1.0, 1.5, 0.8, 0, (10, 15)),
        ]
        errors = detect_errors(segments, scenes, spec)  # must not raise
        dup = [e for e in errors if e["type"] == "DUPLICATED_ACTION"]
        self.assertEqual(len(dup), 1)


if __name__ == "__main__":
    unittest.main()
