"""Minimal regression tests for the matching/report core logic.

Run with: .venv/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.io.schemas import OperationSpec, SceneSpec, TaskSpec
from src.matching.similarity import (FEATURE_CHANNELS, FLOW_CHANNELS, KEYPOINT_CHANNELS, WEIGHTS,
                                     SceneTemplate, dtw_distance, embed_nn_score, frame_nn_score,
                                     window_scores)
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


def _scene_df(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, columns=FEATURE_CHANNELS)


class FrameNnScoreTest(unittest.TestCase):
    def _template(self, values: np.ndarray) -> SceneTemplate:
        return SceneTemplate(0, "s", ["op"], 0.0, 1.0, _scene_df(values))

    def test_window_identical_to_scene_scores_near_one(self):
        rng = np.random.RandomState(0)
        vals = rng.randn(30, len(FEATURE_CHANNELS))
        tpl = self._template(vals)
        win = _scene_df(vals)
        self.assertAlmostEqual(frame_nn_score(win, tpl), 1.0, places=6)

    def test_window_far_from_scene_scores_near_zero(self):
        rng = np.random.RandomState(0)
        near = rng.randn(30, len(FEATURE_CHANNELS))
        far = near + 100.0  # far away in every channel
        tpl = self._template(near)
        win = _scene_df(far)
        self.assertLess(frame_nn_score(win, tpl), 1e-6)

    def test_prefers_the_closer_of_two_scenes(self):
        rng = np.random.RandomState(0)
        cluster_a = rng.randn(20, len(FEATURE_CHANNELS))
        cluster_b = cluster_a + 10.0
        tpl_a = self._template(cluster_a)
        tpl_b = self._template(cluster_b)
        probe = _scene_df(cluster_a + 0.1)  # close to A, far from B
        self.assertGreater(frame_nn_score(probe, tpl_a), frame_nn_score(probe, tpl_b))

    def test_empty_window_scores_zero(self):
        tpl = self._template(np.random.RandomState(0).randn(10, len(FEATURE_CHANNELS)))
        self.assertEqual(frame_nn_score(_scene_df(np.empty((0, len(FEATURE_CHANNELS)))), tpl), 0.0)


class EmbedNnScoreTest(unittest.TestCase):
    def test_none_when_template_has_no_embeddings(self):
        tpl = SceneTemplate(0, "s", ["op"], 0.0, 1.0,
                            _scene_df(np.random.RandomState(0).randn(10, len(FEATURE_CHANNELS))))
        win_embed = np.random.RandomState(0).randn(5, 384)
        self.assertIsNone(embed_nn_score(win_embed, tpl))

    def test_none_when_window_has_no_embeddings(self):
        rng = np.random.RandomState(0)
        tpl = SceneTemplate(0, "s", ["op"], 0.0, 1.0,
                            _scene_df(rng.randn(10, len(FEATURE_CHANNELS))),
                            embed_seg=rng.randn(10, 384))
        self.assertIsNone(embed_nn_score(None, tpl))
        self.assertIsNone(embed_nn_score(np.empty((0, 384)), tpl))

    def test_prefers_the_closer_embedding_cluster(self):
        rng = np.random.RandomState(0)
        pose = rng.randn(10, len(FEATURE_CHANNELS))
        embed_a = rng.randn(10, 384)
        embed_b = embed_a + 10.0
        tpl_a = SceneTemplate(0, "a", ["op"], 0.0, 1.0, _scene_df(pose), embed_seg=embed_a)
        tpl_b = SceneTemplate(1, "b", ["op"], 0.0, 1.0, _scene_df(pose), embed_seg=embed_b)
        probe = embed_a + 0.1
        self.assertGreater(embed_nn_score(probe, tpl_a), embed_nn_score(probe, tpl_b))


class WindowScoresWeightRenormalizationTest(unittest.TestCase):
    def test_total_is_plain_average_of_ones_regardless_of_embeddings(self):
        # keypoint/flow/duration/frame_nn/embed all maxed out (window == scene,
        # matching duration) -> total should be 1.0 whether or not the
        # image_embed term is present, since weights are renormalized over
        # whichever terms are actually available.
        rng = np.random.RandomState(0)
        pose = rng.randn(24, len(KEYPOINT_CHANNELS) + len(FLOW_CHANNELS))
        seg = pd.DataFrame(pose, columns=KEYPOINT_CHANNELS + FLOW_CHANNELS)
        embed = rng.randn(24, 384)
        tpl_with_embed = SceneTemplate(0, "s", ["op"], 0.0, 1.0, seg, embed_seg=embed)
        tpl_without_embed = SceneTemplate(0, "s", ["op"], 0.0, 1.0, seg)

        s_with = window_scores(seg, 1.0, tpl_with_embed, win_embed=embed)
        s_without = window_scores(seg, 1.0, tpl_without_embed, win_embed=None)

        self.assertAlmostEqual(s_with["total"], 1.0, places=3)
        self.assertAlmostEqual(s_without["total"], 1.0, places=3)
        self.assertIn("image_embed", s_with)
        self.assertNotIn("image_embed", s_without)
        self.assertEqual(sum(WEIGHTS.values()), 1.0)


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
