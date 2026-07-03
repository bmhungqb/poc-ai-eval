"""Tests for the work-area ROI, the VLM emission layer (tier 1) and the
aux-operation checklist (tier 2) — all VLM calls mocked.

Run with: .venv/bin/python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.io.schemas import OperationSpec, SceneSpec, TaskSpec
from src.matching.viterbi_decoder import DecodedSegment
from src.reporting.build_report import (LOW_CONF_REL, UNMATCHED_REL, detect_errors,
                                        segment_status)
from src.vision.work_area_roi import (MIN_ROI_SIZE, RoiMapper, load_roi,
                                      roi_from_hand_boxes, save_roi)
from src.vlm.aux_check import (ABSENT, PRESENT, UNCERTAIN, AuxChecker,
                               motion_spike_clusters, split_main_aux, zone_for_op)
from src.vlm.cache import VlmCache
from src.vlm.emission import (VlmEmissionScorer, blend_emissions, candidate_segments,
                              sample_keyframes, vlm_emission_matrix)
from src.vlm.openrouter_client import VlmError, parse_json_reply
from src.vlm.scene_prompts import (build_classify_messages, describe_op,
                                   load_op_descriptions, scene_catalog_text)


class RoiFromHandBoxesTest(unittest.TestCase):
    def test_union_covers_boxes_with_padding(self):
        boxes = np.array([[0.3, 0.4, 0.5, 0.6], [0.35, 0.45, 0.6, 0.7]] * 50)
        x1, y1, x2, y2 = roi_from_hand_boxes(boxes)
        self.assertLess(x1, 0.3)
        self.assertGreater(x2, 0.6)
        self.assertLess(y1, 0.4)
        self.assertGreater(y2, 0.7)
        self.assertTrue(0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0)

    def test_outlier_boxes_are_trimmed(self):
        boxes = np.array([[0.4, 0.4, 0.6, 0.6]] * 200 + [[0.0, 0.0, 0.05, 0.05]])
        x1, y1, _, _ = roi_from_hand_boxes(boxes)
        self.assertGreater(x1, 0.2)  # the corner outlier must not drag the ROI to 0
        self.assertGreater(y1, 0.2)

    def test_empty_boxes_full_frame(self):
        self.assertEqual(roi_from_hand_boxes(np.empty((0, 4))), [0.0, 0.0, 1.0, 1.0])

    def test_minimum_size_enforced(self):
        boxes = np.array([[0.5, 0.5, 0.51, 0.51]] * 10)
        x1, y1, x2, y2 = roi_from_hand_boxes(boxes)
        self.assertGreaterEqual(x2 - x1, MIN_ROI_SIZE - 1e-6)
        self.assertGreaterEqual(y2 - y1, MIN_ROI_SIZE - 1e-6)


class RoiMapperTest(unittest.TestCase):
    def test_keypoint_roundtrip(self):
        mapper = RoiMapper([0.25, 0.25, 0.75, 0.75], 480, 368, upscale_width=960)
        full_px = np.array([[240.0, 184.0], [130.0, 100.0]])  # inside the ROI
        crop_px = (full_px - [mapper.x0, mapper.y0]) * mapper.scale
        back = mapper.to_full_px(crop_px)
        np.testing.assert_allclose(back, full_px, atol=1e-6)

    def test_crop_upscales(self):
        mapper = RoiMapper([0.0, 0.0, 0.5, 0.5], 480, 368, upscale_width=960)
        frame = np.zeros((368, 480, 3), dtype=np.uint8)
        crop = mapper.crop(frame)
        self.assertEqual(crop.shape[1], mapper.out_w)
        self.assertEqual(mapper.out_w, 960)

    def test_never_downscales(self):
        mapper = RoiMapper([0.0, 0.0, 1.0, 1.0], 1920, 1080, upscale_width=960)
        self.assertEqual(mapper.scale, 1.0)

    def test_full_norm_bbox_to_crop_px_clips(self):
        mapper = RoiMapper([0.25, 0.25, 0.75, 0.75], 400, 400, upscale_width=300)
        x1, y1, x2, y2 = mapper.full_norm_bbox_to_crop_px([0.0, 0.0, 1.0, 1.0])
        self.assertEqual((x1, y1), (0, 0))
        self.assertEqual((x2, y2), (mapper.out_w, mapper.out_h))


class RoiFileTest(unittest.TestCase):
    def test_save_and_load_current_format(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "work_area_roi.json"
            save_roi(p, [0.1, 0.2, 0.9, 0.8], zones={"lever": [0.4, 0.3, 0.6, 0.5]})
            data = load_roi(p)
            self.assertEqual(data["work_area"], [0.1, 0.2, 0.9, 0.8])
            self.assertIn("lever", data["zones"])

    def test_load_legacy_roi_auto(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "roi_auto.json"
            p.write_text(json.dumps({
                "needle": [0.3, 0.4, 0.45, 0.58],
                "fabric_area": [0.1, 0.39, 0.66, 0.84],
                "machine_button": [0.36, 0.14, 0.54, 0.37],
            }))
            data = load_roi(p)
            wa = data["work_area"]
            self.assertEqual(wa, [0.1, 0.39, 0.66, 0.84])  # union of needle+fabric
            self.assertIn("machine_button", data["zones"])
            self.assertEqual(data["source"], "legacy")


def _motion_df(n: int, spikes: list[int] | None = None) -> pd.DataFrame:
    left = np.full(n, 0.1)
    right = np.full(n, 0.1)
    for s in spikes or []:
        left[s] = 5.0
    return pd.DataFrame({
        "left_hand_flow_mean_mag": left, "right_hand_flow_mean_mag": right,
        "left_hand_speed": np.zeros(n), "right_hand_speed": np.zeros(n),
    })


class CandidateSegmentsTest(unittest.TestCase):
    def test_covers_video_and_respects_min_duration(self):
        rng = np.random.RandomState(0)
        n, fps = 600, 30.0
        df = pd.DataFrame({
            "left_hand_flow_mean_mag": rng.rand(n),
            "right_hand_flow_mean_mag": rng.rand(n),
            "left_hand_speed": rng.rand(n),
            "right_hand_speed": rng.rand(n),
        })
        segs = candidate_segments(df, fps, min_dur_s=1.0, max_segments=10)
        self.assertLessEqual(len(segs), 10)
        self.assertEqual(segs[0][0], 0)
        self.assertEqual(segs[-1][1], n)
        for (a0, a1), (b0, b1) in zip(segs[:-1], segs[1:]):
            self.assertEqual(a1, b0)  # contiguous
            self.assertGreaterEqual(a1 - a0, 3)


class SampleKeyframesTest(unittest.TestCase):
    def test_interior_and_sorted(self):
        kfs = sample_keyframes(100, 200, n=4)
        self.assertEqual(kfs, sorted(kfs))
        self.assertTrue(all(100 <= k < 200 for k in kfs))
        self.assertEqual(len(kfs), 4)

    def test_tiny_segment(self):
        kfs = sample_keyframes(10, 13, n=4)
        self.assertTrue(all(10 <= k < 13 for k in kfs))
        self.assertGreaterEqual(len(kfs), 1)


class EmissionMatrixTest(unittest.TestCase):
    def _result(self, f0, f1, scores):
        from src.vlm.emission import VlmSegmentResult
        return VlmSegmentResult("s", f0, f1, 0.0, 0.0, [], scores)

    def test_uncovered_steps_are_uniform(self):
        steps = [0, 10, 20, 30]
        mat = vlm_emission_matrix([self._result(5, 15, {0: 1.0, 1: 0.0, 2: 0.0})], steps, 3)
        np.testing.assert_allclose(mat[0], [1 / 3] * 3)
        self.assertGreater(mat[1][0], 0.9)  # step 10 covered, scene 0 dominant
        np.testing.assert_allclose(mat[2], [1 / 3] * 3)

    def test_failed_segment_stays_uniform(self):
        steps = [0, 10]
        mat = vlm_emission_matrix([self._result(0, 20, {0: 0.0, 1: 0.0, 2: 0.0})], steps, 3)
        np.testing.assert_allclose(mat, np.full((2, 3), 1 / 3))

    def test_rows_sum_to_one(self):
        steps = list(range(0, 100, 10))
        mat = vlm_emission_matrix([self._result(0, 50, {0: 0.7, 1: 0.2, 2: 0.0}),
                                   self._result(50, 100, {0: 0.0, 1: 0.1, 2: 0.9})], steps, 3)
        np.testing.assert_allclose(mat.sum(axis=1), np.ones(len(steps)), atol=1e-9)

    def test_blend_weights_and_normalization(self):
        base = np.array([[0.8, 0.1, 0.1]])
        vlm = np.array([[0.0, 1.0, 0.0]])
        out = blend_emissions(base, vlm, weight=0.6)
        np.testing.assert_allclose(out.sum(axis=1), [1.0])
        self.assertGreater(out[0, 1], out[0, 0])  # VLM term dominates at w=0.6
        np.testing.assert_allclose(blend_emissions(base, vlm, weight=0.0), base)


class ParseJsonReplyTest(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(parse_json_reply('{"a": 1}'), {"a": 1})

    def test_fenced_json(self):
        self.assertEqual(parse_json_reply('Here:\n```json\n{"a": 1}\n```\nthanks'), {"a": 1})

    def test_prose_wrapped_json(self):
        self.assertEqual(parse_json_reply('Sure! {"a": {"b": 2}} hope that helps'),
                         {"a": {"b": 2}})

    def test_garbage_raises(self):
        with self.assertRaises(VlmError):
            parse_json_reply("no json here")


class VlmCacheTest(unittest.TestCase):
    def test_roundtrip_and_stable_keys(self):
        with tempfile.TemporaryDirectory() as d:
            cache = VlmCache(d)
            k1 = cache.key(video="abc", frames=[1, 2], prompt_version="v1", model="m")
            k2 = cache.key(video="abc", frames=[1, 2], prompt_version="v1", model="m")
            k3 = cache.key(video="abc", frames=[1, 3], prompt_version="v1", model="m")
            self.assertEqual(k1, k2)
            self.assertNotEqual(k1, k3)
            self.assertIsNone(cache.get(k1))
            cache.put(k1, {"x": 1})
            self.assertEqual(cache.get(k1), {"x": 1})
            self.assertEqual(cache.hits, 1)
            self.assertEqual(cache.misses, 1)


class _FakeSampler:
    fps = 30.0

    def jpeg_b64_many(self, idxs):
        return ["ZmFrZQ=="] * len(idxs)

    def jpeg_b64(self, idx):
        return "ZmFrZQ=="


class _FakeClient:
    model = "fake/model"
    available = True

    def __init__(self, reply: dict | Exception):
        self.reply = reply
        self.calls = 0

    def chat_json(self, messages):
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _scenes(n=3):
    return [SceneSpec(i, float(i), float(i + 1), [f"op{i}"]) for i in range(n)]


class VlmEmissionScorerTest(unittest.TestCase):
    def _scorer(self, client, cache):
        return VlmEmissionScorer(client, cache, _FakeSampler(), _scenes(),
                                 load_op_descriptions(config_path="/nonexistent"),
                                 video_fp="fp")

    def test_parses_scores_and_caches(self):
        client = _FakeClient({"scores": {"0": 0.9, "1": 0.2, "2": "0.05"}, "evidence": "e"})
        with tempfile.TemporaryDirectory() as d:
            cache = VlmCache(d)
            scorer = self._scorer(client, cache)
            r1 = scorer.score_segment("seg00", 0, 90, 30.0)
            self.assertAlmostEqual(r1.scores[0], 0.9)
            self.assertAlmostEqual(r1.scores[2], 0.05)
            self.assertFalse(r1.cached)
            r2 = scorer.score_segment("seg00", 0, 90, 30.0)
            self.assertTrue(r2.cached)
            self.assertEqual(client.calls, 1)
            self.assertEqual(r2.scores, r1.scores)

    def test_vlm_error_yields_zero_scores(self):
        client = _FakeClient(VlmError("boom"))
        with tempfile.TemporaryDirectory() as d:
            scorer = self._scorer(client, VlmCache(d))
            r = scorer.score_segment("seg00", 0, 90, 30.0)
            self.assertEqual(sum(r.scores.values()), 0.0)
            self.assertIn("boom", r.error)


class ScenePromptsTest(unittest.TestCase):
    def test_catalog_lists_all_scenes_in_order(self):
        scenes = _scenes(4)
        text = scene_catalog_text(scenes, {})
        for sc in scenes:
            self.assertIn(f"{sc.scene_index}. ", text)

    def test_describe_op_case_insensitive_with_fallback(self):
        d = {"Cắt chỉ": "cắt chỉ cuối đường may"}
        self.assertEqual(describe_op("cắt chỉ", d), "cắt chỉ cuối đường may")
        self.assertEqual(describe_op("op lạ", d), "op lạ")

    def test_classify_messages_contain_frames_and_contract(self):
        msgs = build_classify_messages(_scenes(), {}, "seg00", ["AA==", "BB=="])
        content = msgs[1]["content"]
        images = [c for c in content if c.get("type") == "image_url"]
        self.assertEqual(len(images), 2)
        text = " ".join(c["text"] for c in content if c.get("type") == "text")
        self.assertIn('"scores"', text)
        self.assertIn("seg00", text)


class SplitMainAuxTest(unittest.TestCase):
    def _spec(self):
        return TaskSpec(task_id="t", task_name="t", operations=[
            OperationSpec("Diễu cạnh dài", 1, 10.0, 50.0, 0.0),
            OperationSpec("Điều chỉnh mép", 1, 5.0, 0.0, 0.0),
            OperationSpec("Cắt chỉ", 1, 2.0, 1.0, 0.0),
        ])

    def test_single_op_scene_has_no_aux(self):
        scene = SceneSpec(0, 0, 1, ["Diễu cạnh dài"])
        main, aux = split_main_aux(scene, self._spec())
        self.assertEqual(main, "Diễu cạnh dài")
        self.assertEqual(aux, [])

    def test_tmu_heuristic_picks_biggest_as_main(self):
        scene = SceneSpec(0, 0, 1, ["Điều chỉnh mép", "Diễu cạnh dài"])
        main, aux = split_main_aux(scene, self._spec())
        self.assertEqual(main, "Diễu cạnh dài")
        self.assertEqual(aux, ["Điều chỉnh mép"])

    def test_always_aux_config_overrides(self):
        scene = SceneSpec(0, 0, 1, ["Cắt chỉ", "Điều chỉnh mép"])
        main, aux = split_main_aux(scene, self._spec(),
                                   {"always_aux": ["Cắt chỉ"]})
        self.assertEqual(main, "Điều chỉnh mép")
        self.assertEqual(aux, ["Cắt chỉ"])

    def test_unknown_only_scene(self):
        scene = SceneSpec(0, 0, 1, ["UNKNOWN"])
        main, aux = split_main_aux(scene, self._spec())
        self.assertEqual((main, aux), ("UNKNOWN", []))


class MotionSpikeClustersTest(unittest.TestCase):
    def test_clusters_centered_on_spikes(self):
        df = _motion_df(300, spikes=[100, 200])
        clusters = motion_spike_clusters(df, 50, 250, fps=30.0, n_clusters=2, cluster_len=6)
        self.assertEqual(len(clusters), 2)
        centers = [cl[len(cl) // 2] for cl in clusters]
        self.assertTrue(any(abs(c - 100) <= 3 for c in centers))
        self.assertTrue(any(abs(c - 200) <= 3 for c in centers))
        for cl in clusters:
            self.assertEqual(cl, list(range(cl[0], cl[0] + len(cl))))  # consecutive full-fps

    def test_flat_segment_still_yields_a_cluster(self):
        df = _motion_df(100)
        clusters = motion_spike_clusters(df, 10, 90, fps=30.0)
        self.assertGreaterEqual(len(clusters), 1)

    def test_zone_for_op_keyword_match(self):
        zones = {"lever": [0.1, 0.1, 0.2, 0.2], "machine_button": [0.3, 0.3, 0.4, 0.4]}
        self.assertEqual(zone_for_op("Lại mũi bằng cần gạt", zones), zones["lever"])
        self.assertEqual(zone_for_op("Lại mũi bằng nút nhấn", zones), zones["machine_button"])
        self.assertIsNone(zone_for_op("Diễu cạnh dài", zones))


class AuxCombineTest(unittest.TestCase):
    def test_confident_yes_wins(self):
        v, c, _ = AuxChecker._combine(
            [{"answer": "no", "confidence": 0.9, "evidence": ""},
             {"answer": "yes", "confidence": 0.8, "evidence": "saw lever"}], None)
        self.assertEqual(v, PRESENT)
        self.assertAlmostEqual(c, 0.8)

    def test_all_confident_no_is_absent(self):
        v, _, _ = AuxChecker._combine(
            [{"answer": "no", "confidence": 0.9, "evidence": ""},
             {"answer": "no", "confidence": 0.8, "evidence": ""}], None)
        self.assertEqual(v, ABSENT)

    def test_low_confidence_answers_are_uncertain(self):
        v, _, _ = AuxChecker._combine(
            [{"answer": "yes", "confidence": 0.3, "evidence": ""}], None)
        self.assertEqual(v, UNCERTAIN)

    def test_zone_detector_alone(self):
        v, _, _ = AuxChecker._combine([], 3.5)
        self.assertEqual(v, PRESENT)
        v, _, _ = AuxChecker._combine([], 1.0)
        self.assertEqual(v, UNCERTAIN)

    def test_nothing_available_is_uncertain(self):
        v, c, _ = AuxChecker._combine([], None)
        self.assertEqual(v, UNCERTAIN)
        self.assertEqual(c, 0.0)


class SegmentStatusTest(unittest.TestCase):
    def _seg(self, conf, idx=0):
        return DecodedSegment("E0", ["op"], 0.0, 1.0, conf, idx)

    def test_thresholds_relative_to_uniform(self):
        n = 10  # uniform = 0.1
        self.assertEqual(segment_status(self._seg(0.10), n), "UNMATCHED")
        self.assertEqual(segment_status(self._seg(UNMATCHED_REL / n + 1e-4), n), "LOW_CONFIDENCE")
        self.assertEqual(segment_status(self._seg(LOW_CONF_REL / n + 1e-4), n), "MATCHED")
        self.assertEqual(segment_status(self._seg(0.9, idx=None), n), "EXTRA")

    def test_unmatched_segment_makes_scene_missing(self):
        scenes = [SceneSpec(0, 0.0, 1.0, ["op0"]), SceneSpec(1, 1.0, 2.0, ["op1"])]
        spec = TaskSpec(task_id="t", task_name="t", operations=[], expert_scenes=scenes)
        # scene 0 matched confidently, scene 1 only via a no-confidence segment
        segments = [
            DecodedSegment("E0", ["op0"], 0.0, 1.0, 0.9, 0),
            DecodedSegment("E1", ["op1"], 1.0, 2.0, 1.0 / len(scenes), 1),
        ]
        errors = detect_errors(segments, scenes, spec)
        missing = [e for e in errors if e["type"] == "MISSING"]
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0]["expert_scene_index"], 1)


if __name__ == "__main__":
    unittest.main()
