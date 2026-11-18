"""Unit tests for the HMM map-matching pipeline (runs without the full dataset)."""

import unittest
import numpy as np

from baseline_hmm import (
    project_to_meters,
    smooth_track,
    subsample_track,
    hmm_map_match,
    evaluate_route,
)


class TestHMMMapMatcher(unittest.TestCase):
    def test_project_to_meters_centering(self):
        nodes = np.array([[50.0, 50.0], [50.001, 50.001]], dtype=np.float64)
        track = np.array([[50.0005, 50.0005]], dtype=np.float64)
        nodes_xy, track_xy = project_to_meters(nodes, track)
        self.assertAlmostEqual(float(np.mean(nodes_xy[:, 0])), 0.0, places=6)
        self.assertAlmostEqual(float(np.mean(nodes_xy[:, 1])), 0.0, places=6)
        self.assertAlmostEqual(float(track_xy[0, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(track_xy[0, 1]), 0.0, places=6)

    def test_smooth_track_respects_gaps_and_endpoints(self):
        # Straight line with jitter + a 20s time gap in the middle
        t = np.array([1, 2, 3, 4, 5, 6, 7, 8, 28, 29, 30, 31, 32, 33, 34], dtype=np.float64)
        xy = np.column_stack([np.arange(len(t)) * 10.0, np.sin(np.arange(len(t))) * 2.0])
        smoothed = smooth_track(xy, timestamps=t, window=7, polyorder=2)
        np.testing.assert_allclose(smoothed[0], xy[0])
        np.testing.assert_allclose(smoothed[7], xy[7])
        np.testing.assert_allclose(smoothed[8], xy[8])
        np.testing.assert_allclose(smoothed[-1], xy[-1])

    def test_subsample_track_preserves_endpoints(self):
        xy = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0], [25.0, 0.0], [30.0, 0.0]])
        sub = subsample_track(xy, min_step_m=20.0)
        self.assertEqual(len(sub), 3)
        np.testing.assert_allclose(sub[0], xy[0])
        np.testing.assert_allclose(sub[-1], xy[-1])

    def test_synthetic_grid_map_matching(self):
        # 3-node L-shaped road: (0,0) -> (100,0) -> (100,100), plus a decoy parallel road
        nodes_xy = np.array(
            [
                [0.0, 0.0],
                [100.0, 0.0],
                [100.0, 100.0],
                [0.0, 50.0],
                [100.0, 50.0],
            ],
            dtype=np.float64,
        )
        arcs = np.array(
            [
                [0, 1],  # arc 0 (true)
                [1, 2],  # arc 1 (true)
                [3, 4],  # arc 2 (decoy)
                [4, 2],  # arc 3 (decoy)
            ],
            dtype=np.int32,
        )
        # Noisy 1 Hz GPS along (0,0) -> (100,0) -> (100,100)
        rng = np.random.default_rng(42)
        seg1 = np.column_stack([np.linspace(5, 95, 10), np.zeros(10)])
        seg2 = np.column_stack([np.full(10, 100.0), np.linspace(5, 95, 10)])
        track_xy = np.vstack([seg1, seg2]) + rng.normal(0.0, 2.0, size=(20, 2))
        timestamps = np.arange(1, 21, dtype=np.float64)

        pred, arc_lens = hmm_map_match(nodes_xy, arcs, track_xy, timestamps=timestamps)
        metrics = evaluate_route(pred, [0, 1], arc_lens)
        self.assertEqual(pred, [0, 1])
        self.assertAlmostEqual(metrics["f1_cnt"], 1.0)
        self.assertAlmostEqual(metrics["f1_len"], 1.0)


if __name__ == "__main__":
    unittest.main()
