import unittest

import numpy as np

from utils.grasp_eval import detect_grasps
from utils.offset_eval import resample_grasp_geometry


class GraspSizeScalingTest(unittest.TestCase):
    def setUp(self):
        self.quality = np.zeros((11, 11), dtype=np.float32)
        self.quality[5, 5] = 1.0
        self.sine = np.zeros_like(self.quality)
        self.cosine = np.ones_like(self.quality)
        self.width = np.full_like(self.quality, 0.5)

    def test_inverse_resize_restores_width_but_keeps_protocol_height(self):
        grasps, _ = detect_grasps(
            self.quality,
            self.sine,
            self.cosine,
            self.width,
            num_grasps=1,
            size_scale=3.0,
            size_factor=100.0,
        )
        self.assertEqual(len(grasps), 1)
        self.assertAlmostEqual(grasps[0][2], 150.0)
        self.assertAlmostEqual(grasps[0][3], 20.0)

    def test_predicted_short_side_scales_with_width(self):
        short_side = np.full_like(self.quality, 0.2)
        grasps, _ = detect_grasps(
            self.quality,
            self.sine,
            self.cosine,
            self.width,
            num_grasps=1,
            grasp_short_mask=short_side,
            size_scale=3.0,
            size_factor=100.0,
        )
        self.assertEqual(len(grasps), 1)
        self.assertAlmostEqual(grasps[0][2], 150.0)
        self.assertAlmostEqual(grasps[0][3], 60.0, delta=1e-5)

    def test_offset_resampling_keeps_fixed_protocol_height(self):
        rectangles = [[5.0, 5.0, 150.0, 20.0, 0.0]]
        restored = resample_grasp_geometry(
            rectangles,
            self.sine,
            self.cosine,
            self.width,
            size_scale=3.0,
            width_factor=100.0,
        )
        self.assertAlmostEqual(float(restored[0][2]), 150.0)
        self.assertAlmostEqual(float(restored[0][3]), 20.0)


if __name__ == "__main__":
    unittest.main()
