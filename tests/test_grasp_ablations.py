import unittest

import numpy as np

from utils.grasp_ablation import filter_grasp_centres, mask_grasp_quality


class GraspAblationTest(unittest.TestCase):
    def test_quality_is_zeroed_outside_the_predicted_mask(self):
        quality = np.array([[0.9, 0.8], [0.7, 0.6]], dtype=np.float32)
        segmentation = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        filtered = mask_grasp_quality(quality, segmentation)
        np.testing.assert_array_equal(
            filtered,
            np.array([[0.0, 0.8], [0.7, 0.0]], dtype=np.float32),
        )

    def test_only_final_centres_inside_the_mask_are_retained(self):
        segmentation = np.zeros((3, 4), dtype=np.uint8)
        segmentation[1, 2] = 1
        inside = [2.0, 1.0, 30.0, 20.0, 5.0]
        outside = [1.0, 1.0, 30.0, 20.0, 5.0]
        out_of_bounds = [4.0, 1.0, 30.0, 20.0, 5.0]
        invalid = [float("nan"), 1.0, 30.0, 20.0, 5.0]
        self.assertEqual(
            filter_grasp_centres(
                [inside, outside, out_of_bounds, invalid], segmentation
            ),
            [inside],
        )

    def test_filter_requires_one_shared_canvas(self):
        with self.assertRaises(ValueError):
            mask_grasp_quality(np.zeros((2, 2)), np.zeros((3, 2)))
        with self.assertRaises(ValueError):
            filter_grasp_centres([], np.zeros((1, 2, 2)))


if __name__ == "__main__":
    unittest.main()