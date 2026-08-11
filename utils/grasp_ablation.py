"""NumPy post-processing primitives for DROG-OFF inference ablations."""

import numpy as np


def mask_grasp_quality(quality_map, segmentation_mask):
    """Zero grasp quality outside a predicted segmentation mask."""
    quality = np.asarray(quality_map)
    segmentation = np.asarray(segmentation_mask).astype(bool)
    if quality.shape != segmentation.shape:
        raise ValueError(
            "Grasp quality and segmentation masks must share the original "
            f"image canvas, got {quality.shape} and {segmentation.shape}."
        )
    return np.where(segmentation, quality, 0.0)


def filter_grasp_centres(grasps, segmentation_mask):
    """Retain final rectangles whose (x, y) centre lies inside the mask."""
    segmentation = np.asarray(segmentation_mask).astype(bool)
    if segmentation.ndim != 2:
        raise ValueError(
            f"Segmentation mask must be 2-D, got {segmentation.shape}."
        )
    height, width = segmentation.shape
    retained = []
    for grasp in grasps:
        center_x, center_y = float(grasp[0]), float(grasp[1])
        if not (np.isfinite(center_x) and np.isfinite(center_y)):
            continue
        column, row = int(round(center_x)), int(round(center_y))
        if 0 <= row < height and 0 <= column < width:
            if segmentation[row, column]:
                retained.append(grasp)
    return retained