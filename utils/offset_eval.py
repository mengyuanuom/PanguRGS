"""Geometry adapters shared by validation and real-world grasp evaluation."""

from typing import Iterable

import cv2
import numpy as np


def inverse_warp(array, inverse_matrix, original_hw, interpolation=cv2.INTER_NEAREST):
    """Warp one input-space map back to the original image resolution."""
    original_h, original_w = (int(value) for value in original_hw)
    return cv2.warpAffine(
        np.asarray(array),
        np.asarray(inverse_matrix, dtype=np.float32),
        (original_w, original_h),
        flags=interpolation,
        borderValue=0.0,
    )


def apply_affine(points_xy, matrix):
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if not len(points):
        return points
    homogeneous = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1
    )
    return (np.asarray(matrix, dtype=np.float32) @ homogeneous.T).T.astype(np.float32)


def five_to_corners(rectangle):
    center_x, center_y, width, height, theta = rect_to_five(rectangle)
    radians = np.deg2rad(theta)
    cosine, sine = np.cos(radians), np.sin(radians)
    half_width, half_height = width / 2.0, height / 2.0
    corners = np.array(
        [
            [-half_width, -half_height],
            [half_width, -half_height],
            [half_width, half_height],
            [-half_width, half_height],
        ],
        dtype=np.float32,
    )
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    corners = corners @ rotation.T
    corners[:, 0] += center_x
    corners[:, 1] += center_y
    return corners


def corners_to_five(corners):
    points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    (center_x, center_y), (width, height), angle = cv2.minAreaRect(points)
    if width < height:
        width, height = height, width
        angle += 90.0
    if angle <= -90.0:
        angle += 180.0
    if angle > 90.0:
        angle -= 180.0
    return np.array([center_x, center_y, width, height, angle], dtype=np.float32)


def rect_to_five(rectangle):
    """Normalize common 4/5/6/8-value grasp formats to ``cx,cy,w,h,theta``."""
    values = np.asarray(rectangle, dtype=np.float32).reshape(-1)
    if values.size == 5:
        return values.copy()
    if values.size == 6:
        candidate = values[:5].copy()
        theta = float(candidate[4])
        if not (abs(theta) <= 180.0 or abs(theta) <= 2.0 * np.pi):
            center_x, center_y, theta, width, height = values[:5]
            candidate = np.array(
                [center_x, center_y, width, height, theta], dtype=np.float32
            )
        return candidate
    if values.size == 8:
        return corners_to_five(values.reshape(4, 2))
    if values.size == 4:
        minimum_x, minimum_y, maximum_x, maximum_y = values
        return np.array(
            [
                (minimum_x + maximum_x) / 2.0,
                (minimum_y + maximum_y) / 2.0,
                max(1e-6, maximum_x - minimum_x),
                max(1e-6, maximum_y - minimum_y),
                0.0,
            ],
            dtype=np.float32,
        )
    raise ValueError(
        f"Unsupported grasp rectangle with {values.size} values: {values.tolist()}"
    )


def rectangles_to_five(rectangles: Iterable):
    return [rect_to_five(rectangle) for rectangle in rectangles]


def targets_to_six(rectangles: Iterable):
    result = []
    for rectangle in rectangles:
        values = np.asarray(rectangle, dtype=np.float32).reshape(-1)
        if values.size == 6:
            result.append(values.copy())
        else:
            result.append(
                np.concatenate([rect_to_five(values), np.array([0.0], dtype=np.float32)])
            )
    return result


def _sample_offset(offset, x, y):
    values = np.asarray(offset, dtype=np.float32)
    if values.ndim == 4:
        if values.shape[0] != 1:
            raise ValueError(f"Expected one offset sample, got {values.shape}")
        values = values[0]
    if values.ndim != 3 or values.shape[0] != 2:
        raise ValueError(f"Offset map must have shape (2,H,W), got {values.shape}")
    height, width = values.shape[-2:]
    column = int(np.clip(round(float(x)), 0, width - 1))
    row = int(np.clip(round(float(y)), 0, height - 1))
    return float(values[0, row, column]), float(values[1, row, column])


def _sample_bilinear(array, x, y):
    """Sample one 2-D map at a floating-point image coordinate."""
    values = np.asarray(array, dtype=np.float32).squeeze()
    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D map, got {values.shape}")
    height, width = values.shape
    x = float(np.clip(x, 0.0, width - 1.0))
    y = float(np.clip(y, 0.0, height - 1.0))
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, width - 1), min(y0 + 1, height - 1)
    wx, wy = x - x0, y - y0
    top = (1.0 - wx) * values[y0, x0] + wx * values[y0, x1]
    bottom = (1.0 - wx) * values[y1, x0] + wx * values[y1, x1]
    return float((1.0 - wy) * top + wy * bottom)


def refine_with_offset(rectangles, offset, inverse_matrix, radius):
    """Apply normalized input-space center offsets and return five-value grasps."""
    if hasattr(offset, "detach"):
        offset = offset.detach().float().cpu().numpy()
    inverse = np.asarray(inverse_matrix, dtype=np.float32)
    forward = cv2.invertAffineTransform(inverse)
    refined = []
    for rectangle in rectangles:
        corners = five_to_corners(rectangle)
        original_center = corners.mean(axis=0, keepdims=True)
        input_center = apply_affine(original_center, forward)
        delta_x, delta_y = _sample_offset(
            offset, input_center[0, 0], input_center[0, 1]
        )
        shifted_input = input_center + np.array(
            [[delta_x * float(radius), delta_y * float(radius)]], dtype=np.float32
        )
        shifted_original = apply_affine(shifted_input, inverse)
        translation = shifted_original[0] - original_center[0]
        corners[:, 0] += translation[0]
        corners[:, 1] += translation[1]
        refined.append(corners_to_five(corners))
    return refined


def resample_grasp_geometry(
    rectangles,
    sine,
    cosine,
    width,
    short_side=None,
    width_factor=100.0,
    size_scale=1.0,
):
    """Resample angle and both rectangle sides at refined grasp centers."""
    sine = np.asarray(sine, dtype=np.float32).squeeze()
    cosine = np.asarray(cosine, dtype=np.float32).squeeze()
    width = np.asarray(width, dtype=np.float32).squeeze()
    short_side = (
        None
        if short_side is None
        else np.asarray(short_side, dtype=np.float32).squeeze()
    )
    geometry_maps = [sine, cosine, width]
    if short_side is not None:
        geometry_maps.append(short_side)
    if any(value.ndim != 2 for value in geometry_maps):
        raise ValueError("grasp geometry maps must be 2-D")
    if any(value.shape != sine.shape for value in geometry_maps[1:]):
        raise ValueError("grasp geometry maps must share one shape")

    refined = []
    for rectangle in rectangles:
        center_x, center_y, _old_width, height, _old_angle = rect_to_five(rectangle)
        sampled_sine = _sample_bilinear(sine, center_x, center_y)
        sampled_cosine = _sample_bilinear(cosine, center_x, center_y)
        sampled_width = _sample_bilinear(width, center_x, center_y)
        decoded_short = (
            height
            if short_side is None
            else _sample_bilinear(short_side, center_x, center_y)
            * width_factor
            * float(size_scale)
        )
        angle_degrees = float(
            0.5 * np.arctan2(sampled_sine, sampled_cosine) / np.pi * 180.0
        )
        refined.append(
            np.array(
                [
                    center_x,
                    center_y,
                    max(1.0, sampled_width * float(width_factor) * float(size_scale)),
                    max(1.0, decoded_short),
                    angle_degrees,
                ],
                dtype=np.float32,
            )
        )
    return refined
