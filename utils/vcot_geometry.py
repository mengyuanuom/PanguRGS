"""Pure metadata and geometry helpers for VCoT/Grasp-Anything."""

from pathlib import Path

import numpy as np


_SPLIT_FILES = {
    "train": "train.csv",
    "seen": "test_seen.csv",
    "test_seen": "test_seen.csv",
    "unseen": "test_unseen.csv",
    "test_unseen": "test_unseen.csv",
}


def resolve_vcot_split(split):
    key = str(split).strip().lower().replace("-", "_")
    if key not in _SPLIT_FILES:
        choices = ", ".join(sorted(_SPLIT_FILES))
        raise ValueError(f"Unknown VCoT split {split!r}; choose one of: {choices}")
    return _SPLIT_FILES[key]


def resolve_vcot_grasp_root(root_dir):
    root_dir = Path(root_dir).expanduser()
    candidates = (
        root_dir / "grasp_label_positive",
        root_dir / "positive_grasp",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


def grasp_anything_to_quads(grasps):
    """Convert [score, x, y, length, width, theta_deg] rows to XY quads."""
    values = np.asarray(grasps, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] < 6:
        raise ValueError(
            "VCoT grasp tensor must have shape [N, 6+] with "
            "[score, x, y, w, h, theta_deg]"
        )
    values = values[:, :6]
    valid = np.isfinite(values).all(axis=1) & (values[:, 3] > 0) & (values[:, 4] > 0)
    values = values[valid]
    if not len(values):
        return np.zeros((0, 4, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    score, x, y, length, width, theta = values.T
    angle = np.deg2rad(theta)
    xo = np.cos(angle)
    yo = np.sin(angle)
    y1 = y + length / 2.0 * yo
    x1 = x - length / 2.0 * xo
    y2 = y - length / 2.0 * yo
    x2 = x + length / 2.0 * xo
    row_col = np.stack(
        [
            np.stack([y1 - width / 2.0 * xo, x1 - width / 2.0 * yo], axis=1),
            np.stack([y2 - width / 2.0 * xo, x2 - width / 2.0 * yo], axis=1),
            np.stack([y2 + width / 2.0 * xo, x2 + width / 2.0 * yo], axis=1),
            np.stack([y1 + width / 2.0 * xo, x1 + width / 2.0 * yo], axis=1),
        ],
        axis=1,
    )
    toolrgs_order = row_col[:, [0, 3, 2, 1], ::-1]
    return toolrgs_order.astype(np.float32), score.astype(np.float32)