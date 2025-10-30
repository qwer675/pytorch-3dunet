"""Preview image utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)


def _normalize_slice(slice_: np.ndarray) -> np.ndarray:
    slice_float = slice_.astype(np.float32)
    min_val = float(slice_float.min())
    max_val = float(slice_float.max())
    if max_val - min_val < 1e-6:
        return np.zeros_like(slice_float, dtype=np.uint8)
    norm = (slice_float - min_val) / (max_val - min_val)
    return (norm * 255).astype(np.uint8)


def _label_to_color(label_slice: np.ndarray) -> np.ndarray:
    """Map label values to RGB colors."""

    label = label_slice.astype(np.uint32)
    unique_labels = np.unique(label)
    color_map = {0: np.array([0, 0, 0], dtype=np.uint8)}
    rng = np.random.default_rng(42)
    for value in unique_labels:
        if value == 0:
            continue
        if value not in color_map:
            color = rng.integers(0, 255, size=3, dtype=np.uint8)
            color_map[int(value)] = color
    colored = np.zeros((*label.shape, 3), dtype=np.uint8)
    for value, color in color_map.items():
        mask = label == value
        colored[mask] = color
    return colored


def _blend(raw_slice: np.ndarray, label_slice: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    raw_img = _normalize_slice(raw_slice)
    raw_rgb = np.stack([raw_img] * 3, axis=-1)
    label_rgb = _label_to_color(label_slice)
    mask = label_slice > 0
    blended = raw_rgb.copy()
    blended[mask] = (
        raw_rgb[mask].astype(np.float32) * (1.0 - alpha)
        + label_rgb[mask].astype(np.float32) * alpha
    ).astype(np.uint8)
    return blended


def save_previews(
    raw_volume: np.ndarray,
    label_volume: np.ndarray,
    case_id: str,
    output_dir: Path,
    num_slices: int,
) -> List[Path]:
    """Save preview PNGs for selected slices."""

    output_dir.mkdir(parents=True, exist_ok=True)
    depth = raw_volume.shape[0]
    if depth == 0:
        LOGGER.warning("Volume has zero depth; skipping previews for case %s", case_id)
        return []
    slice_indices = np.linspace(0, depth - 1, num=num_slices, dtype=int)
    saved_paths: List[Path] = []
    for idx in slice_indices:
        preview = _blend(raw_volume[idx], label_volume[idx])
        filename = output_dir / f"{case_id}_z{idx:04d}.png"
        Image.fromarray(preview).save(filename)
        saved_paths.append(filename)
    return saved_paths
