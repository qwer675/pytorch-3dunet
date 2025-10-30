"""Input/output helpers for TIFF and NIfTI data."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from tifffile import TiffFile
from tifffile.ome import OME

from .exceptions import DataValidationError

LOGGER = logging.getLogger(__name__)


@dataclass
class ImageMeta:
    """Metadata describing a volumetric image."""

    spacing: Tuple[float, float, float]
    direction: Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]
    origin: Tuple[float, float, float]


@dataclass
class LoadedVolume:
    """Container for image data and metadata."""

    data: np.ndarray
    meta: ImageMeta


EPS = 1e-5


def _normalize_direction(matrix: np.ndarray) -> np.ndarray:
    """Return a unit-length direction matrix."""

    normed = np.array(matrix, dtype=np.float64)
    for i in range(3):
        vec = normed[:, i]
        length = np.linalg.norm(vec)
        if length > 0:
            normed[:, i] = vec / length
    return normed


def _extract_spacing_direction_from_ome(ome: OME) -> tuple[Optional[Tuple[float, float, float]], Optional[np.ndarray]]:
    pixels = ome.images[0].pixels
    spacing = None
    direction = None
    if pixels.physical_size_z and pixels.physical_size_y and pixels.physical_size_x:
        spacing = (
            float(pixels.physical_size_z),
            float(pixels.physical_size_y),
            float(pixels.physical_size_x),
        )
    if pixels.rotation is not None:
        try:
            rotation_matrix = np.array(json.loads(pixels.rotation))
            if rotation_matrix.shape == (3, 3):
                direction = rotation_matrix
        except (json.JSONDecodeError, TypeError, ValueError):
            LOGGER.debug("Failed to parse rotation matrix from OME metadata")
    return spacing, direction


def _infer_spacing_direction_from_tiff(tiff: TiffFile) -> tuple[Optional[Tuple[float, float, float]], Optional[np.ndarray]]:
    spacing = None
    direction = None
    if tiff.ome_metadata:
        ome = OME.from_xml(tiff.ome_metadata)
        spacing, direction = _extract_spacing_direction_from_ome(ome)
    if spacing is None:
        try:
            page0 = tiff.pages[0]
            xres = page0.tags.get("XResolution")
            yres = page0.tags.get("YResolution")
            if xres and yres:
                spacing_xy = (xres.value[1] / xres.value[0], yres.value[1] / yres.value[0])
                spacing = (1.0, float(spacing_xy[1]), float(spacing_xy[0]))
        except Exception:  # noqa: BLE001 - fallback best effort
            LOGGER.debug("Could not infer XY spacing from TIFF tags", exc_info=True)
    if direction is None:
        direction = np.eye(3)
    return spacing, direction


def load_raw_tiff(path: Path, *, provided_spacing: Optional[Sequence[float]] = None, provided_direction: Optional[Sequence[Sequence[float]]] = None) -> LoadedVolume:
    """Load a multi-page TIFF stack and return the volume with metadata."""

    with TiffFile(str(path)) as tif:
        data = tif.asarray().astype(np.float32)
        spacing, direction = _infer_spacing_direction_from_tiff(tif)

    if provided_spacing is not None:
        spacing = tuple(float(v) for v in provided_spacing)
    if spacing is None:
        raise DataValidationError("Voxel spacing missing for raw volume", case_id=str(path))

    if provided_direction is not None:
        direction_matrix = np.array(provided_direction, dtype=np.float64)
    else:
        direction_matrix = np.array(direction, dtype=np.float64)

    if direction_matrix.shape != (3, 3):
        raise DataValidationError(
            f"Invalid direction matrix shape for raw volume: {direction_matrix.shape}",
            case_id=str(path),
        )

    direction_matrix = _normalize_direction(direction_matrix)

    meta = ImageMeta(
        spacing=(float(spacing[0]), float(spacing[1]), float(spacing[2])),
        direction=tuple(tuple(float(v) for v in row) for row in direction_matrix),
        origin=(0.0, 0.0, 0.0),
    )
    return LoadedVolume(data=data, meta=meta)


def load_label_nifti(path: Path, *, provided_spacing: Optional[Sequence[float]] = None, provided_direction: Optional[Sequence[Sequence[float]]] = None) -> LoadedVolume:
    """Load a NIfTI file and return the segmentation volume."""

    img = nib.load(str(path))
    canonical_img = nib.as_closest_canonical(img)
    data = np.asanyarray(canonical_img.dataobj).astype(np.uint16)

    affine = canonical_img.affine
    spacing = provided_spacing or tuple(float(v) for v in nib.affines.voxel_sizes(affine))

    direction_matrix = np.array(provided_direction, dtype=np.float64) if provided_direction is not None else affine[:3, :3] / spacing
    direction_matrix = _normalize_direction(direction_matrix)

    origin = tuple(float(v) for v in canonical_img.affine[:3, 3])

    meta = ImageMeta(
        spacing=(float(spacing[0]), float(spacing[1]), float(spacing[2])),
        direction=tuple(tuple(float(v) for v in row) for row in direction_matrix),
        origin=origin,
    )
    return LoadedVolume(data=data, meta=meta)


def validate_compatibility(raw: LoadedVolume, label: LoadedVolume, *, case_id: str) -> None:
    """Ensure the raw and label volumes are compatible."""

    if raw.data.ndim != 3:
        raise DataValidationError("Raw volume must be 3D", case_id=case_id)
    if label.data.ndim != 3:
        raise DataValidationError("Label volume must be 3D", case_id=case_id)

    if raw.data.shape != label.data.shape:
        raise DataValidationError(
            f"Shape mismatch: raw {raw.data.shape} vs label {label.data.shape}",
            case_id=case_id,
        )

    if not np.allclose(raw.meta.spacing, label.meta.spacing, rtol=1e-3, atol=1e-3):
        raise DataValidationError(
            f"Spacing mismatch: raw {raw.meta.spacing} vs label {label.meta.spacing}",
            case_id=case_id,
        )

    raw_direction = np.array(raw.meta.direction)
    label_direction = np.array(label.meta.direction)
    if not np.allclose(raw_direction, label_direction, atol=1e-4):
        raise DataValidationError(
            "Direction matrix mismatch between raw and label", case_id=case_id
        )


def compute_label_histogram(data: np.ndarray) -> dict[str, int]:
    """Compute a histogram of label values."""

    values, counts = np.unique(data, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(values, counts)}


def format_direction(direction: Sequence[Sequence[float]]) -> str:
    matrix = np.array(direction)
    return np.array2string(matrix, precision=5, separator=", ")
