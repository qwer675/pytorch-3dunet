"""Core dataset building workflow."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np

from .config import BuilderConfig, CaseConfig
from .exceptions import DataValidationError, DatasetBuilderError
from .io_utils import (
    LoadedVolume,
    compute_label_histogram,
    format_direction,
    load_label_nifti,
    load_raw_tiff,
    validate_compatibility,
)
from .preview import save_previews

LOGGER = logging.getLogger(__name__)


@dataclass
class CaseResult:
    """Result metadata for a processed case."""

    case_id: str
    shape: tuple[int, int, int, int]
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    histogram: Dict[str, int]
    preview_paths: List[str]
    checks: Dict[str, bool]


@dataclass
class CaseData:
    """Containers to hold processed case arrays and metadata."""

    raw: np.ndarray
    label: np.ndarray
    raw_volume: LoadedVolume
    label_volume: LoadedVolume


@dataclass
class BuildArtifacts:
    """Aggregated results for the build."""

    cases: List[CaseResult]
    global_histogram: Dict[str, int]
    output_path: Path
    report_path: Optional[Path]


def _merge_histograms(histograms: List[Dict[str, int]]) -> Dict[str, int]:
    merged: Dict[str, int] = {}
    for hist in histograms:
        for key, value in hist.items():
            merged[key] = merged.get(key, 0) + value
    return merged


def _prepare_case(case: CaseConfig) -> CaseData:
    LOGGER.info("Loading case %s", case.case_id)
    raw_volume = load_raw_tiff(
        case.raw_path,
        provided_spacing=case.raw_spacing,
        provided_direction=case.raw_direction,
    )
    label_volume = load_label_nifti(
        case.label_path,
        provided_spacing=case.label_spacing,
        provided_direction=case.label_direction,
    )
    validate_compatibility(raw_volume, label_volume, case_id=case.case_id)

    raw = raw_volume.data[np.newaxis, ...]
    label = label_volume.data[np.newaxis, ...]
    return CaseData(raw=raw, label=label, raw_volume=raw_volume, label_volume=label_volume)


def _write_case_to_hdf5(
    file: h5py.File,
    case: CaseConfig,
    data: CaseData,
    config: BuilderConfig,
) -> CaseResult:
    group = file.create_group(case.case_id)

    group.create_dataset(
        "raw",
        data=data.raw,
        dtype=np.float32,
        chunks=tuple(int(v) for v in config.chunk_size),
        compression=config.compression,
        compression_opts=config.compression_opts,
    )
    group.create_dataset(
        "label",
        data=data.label,
        dtype=np.uint16,
        chunks=tuple(int(v) for v in config.chunk_size),
        compression=config.compression,
        compression_opts=config.compression_opts,
    )

    spacing = data.raw_volume.meta.spacing
    origin = data.label_volume.meta.origin
    direction = data.raw_volume.meta.direction

    group.attrs["spacing"] = spacing
    group.attrs["origin"] = origin
    group.attrs["direction"] = np.array(direction).flatten()

    histogram = compute_label_histogram(data.label_volume.data)
    preview_paths: List[str] = []
    if config.preview.enabled:
        preview_paths = [str(p) for p in save_previews(
            data.raw_volume.data,
            data.label_volume.data,
            case.case_id,
            config.preview.output_dir,
            config.preview.num_slices,
        )]

    result = CaseResult(
        case_id=case.case_id,
        shape=data.raw.shape,
        spacing=spacing,
        origin=origin,
        direction=direction,
        histogram=histogram,
        preview_paths=preview_paths,
        checks={
            "shape_match": True,
            "spacing_match": True,
            "direction_match": True,
        },
    )
    return result


def build_dataset(config: BuilderConfig) -> BuildArtifacts:
    LOGGER.info("Writing dataset to %s", config.output_path)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)

    case_results: List[CaseResult] = []
    histograms: List[Dict[str, int]] = []

    with h5py.File(config.output_path, "w") as h5_file:
        for case in config.cases:
            try:
                data = _prepare_case(case)
                result = _write_case_to_hdf5(h5_file, case, data, config)
            except DatasetBuilderError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise DatasetBuilderError(str(exc), case_id=case.case_id) from exc
            case_results.append(result)
            histograms.append(result.histogram)

    global_histogram = _merge_histograms(histograms)
    return BuildArtifacts(
        cases=case_results,
        global_histogram=global_histogram,
        output_path=config.output_path,
        report_path=config.report_path,
    )


def emit_report(artifacts: BuildArtifacts) -> dict:
    report = {
        "output_path": str(artifacts.output_path),
        "global_histogram": artifacts.global_histogram,
        "cases": [
            {
                "case_id": case.case_id,
                "shape": list(case.shape),
                "spacing": list(case.spacing),
                "origin": list(case.origin),
                "direction": [list(row) for row in case.direction],
                "checks": case.checks,
                "histogram": case.histogram,
                "preview_paths": case.preview_paths,
            }
            for case in artifacts.cases
        ],
    }

    LOGGER.info("\n=== Dataset Build Report ===")
    LOGGER.info("Output file: %s", artifacts.output_path)
    LOGGER.info("Global label histogram: %s", artifacts.global_histogram)
    for case in artifacts.cases:
        LOGGER.info("--- Case %s ---", case.case_id)
        LOGGER.info("Shape: %s", case.shape)
        LOGGER.info("Spacing: %s", case.spacing)
        LOGGER.info("Origin: %s", case.origin)
        LOGGER.info("Direction: %s", format_direction(case.direction))
        LOGGER.info("Checks: %s", case.checks)
        LOGGER.info("Histogram: %s", case.histogram)
        if case.preview_paths:
            LOGGER.info("Preview PNGs: %s", ", ".join(case.preview_paths))

    if artifacts.report_path:
        artifacts.report_path.parent.mkdir(parents=True, exist_ok=True)
        with artifacts.report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        LOGGER.info("Report written to %s", artifacts.report_path)
    return report
