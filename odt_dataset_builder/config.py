"""Configuration models for the ODT dataset builder."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import yaml

from .exceptions import ConfigurationError


@dataclass
class PreviewConfig:
    """Configuration for preview PNG generation."""

    enabled: bool = True
    num_slices: int = 3
    output_dir: Path = Path("previews")


@dataclass
class CaseConfig:
    """Configuration for a single case (raw/label pair)."""

    case_id: str
    raw_path: Path
    label_path: Path
    raw_spacing: Optional[Sequence[float]] = None
    raw_direction: Optional[Sequence[Sequence[float]]] = None
    label_spacing: Optional[Sequence[float]] = None
    label_direction: Optional[Sequence[Sequence[float]]] = None

    def validate(self) -> None:
        if not self.case_id:
            raise ConfigurationError("case_id must be provided", case_id=self.case_id)
        if not self.raw_path.exists():
            raise ConfigurationError(
                f"Raw TIFF not found at {self.raw_path}", case_id=self.case_id
            )
        if not self.label_path.exists():
            raise ConfigurationError(
                f"Label NIfTI not found at {self.label_path}", case_id=self.case_id
            )


@dataclass
class BuilderConfig:
    """Top-level configuration for the dataset builder."""

    output_path: Path
    cases: List[CaseConfig]
    chunk_size: Sequence[int] = (1, 64, 128, 128)
    compression: str = "gzip"
    compression_opts: Optional[int] = 4
    preview: PreviewConfig = field(default_factory=PreviewConfig)
    report_path: Optional[Path] = Path("report.json")

    def validate(self) -> None:
        if not self.cases:
            raise ConfigurationError("At least one case must be provided")
        for case in self.cases:
            case.validate()


def _load_raw_config(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            if path.suffix.lower() in {".yaml", ".yml"}:
                return yaml.safe_load(f)
            if path.suffix.lower() == ".json":
                import json

                return json.load(f)
            raise ConfigurationError(
                f"Unsupported config extension: {path.suffix}. Use .yaml/.yml/.json"
            )
    except OSError as exc:
        raise ConfigurationError(f"Failed to read config: {exc}") from exc


def load_config(path: Path) -> BuilderConfig:
    """Load the builder configuration from YAML or JSON."""

    raw_config = _load_raw_config(path)
    try:
        cases_cfg = []
        for entry in raw_config.get("cases", []):
            cases_cfg.append(
                CaseConfig(
                    case_id=str(entry["id"]),
                    raw_path=Path(entry["raw"]).expanduser().resolve(),
                    label_path=Path(entry["label"]).expanduser().resolve(),
                    raw_spacing=entry.get("raw_spacing"),
                    raw_direction=entry.get("raw_direction"),
                    label_spacing=entry.get("label_spacing"),
                    label_direction=entry.get("label_direction"),
                )
            )
        preview_cfg = raw_config.get("preview", {})
        preview = PreviewConfig(
            enabled=bool(preview_cfg.get("enabled", True)),
            num_slices=int(preview_cfg.get("num_slices", 3)),
            output_dir=Path(preview_cfg.get("output_dir", "previews")).expanduser(),
        )
        config = BuilderConfig(
            output_path=Path(raw_config["output_path"]).expanduser(),
            cases=cases_cfg,
            chunk_size=tuple(int(v) for v in raw_config.get("chunk_size", (1, 64, 128, 128))),
            compression=raw_config.get("compression", "gzip"),
            compression_opts=raw_config.get("compression_opts", 4),
            preview=preview,
            report_path=(
                Path(raw_config["report_path"]).expanduser()
                if raw_config.get("report_path")
                else None
            ),
        )
    except KeyError as exc:
        raise ConfigurationError(f"Missing required config key: {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid configuration value: {exc}") from exc

    config.validate()
    return config


def iter_cases(config: BuilderConfig) -> Iterable[CaseConfig]:
    """Yield case configurations."""

    yield from config.cases
