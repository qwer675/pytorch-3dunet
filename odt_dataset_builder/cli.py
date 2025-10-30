"""Command-line interface for the ODT dataset builder."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from .builder import build_dataset, emit_report
from .config import load_config
from .exceptions import DatasetBuilderError


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ODT TIFF stacks and Slicer labels to an HDF5 dataset compatible with pytorch-3dunet.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML/JSON configuration file defining cases and output",
    )
    parser.add_argument(
        "--verbosity",
        "-v",
        action="count",
        default=0,
        help="Increase log verbosity (use -v or -vv)",
    )
    return parser.parse_args(args)


def main(argv: list[str] | None = None) -> int:
    parsed = parse_args(argv)
    _configure_logging(parsed.verbosity)
    try:
        config = load_config(parsed.config)
        artifacts = build_dataset(config)
        emit_report(artifacts)
    except DatasetBuilderError as exc:
        logging.getLogger(__name__).error("%s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).exception("Unexpected failure: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
