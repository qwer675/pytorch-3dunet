"""Custom exceptions for the ODT dataset builder."""

from __future__ import annotations


class DatasetBuilderError(Exception):
    """Base class for dataset builder errors."""

    def __init__(self, message: str, *, case_id: str | None = None) -> None:
        prefix = f"[Case: {case_id}] " if case_id else ""
        super().__init__(f"{prefix}{message}")
        self.case_id = case_id


class DataValidationError(DatasetBuilderError):
    """Raised when validation of raw/label pairing fails."""


class ConfigurationError(DatasetBuilderError):
    """Raised when configuration file is invalid."""
