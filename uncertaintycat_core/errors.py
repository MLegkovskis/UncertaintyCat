"""Stable error taxonomy shared by the compute service and API clients."""

from __future__ import annotations

from typing import Any


class UncertaintyCatError(Exception):
    """Base class for errors safe to translate into a public API response."""

    code = "uncertaintycat_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class InvalidModelError(UncertaintyCatError):
    code = "invalid_model"


class UnsafeModelError(UncertaintyCatError):
    code = "unsafe_model"


class UnknownAnalysisError(UncertaintyCatError):
    code = "unknown_analysis"


class IncompatibleAnalysisError(UncertaintyCatError):
    code = "incompatible_analysis"


class AnalysisExecutionError(UncertaintyCatError):
    code = "analysis_execution_failed"
