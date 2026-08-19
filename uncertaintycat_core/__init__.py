"""Versioned, UI-independent numerical engine for UncertaintyCat."""

from uncertaintycat_core.catalog import analysis_catalog, get_plugin
from uncertaintycat_core.contracts import (
    AnalysisRequest,
    AnalysisResultEnvelope,
    ModelMetadata,
    RunRequest,
)
from uncertaintycat_core.model import compile_model, validate_model_source
from uncertaintycat_core.runner import run_analysis, run_suite

__all__ = [
    "AnalysisRequest",
    "AnalysisResultEnvelope",
    "ModelMetadata",
    "RunRequest",
    "analysis_catalog",
    "compile_model",
    "get_plugin",
    "run_analysis",
    "run_suite",
    "validate_model_source",
]

__version__ = "0.2.0"
