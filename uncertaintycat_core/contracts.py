"""Serializable public contracts for models, analyses, runs, and results."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

JsonScalar = str | int | float | bool | None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AccuracyProfile(str, Enum):
    preview = "preview"
    standard = "standard"
    high = "high"


class ResultStatus(str, Enum):
    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


class VariableMetadata(StrictModel):
    index: int = Field(ge=0)
    name: str
    distribution: str | None = None
    parameters: list[float] = Field(default_factory=list)
    kind: Literal["continuous", "discrete", "mixed", "unknown"] = "unknown"
    mean: float | None = None
    standard_deviation: float | None = Field(default=None, ge=0)


class OutputMetadata(StrictModel):
    index: int = Field(ge=0)
    name: str


class ModelEquation(StrictModel):
    output_name: str = Field(min_length=1, max_length=120)
    latex: str = Field(min_length=1, max_length=4_000)
    representation: Literal["closed_form", "declared", "formal_mapping"]


class ModelMetadata(StrictModel):
    source_hash: str
    input_dimension: int = Field(gt=0)
    output_dimension: int = Field(gt=0)
    inputs: list[VariableMetadata]
    outputs: list[OutputMetadata]
    equations: list[ModelEquation] = Field(default_factory=list, max_length=6)
    openturns_version: str
    batch_evaluation_supported: bool
    validation_sample_size: int = Field(ge=1)
    validation_runtime_ms: float = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)
    function_type: str = "OpenTURNS Function"
    exact_gradient_available: bool = False
    exact_hessian_available: bool = False
    copula: str = "Unknown"
    dependent_inputs: bool = False


class PilotOutputSummary(StrictModel):
    output_index: int = Field(ge=0)
    output_name: str
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float = Field(ge=0)
    quantile_05: float
    quantile_95: float
    variable: bool


class ModelProfile(StrictModel):
    input_dimension: int = Field(gt=0)
    output_dimension: int = Field(gt=0)
    continuous_marginals: int = Field(ge=0)
    discrete_marginals: int = Field(ge=0)
    copula: str
    dependent_inputs: bool
    function_type: str
    batch_support: bool
    validation_evaluation_runtime_ms: float = Field(ge=0)
    projected_1000_evaluation_runtime_ms: float = Field(ge=0)
    pilot_sample_size: int = Field(gt=0)
    pilot_outputs: list[PilotOutputSummary]


class AnalysisRecommendation(StrictModel):
    capability: str
    status: Literal["recommended", "available", "incompatible"]
    priority: int = Field(ge=1)
    rationale_codes: list[str]
    projected_evaluations: int | None = Field(default=None, ge=0)
    projected_runtime_ms: float | None = Field(default=None, ge=0)
    compatibility_warnings: list[str] = Field(default_factory=list)
    safe_config: dict[str, JsonScalar] = Field(default_factory=dict)


class WorkflowRecommendation(StrictModel):
    path: Literal["direct", "dimensionality_reduction", "surrogate"]
    rationale_codes: list[str]


class ModelAssessment(StrictModel):
    version: str = "1.4.0"
    profile: ModelProfile
    workflow: WorkflowRecommendation
    recommendations: list[AnalysisRecommendation]


class TableData(StrictModel):
    columns: list[str]
    rows: list[list[JsonScalar]]
    row_count: int = Field(ge=0)
    truncated: bool = False

    @field_validator("rows")
    @classmethod
    def rows_match_columns(cls, rows: list[list[JsonScalar]], info: Any) -> list[list[JsonScalar]]:
        columns = info.data.get("columns", [])
        if columns and any(len(row) != len(columns) for row in rows):
            raise ValueError("Every row must contain one value per column")
        return rows


class SeriesData(StrictModel):
    name: str
    x: list[JsonScalar]
    y: list[JsonScalar]
    x_label: str | None = None
    y_label: str | None = None

    @field_validator("y")
    @classmethod
    def equal_axis_lengths(cls, values: list[JsonScalar], info: Any) -> list[JsonScalar]:
        x_values = info.data.get("x", [])
        if len(values) != len(x_values):
            raise ValueError("Series x and y arrays must be the same length")
        return values


class MatrixData(StrictModel):
    row_labels: list[str]
    column_labels: list[str]
    values: list[list[float | None]]


class ArtifactReference(StrictModel):
    id: str
    media_type: str
    filename: str
    size_bytes: int = Field(ge=0)
    sha256: str


class AnalysisPayload(StrictModel):
    metrics: dict[str, JsonScalar] = Field(default_factory=dict)
    tables: dict[str, TableData] = Field(default_factory=dict)
    series: dict[str, SeriesData] = Field(default_factory=dict)
    matrices: dict[str, MatrixData] = Field(default_factory=dict)
    facts: dict[str, JsonScalar] = Field(default_factory=dict)
    artifacts: list[ArtifactReference] = Field(default_factory=list)


class AnalysisRequest(StrictModel):
    analysis_key: str
    plugin_version: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    output_targets: list[int] = Field(default_factory=list)


class RunRequest(StrictModel):
    model_version_id: UUID | None = None
    model_source: str | None = None
    analyses: list[AnalysisRequest]
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    accuracy_profile: AccuracyProfile = AccuracyProfile.standard
    idempotency_key: str | None = Field(default=None, max_length=128)


class RuntimeMetrics(StrictModel):
    duration_ms: float = Field(ge=0)
    model_evaluations: int = Field(default=0, ge=0)
    sample_size: int | None = Field(default=None, ge=0)


class AnalysisResultEnvelope(StrictModel):
    run_id: UUID = Field(default_factory=uuid4)
    task_id: UUID = Field(default_factory=uuid4)
    analysis_key: str
    plugin_version: str
    result_schema_version: str
    model_hash: str
    seed: int
    uq_core_version: str
    openturns_version: str
    status: Literal["succeeded"] = "succeeded"
    started_at: datetime
    completed_at: datetime = Field(default_factory=utc_now)
    runtime: RuntimeMetrics
    warnings: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    payload: AnalysisPayload


class AnalysisCatalogEntry(StrictModel):
    key: str
    version: str
    result_schema_version: str
    name: str
    category: str
    description: str
    assumptions: list[str]
    supports_dependent_inputs: bool
    requires_dependent_inputs: bool
    supports_multi_output: bool
    resource_class: Literal["lite", "standard", "heavy"]
    config_schema: dict[str, Any]
