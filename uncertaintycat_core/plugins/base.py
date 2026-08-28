"""Analysis plugin protocol and shared helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Literal, TypeAlias, TypeVar

from pydantic import BaseModel

from uncertaintycat_core.contracts import AnalysisCatalogEntry, AnalysisPayload
from uncertaintycat_core.model import ModelRuntime

ConfigT = TypeVar("ConfigT", bound=BaseModel)
ApplicabilityIssue: TypeAlias = tuple[str, str]


class AnalysisPlugin(ABC, Generic[ConfigT]):
    key: str
    version: str
    result_schema_version = "1.0.0"
    name: str
    category: str
    description: str
    assumptions: tuple[str, ...] = ()
    supports_dependent_inputs = True
    requires_dependent_inputs = False
    supports_multi_output = True
    resource_class: Literal["lite", "standard", "heavy"] = "standard"
    config_model: type[ConfigT]

    def parse_config(self, raw: dict[str, Any], *, seed: int, output_targets: list[int]) -> ConfigT:
        values = dict(raw)
        values.setdefault("seed", seed)
        if output_targets:
            values.setdefault("output_targets", output_targets)
        return self.config_model.model_validate(values)

    def catalog_entry(self) -> AnalysisCatalogEntry:
        return AnalysisCatalogEntry(
            key=self.key,
            version=self.version,
            result_schema_version=self.result_schema_version,
            name=self.name,
            category=self.category,
            description=self.description,
            assumptions=list(self.assumptions),
            supports_dependent_inputs=self.supports_dependent_inputs,
            requires_dependent_inputs=self.requires_dependent_inputs,
            supports_multi_output=self.supports_multi_output,
            resource_class=self.resource_class,
            config_schema=self.config_model.model_json_schema(),
        )

    def applicability_warnings(self, runtime: ModelRuntime, config: ConfigT) -> list[str]:
        return []

    def model_incompatibility_reasons(self, runtime: ModelRuntime) -> list[ApplicabilityIssue]:
        """Return model-level blockers that do not depend on a user's run configuration."""

        issues: list[ApplicabilityIssue] = []
        if self.requires_dependent_inputs and not runtime.metadata.dependent_inputs:
            issues.append(
                (
                    "DEPENDENT_INPUTS_REQUIRED",
                    f"{self.name} requires a dependent input copula; this model "
                    "declares independent inputs.",
                )
            )
        if not self.supports_dependent_inputs and runtime.metadata.dependent_inputs:
            issues.append(
                (
                    "INDEPENDENT_INPUTS_REQUIRED",
                    f"{self.name} requires independent inputs; this model declares "
                    "a dependent copula.",
                )
            )
        return issues

    def safe_model_config(
        self, runtime: ModelRuntime
    ) -> dict[str, str | int | float | bool | None]:
        """Expose model-dependent UI limits while keeping core execution authoritative."""

        return {}

    @abstractmethod
    def run(self, runtime: ModelRuntime, config: ConfigT) -> tuple[AnalysisPayload, int]:
        """Return a serializable payload and the number of model evaluations."""
