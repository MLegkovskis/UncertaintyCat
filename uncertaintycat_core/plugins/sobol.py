"""OpenTURNS Saltelli Sobol analysis with explicit applicability checks."""

from __future__ import annotations

import openturns as ot
from pydantic import Field

from uncertaintycat_core.contracts import AnalysisPayload, MatrixData, StrictModel, TableData
from uncertaintycat_core.errors import IncompatibleAnalysisError
from uncertaintycat_core.model import ModelRuntime
from uncertaintycat_core.plugins.base import AnalysisPlugin


class SobolConfig(StrictModel):
    base_sample_size: int = Field(default=1024, ge=64, le=131_072)
    seed: int = Field(default=42, ge=0)
    output_targets: list[int] = Field(default_factory=list, max_length=1)
    second_order: bool | None = None


class SobolPlugin(AnalysisPlugin[SobolConfig]):
    key = "sobol"
    version = "2.0.0"
    name = "Sobol Sensitivity Analysis"
    category = "Sensitivity"
    description = "Estimate first-, total-, and optional second-order variance contributions."
    assumptions = (
        "Input variables must be independent for the classical Sobol interpretation.",
        "The selected output must have non-zero variance.",
    )
    supports_dependent_inputs = False
    supports_multi_output = False
    resource_class = "heavy"
    config_model = SobolConfig

    @staticmethod
    def _is_independent(problem: ot.Distribution) -> bool:
        try:
            return bool(problem.hasIndependentCopula())
        except Exception:
            try:
                return isinstance(problem.getCopula(), ot.IndependentCopula)
            except Exception:
                return problem.getDimension() == 1

    def applicability_warnings(self, runtime: ModelRuntime, config: SobolConfig) -> list[str]:
        if not self._is_independent(runtime.problem):
            raise IncompatibleAnalysisError(
                "Classical Sobol analysis requires independent inputs; "
                "the model distribution has a dependent copula."
            )
        return []

    def run(self, runtime: ModelRuntime, config: SobolConfig) -> tuple[AnalysisPayload, int]:
        self.applicability_warnings(runtime, config)
        target = config.output_targets[0] if config.output_targets else 0
        if target >= runtime.metadata.output_dimension:
            raise IncompatibleAnalysisError("The requested output target does not exist.")
        dimension = runtime.metadata.input_dimension
        compute_second_order = (
            config.second_order if config.second_order is not None else dimension <= 10
        )
        ot.RandomGenerator.SetSeed(config.seed)
        experiment = ot.SobolIndicesExperiment(
            runtime.problem, config.base_sample_size, compute_second_order
        )
        input_design = experiment.generate()
        complete_output = runtime.model(input_design)
        output_design = complete_output.getMarginal(target)
        if float(output_design.computeVariance()[0]) <= ot.SpecFunc.ScalarEpsilon:
            raise IncompatibleAnalysisError("Sobol analysis is undefined for a constant output.")
        algorithm = ot.SaltelliSensitivityAlgorithm(
            input_design, output_design, config.base_sample_size
        )
        first = algorithm.getFirstOrderIndices()
        total = algorithm.getTotalOrderIndices()
        first_interval = algorithm.getFirstOrderIndicesInterval()
        total_interval = algorithm.getTotalOrderIndicesInterval()
        names = [item.name for item in runtime.metadata.inputs]
        rows: list[list[float | str]] = []
        for index, name in enumerate(names):
            rows.append(
                [
                    name,
                    float(first[index]),
                    float(first_interval.getLowerBound()[index]),
                    float(first_interval.getUpperBound()[index]),
                    float(total[index]),
                    float(total_interval.getLowerBound()[index]),
                    float(total_interval.getUpperBound()[index]),
                    float(total[index] - first[index]),
                ]
            )
        matrices: dict[str, MatrixData] = {}
        if compute_second_order:
            second = algorithm.getSecondOrderIndices()
            matrix = [
                [float(second[i, j]) if i != j else 0.0 for j in range(dimension)]
                for i in range(dimension)
            ]
            matrices["second_order"] = MatrixData(
                row_labels=names, column_labels=names, values=matrix
            )
        top_index = max(range(dimension), key=lambda index: float(total[index]))
        evaluations = input_design.getSize()
        return (
            AnalysisPayload(
                metrics={
                    "base_sample_size": config.base_sample_size,
                    "model_evaluations": evaluations,
                    "sum_first_order": float(sum(first)),
                    "sum_total_order": float(sum(total)),
                },
                tables={
                    "indices": TableData(
                        columns=[
                            "Variable",
                            "First Order",
                            "First Lower",
                            "First Upper",
                            "Total Order",
                            "Total Lower",
                            "Total Upper",
                            "Interaction",
                        ],
                        rows=rows,
                        row_count=len(rows),
                    )
                },
                matrices=matrices,
                facts={
                    "output": runtime.metadata.outputs[target].name,
                    "most_influential_input": names[top_index],
                    "largest_total_order_index": float(total[top_index]),
                },
            ),
            evaluations,
        )


plugin = SobolPlugin()
