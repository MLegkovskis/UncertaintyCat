"""OpenTURNS Taylor-expansion moments and local importance factors."""

from __future__ import annotations

import openturns as ot
from pydantic import Field

from uncertaintycat_core.contracts import AnalysisPayload, StrictModel, TableData
from uncertaintycat_core.errors import IncompatibleAnalysisError
from uncertaintycat_core.model import ModelRuntime
from uncertaintycat_core.plugins.base import AnalysisPlugin


class TaylorConfig(StrictModel):
    # Retained for old saved configurations; OpenTURNS controls differentiation.
    relative_step: float = Field(default=1e-5, gt=0, le=0.1)
    validation_size: int = Field(default=500, ge=20, le=20_000)
    seed: int = Field(default=42, ge=0)
    output_targets: list[int] = Field(default_factory=list, max_length=1)


class TaylorPlugin(AnalysisPlugin[TaylorConfig]):
    key = "taylor"
    version = "2.0.0"
    name = "Taylor Expansion Moments"
    category = "Sensitivity"
    description = "Approximate moments and local importance with TaylorExpansionMoments."
    assumptions = (
        "The response is differentiable near the input mean.",
        "Taylor importance is local and may not represent a strongly nonlinear global response.",
    )
    supports_multi_output = False
    config_model = TaylorConfig

    def run(self, runtime: ModelRuntime, config: TaylorConfig) -> tuple[AnalysisPayload, int]:
        target = config.output_targets[0] if config.output_targets else 0
        if target >= runtime.metadata.output_dimension:
            raise IncompatibleAnalysisError("The requested output target does not exist.")
        selected_model = runtime.model.getMarginal(target)
        output_vector = ot.CompositeRandomVector(selected_model, ot.RandomVector(runtime.problem))
        calls_before = runtime.model.getEvaluationCallsNumber()
        try:
            expansion = ot.TaylorExpansionMoments(output_vector)
            mean_first = float(expansion.getMeanFirstOrder()[0])
            mean_second = float(expansion.getMeanSecondOrder()[0])
            variance = float(expansion.getCovariance()[0, 0])
            gradient = expansion.getGradientAtMean()
            importance = expansion.getImportanceFactors()
        except Exception as exc:
            raise IncompatibleAnalysisError(
                f"OpenTURNS Taylor expansion could not evaluate this model: {exc}"
            ) from exc
        covariance = runtime.problem.getCovariance()
        names = [item.name for item in runtime.metadata.inputs]
        rows = [
            [
                name,
                float(runtime.problem.getMean()[index]),
                float(gradient[index, 0]),
                float(covariance[index, index]),
                float(importance[index]),
            ]
            for index, name in enumerate(names)
        ]
        top = max(range(len(importance)), key=lambda index: abs(float(importance[index])))
        evaluations = max(0, runtime.model.getEvaluationCallsNumber() - calls_before)
        return AnalysisPayload(
            metrics={
                "first_order_mean": mean_first,
                "second_order_mean": mean_second,
                "first_order_variance": variance,
            },
            tables={
                "indices": TableData(
                    columns=[
                        "Variable",
                        "Input Mean",
                        "Gradient at Mean",
                        "Input Variance",
                        "Taylor Importance Factor",
                    ],
                    rows=rows,
                    row_count=len(rows),
                )
            },
            facts={
                "output": runtime.metadata.outputs[target].name,
                "most_influential_input": names[top],
                "largest_taylor_importance": float(importance[top]),
                "authority": "OpenTURNS TaylorExpansionMoments",
            },
        ), evaluations


plugin = TaylorPlugin()
