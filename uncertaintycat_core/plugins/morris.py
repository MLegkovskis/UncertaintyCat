"""Distribution-aware screening through the official OTMorris module."""

from __future__ import annotations

import openturns as ot
import otmorris
from pydantic import Field

from uncertaintycat_core.contracts import AnalysisPayload, StrictModel, TableData
from uncertaintycat_core.errors import IncompatibleAnalysisError
from uncertaintycat_core.model import ModelRuntime
from uncertaintycat_core.plugins.base import AnalysisPlugin
from uncertaintycat_core.plugins.sobol import SobolPlugin


class MorrisConfig(StrictModel):
    trajectories: int = Field(default=10, ge=4, le=1_000)
    levels: int = Field(default=6, ge=4, le=50)
    candidate_threshold_fraction: float = Field(default=0.05, ge=0, le=1)
    seed: int = Field(default=42, ge=0)
    output_targets: list[int] = Field(default_factory=list, max_length=1)


class MorrisPlugin(AnalysisPlugin[MorrisConfig]):
    key = "morris"
    version = "2.0.0"
    name = "Morris Screening"
    category = "Sensitivity"
    description = "Screen inputs using the pinned official OTMorris experiment and estimator."
    assumptions = (
        "Inputs must be independent.",
        "Elementary effects are computed along OTMorris trajectories in probability space.",
        "The candidate threshold is a user-adjustable screening rule, not proof of irrelevance.",
    )
    supports_dependent_inputs = False
    supports_multi_output = False
    config_model = MorrisConfig

    def applicability_warnings(self, runtime: ModelRuntime, config: MorrisConfig) -> list[str]:
        if not SobolPlugin._is_independent(runtime.problem):
            raise IncompatibleAnalysisError(
                "OTMorris probability-space trajectories require independent inputs."
            )
        return []

    def run(self, runtime: ModelRuntime, config: MorrisConfig) -> tuple[AnalysisPayload, int]:
        self.applicability_warnings(runtime, config)
        target = config.output_targets[0] if config.output_targets else 0
        if target >= runtime.metadata.output_dimension:
            raise IncompatibleAnalysisError("The requested output target does not exist.")
        dimension = runtime.metadata.input_dimension
        ot.RandomGenerator.SetSeed(config.seed)
        experiment = otmorris.MorrisExperimentGrid([config.levels] * dimension, config.trajectories)
        unit_sample = experiment.generate()
        physical_sample = ot.Sample(unit_sample.getSize(), dimension)
        for row_index, unit_point in enumerate(unit_sample):
            physical_sample[row_index] = ot.Point(
                [
                    runtime.problem.getMarginal(column).computeQuantile(
                        min(max(float(unit_point[column]), 1e-12), 1.0 - 1e-12)
                    )[0]
                    for column in range(dimension)
                ]
            )
        output_sample = runtime.model(physical_sample).getMarginal(target)
        estimator = otmorris.Morris(
            unit_sample,
            output_sample,
            experiment.getBounds(),
        )
        signed = estimator.getMeanElementaryEffects()
        absolute = estimator.getMeanAbsoluteElementaryEffects()
        dispersion = estimator.getStandardDeviationElementaryEffects()
        names = [item.name for item in runtime.metadata.inputs]
        ranking = sorted(range(dimension), key=lambda index: float(absolute[index]), reverse=True)
        ranks = {index: rank + 1 for rank, index in enumerate(ranking)}
        largest = float(absolute[ranking[0]])
        threshold = config.candidate_threshold_fraction * largest
        rows: list[list[str | float | int | bool]] = [
            [
                name,
                float(signed[index]),
                float(absolute[index]),
                float(dispersion[index]),
                ranks[index],
                float(absolute[index]) >= threshold,
            ]
            for index, name in enumerate(names)
        ]
        evaluations = int(unit_sample.getSize())
        return AnalysisPayload(
            metrics={
                "trajectories": config.trajectories,
                "levels": config.levels,
                "model_evaluations": evaluations,
                "candidate_threshold_fraction": config.candidate_threshold_fraction,
                "candidate_threshold": threshold,
            },
            tables={
                "effects": TableData(
                    columns=[
                        "Variable",
                        "Signed Mean Effect",
                        "Mean Absolute Effect",
                        "Effect Dispersion",
                        "Rank",
                        "Candidate Retained",
                    ],
                    rows=rows,
                    row_count=len(rows),
                )
            },
            facts={
                "output": runtime.metadata.outputs[target].name,
                "most_influential_input": names[ranking[0]],
                "largest_mean_absolute_effect": largest,
                "threshold_is_proof_of_irrelevance": False,
                "authority": f"otmorris {otmorris.__version__}",
            },
        ), evaluations


plugin = MorrisPlugin()
