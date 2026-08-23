"""Gaussian process surrogate using the stable OpenTURNS 1.27 GPR API."""

from __future__ import annotations

import math
from typing import Literal

import openturns as ot
from pydantic import Field

from uncertaintycat_core.contracts import AnalysisPayload, SeriesData, StrictModel, TableData
from uncertaintycat_core.errors import IncompatibleAnalysisError
from uncertaintycat_core.model import ModelRuntime
from uncertaintycat_core.plugins.base import AnalysisPlugin

Kernel = Literal["MATERN_1_5", "MATERN_2_5", "SQUARED_EXPONENTIAL"]
Trend = Literal["CONSTANT", "LINEAR"]


class GprConfig(StrictModel):
    # Exact GPR fitting is cubic in the training size. Keep this deliberately
    # below the broader application sample budget so a valid config is also a
    # defensible request for the 2 CPU / 2 GiB compute boundary.
    training_size: int = Field(default=128, ge=16, le=512)
    validation_size: int = Field(default=256, ge=20, le=5_000)
    kernel: Kernel = "MATERN_2_5"
    trend: Trend = "CONSTANT"
    seed: int = Field(default=42, ge=0)
    output_targets: list[int] = Field(default_factory=list, max_length=1)
    inline_validation_limit: int = Field(default=500, ge=0, le=5_000)


class GprPlugin(AnalysisPlugin[GprConfig]):
    key = "gpr"
    version = "2.0.0"
    name = "Gaussian Process Surrogate"
    category = "Surrogate"
    description = (
        "Fit a Gaussian process metamodel and independently validate predictions "
        "and model-based uncertainty intervals."
    )
    assumptions = (
        "Inputs are continuous and the training design covers the probability region of interest.",
        "The selected stationary kernel and trend are suitable for the response regularity.",
        "Hold-out R2 and RMSE must be checked before the surrogate is used downstream.",
        "The reported 95% intervals are conditional Gaussian-process model intervals, not "
        "guaranteed frequentist confidence intervals.",
    )
    supports_dependent_inputs = True
    supports_multi_output = False
    resource_class = "heavy"
    config_model = GprConfig

    def applicability_warnings(self, runtime: ModelRuntime, config: GprConfig) -> list[str]:
        if not runtime.problem.isContinuous():
            raise IncompatibleAnalysisError(
                "The Gaussian process surrogate currently supports continuous input "
                "distributions only."
            )
        if (
            config.trend == "LINEAR"
            and config.training_size <= runtime.metadata.input_dimension + 1
        ):
            raise IncompatibleAnalysisError(
                "A linear GPR trend requires more training points than trend coefficients."
            )
        if config.training_size < 10 * runtime.metadata.input_dimension:
            return [
                "The GPR design has fewer than 10 training points per input; "
                "treat hold-out accuracy and interval coverage cautiously."
            ]
        return []

    def run(self, runtime: ModelRuntime, config: GprConfig) -> tuple[AnalysisPayload, int]:
        self.applicability_warnings(runtime, config)
        target = config.output_targets[0] if config.output_targets else 0
        if target >= runtime.metadata.output_dimension:
            raise IncompatibleAnalysisError("The requested output target does not exist.")

        dimension = runtime.metadata.input_dimension
        result, training_x, training_y = fit_gpr(runtime, config)
        input_spread = [float(value) for value in training_x.computeStandardDeviation()]

        metamodel = result.getMetaModel()
        training_predictions = metamodel(training_x)
        training_validation = ot.MetaModelValidation(training_y, training_predictions)

        ot.RandomGenerator.SetSeed(config.seed + 1)
        validation_x = runtime.problem.getSample(config.validation_size)
        validation_y = runtime.model(validation_x).getMarginal(target)
        predictions_sample = metamodel(validation_x)
        observed = [float(row[0]) for row in validation_y]
        predicted = [float(row[0]) for row in predictions_sample]
        if not all(math.isfinite(value) for value in [*observed, *predicted]):
            raise IncompatibleAnalysisError(
                "The selected output or GPR metamodel produced non-finite validation values."
            )
        observed_mean = float(validation_y.computeMean()[0])
        if float(validation_y.computeVariance()[0]) <= ot.SpecFunc.ScalarEpsilon * max(
            1.0, observed_mean**2
        ):
            raise IncompatibleAnalysisError(
                "GPR hold-out R2 is undefined because the validation output is constant."
            )
        validation = ot.MetaModelValidation(validation_y, predictions_sample)
        residual_sample = validation.getResidualSample()
        residuals = [float(row[0]) for row in residual_sample]
        r2 = float(validation.computeR2Score()[0])
        mse = float(validation.computeMeanSquaredError()[0])
        rmse = mse**0.5
        normalized_rmse = rmse / float(validation_y.computeStandardDeviation()[0])
        absolute_residuals = ot.SymbolicFunction(["r"], ["abs(r)"])(residual_sample)
        validation_mae = float(absolute_residuals.computeMean()[0])

        conditional = ot.GaussianProcessConditionalCovariance(result)
        conditional_variances = [
            float(row[0]) for row in conditional.getConditionalMarginalVariance(validation_x)
        ]
        conditional_std = [math.sqrt(max(value, 0.0)) for value in conditional_variances]
        normal_975 = float(ot.Normal().computeQuantile(0.975)[0])
        lower = [
            value - normal_975 * spread
            for value, spread in zip(predicted, conditional_std, strict=True)
        ]
        upper = [
            value + normal_975 * spread
            for value, spread in zip(predicted, conditional_std, strict=True)
        ]
        covered = [
            low <= value <= high for value, low, high in zip(observed, lower, upper, strict=True)
        ]
        coverage_sample = ot.Sample([[1.0 if value else 0.0] for value in covered])

        covariance_result = result.getCovarianceModel()
        scales = [float(value) for value in covariance_result.getScale()]
        normalized_scales = [scales[index] / input_spread[index] for index in range(dimension)]
        input_names = [item.name for item in runtime.metadata.inputs]
        hyperparameter_rows = [
            [name, scales[index], normalized_scales[index]]
            for index, name in enumerate(input_names)
        ]
        trend_coefficients = [float(value) for value in result.getTrendCoefficients()]
        trend_names = ["Intercept"] if config.trend == "CONSTANT" else ["Intercept", *input_names]
        trend_rows = [
            [name, float(coefficient)]
            for name, coefficient in zip(trend_names, trend_coefficients, strict=True)
        ]

        inline_size = min(config.validation_size, config.inline_validation_limit)
        validation_rows = [
            [
                int(index),
                float(observed[index]),
                float(predicted[index]),
                float(residuals[index]),
                float(conditional_std[index]),
                float(lower[index]),
                float(upper[index]),
                bool(covered[index]),
            ]
            for index in range(inline_size)
        ]
        return (
            AnalysisPayload(
                metrics={
                    "training_size": config.training_size,
                    "validation_size": config.validation_size,
                    "validation_r2": r2,
                    "validation_rmse": rmse,
                    "validation_normalized_rmse": normalized_rmse,
                    "validation_mae": validation_mae,
                    "training_interpolation_rmse": float(
                        training_validation.computeMeanSquaredError()[0]
                    )
                    ** 0.5,
                    "nominal_95_interval_coverage": float(coverage_sample.computeMean()[0]),
                    "optimized_log_likelihood": float(result.getOptimalLogLikelihood()),
                    "optimized_amplitude": float(covariance_result.getAmplitude()[0]),
                    "nugget_factor": float(covariance_result.getNuggetFactor()),
                },
                tables={
                    "kernel_length_scales": TableData(
                        columns=["Input", "Optimized Scale", "Scale / Training Std"],
                        rows=hyperparameter_rows,
                        row_count=len(hyperparameter_rows),
                    ),
                    "trend_coefficients": TableData(
                        columns=["Trend Term", "Coefficient"],
                        rows=trend_rows,
                        row_count=len(trend_rows),
                    ),
                    "validation_predictions": TableData(
                        columns=[
                            "Sample",
                            "Observed",
                            "Predicted",
                            "Residual",
                            "Conditional Std",
                            "95% Lower",
                            "95% Upper",
                            "Covered",
                        ],
                        rows=validation_rows,
                        row_count=config.validation_size,
                        truncated=inline_size < config.validation_size,
                    ),
                },
                series={
                    "validation": SeriesData(
                        name="GPR validation",
                        x=[float(value) for value in observed],
                        y=[float(value) for value in predicted],
                        x_label="Observed",
                        y_label="Predicted",
                    )
                },
                facts={
                    "output": runtime.metadata.outputs[target].name,
                    "kernel": _kernel_label(config.kernel),
                    "trend": config.trend.title(),
                    "interval_level": "95% conditional Gaussian-process model interval",
                },
            ),
            config.training_size + config.validation_size,
        )


def fit_gpr(
    runtime: ModelRuntime, config: GprConfig
) -> tuple[ot.GaussianProcessRegressionResult, ot.Sample, ot.Sample]:
    """Fit the exact GPR persisted by Surrogate Studio."""
    plugin.applicability_warnings(runtime, config)
    target = config.output_targets[0] if config.output_targets else 0
    if target >= runtime.metadata.output_dimension:
        raise IncompatibleAnalysisError("The requested output target does not exist.")
    dimension = runtime.metadata.input_dimension
    ot.RandomGenerator.SetSeed(config.seed)
    training_x = runtime.problem.getSample(config.training_size)
    training_y = runtime.model(training_x).getMarginal(target)
    training_values = [float(row[0]) for row in training_y]
    if not all(math.isfinite(value) for value in training_values):
        raise IncompatibleAnalysisError(
            "The selected output produced non-finite GPR training values."
        )
    training_mean = float(training_y.computeMean()[0])
    variance_scale = max(1.0, training_mean**2)
    if float(training_y.computeVariance()[0]) <= ot.SpecFunc.ScalarEpsilon * variance_scale:
        raise IncompatibleAnalysisError(
            "Gaussian process fitting is undefined for a constant selected output."
        )
    input_spread = [float(value) for value in training_x.computeStandardDeviation()]
    if not all(math.isfinite(value) and value > 0.0 for value in input_spread):
        raise IncompatibleAnalysisError(
            "The GPR training design has an input with no observed variation."
        )
    covariance = _build_covariance(config.kernel, dimension)
    basis = _build_basis(config.trend, dimension)
    try:
        fitter = ot.GaussianProcessFitter(training_x, training_y, covariance, basis)
        fitter.run()
        regression = ot.GaussianProcessRegression(fitter.getResult())
        regression.run()
        return regression.getResult(), training_x, training_y
    except Exception as exc:
        raise IncompatibleAnalysisError(
            f"Gaussian process construction failed for this model and configuration: {exc}"
        ) from exc


def _build_covariance(kernel: Kernel, dimension: int) -> ot.CovarianceModel:
    if kernel == "SQUARED_EXPONENTIAL":
        return ot.SquaredExponential(dimension)
    nu = 1.5 if kernel == "MATERN_1_5" else 2.5
    return ot.MaternModel([1.0] * dimension, nu)


def _build_basis(trend: Trend, dimension: int) -> ot.Basis:
    if trend == "LINEAR":
        return ot.LinearBasisFactory(dimension).build()
    return ot.ConstantBasisFactory(dimension).build()


def _kernel_label(kernel: Kernel) -> str:
    return {
        "MATERN_1_5": "Matern 3/2",
        "MATERN_2_5": "Matern 5/2",
        "SQUARED_EXPONENTIAL": "Squared exponential",
    }[kernel]


plugin = GprPlugin()
