"""Bounded deterministic parameter calibration with stable OpenTURNS APIs."""

from __future__ import annotations

import math
from typing import Annotated, Any

import openturns as ot
from pydantic import Field, FiniteFloat, field_validator, model_validator

from uncertaintycat_core.contracts import (
    AnalysisPayload,
    MatrixData,
    SeriesData,
    StrictModel,
    TableData,
)
from uncertaintycat_core.errors import IncompatibleAnalysisError
from uncertaintycat_core.model import ModelRuntime
from uncertaintycat_core.plugins.base import AnalysisPlugin

MAX_CALIBRATED_PARAMETERS = 8
MAX_MODEL_INPUTS = 32
MAX_OBSERVATION_ROWS = 250
MAX_OPTIMIZER_CALLS = 500
DEFAULT_OPTIMIZER_CALLS = 250
MAX_REPORT_JSON_BYTES = 1_000_000
ILL_CONDITIONED_THRESHOLD = 1.0e10

NonNegativeInt = Annotated[int, Field(ge=0)]


class CalibrationConfig(StrictModel):
    parameter_indices: list[NonNegativeInt] = Field(
        min_length=1, max_length=MAX_CALIBRATED_PARAMETERS
    )
    starting_values: list[FiniteFloat] = Field(min_length=1, max_length=MAX_CALIBRATED_PARAMETERS)
    observed_input_names: list[str] = Field(default_factory=list, max_length=MAX_MODEL_INPUTS)
    observed_output_name: str = Field(min_length=1, max_length=200)
    observed_inputs: list[list[FiniteFloat]] = Field(min_length=3, max_length=MAX_OBSERVATION_ROWS)
    observed_outputs: list[FiniteFloat] = Field(min_length=3, max_length=MAX_OBSERVATION_ROWS)
    maximum_calls: int = Field(default=DEFAULT_OPTIMIZER_CALLS, ge=10, le=MAX_OPTIMIZER_CALLS)
    seed: int = Field(default=42, ge=0, le=2_147_483_647)
    output_targets: list[NonNegativeInt] = Field(default_factory=list, max_length=1)

    @field_validator("parameter_indices")
    @classmethod
    def unique_parameter_indices(cls, values: list[int]) -> list[int]:
        if len(values) != len(set(values)):
            raise ValueError("Calibration parameter indices must be unique.")
        return values

    @field_validator("observed_input_names")
    @classmethod
    def valid_observed_input_names(cls, values: list[str]) -> list[str]:
        stripped = [value.strip() for value in values]
        if any(not value for value in stripped):
            raise ValueError("Observed input names cannot be empty.")
        if len(stripped) != len(set(stripped)):
            raise ValueError("Observed input names must be unique.")
        return stripped

    @field_validator("observed_output_name")
    @classmethod
    def valid_observed_output_name(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("The observed output name cannot be empty.")
        return stripped

    @model_validator(mode="after")
    def consistent_shapes(self) -> CalibrationConfig:
        if len(self.parameter_indices) != len(self.starting_values):
            raise ValueError("Provide one starting value per calibration parameter.")
        if len(self.observed_inputs) != len(self.observed_outputs):
            raise ValueError("Observed input and output row counts must match.")
        if self.observed_output_name in self.observed_input_names:
            raise ValueError("Observed input and output column names must be unique.")
        expected_dimension = len(self.observed_input_names)
        if any(len(row) != expected_dimension for row in self.observed_inputs):
            raise ValueError("Every observed input row must match the named input columns.")
        return self


def _singular_values(residual_function: ot.Function, parameters: list[float]) -> list[float]:
    try:
        values = residual_function.gradient(parameters).computeSingularValues()
    except Exception as exc:
        raise IncompatibleAnalysisError(
            f"OpenTURNS could not evaluate the calibration parameter Jacobian: {exc}"
        ) from exc
    return [float(value) for value in values]


def _jacobian_condition(singular_values: list[float], row_count: int) -> float:
    if not singular_values or not all(math.isfinite(value) for value in singular_values):
        raise IncompatibleAnalysisError(
            "The calibration parameter Jacobian did not produce finite singular values."
        )
    largest = max(singular_values)
    smallest = min(singular_values)
    tolerance = max(row_count, len(singular_values)) * math.ulp(1.0) * largest
    if largest <= 0.0 or smallest <= tolerance:
        raise IncompatibleAnalysisError(
            "The calibration parameter Jacobian is rank-deficient. The selected parameters "
            "cannot be estimated separately from these observations and starting values."
        )
    return largest / smallest


def _fit_metrics(observed: list[float], predicted: list[float]) -> dict[str, float | None]:
    residuals = [actual - estimate for actual, estimate in zip(observed, predicted, strict=True)]
    count = len(residuals)
    squared_error = sum(value * value for value in residuals)
    absolute_error = sum(abs(value) for value in residuals)
    mean_residual = sum(residuals) / count
    residual_variance = sum((value - mean_residual) ** 2 for value in residuals) / (count - 1)
    observed_mean = sum(observed) / count
    total_squares = sum((value - observed_mean) ** 2 for value in observed)
    return {
        "rmse": math.sqrt(squared_error / count),
        "mae": absolute_error / count,
        "mean_residual": mean_residual,
        "residual_standard_deviation": math.sqrt(max(0.0, residual_variance)),
        "maximum_absolute_residual": max(abs(value) for value in residuals),
        "r_squared": 1.0 - squared_error / total_squares if total_squares > 0.0 else None,
    }


class CalibrationPlugin(AnalysisPlugin[CalibrationConfig]):
    key = "calibration_nlls"
    version = "1.0.0"
    name = "Nonlinear Least-Squares Calibration"
    category = "Calibration"
    description = (
        "Estimate selected model inputs from named observations with bounded OpenTURNS nonlinear "
        "least squares."
    )
    assumptions = (
        "Selected continuous model inputs are treated as constant calibration parameters; "
        "remaining inputs are observed explanatory variables.",
        "The fit is ordinary unweighted nonlinear least squares with independent, "
        "equal-variance residuals for its usual statistical interpretation.",
        "Parameter uncertainty is OpenTURNS' local linear Gaussian (Laplace) approximation "
        "at the optimum with bootstrap disabled; it is not an exact confidence guarantee.",
        "Calibration fit does not by itself establish parameter identifiability, causal "
        "validity, or predictive validity outside the observed domain.",
        "The project's input distribution and copula are not sampled in this "
        "observation-conditioned fit.",
    )
    supports_multi_output = False
    resource_class = "heavy"
    config_model = CalibrationConfig

    def applicability_warnings(self, runtime: ModelRuntime, config: CalibrationConfig) -> list[str]:
        if runtime.metadata.input_dimension > MAX_MODEL_INPUTS:
            raise IncompatibleAnalysisError(
                f"Calibration supports at most {MAX_MODEL_INPUTS} model inputs."
            )
        target = config.output_targets[0] if config.output_targets else 0
        if target >= runtime.metadata.output_dimension:
            raise IncompatibleAnalysisError("The requested calibration output does not exist.")
        if any(index >= runtime.metadata.input_dimension for index in config.parameter_indices):
            raise IncompatibleAnalysisError("A selected calibration parameter does not exist.")
        input_names = [item.name for item in runtime.metadata.inputs]
        if len(input_names) != len(set(input_names)):
            raise IncompatibleAnalysisError(
                "Calibration requires unique model input names for named observations."
            )
        selected = set(config.parameter_indices)
        if any(runtime.metadata.inputs[index].kind != "continuous" for index in selected):
            raise IncompatibleAnalysisError(
                "Calibration parameters must be continuous model inputs."
            )
        expected_inputs = [
            item.name for item in runtime.metadata.inputs if item.index not in selected
        ]
        if config.observed_input_names != expected_inputs:
            raise IncompatibleAnalysisError(
                "Observed input names must exactly match the non-calibrated model inputs "
                "in model order."
            )
        expected_output = runtime.metadata.outputs[target].name
        if expected_output in expected_inputs:
            raise IncompatibleAnalysisError(
                "The selected output name must differ from every observed input name."
            )
        if config.observed_output_name != expected_output:
            raise IncompatibleAnalysisError(
                f"The observed output column must be named '{expected_output}'."
            )
        minimum_rows = len(config.parameter_indices) + 2
        if len(config.observed_outputs) < minimum_rows:
            raise IncompatibleAnalysisError(
                f"At least {minimum_rows} observations are required for "
                f"{len(config.parameter_indices)} calibration parameters."
            )
        output_scale = max(1.0, *(abs(float(value)) for value in config.observed_outputs))
        if (
            max(config.observed_outputs) - min(config.observed_outputs)
            <= math.ulp(1.0) * output_scale
        ):
            raise IncompatibleAnalysisError(
                "Observed output values must vary for this calibration and "
                "residual-uncertainty report."
            )
        return [
            "Approximate parameter intervals use OpenTURNS' local linear Gaussian "
            "(Laplace) approximation and are not exact confidence guarantees.",
            "A successful fit and full-rank local Jacobian do not establish global "
            "identifiability, causality, or validity outside the observed domain.",
        ]

    def run(self, runtime: ModelRuntime, config: CalibrationConfig) -> tuple[AnalysisPayload, int]:
        self.applicability_warnings(runtime, config)
        target = config.output_targets[0] if config.output_targets else 0
        parameter_names = [
            runtime.metadata.inputs[index].name for index in config.parameter_indices
        ]
        selected_model = runtime.model.getMarginal(target)
        starting_values = [float(value) for value in config.starting_values]
        row_count = len(config.observed_outputs)
        input_observations = (
            ot.Sample([[float(value) for value in row] for row in config.observed_inputs])
            if config.observed_input_names
            else ot.Sample(row_count, 0)
        )
        input_observations.setDescription(config.observed_input_names)
        output_observations = ot.Sample([[float(value)] for value in config.observed_outputs])
        output_observations.setDescription([config.observed_output_name])
        try:
            parametric_model = ot.ParametricFunction(
                selected_model, config.parameter_indices, starting_values
            )
            residual_function = ot.NonLinearLeastSquaresCalibration.BuildResidualFunction(
                parametric_model, input_observations, output_observations
            )
        except Exception as exc:
            raise IncompatibleAnalysisError(
                f"OpenTURNS could not construct the calibration problem: {exc}"
            ) from exc

        calls_before = runtime.model.getEvaluationCallsNumber()
        starting_singular_values = _singular_values(residual_function, starting_values)
        _jacobian_condition(starting_singular_values, row_count)
        try:
            algorithm = ot.NonLinearLeastSquaresCalibration(
                parametric_model,
                input_observations,
                output_observations,
                starting_values,
            )
            algorithm.setBootstrapSize(0)
            optimizer = algorithm.getOptimizationAlgorithm()
            optimizer.setMaximumCallsNumber(config.maximum_calls)
            optimizer.setMaximumIterationNumber(config.maximum_calls)
            algorithm.setOptimizationAlgorithm(optimizer)
            ot.RandomGenerator.SetSeed(config.seed)
            algorithm.run()
            result = algorithm.getResult()
            optimizer = algorithm.getOptimizationAlgorithm()
            optimization_result = optimizer.getResult()
        except Exception as exc:
            raise IncompatibleAnalysisError(
                f"OpenTURNS nonlinear least-squares calibration failed: {exc}"
            ) from exc

        status = int(optimization_result.getStatus())
        status_message = str(optimization_result.getStatusMessage())
        if status != int(ot.OptimizationResult.SUCCESS):
            raise IncompatibleAnalysisError(
                "OpenTURNS nonlinear least-squares calibration did not converge: "
                f"{status_message or status}."
            )

        calibrated_values = [float(value) for value in result.getParameterMAP()]
        if not all(math.isfinite(value) for value in calibrated_values):
            raise IncompatibleAnalysisError(
                "OpenTURNS did not produce finite calibrated parameter values."
            )
        optimum_singular_values = _singular_values(result.getResidualFunction(), calibrated_values)
        condition_number = _jacobian_condition(optimum_singular_values, row_count)
        posterior = result.getParameterPosterior()
        approximate_sd = [float(value) for value in posterior.getStandardDeviation()]
        if not all(math.isfinite(value) and value >= 0.0 for value in approximate_sd):
            raise IncompatibleAnalysisError(
                "OpenTURNS did not produce finite local parameter-uncertainty estimates."
            )
        before_predictions = [float(row[0]) for row in result.getOutputAtPriorMean()]
        after_predictions = [float(row[0]) for row in result.getOutputAtPosteriorMean()]
        if not all(math.isfinite(value) for value in before_predictions + after_predictions):
            raise IncompatibleAnalysisError(
                "OpenTURNS did not produce finite calibration predictions."
            )
        evaluations = max(0, runtime.model.getEvaluationCallsNumber() - calls_before)

        observed = [float(value) for value in config.observed_outputs]
        before_metrics = _fit_metrics(observed, before_predictions)
        after_metrics = _fit_metrics(observed, after_predictions)
        before_residuals = [
            actual - estimate for actual, estimate in zip(observed, before_predictions, strict=True)
        ]
        after_residuals = [
            actual - estimate for actual, estimate in zip(observed, after_predictions, strict=True)
        ]
        parameter_rows: list[list[Any]] = []
        for index, name in enumerate(parameter_names):
            marginal = posterior.getMarginal(index)
            lower = float(marginal.computeQuantile(0.025)[0])
            upper = float(marginal.computeQuantile(0.975)[0])
            if not math.isfinite(lower) or not math.isfinite(upper):
                raise IncompatibleAnalysisError(
                    "OpenTURNS did not produce finite approximate parameter intervals."
                )
            parameter_rows.append(
                [
                    name,
                    starting_values[index],
                    calibrated_values[index],
                    calibrated_values[index] - starting_values[index],
                    approximate_sd[index],
                    lower,
                    upper,
                ]
            )
        observation_rows = [
            [
                row_index + 1,
                *[float(value) for value in config.observed_inputs[row_index]],
                observed[row_index],
                before_predictions[row_index],
                after_predictions[row_index],
                before_residuals[row_index],
                after_residuals[row_index],
            ]
            for row_index in range(row_count)
        ]
        residual_diagnostic_rows = [
            ["RMSE", before_metrics["rmse"], after_metrics["rmse"]],
            ["MAE", before_metrics["mae"], after_metrics["mae"]],
            ["Mean residual", before_metrics["mean_residual"], after_metrics["mean_residual"]],
            [
                "Residual standard deviation",
                before_metrics["residual_standard_deviation"],
                after_metrics["residual_standard_deviation"],
            ],
            [
                "Maximum absolute residual",
                before_metrics["maximum_absolute_residual"],
                after_metrics["maximum_absolute_residual"],
            ],
            ["R squared", before_metrics["r_squared"], after_metrics["r_squared"]],
        ]
        correlation = posterior.getCorrelation()
        correlation_values = [
            [float(correlation[row, column]) for column in range(len(parameter_names))]
            for row in range(len(parameter_names))
        ]
        if not all(math.isfinite(value) for row in correlation_values for value in row):
            raise IncompatibleAnalysisError(
                "OpenTURNS did not produce a finite approximate parameter correlation matrix."
            )
        optimizer_name = optimizer.getImplementation().getClassName()
        optimizer_calls = int(optimization_result.getCallsNumber())
        optimizer_iterations = int(optimization_result.getIterationNumber())
        payload = AnalysisPayload(
            metrics={
                "observations": row_count,
                "parameters": len(parameter_names),
                "rmse_before": before_metrics["rmse"],
                "rmse_after": after_metrics["rmse"],
                "mae_after": after_metrics["mae"],
                "mean_residual_after": after_metrics["mean_residual"],
                "residual_standard_deviation_after": after_metrics["residual_standard_deviation"],
                "maximum_absolute_residual_after": after_metrics["maximum_absolute_residual"],
                "r_squared_after": after_metrics["r_squared"],
                "jacobian_condition_number": condition_number,
                "optimizer_calls": optimizer_calls,
                "optimizer_iterations": optimizer_iterations,
                "model_evaluations": evaluations,
            },
            tables={
                "calibrated_parameters": TableData(
                    columns=[
                        "Parameter",
                        "Starting Value",
                        "Calibrated Value",
                        "Change",
                        "Approximate SD (Local Linear Gaussian)",
                        "Approximate 95% Lower",
                        "Approximate 95% Upper",
                    ],
                    rows=parameter_rows,
                    row_count=len(parameter_rows),
                ),
                "observations_and_predictions": TableData(
                    columns=[
                        "Observation",
                        *config.observed_input_names,
                        f"Observed {config.observed_output_name}",
                        "Predicted Before",
                        "Predicted After",
                        "Residual Before (Observed - Predicted)",
                        "Residual After (Observed - Predicted)",
                    ],
                    rows=observation_rows,
                    row_count=row_count,
                ),
                "residual_diagnostics": TableData(
                    columns=["Diagnostic", "Before Calibration", "After Calibration"],
                    rows=residual_diagnostic_rows,
                    row_count=len(residual_diagnostic_rows),
                ),
            },
            series={
                "observed_vs_predicted_before": SeriesData(
                    name="Before calibration",
                    x=observed,
                    y=before_predictions,
                    x_label=f"Observed {config.observed_output_name}",
                    y_label=f"Predicted {config.observed_output_name}",
                ),
                "observed_vs_predicted_after": SeriesData(
                    name="After calibration",
                    x=observed,
                    y=after_predictions,
                    x_label=f"Observed {config.observed_output_name}",
                    y_label=f"Predicted {config.observed_output_name}",
                ),
            },
            matrices={
                "approximate_parameter_correlation": MatrixData(
                    row_labels=parameter_names,
                    column_labels=parameter_names,
                    values=correlation_values,
                )
            },
            facts={
                "authority": (
                    "OpenTURNS ParametricFunction + NonLinearLeastSquaresCalibration + "
                    "CalibrationResult"
                ),
                "method": "ordinary nonlinear least squares",
                "output": config.observed_output_name,
                "calibration_parameters": ", ".join(parameter_names),
                "observed_explanatory_inputs": ", ".join(config.observed_input_names) or "none",
                "optimizer": optimizer_name,
                "optimizer_status": status_message or "SUCCESS",
                "optimizer_status_code": status,
                "optimizer_maximum_calls": config.maximum_calls,
                "bootstrap_size": 0,
                "parameter_uncertainty": (
                    "Local linear Gaussian (Laplace) approximation at the optimum; "
                    "not an exact confidence guarantee."
                ),
                "residual_definition": (
                    "Displayed residuals are observed minus predicted; OpenTURNS internally "
                    "minimizes predicted minus observed."
                ),
                "local_jacobian_full_rank": True,
                "local_jacobian_ill_conditioned": condition_number >= ILL_CONDITIONED_THRESHOLD,
                "evaluation_accounting": (
                    "Exact delta of OpenTURNS atomic model evaluation calls, including "
                    "derivative and result-construction work."
                ),
                "stored_prediction_rows": row_count,
                "prediction_rows_truncated": False,
                "report_payload_limit_bytes": MAX_REPORT_JSON_BYTES,
            },
        )
        if len(payload.model_dump_json().encode()) > MAX_REPORT_JSON_BYTES:
            raise IncompatibleAnalysisError(
                f"The calibration result exceeds the {MAX_REPORT_JSON_BYTES}-byte report limit."
            )
        return payload, evaluations


plugin = CalibrationPlugin()
