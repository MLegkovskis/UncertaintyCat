"""Gaussian-process surrogates trained directly from paired empirical data."""

from __future__ import annotations

import base64
import hashlib
import math
import tempfile
from pathlib import Path
from typing import Any, Literal

import openturns as ot
import pandas as pd
from pydantic import Field, field_validator

from uncertaintycat_core.data_lab import DatasetContent, _dataframe
from uncertaintycat_core.errors import InvalidModelError

Kernel = Literal["MATERN_1_5", "MATERN_2_5", "SQUARED_EXPONENTIAL"]
Trend = Literal["CONSTANT", "LINEAR"]


class DataSurrogateRequest(DatasetContent):
    input_columns: list[str] = Field(min_length=1, max_length=40)
    output_column: str = Field(min_length=1, max_length=200)
    validation_fraction: float = Field(default=0.2, ge=0.1, le=0.5)
    kernel: Kernel = "MATERN_2_5"
    trend: Trend = "CONSTANT"
    seed: int = Field(default=42, ge=0, le=2_147_483_647)

    @field_validator("input_columns")
    @classmethod
    def unique_inputs(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("Input columns must be unique.")
        return value


def fit_data_surrogate(request: DataSurrogateRequest) -> dict[str, Any]:
    """Fit and hold-out validate an OpenTURNS GPR from an empirical X/Y sample."""
    frame = _dataframe(request)
    selected = [*request.input_columns, request.output_column]
    if request.output_column in request.input_columns:
        raise InvalidModelError("The output column cannot also be an input column.")
    missing = [column for column in selected if column not in frame.columns]
    if missing:
        raise InvalidModelError(f"Unknown surrogate columns: {', '.join(missing)}")
    numeric = frame[selected].apply(pd.to_numeric, errors="coerce").dropna()
    numeric = numeric[
        numeric.apply(lambda row: all(math.isfinite(float(value)) for value in row), axis=1)
    ]
    dimension = len(request.input_columns)
    minimum_rows = max(20, 2 * (dimension + 2))
    if len(numeric) < minimum_rows:
        raise InvalidModelError(
            f"Data-driven GPR requires at least {minimum_rows} complete finite rows "
            f"for {dimension} input(s)."
        )
    input_sample = ot.Sample(
        [[float(row[column]) for column in request.input_columns] for _, row in numeric.iterrows()]
    )
    input_sample.setDescription(request.input_columns)
    output_sample = ot.Sample([[float(value)] for value in numeric[request.output_column]])
    output_sample.setDescription([request.output_column])
    if any(
        not math.isfinite(float(value)) or float(value) <= 0.0
        for value in input_sample.computeStandardDeviation()
    ):
        raise InvalidModelError("Every surrogate input must vary across the complete rows.")
    if float(output_sample.computeVariance()[0]) <= ot.SpecFunc.ScalarEpsilon:
        raise InvalidModelError("The surrogate output must vary across the complete rows.")

    folds = max(2, min(10, round(1.0 / request.validation_fraction)))
    ot.RandomGenerator.SetSeed(request.seed)
    splitter = ot.KFoldSplitter(input_sample.getSize(), folds)
    splitter.setRandomize(True)
    training_indices, validation_indices = next(iter(splitter))
    training_x = input_sample[training_indices]
    training_y = output_sample[training_indices]
    validation_x = input_sample[validation_indices]
    validation_y = output_sample[validation_indices]
    if request.trend == "LINEAR" and training_x.getSize() <= dimension + 1:
        raise InvalidModelError(
            "A linear GPR trend requires more training rows than trend coefficients."
        )

    covariance = _covariance(request.kernel, dimension)
    basis = (
        ot.LinearBasisFactory(dimension).build()
        if request.trend == "LINEAR"
        else ot.ConstantBasisFactory(dimension).build()
    )
    try:
        fitter = ot.GaussianProcessFitter(training_x, training_y, covariance, basis)
        fitter.run()
        regression = ot.GaussianProcessRegression(fitter.getResult())
        regression.run()
        result = regression.getResult()
    except Exception as exc:
        raise InvalidModelError(f"OpenTURNS could not fit the data-driven GPR: {exc}") from exc

    predicted_sample = result.getMetaModel()(validation_x)
    validation = ot.MetaModelValidation(validation_y, predicted_sample)
    r2 = float(validation.computeR2Score()[0])
    rmse = float(validation.computeMeanSquaredError()[0]) ** 0.5
    validation_std = float(validation_y.computeStandardDeviation()[0])
    if not math.isfinite(validation_std) or validation_std <= ot.SpecFunc.ScalarEpsilon:
        raise InvalidModelError(
            "The held-out response values do not vary enough to validate this surrogate."
        )
    normalized_rmse = rmse / validation_std
    observed = [float(row[0]) for row in validation_y]
    predicted = [float(row[0]) for row in predicted_sample]
    if not all(math.isfinite(value) for value in [r2, rmse, normalized_rmse, *predicted]):
        raise InvalidModelError("The data-driven GPR produced non-finite validation evidence.")

    with tempfile.TemporaryDirectory(prefix="uncertaintycat-data-surrogate-") as directory:
        path = Path(directory) / "surrogate.xml"
        study = ot.Study()
        study.setStorageManager(ot.XMLStorageManager(str(path)))
        study.add("surrogate_result", result)
        study.save()
        xml = path.read_bytes()
    return {
        "method": "gpr",
        "pluginVersion": "1.0.0",
        "openturnsVersion": ot.__version__,
        "inputColumns": request.input_columns,
        "outputColumn": request.output_column,
        "config": {
            "kernel": request.kernel,
            "trend": request.trend,
            "seed": request.seed,
            "validationFraction": request.validation_fraction,
        },
        "validation": {
            "trainingSize": training_x.getSize(),
            "validationSize": validation_x.getSize(),
            "r2": r2,
            "rmse": rmse,
            "normalizedRmse": normalized_rmse,
            "meetsDefault": r2 >= 0.95 and normalized_rmse <= 0.1,
            "observed": observed[:500],
            "predicted": predicted[:500],
        },
        "artifact": {
            "xmlBase64": base64.b64encode(xml).decode("ascii"),
            "sha256": hashlib.sha256(xml).hexdigest(),
            "sizeBytes": len(xml),
            "resultType": "GaussianProcessRegressionResult",
        },
    }


def _covariance(kernel: Kernel, dimension: int) -> ot.CovarianceModel:
    if kernel == "SQUARED_EXPONENTIAL":
        return ot.SquaredExponential(dimension)
    return ot.MaternModel([1.0] * dimension, 1.5 if kernel == "MATERN_1_5" else 2.5)
