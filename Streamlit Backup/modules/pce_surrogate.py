"""Polynomial Chaos Expansion surrogate utilities."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import openturns as ot
import pandas as pd
import plotly.graph_objects as go
import math


def _build_polynomial_basis(distribution: ot.Distribution) -> ot.OrthogonalProductPolynomialFactory:
    """Create an orthogonal polynomial basis compatible with the input distribution."""
    dimension = distribution.getDimension()
    poly_collection = ot.PolynomialFamilyCollection(dimension)
    for i in range(dimension):
        marginal = distribution.getMarginal(i)
        poly_collection[i] = ot.StandardDistributionPolynomialFactory(marginal)
    return ot.OrthogonalProductPolynomialFactory(poly_collection)


def build_pce_surrogate(
    model: ot.Function,
    problem: ot.Distribution,
    estimation_method: str,
    degree: int,
    sample_size: int,
    quadrature_order: int,
    sparsity_strategy: str,
) -> Tuple[ot.FunctionalChaosResult, int]:
    """Construct a polynomial chaos surrogate and return the result plus training size."""
    basis = _build_polynomial_basis(problem)
    enumerate_function = basis.getEnumerateFunction()
    basis_size = enumerate_function.getBasisSizeFromTotalDegree(degree)
    adaptive_strategy = ot.FixedStrategy(basis, basis_size)

    if estimation_method == "Least Squares (Regression)":
        input_train = problem.getSample(sample_size)
        output_train = model(input_train)
        if sparsity_strategy == "Sparse (LARS)":
            selection_factory = ot.LeastSquaresMetaModelSelectionFactory()
            projection_strategy = ot.LeastSquaresStrategy(
                input_train, output_train, selection_factory
            )
        else:
            projection_strategy = ot.LeastSquaresStrategy(input_train, output_train)
        chaos_algo = ot.FunctionalChaosAlgorithm(
            input_train, output_train, problem, adaptive_strategy, projection_strategy
        )
        training_size = input_train.getSize()
    elif estimation_method == "Integration (Quadrature)":
        dimension = problem.getDimension()
        marginal_degrees = [quadrature_order] * dimension
        experiment = ot.GaussProductExperiment(problem, marginal_degrees)
        x_quad, weights = experiment.generateWithWeights()
        y_quad = model(x_quad)
        projection_strategy = ot.IntegrationStrategy()
        chaos_algo = ot.FunctionalChaosAlgorithm(
            x_quad, weights, y_quad, problem, adaptive_strategy, projection_strategy
        )
        training_size = x_quad.getSize()
    else:
        raise ValueError(f"Unknown estimation method: {estimation_method}")

    chaos_algo.run()
    return chaos_algo.getResult(), training_size


def compute_pce_validation(
    chaos_result: ot.FunctionalChaosResult,
    model: ot.Function,
    problem: ot.Distribution,
    n_test: int = 1000,
) -> Dict[str, Any]:
    """Compute validation metrics and scatter plot comparing model vs surrogate."""
    metamodel = chaos_result.getMetaModel()
    x_test = problem.getSample(n_test)
    y_true = model(x_test)
    y_pred = metamodel(x_test)

    validation = ot.MetaModelValidation(y_true, y_pred)
    r2_score = validation.computeR2Score()[0]

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    min_val = float(min(y_true_np.min(), y_pred_np.min()))
    max_val = float(max(y_true_np.max(), y_pred_np.max()))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_true_np.flatten(),
            y=y_pred_np.flatten(),
            mode="markers",
            name="Predictions",
            marker=dict(color="#1f77b4", opacity=0.7),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Ideal",
            line=dict(color="#d62728", dash="dash"),
        )
    )
    fig.update_layout(
        title=f"Metamodel Validation (Q² on test sample = {r2_score*100:.2f}%)",
        xaxis_title="True output",
        yaxis_title="PCE prediction",
        legend=dict(orientation="h"),
    )

    return {"r2_score": r2_score, "validation_plot": fig}


def _compute_sparse_least_squares_for_cv(
    input_train: ot.Sample,
    output_train: ot.Sample,
    basis: ot.OrthogonalProductPolynomialFactory,
    total_degree: int,
    distribution: ot.Distribution,
) -> ot.FunctionalChaosResult:
    selection = ot.LeastSquaresMetaModelSelectionFactory()
    projection = ot.LeastSquaresStrategy(input_train, output_train, selection)
    enum_function = basis.getEnumerateFunction()
    basis_size = enum_function.getBasisSizeFromTotalDegree(total_degree)
    strategy = ot.FixedStrategy(basis, basis_size)
    chaos = ot.FunctionalChaosAlgorithm(
        input_train, output_train, distribution, strategy, projection
    )
    chaos.run()
    return chaos.getResult()


def _compute_r2_score_by_kfold(
    input_sample: ot.Sample,
    output_sample: ot.Sample,
    basis: ot.OrthogonalProductPolynomialFactory,
    total_degree: int,
    distribution: ot.Distribution,
    k_parameter: int = 5,
) -> float:
    sample_size = input_sample.getSize()
    splitter = ot.KFoldSplitter(sample_size, k_parameter)
    r2_scores = ot.Sample(0, output_sample.getDimension())

    for indices_train, indices_test in splitter:
        x_train, x_test = input_sample[indices_train], input_sample[indices_test]
        y_train, y_test = output_sample[indices_train], output_sample[indices_test]
        chaos_result = _compute_sparse_least_squares_for_cv(
            x_train, y_train, basis, total_degree, distribution
        )
        metamodel = chaos_result.getMetaModel()
        predictions = metamodel(x_test)
        validation = ot.MetaModelValidation(y_test, predictions)
        r2_scores.add(validation.computeR2Score())

    r2_mean = r2_scores.computeMean()[0]
    return float(r2_mean)


def compute_degree_sensitivity_kfold(
    model: ot.Function,
    problem: ot.Distribution,
    sample_size: int,
    max_degree: int = 10,
    k_fold: int = 5,
) -> go.Figure:
    """Compute R² vs polynomial degree using k-fold validation."""
    x_sample = problem.getSample(sample_size)
    y_sample = model(x_sample)
    basis = _build_polynomial_basis(problem)

    degrees = list(range(1, max_degree + 1))
    r2_scores = []
    for deg in degrees:
        score = _compute_r2_score_by_kfold(
            x_sample, y_sample, basis, deg, problem, k_fold
        )
        r2_scores.append(score)

    valid_scores = [s for s in r2_scores if np.isfinite(s)]
    ymin = -0.1
    if valid_scores:
        ymin = min(-0.1, min(valid_scores) - 0.05)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=degrees,
            y=r2_scores,
            mode="lines+markers",
            name="Q² score",
        )
    )
    fig.update_layout(
        title=f"{k_fold}-Fold Cross-Validation (N={sample_size})",
        xaxis_title="Polynomial degree",
        yaxis_title="Q² score",
        yaxis=dict(range=[ymin, 1.05]),
    )
    return fig


def compute_pce_sobol_indices(
    chaos_result: ot.FunctionalChaosResult,
    model: ot.Function,
    problem: ot.Distribution,
    build_config: Dict[str, Any],
    bootstrap_size: int = 0,
) -> go.Figure:
    """Compute Sobol' indices from the PCE, optionally with bootstrap confidence intervals."""
    dim_input = chaos_result.getDistribution().getDimension()
    input_names = [str(name) for name in chaos_result.getDistribution().getDescription()]

    chaos_si = ot.FunctionalChaosSobolIndices(chaos_result)
    fo = np.array([chaos_si.getSobolIndex(i) for i in range(dim_input)])
    to = np.array([chaos_si.getSobolTotalIndex(i) for i in range(dim_input)])
    lower_fo = upper_fo = lower_to = upper_to = None

    if bootstrap_size > 0:
        fo_samples = []
        to_samples = []
        for _ in range(bootstrap_size):
            boot_result, _ = build_pce_surrogate(
                model,
                problem,
                build_config.get("estimation_method", "Least Squares (Regression)"),
                build_config.get("degree", 5),
                build_config.get("sample_size", 100),
                build_config.get("quadrature_order", 2),
                build_config.get("sparsity_strategy", "Sparse (LARS)"),
            )
            boot_si = ot.FunctionalChaosSobolIndices(boot_result)
            fo_samples.append([boot_si.getSobolIndex(i) for i in range(dim_input)])
            to_samples.append([boot_si.getSobolTotalIndex(i) for i in range(dim_input)])
        fo_arr = np.array(fo_samples)
        to_arr = np.array(to_samples)
        fo = fo_arr.mean(axis=0)
        to = to_arr.mean(axis=0)
        lower_fo = np.quantile(fo_arr, 0.025, axis=0)
        upper_fo = np.quantile(fo_arr, 0.975, axis=0)
        lower_to = np.quantile(to_arr, 0.025, axis=0)
        upper_to = np.quantile(to_arr, 0.975, axis=0)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=input_names,
            y=fo,
            name="First Order",
            error_y=dict(
                type="data",
                array=None if lower_fo is None else upper_fo - fo,
                arrayminus=None if lower_fo is None else fo - lower_fo,
            ),
        )
    )
    fig.add_trace(
        go.Bar(
            x=input_names,
            y=to,
            name="Total Order",
            opacity=0.6,
            error_y=dict(
                type="data",
                array=None if lower_to is None else upper_to - to,
                arrayminus=None if lower_to is None else to - lower_to,
            ),
        )
    )
    title = "Sobol' Indices from PCE"
    if bootstrap_size > 0:
        title += f" (Bootstrap N={bootstrap_size})"
    training_size = build_config.get("training_sample_size")
    if training_size:
        title += f" (N={training_size})"
    fig.update_layout(
        title=title,
        xaxis_title="Input",
        yaxis_title="Sobol' Index",
        barmode="group",
    )
    return fig


def _mean_parametric_pce(
    chaos_result: ot.FunctionalChaosResult, excluded_indices: Iterable[int]
) -> ot.ParametricFunction:
    distribution = chaos_result.getDistribution()
    metamodel = chaos_result.getMetaModel()
    mean_point = distribution.getMean()
    reference = mean_point[excluded_indices]
    return ot.ParametricFunction(metamodel, excluded_indices, reference)


def get_pce_conditional_expectation_plots(
    chaos_result: ot.FunctionalChaosResult,
    model: ot.Function,
    problem: ot.Distribution,
    sample_size: int = 1000,
) -> List[go.Figure]:
    """Return conditional expectation plots comparing true data vs parametric PCE."""
    if not problem.hasIndependentCopula():
        raise RuntimeError(
            "Conditional expectation plots require the input copula to be independent."
        )
    input_dimension = problem.getDimension()
    input_sample = problem.getSample(sample_size)
    output_sample = model(input_sample)
    metamodel = chaos_result.getMetaModel()
    input_names = problem.getDescription()

    sample_np = np.array(input_sample)
    output_np = np.array(output_sample).flatten()
    distribution = problem
    bounds = distribution.getRange()
    lower = np.array(bounds.getLowerBound())
    upper = np.array(bounds.getUpperBound())
    mean_conditional = chaos_result.getConditionalExpectation

    figures: List[go.Figure] = []
    for idx in range(input_dimension):
        excluded = [i for i in range(input_dimension) if i != idx]
        parametric_pce = _mean_parametric_pce(chaos_result, excluded)
        try:
            conditional = mean_conditional([idx]).getMetaModel()
        except Exception as err:
            raise RuntimeError(
                f"OpenTURNS failed to compute the conditional expectation for X_{idx}. ({err})"
            ) from err

        grid = np.linspace(lower[idx], upper[idx], 120)
        parametric_vals = [parametric_pce([val]) for val in grid]
        conditional_vals = [conditional([val]) for val in grid]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=sample_np[:, idx],
                y=output_np,
                mode="markers",
                name="Model samples",
                marker=dict(size=4, opacity=0.4, color="#8c8c8c"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=np.array(parametric_vals).flatten(),
                mode="lines",
                name="PCE (mean-fixed)",
                line=dict(dash="dash", color="#2ca02c"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=np.array(conditional_vals).flatten(),
                mode="lines",
                name="Conditional expectation",
                line=dict(color="#ff7f0e"),
            )
        )
        fig.update_layout(
            title=f"Conditional Expectation vs {input_names[idx]}",
            xaxis_title=input_names[idx],
            yaxis_title="Output",
        )
        figures.append(fig)

    return figures


def get_pce_coefficients_table(
    chaos_result: ot.FunctionalChaosResult,
) -> pd.DataFrame:
    """Return coefficients sorted by magnitude."""
    basis = chaos_result.getOrthogonalBasis()
    enum_function = basis.getEnumerateFunction()
    indices = chaos_result.getIndices()
    coefficients = chaos_result.getCoefficients()
    data = []
    for k in range(indices.getSize()):
        multiindex = enum_function(indices[k])
        coeff = coefficients[k][0]
        data.append(
            {
                "Multi-index": str(list(multiindex)),
                "Degree": int(sum(multiindex)),
                "Coefficient": float(coeff),
            }
        )
    df = pd.DataFrame(data)
    if df.empty:
        return df
    return df.reindex(df["Coefficient"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def count_pce_coefficients(dimension: int, degree: int) -> int:
    """Return the number of coefficients in a total-degree expansion."""
    if degree < 0:
        return 0
    return math.comb(dimension + degree, degree)


def suggest_degree_from_ratio(
    sample_size: int,
    dimension: int,
    proportion: float,
    max_degree: int = 30,
) -> int | None:
    """Return the largest degree whose coefficient count <= proportion * sample size."""
    if sample_size <= 0:
        return None
    proportion = max(0.0, min(1.0, proportion))
    if proportion == 0:
        return 0
    threshold = sample_size * proportion
    degree = 0
    while degree <= max_degree:
        coeffs = count_pce_coefficients(dimension, degree)
        if coeffs > threshold:
            return max(degree - 1, 0)
        degree += 1
    return max_degree
