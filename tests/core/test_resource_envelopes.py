from __future__ import annotations

from pathlib import Path

import pytest

from uncertaintycat_core.errors import IncompatibleAnalysisError
from uncertaintycat_core.model import compile_model
from uncertaintycat_core.plugins.hsic import (
    HsicConfig,
    maximum_hsic_sample_size,
)
from uncertaintycat_core.plugins.hsic import (
    estimate_hsic_work_units as estimate_global_hsic_work_units,
)
from uncertaintycat_core.plugins.hsic import (
    plugin as hsic_plugin,
)
from uncertaintycat_core.plugins.target_hsic import (
    MAXIMUM_HSIC_WORK_UNITS,
    TargetHsicConfig,
    estimate_hsic_work_units,
    plugin,
)

TWENTY_DIMENSIONAL_MODEL = """
import openturns as ot
names = [f"x{index}" for index in range(20)]
model = ot.SymbolicFunction(names, [" + ".join(names)])
problem = ot.Normal(20)
problem.setDescription(names)
"""


def _independent_target_hsic_loop_oracle(
    sample_size: int, input_dimension: int, permutations: int
) -> int:
    """Count the pinned implementation's nested passes without using production algebra."""

    units = 0
    for _ in range(permutations + 4):
        for _ in range(input_dimension + 1):
            units += sample_size * sample_size
    return units


def _independent_global_hsic_loop_oracle(
    sample_size: int, input_dimension: int, permutations: int
) -> int:
    """Count OpenTURNS' all-variable quadratic passes without production algebra."""

    units = 0
    for _ in range(permutations + 4):
        for _ in range(input_dimension + 1):
            units += sample_size * sample_size
    return units


@pytest.mark.scientific
@pytest.mark.parametrize(
    ("sample_size", "input_dimension", "permutations"),
    [
        (50, 1, 0),
        (100, 3, 100),
        (250, 20, 100),
        (500, 20, 200),
    ],
)
def test_target_hsic_resource_estimator_matches_independent_loop_oracle(
    sample_size: int, input_dimension: int, permutations: int
) -> None:
    assert estimate_hsic_work_units(
        sample_size, input_dimension, permutations
    ) == _independent_target_hsic_loop_oracle(sample_size, input_dimension, permutations)


@pytest.mark.scientific
def test_target_hsic_default_fits_maximum_supported_dimension_budget() -> None:
    config = TargetHsicConfig()
    estimated = estimate_hsic_work_units(config.sample_size, 20, config.permutations)

    assert config.sample_size == 250
    assert config.permutations == 100
    assert estimated == 136_500_000
    assert estimated <= MAXIMUM_HSIC_WORK_UNITS


@pytest.mark.scientific
def test_target_hsic_rejects_first_permutation_count_above_budget() -> None:
    runtime = compile_model(TWENTY_DIMENSIONAL_MODEL)
    within_budget = TargetHsicConfig(sample_size=250, permutations=110)
    above_budget = TargetHsicConfig(sample_size=250, permutations=111)

    assert estimate_hsic_work_units(250, 20, 110) == 149_625_000
    assert estimate_hsic_work_units(250, 20, 111) == 150_937_500
    assert plugin.applicability_warnings(runtime, within_budget)
    with pytest.raises(IncompatibleAnalysisError, match="workload exceeds"):
        plugin.applicability_warnings(runtime, above_budget)


@pytest.mark.scientific
@pytest.mark.parametrize(
    ("sample_size", "input_dimension", "permutations"),
    [(30, 1, 0), (100, 3, 1000), (400, 8, 100), (1000, 8, 100)],
)
def test_hsic_resource_estimator_matches_independent_loop_oracle(
    sample_size: int, input_dimension: int, permutations: int
) -> None:
    assert estimate_global_hsic_work_units(
        sample_size, input_dimension, permutations
    ) == _independent_global_hsic_loop_oracle(sample_size, input_dimension, permutations)


@pytest.mark.scientific
def test_damped_oscillator_hsic_boundary_admits_400_and_rejects_401_samples() -> None:
    runtime = compile_model(Path("examples/Damped_Oscillator.py").read_text())

    assert runtime.metadata.input_dimension == 8
    assert maximum_hsic_sample_size(8, 100) == 400
    assert estimate_global_hsic_work_units(400, 8, 100) == 149_760_000
    assert estimate_global_hsic_work_units(401, 8, 100) == 150_509_736
    assert (
        hsic_plugin.applicability_warnings(runtime, HsicConfig(sample_size=400, permutations=100))
        == []
    )
    with pytest.raises(IncompatibleAnalysisError, match="use at most 400 samples"):
        hsic_plugin.applicability_warnings(runtime, HsicConfig(sample_size=401, permutations=100))


def test_oversized_hsic_is_rejected_before_any_model_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = compile_model(Path("examples/Damped_Oscillator.py").read_text())
    sampled = False

    def unexpected_sampling(*_args: object, **_kwargs: object) -> None:
        nonlocal sampled
        sampled = True
        raise AssertionError("sampling must not start for an inadmissible HSIC request")

    monkeypatch.setattr(runtime, "sample_and_evaluate", unexpected_sampling)
    with pytest.raises(IncompatibleAnalysisError, match="workload exceeds"):
        hsic_plugin.run(runtime, HsicConfig(sample_size=1000, permutations=100))
    assert sampled is False
