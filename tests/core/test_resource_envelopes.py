from __future__ import annotations

import pytest

from uncertaintycat_core.errors import IncompatibleAnalysisError
from uncertaintycat_core.model import compile_model
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
