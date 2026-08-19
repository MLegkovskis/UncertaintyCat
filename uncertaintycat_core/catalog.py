"""Explicit, versioned catalog of supported analyses."""

from __future__ import annotations

from uncertaintycat_core.contracts import AnalysisCatalogEntry
from uncertaintycat_core.errors import UnknownAnalysisError
from uncertaintycat_core.plugins.base import AnalysisPlugin
from uncertaintycat_core.plugins.convergence import plugin as convergence_plugin
from uncertaintycat_core.plugins.correlation import plugin as correlation_plugin
from uncertaintycat_core.plugins.eda import plugin as eda_plugin
from uncertaintycat_core.plugins.fast import plugin as fast_plugin
from uncertaintycat_core.plugins.hsic import plugin as hsic_plugin
from uncertaintycat_core.plugins.monte_carlo import plugin as monte_carlo_plugin
from uncertaintycat_core.plugins.morris import plugin as morris_plugin
from uncertaintycat_core.plugins.pce import plugin as pce_plugin
from uncertaintycat_core.plugins.reliability import plugin as reliability_plugin
from uncertaintycat_core.plugins.sobol import plugin as sobol_plugin
from uncertaintycat_core.plugins.taylor import plugin as taylor_plugin

_PLUGINS: dict[str, AnalysisPlugin] = {
    plugin.key: plugin
    for plugin in (
        monte_carlo_plugin,
        eda_plugin,
        correlation_plugin,
        sobol_plugin,
        fast_plugin,
        hsic_plugin,
        taylor_plugin,
        morris_plugin,
        convergence_plugin,
        reliability_plugin,
        pce_plugin,
    )
}


def get_plugin(key: str) -> AnalysisPlugin:
    try:
        return _PLUGINS[key]
    except KeyError as exc:
        raise UnknownAnalysisError(f"Unknown analysis key: {key}") from exc


def analysis_catalog() -> list[AnalysisCatalogEntry]:
    return [plugin.catalog_entry() for plugin in _PLUGINS.values()]
