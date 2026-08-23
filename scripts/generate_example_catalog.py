"""Generate the typed public example catalog from canonical examples/*.py sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "packages/contracts/src/example-catalog.generated.ts"

METADATA: dict[str, tuple[str, str, int, str, str, list[str]]] = {
    "Beam": ("Simply supported beam", "Structural mechanics", 4, "Beam deflection under uncertain geometry, material stiffness, and loading.", "introductory", ["monte_carlo", "eda", "sobol"]),
    "Bike_Speed": ("Cycling speed", "Transport dynamics", 8, "Cycling-speed response under rider, machine, and environmental uncertainty.", "intermediate", ["monte_carlo", "morris", "sobol"]),
    "Borehole": ("Borehole flow", "Hydrology", 8, "Classical groundwater-flow benchmark with eight uncertain physical inputs.", "intermediate", ["monte_carlo", "morris", "sobol", "pce"]),
    "Chaboche_Model": ("Chaboche material model", "Material mechanics", 4, "Nonlinear material-response benchmark based on a Chaboche constitutive relation.", "advanced", ["monte_carlo", "sobol", "gpr"]),
    "Chemical_Reactor": ("Chemical reactor", "Process engineering", 6, "Reactor-response model with uncertain kinetic and operating inputs.", "intermediate", ["monte_carlo", "morris", "reliability"]),
    "Cylinder_heating": ("Cylinder heating", "Thermal engineering", 5, "Transient heating response for a cylinder with uncertain thermal properties.", "intermediate", ["monte_carlo", "taylor", "pce"]),
    "Damped_Oscillator": ("Damped oscillator", "Structural dynamics", 8, "Dynamic oscillator response with uncertain mass, damping, forcing, and stiffness.", "advanced", ["monte_carlo", "morris", "gpr", "reliability"]),
    "Epidemic_Model": ("Epidemic response", "Population dynamics", 7, "Compartmental epidemic-response benchmark with uncertain transition parameters.", "intermediate", ["monte_carlo", "morris", "sobol"]),
    "Flood_Model": ("Flood model", "Flood risk", 8, "River-overflow benchmark linking hydrological and geometric uncertainty.", "intermediate", ["monte_carlo", "sobol", "reliability"]),
    "Ishigami": ("Ishigami function", "Sensitivity benchmark", 3, "Nonlinear, non-monotonic benchmark with a strong input interaction.", "introductory", ["monte_carlo", "eda", "sobol", "fast", "hsic"]),
    "Logistic_Model": ("Logistic growth", "Population dynamics", 3, "Logistic growth response with uncertain growth and capacity parameters.", "introductory", ["monte_carlo", "taylor", "pce"]),
    "Material_Stress": ("Material stress", "Material mechanics", 5, "Stress response under uncertain loading, geometry, and material properties.", "introductory", ["monte_carlo", "sobol", "reliability"]),
    "Morris_Function": ("Morris 20-function", "Screening benchmark", 20, "High-dimensional nonlinear benchmark designed for factor screening.", "advanced", ["morris", "monte_carlo", "gpr"]),
    "Portfolio_Risk": ("Portfolio risk", "Financial risk", 4, "Portfolio outcome benchmark with uncertain returns and dependence parameters.", "intermediate", ["monte_carlo", "eda", "reliability"]),
    "Rocket_Trajectory": ("Rocket trajectory", "Aerospace", 5, "Trajectory-response benchmark with uncertain propulsion and vehicle parameters.", "intermediate", ["monte_carlo", "morris", "gpr"]),
    "Solar_Panel_Output": ("Solar panel output", "Renewable energy", 6, "Power-output response under uncertain irradiance, temperature, and component properties.", "introductory", ["monte_carlo", "sobol", "pce"]),
    "Stiffened_Panel": ("Stiffened panel", "Structural mechanics", 10, "Structural-response benchmark for a stiffened panel with ten uncertain inputs.", "advanced", ["morris", "monte_carlo", "gpr", "reliability"]),
    "Trump_Tariff": ("Tariff scenario", "Economic modelling", 4, "Illustrative tariff-response model with uncertain economic assumptions.", "introductory", ["monte_carlo", "eda", "sobol"]),
    "Truss_Model": ("Truss structure", "Structural mechanics", 10, "Truss-response model combining load, geometry, and stiffness uncertainty.", "advanced", ["morris", "monte_carlo", "reliability"]),
    "Tube_Deflection": ("Tube deflection", "Structural mechanics", 6, "Tube-deflection response under uncertain geometry, stiffness, and load.", "intermediate", ["monte_carlo", "taylor", "sobol"]),
    "Undamped_Oscillator": ("Undamped oscillator", "Structural dynamics", 6, "Oscillator-response model without damping under uncertain forcing and structure.", "intermediate", ["monte_carlo", "sobol", "reliability"]),
    "Viscous_Freefall": ("Viscous free fall", "Fluid dynamics", 4, "Falling-body response with uncertain drag, gravity, mass, and initial state.", "introductory", ["monte_carlo", "taylor", "pce"]),
    "Wind_Turbine_Power": ("Wind turbine power", "Renewable energy", 6, "Turbine power-response model with uncertain wind and machine parameters.", "intermediate", ["monte_carlo", "morris", "gpr"]),
}


def build() -> str:
    entries: list[dict[str, object]] = []
    sources = sorted((ROOT / "examples").glob("*.py"))
    found = {path.stem for path in sources}
    if found != set(METADATA):
        raise SystemExit(f"Catalog metadata mismatch: missing={found - set(METADATA)}, stale={set(METADATA) - found}")
    for path in sources:
        title, domain, input_dimension, summary, difficulty, analyses = METADATA[path.stem]
        source = path.read_text(encoding="utf-8")
        entries.append(
            {
                "id": path.stem.lower(),
                "title": title,
                "filename": path.name,
                "domain": domain,
                "inputDimension": input_dimension,
                "outputDimension": 1,
                "summary": summary,
                "difficulty": difficulty,
                "suggestedAnalyses": analyses,
                "source": source,
                "sha256": hashlib.sha256(source.encode()).hexdigest(),
            }
        )
    payload = json.dumps(entries, indent=2, ensure_ascii=False)
    return (
        "// Generated by scripts/generate_example_catalog.py. Do not edit by hand.\n"
        'import type { ExampleCatalogEntry } from "./index";\n\n'
        f"export const EXAMPLE_CATALOG = {payload} as const satisfies readonly ExampleCatalogEntry[];\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    generated = build()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != generated:
            raise SystemExit("Generated example catalog is stale. Run npm run generate:examples.")
        return
    OUTPUT.write_text(generated, encoding="utf-8")


if __name__ == "__main__":
    main()
