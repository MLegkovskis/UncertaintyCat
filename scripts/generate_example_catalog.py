"""Generate the typed authenticated example catalog from canonical examples/*.py sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "packages/contracts/src/example-catalog.generated.ts"

METADATA: dict[str, tuple[str, str, int, str, str, list[str]]] = {
    "Beam": (
        "Beam deflection",
        "Structural mechanics",
        4,
        "Beam deflection under uncertain geometry, material stiffness, and loading.",
        "introductory",
        ["monte_carlo", "eda", "sobol"],
    ),
    "Bike_Speed": (
        "Cycling speed",
        "Transport dynamics",
        8,
        "Cycling-speed response under rider, machine, and environmental uncertainty.",
        "intermediate",
        ["monte_carlo", "morris", "sobol"],
    ),
    "Borehole": (
        "Borehole flow",
        "Hydrology",
        8,
        "Classical groundwater-flow benchmark with eight uncertain physical inputs.",
        "intermediate",
        ["monte_carlo", "morris", "sobol", "pce"],
    ),
    "Chaboche_Model": (
        "Chaboche material model",
        "Material mechanics",
        4,
        "Nonlinear material-response benchmark based on a Chaboche constitutive relation.",
        "advanced",
        ["monte_carlo", "sobol", "gpr"],
    ),
    "Calibration_Exponential": (
        "Nonlinear exponential calibration",
        "Calibration benchmark",
        4,
        "Official OpenTURNS nonlinear least-squares family with three calibration parameters.",
        "introductory",
        ["calibration_nlls"],
    ),
    "Chemical_Reactor": (
        "Chemical reactor",
        "Process engineering",
        6,
        "Reactor-response model with uncertain kinetic and operating inputs.",
        "intermediate",
        ["monte_carlo", "morris", "reliability"],
    ),
    "Cylinder_heating": (
        "Cylinder heating",
        "Thermal engineering",
        5,
        "Transient heating response for a cylinder with uncertain thermal properties.",
        "intermediate",
        ["monte_carlo", "taylor", "pce"],
    ),
    "Damped_Oscillator": (
        "Damped oscillator",
        "Structural dynamics",
        8,
        "Dynamic oscillator response with uncertain mass, damping, forcing, and stiffness.",
        "advanced",
        ["monte_carlo", "morris", "gpr", "reliability"],
    ),
    "Epidemic_Model": (
        "Epidemic response",
        "Population dynamics",
        7,
        "Compartmental epidemic-response benchmark with uncertain transition parameters.",
        "intermediate",
        ["monte_carlo", "morris", "sobol"],
    ),
    "Flood_Model": (
        "Flood model",
        "Flood risk",
        8,
        "River-overflow benchmark linking hydrological and geometric uncertainty.",
        "intermediate",
        ["monte_carlo", "sobol", "reliability"],
    ),
    "Ishigami": (
        "Ishigami function",
        "Sensitivity benchmark",
        3,
        "Nonlinear, non-monotonic benchmark with a strong input interaction.",
        "introductory",
        ["monte_carlo", "eda", "sobol", "fast", "hsic"],
    ),
    "Logistic_Model": (
        "Logistic growth",
        "Population dynamics",
        3,
        "Logistic growth response with uncertain growth and capacity parameters.",
        "introductory",
        ["monte_carlo", "taylor", "pce"],
    ),
    "Material_Stress": (
        "Material stress",
        "Material mechanics",
        5,
        "Stress response under uncertain loading, geometry, and material properties.",
        "introductory",
        ["monte_carlo", "sobol", "reliability"],
    ),
    "Morris_Function": (
        "Morris 20-function",
        "Screening benchmark",
        20,
        "High-dimensional nonlinear benchmark designed for factor screening.",
        "advanced",
        ["morris", "monte_carlo", "gpr"],
    ),
    "Portfolio_Risk": (
        "Portfolio risk",
        "Financial risk",
        4,
        "Portfolio outcome benchmark with uncertain returns and dependence parameters.",
        "intermediate",
        ["monte_carlo", "eda", "reliability"],
    ),
    "Rocket_Trajectory": (
        "Rocket trajectory",
        "Aerospace",
        5,
        "Trajectory-response benchmark with uncertain propulsion and vehicle parameters.",
        "intermediate",
        ["monte_carlo", "morris", "gpr"],
    ),
    "Solar_Panel_Output": (
        "Solar panel output",
        "Renewable energy",
        6,
        "Power-output response under uncertain irradiance, temperature, and component properties.",
        "introductory",
        ["monte_carlo", "sobol", "pce"],
    ),
    "Stiffened_Panel": (
        "Stiffened panel",
        "Structural mechanics",
        10,
        "Structural-response benchmark for a stiffened panel with ten uncertain inputs.",
        "advanced",
        ["morris", "monte_carlo", "gpr", "reliability"],
    ),
    "Trump_Tariff": (
        "Tariff scenario",
        "Economic modelling",
        4,
        "Illustrative tariff-response model with uncertain economic assumptions.",
        "introductory",
        ["monte_carlo", "eda", "sobol"],
    ),
    "Truss_Model": (
        "Truss structure",
        "Structural mechanics",
        10,
        "Truss-response model combining load, geometry, and stiffness uncertainty.",
        "advanced",
        ["morris", "monte_carlo", "reliability"],
    ),
    "Tube_Deflection": (
        "Tube deflection",
        "Structural mechanics",
        6,
        "Tube-deflection response under uncertain geometry, stiffness, and load.",
        "intermediate",
        ["monte_carlo", "taylor", "sobol"],
    ),
    "Undamped_Oscillator": (
        "Undamped oscillator",
        "Structural dynamics",
        6,
        "Oscillator-response model without damping under uncertain forcing and structure.",
        "intermediate",
        ["monte_carlo", "sobol", "reliability"],
    ),
    "Viscous_Freefall": (
        "Viscous free fall",
        "Fluid dynamics",
        4,
        "Falling-body response with uncertain drag, gravity, mass, and initial state.",
        "introductory",
        ["monte_carlo", "taylor", "pce"],
    ),
    "Wind_Turbine_Power": (
        "Wind turbine power",
        "Renewable energy",
        6,
        "Turbine power-response model with uncertain wind and machine parameters.",
        "intermediate",
        ["monte_carlo", "morris", "gpr"],
    ),
}

# Equations are curated scientific metadata, not language-model reconstructions.
# Keep this intentionally explicit: absence is safer than a plausible-looking
# equation that does not exactly match the executable reference model.
EQUATIONS: dict[str, list[dict[str, str]]] = {
    "Beam": [
        {
            "outputName": "Y",
            "latex": r"Y = \frac{F L^{3}}{3 E I}",
        }
    ],
    "Bike_Speed": [
        {
            "outputName": "Cycling speed (implicit balance)",
            "latex": (
                r"P_{r}=\frac{1}{2}\rho C_{d}A_{f}v^{3}"
                r"+C_{rr}m g v"
            ),
        }
    ],
    "Borehole": [
        {
            "outputName": "Borehole flow rate",
            "latex": (
                r"Y=\frac{2\pi T_{u}(H_{u}-H_{l})}"
                r"{\ln(r/r_{w})\left[1+\frac{2LT_{u}}"
                r"{\ln(r/r_{w})r_{w}^{2}K_{w}}+\frac{T_{u}}{T_{l}}\right]}"
            ),
        }
    ],
    "Calibration_Exponential": [
        {
            "outputName": "y",
            "latex": r"y = a + b\exp\left(c x\right)",
        }
    ],
    "Chaboche_Model": [
        {
            "outputName": "Stress response",
            "latex": r"\sigma=R-\frac{C}{\Gamma}\left(e^{-\Gamma\varepsilon}-1\right)",
        }
    ],
    "Chemical_Reactor": [
        {
            "outputName": "Rate constant",
            "latex": r"k=k_{0}\exp\left(-\frac{E}{RT}\right)",
        },
        {
            "outputName": "Conversion",
            "latex": (
                r"X=\frac{C_{A0}-C_A}{C_{A0}}=\frac{kV}{Q+kV},"
                r"\qquad Q=1"
            ),
        },
    ],
    "Cylinder_heating": [
        {
            "outputName": "Radial temperature equation",
            "latex": (
                r"\frac{d^{2}T}{dr^{2}}=-\frac{h\left(T-300\right)}{kr}"
                r"+\frac{Q}{k\ell}"
            ),
        },
        {
            "outputName": "Initial conditions and response",
            "latex": (
                r"T(10^{-5})=300,\quad T'(10^{-5})=0,\quad"
                r"Y=\max_{r\in[10^{-5},R]}T(r)"
            ),
        },
    ],
    "Damped_Oscillator": [
        {
            "outputName": "Natural frequencies",
            "latex": (
                r"\omega_p=\sqrt{\frac{k_p}{m_p}},\qquad"
                r"\omega_s=\sqrt{\frac{k_s}{m_s}}"
            ),
        },
        {
            "outputName": "Auxiliary oscillator terms",
            "latex": (
                r"\gamma=\frac{m_s}{m_p},\quad"
                r"\omega_a=\frac{\omega_p+\omega_s}{2},\quad"
                r"\zeta_a=\frac{\zeta_p+\zeta_s}{2},\quad"
                r"\theta=\frac{\omega_p-\omega_s}{\omega_a}"
            ),
        },
        {
            "outputName": "Mean-square relative displacement",
            "latex": (
                r"\mathbb{E}[x_s^2]=\frac{\pi S_0}{4\zeta_s\omega_s^3}"
                r"\frac{\zeta_a\zeta_s}"
                r"{\zeta_p\zeta_s(4\zeta_a^2+\theta^2)+\gamma\zeta_a^2}"
                r"\frac{(\zeta_p\omega_p^3+\zeta_s\omega_s^3)\omega_p}"
                r"{4\zeta_a\omega_a^4}"
            ),
        },
        {
            "outputName": "Performance function",
            "latex": r"g=F_s-3k_s\sqrt{\mathbb{E}[x_s^2]}",
        },
    ],
    "Epidemic_Model": [
        {
            "outputName": "SIR governing equations",
            "latex": (
                r"\frac{dS}{dt}=-\frac{\beta SI}{N},\qquad"
                r"\frac{dI}{dt}=\frac{\beta SI}{N}-\gamma I,\qquad"
                r"\frac{dR}{dt}=\gamma I"
            ),
        },
        {
            "outputName": "Peak infected population",
            "latex": r"Y=\max_{0\leq t\leq t_{\max}} I(t)",
        },
    ],
    "Flood_Model": [
        {
            "outputName": "River slope and depth",
            "latex": (
                r"\alpha=\max\left(\frac{Z_m-Z_v}{L},0\right),\qquad"
                r"H=\begin{cases}\left(\frac{Q}{K_sB\sqrt{\alpha}}\right)^{0.6},"
                r"&Q,K_s,\alpha>0\\0,&\text{otherwise}\end{cases}"
            ),
        },
        {
            "outputName": "Overflow margin",
            "latex": r"S=H+Z_v-(Z_b+H_d)",
        },
    ],
    "Ishigami": [
        {
            "outputName": "Ishigami response",
            "latex": r"Y=\sin(x_1)+7\sin^2(x_2)+0.1x_3^4\sin(x_1)",
        }
    ],
    "Logistic_Model": [
        {
            "outputName": "Logistic response",
            "latex": (
                r"\widetilde y_0=10^6y_0,\quad b=e^c,\quad"
                r"y=\frac{1}{10^6}\frac{a\widetilde y_0}"
                r"{b\widetilde y_0+(a-b\widetilde y_0)e^{-a(2000-0)}}"
            ),
        }
    ],
    "Material_Stress": [
        {
            "outputName": "Critical stress",
            "latex": (
                r"s=\max\left(\frac{8\gamma\phi R_s}{\pi G b^2},0\right),"
                r"\quad b=2.54\times10^{-9},\quad"
                r"\sigma_c=\max\left(\frac{1}{10^6}\left|\frac{M\gamma}{2b}"
                r"(\sqrt{s}-\phi)\right|,10^{-6}\right)"
            ),
        }
    ],
    "Morris_Function": [
        {
            "outputName": "Morris benchmark response",
            "latex": (
                r"Y=\sum_{i=1}^{20}\beta_iw_i+"
                r"\sum_{i<j}\beta_{ij}w_iw_j+"
                r"\sum_{i<j<k}\beta_{ijk}w_iw_jw_k+"
                r"5w_1w_2w_3w_4"
            ),
        },
        {
            "outputName": "Input transformation",
            "latex": (
                r"w_i=2(u_i-0.5),\quad"
                r"w_i=2\left(\frac{1.1u_i}{u_i+0.1}-0.5\right)"
                r"\ \text{for }i\in\{3,5,7\}"
            ),
        },
        {
            "outputName": "Morris coefficients",
            "latex": (
                r"\beta_i=\begin{cases}20,&i\leq10\\(-1)^i,&i>10\end{cases},\quad"
                r"\beta_{ij}=\begin{cases}-15,&i,j\leq6\\(-1)^{i+j},&\text{otherwise}\end{cases},\quad"
                r"\beta_{ijk}=\begin{cases}-10,&i,j,k\leq5\\0,&\text{otherwise}\end{cases}"
            ),
        },
    ],
    "Portfolio_Risk": [
        {
            "outputName": "Correlated return covariance",
            "latex": (
                r"\Delta t=\frac{T}{252},\qquad"
                r"\Sigma=\sigma^2\Delta t\left[\rho\mathbf{1}\mathbf{1}^{\mathsf T}"
                r"+(1-\rho)I\right],\quad \rho=0.5,\quad LL^{\mathsf T}=\Sigma"
            ),
        },
        {
            "outputName": "Portfolio value",
            "latex": (
                r"V=S_0\exp\left(\mathbf{1}^{\mathsf T}"
                r"(\mu\Delta t\,\mathbf{1}+LZ)\right)"
            ),
        },
    ],
    "Rocket_Trajectory": [
        {
            "outputName": "Velocity increment",
            "latex": r"\Delta v=v_e\ln\left(\frac{m_0}{m_0-m_f}\right)",
        },
        {
            "outputName": "Maximum altitude",
            "latex": (r"h_{\max}=\frac{\left[\Delta v\sin(\theta\pi/180)\right]^2}{2g}"),
        },
    ],
    "Solar_Panel_Output": [
        {
            "outputName": "Solar-panel power",
            "latex": r"P=GA\eta\left[1+\beta(T_{cell}-T_{ref})\right]",
        }
    ],
    "Stiffened_Panel": [
        {
            "outputName": "Plate and stiffener terms",
            "latex": (
                r"k_{xy}=5.35+4\left(\frac{b_0}{a}\right)^2,\quad"
                r"D=\frac{Et^3}{12(1-\nu^2)},\quad A=\ell t,\quad"
                r"\bar A=A+t\left[p+\frac{f_1-f_2}{2}\right]"
            ),
        },
        {
            "outputName": "Neutral-axis terms",
            "latex": (
                r"h_0=\frac{A(h_c+2t)+t^2(f_1-f_2)}{2\bar A},\quad"
                r"h=h_c+t"
            ),
        },
        {
            "outputName": "Critical shear response",
            "latex": (
                r"N_{xy}^{\mathrm{MPa}}=10^9\frac{k_{xy}\pi^2D}{b_0^2}"
                r"\left[1+\frac{2p(h-2h_0)-h_c(f_1-f_2)}{4h_0\ell}\right]"
            ),
        },
    ],
    "Trump_Tariff": [
        {
            "outputName": "Reciprocal tariff",
            "latex": r"\Delta\tau=\max\left(\frac{x-m}{\varepsilon\phi m},0\right)",
        }
    ],
    "Truss_Model": [
        {
            "outputName": "Scaled structural variables",
            "latex": (
                r"\widehat E_i=\max(10^9E_i,10^{-12}),\quad"
                r"\widehat A_i=\max(10^{-4}A_i,10^{-12}),\quad"
                r"\widehat P_i=10^3P_i"
            ),
        },
        {
            "outputName": "Midspan deflection",
            "latex": (
                r"y=-\frac{\sqrt{2}(2\widehat P_1+6\widehat P_2+"
                r"10\widehat P_3+10\widehat P_4+6\widehat P_5+2\widehat P_6)}"
                r"{\widehat A_2\widehat E_2}"
                r"-\frac{36\widehat P_1+100\widehat P_2+140\widehat P_3+"
                r"140\widehat P_4+100\widehat P_5+36\widehat P_6}"
                r"{\widehat A_1\widehat E_1}"
            ),
        },
    ],
    "Tube_Deflection": [
        {
            "outputName": "Second moment of area",
            "latex": r"I = \frac{\pi}{32}\left(D_e^4-d_i^4\right)",
        },
        {
            "outputName": "Deflection",
            "latex": r"y = -\frac{F a^2 \left(L-a\right)^2}{3 E L I}",
        },
    ],
    "Undamped_Oscillator": [
        {
            "outputName": "Natural frequency",
            "latex": r"\omega_0=\sqrt{\frac{C_1+C_2}{M}}",
        },
        {
            "outputName": "Performance function",
            "latex": (
                r"Y=3R-\left|\frac{2F_1}{M\omega_0^2}"
                r"\sin\left(\frac{\omega_0T_1}{2}\right)\right|"
            ),
        },
    ],
    "Viscous_Freefall": [
        {
            "outputName": "Vertical trajectory",
            "latex": (
                r"\tau=\frac{m}{c},\quad v_{\infty}=-\frac{mg}{c},\quad"
                r"z(t)=z_0+v_{\infty}t+\tau(v_0-v_{\infty})"
                r"\left(1-e^{-t/\tau}\right),\quad g=9.81"
            ),
        },
        {
            "outputName": "Maximum height",
            "latex": r"Y=\max_{0\leq t\leq12}z(t)",
        },
    ],
    "Wind_Turbine_Power": [
        {
            "outputName": "Wind-turbine power",
            "latex": (
                r"P=\frac{1}{2}\rho_{air}A C_p"
                r"\cos(\beta\pi/180)v_{wind}^{3}\eta"
            ),
        }
    ],
}

if set(EQUATIONS) != set(METADATA):
    raise SystemExit(
        "Every bundled reference model must have curated equations: "
        f"missing={set(METADATA) - set(EQUATIONS)}, stale={set(EQUATIONS) - set(METADATA)}"
    )


def build() -> str:
    entries: list[dict[str, object]] = []
    sources = sorted((ROOT / "examples").glob("*.py"))
    found = {path.stem for path in sources}
    if found != set(METADATA):
        missing = found - set(METADATA)
        stale = set(METADATA) - found
        raise SystemExit(f"Catalog metadata mismatch: missing={missing}, stale={stale}")
    for path in sources:
        title, domain, input_dimension, summary, difficulty, analyses = METADATA[path.stem]
        source = path.read_text(encoding="utf-8")
        entry: dict[str, object] = {
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
        if path.stem in EQUATIONS:
            entry["equations"] = EQUATIONS[path.stem]
        entries.append(entry)
    payload = json.dumps(entries, indent=2, ensure_ascii=False)
    catalog = (
        f"export const EXAMPLE_CATALOG = {payload} as const satisfies "
        "readonly ExampleCatalogEntry[];\n"
    )
    return (
        "// Generated by scripts/generate_example_catalog.py. Do not edit by hand.\n"
        'import type { ExampleCatalogEntry } from "./index";\n\n' + catalog
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
