# UncertaintyCat 🐱 | AI-Assisted Uncertainty Quantification

[![Streamlit](https://img.shields.io/badge/Streamlit-cloud-red)](https://uncertaintycat.streamlit.app/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![OpenTURNS](https://img.shields.io/badge/built%20with-OpenTURNS-4d8ecf)](https://openturns.github.io/openturns/latest/)

UncertaintyCat is a modern, browser-based workspace for engineers, researchers, and data scientists who need to **quantify uncertainty**, **understand model sensitivity**, and **communicate results** fast. It blends battle-tested algorithms from [OpenTURNS](https://openturns.github.io/openturns/latest/) with Streamlit’s interactive UX and an AI assistant that can summarize results or answer questions about any completed analysis.

👉 **Try it now:** https://uncertaintycat.streamlit.app/

---
## 📚 Table of Contents
- [Motivation](#-motivation)
- [Feature Highlights](#-feature-highlights)
- [Workspaces & Pages](#-workspaces--pages)
- [Defining Your Model](#-defining-your-model)
- [Analysis Modules](#-analysis-modules)
- [Architecture at a Glance](#-architecture-at-a-glance)
- [Run Locally with uv](#-run-locally-with-uv)
- [Deploy to Streamlit Community Cloud](#-deploy-to-streamlit-community-cloud)
- [Contributing & Roadmap](#-contributing--roadmap)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)

---
## 💡 Motivation
During a PhD project with **Tata Steel**, a single Sobol analysis routinely revealed which furnace inputs dominated output variability. The bottleneck wasn’t math—it was the repetitive OpenTURNS scripting, plotting, and report writing. UncertaintyCat automates the workflow: drop in a model, push a button, and receive a full suite of analyses plus AI-generated commentary you can hand to stakeholders.

---
## 🌟 Feature Highlights
- **Zero-install, multipage Streamlit app** with dedicated spaces for UQ, dimensionality reduction, distribution fitting, and the new **Surrogate Modeling** lab.
- **Comprehensive UQ toolkit**: Monte Carlo, Sobol/FAST/ANCOVA, Morris, HSIC, Taylor, EDA, correlation, machine-learning Shapley indices, and more.
- **AI everywhere**: Each module can call Groq-hosted LLMs for tailored insights, and the sidebar chat can answer questions about whichever analyses you’ve run—even incrementally.
- **“🚀 Overall Summary” meta-analysis**: generates a consolidated executive summary whenever multiple modules have been executed.
- **Surrogate Modeling (PCE)**: Build a polynomial chaos expansion surrogate, validate it, run degree-vs-R² cross-validation, inspect Sobol’ indices (with optional bootstrap), and visualize conditional expectations.
- **Modern state & registry architecture**: analyses register once and auto-populate buttons, progress bars, and expander sections; compute modules are now Streamlit-free for easier testing.
- **Streamlit Cloud-ready**: dependency management is handled by `pyproject.toml` + `uv.lock`, making deployments deterministic and fast.

---
## 🧭 Workspaces & Pages
| Page | Purpose |
| --- | --- |
| **Home** | Welcome page that keeps the sidebar “frozen” (no accidental model swaps) and explains available workspaces. |
| **📊 UQ Dashboard** | Primary hub for Monte Carlo, Sobol, FAST, ANCOVA, Taylor, HSIC, correlation, expectation convergence, EDA, ML-based Shapley, and the AI-driven Overall Summary. Buttons are generated dynamically from the analysis registry. |
| **📉 Dimensionality Reduction** | Morris screening workflow that highlights non-influential parameters and suggests fixed values for follow-on studies. |
| **📈 Distribution Fitting** | Ingest CSV/Excel data, fit parametric distributions and copulas, and emit ready-to-paste `problem` definitions. |
| **🛠️ Surrogate Modeling (new)** | Build polynomial chaos expansions (regression or quadrature), validate them, explore degree sensitivity, compute PCE-based Sobol’ indices, visualize conditional expectations, and inspect coefficients. |

Each page shares a consistent sidebar (model selector, uploader, LLM picker) and uses centralized session state so results persist while you navigate.

---
## 🐍 Defining Your Model
Provide two objects in the built-in editor:
1. `model`: an `ot.Function` (Python function wrapped via `ot.PythonFunction`, symbolic expression, or OpenTURNS workflow).
2. `problem`: an `ot.Distribution`, typically `ot.JointDistribution`/`ot.ComposedDistribution` of your marginals.

Minimal Ishigami example:
```python
import openturns as ot
import numpy as np

def ishigami(x):
    x1, x2, x3 = x
    return [np.sin(x1) + 7.0*np.sin(x2)**2 + 0.1*x3**4*np.sin(x1)]

model = ot.PythonFunction(3, 1, ishigami)
problem = ot.ComposedDistribution([
    ot.Uniform(-np.pi, np.pi),
    ot.Uniform(-np.pi, np.pi),
    ot.Uniform(-np.pi, np.pi),
])
problem.setDescription(["X1", "X2", "X3"])
```
Paste this into the editor, hit **Validate Model Input**, and you’re ready to run analyses.

---
## 🧪 Analysis Modules
| Module | Description |
| --- | --- |
| **🚀 Overall Summary** | Meta-analysis synthesizing all completed modules via an LLM-powered executive summary. |
| **Model Understanding** | AI narrative describing equations, distributions, and uncertainty propagation. |
| **Monte Carlo & EDA** | Sample generation, summary stats, correlation heatmaps, cross-cuts, and AI commentary. |
| **Expectation Convergence** | Track mean/std convergence with increasing sample size plus AI insights. |
| **Sobol / FAST / ANCOVA / Taylor** | Variance-based and local sensitivity analyses with Plotly visualizations and textual explanations. |
| **HSIC & Correlation** | Kernel and Pearson/Spearman methods for dependency assessment. |
| **ML Shapley** | Train a surrogate and compute Shapley values for attribution. |
| **Morris Screening** | Efficient screening to spot non-influential inputs. |
| **Surrogate Modeling (PCE)** | Build polynomial chaos expansions and run validation, cross-validation, Sobol’, conditional expectation, and coefficient diagnostics. |

Each module writes to `st.session_state.all_results`, so the sidebar chat and summary module always have the latest context.

---
## 🏗 Architecture at a Glance
- **Streamlit multipage app**: `UncertaintyCat.py` is a thin entrypoint; actual content lives under `pages/`.
- **Shared UI layer** (`app/components.py`): renders the sidebar, code editor, and chat widget, and ensures model selections persist across pages.
- **Application logic** (`app/core.py`, `app/config.py`, `app/state.py`): analysis registry, cached model compilation, orchestration helpers, and typed accessors around `st.session_state`.
- **Compute modules** (`modules/`): pure Python/OpenTURNS logic for each analysis, 100% Streamlit-free so they can be reused and unit tested.
- **Presentation layer** (`app/displays.py`): takes raw compute results and renders Plotly figures, tables, and text in the Streamlit UI.
- **Chat utilities** (`app/chat_utils.py`): builds Markdown context for both the per-module AI insights and the sidebar chat, enabling incremental Q&A.

---
## 💻 Run Locally with uv
### Prerequisites
- Python 3.12+
- Git
- [uv](https://github.com/astral-sh/uv) (modern Python package manager from Astral)

### Steps
1. **Clone the repo**
    ```bash
    git clone https://github.com/MLegkovskis/UncertaintyCat.git
    cd UncertaintyCat
    ```
2. **Install uv** (one time)
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # restart your shell so that `uv` is on PATH
    ```
3. **Create & activate the environment**
    ```bash
    uv venv
    source .venv/bin/activate       # Windows: .venv\Scripts\activate
    ```
4. **Install dependencies**
    ```bash
    uv pip sync pyproject.toml
    ```
5. **Optional – configure AI secrets**
    ```toml
    # .streamlit/secrets.toml
    GROQ_API_KEY = "your-key"
    ```
6. **Run the app**
    ```bash
    uv run streamlit run UncertaintyCat.py
    ```

### Generate a `uv.lock`
Community Cloud prefers a deterministic lock file. After installing uv:
```bash
uv pip compile pyproject.toml --output-file uv.lock
```
Commit `uv.lock` and use it locally with:
```bash
uv pip sync uv.lock
```

---
## ☁ Deploy to Streamlit Community Cloud
1. Push your repo with `pyproject.toml` and `uv.lock` at the root (or alongside `UncertaintyCat.py`).
2. In Streamlit Cloud, point the app to `UncertaintyCat.py`.
3. Community Cloud automatically detects `uv.lock`, installs dependencies with uv, and runs the app. No `requirements.txt` needed.
4. Need system packages? Add them to `packages.txt` (one Debian package per line) and commit to the repo root.

Tip: use the same Python version locally (3.12) to avoid surprises.

---
## 🗺️ Contributing & Roadmap
We welcome issues and pull requests! Current areas of interest:
- **Equation-to-model assistant** (convert LaTeX to OpenTURNS code).
- **Distributed backends** for >10⁶ sample Monte Carlo and Sobol runs.
- **Docker bundle** for on-prem deployments and offline LLM hosting.
- **New analyses** (Bayesian calibration, adaptive sampling, etc.).

How to contribute:
1. Fork the repo and create a feature branch.
2. Run `uv pip sync pyproject.toml` (or `uv.lock`) and `uv run streamlit run UncertaintyCat.py` to test.
3. Follow the existing style (type hints, docstrings, no Streamlit in compute modules).
4. Submit a PR targeting `main` and describe your changes.

---
## 🙏 Acknowledgments
- [OpenTURNS](https://openturns.github.io/openturns/latest/) for the numerical backbone.
- [Streamlit](https://streamlit.io/) for the rapid UI framework.
- [Groq Cloud](https://groq.com/) for powering the AI insights and chat assistant.
- The UQ community for open benchmarks (Ishigami, Borehole, etc.) that ship as built-in examples.

For background, see the [OpenTURNS User Day 2025 presentation](https://github.com/openturns/presentation/blob/master/userday2025/JU_OT_2025_UncertaintyCat.pdf).

---
## 📜 License
This project is released under the [MIT License](LICENSE).
