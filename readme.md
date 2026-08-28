# UncertaintyCat

[UncertaintyCat](https://uncertaintycat.com) is an open-source uncertainty-quantification workspace built on [OpenTURNS](https://openturns.github.io/openturns/latest/). It turns a versioned OpenTURNS model into reproducible analyses, durable engineering studies, interactive reports, portable evidence bundles, and tool-grounded report conversations.

The public site is a static product overview. Authentication through Cloudflare is required before the application exposes an analysis catalog or reference-model source, validates a model, accepts a dataset, starts computation, or reads any retained result.

## Current capabilities

- Immutable Python and symbolic-builder model versions with exact source hashes, deterministic assessment, lineage, and OpenTURNS provenance.
- Twenty-three hash-checked reference models available inside authenticated workspaces.
- Twelve versioned analysis plugins: uncertainty propagation by Monte Carlo sampling, exploratory data analysis, correlation, Sobol, FAST, HSIC, Taylor, Morris, expectation convergence, reliability, polynomial chaos, and Gaussian-process regression.
- Multi-output propagation and exploration with explicit output targeting for scalar methods.
- D1-backed studies, datasets, runs, task state, reports, quotas, model explanations, and report conversations.
- R2-backed immutable model source, private uploaded data, promoted model-based surrogates, and retained data-driven OpenTURNS Gaussian-process artifacts.
- Queue-driven, idempotent computation in isolated Cloudflare Sandbox containers with retry, cancellation, and partial-result handling.
- Apache ECharts reports with exact data fallbacks, direct browser PDF downloads, ZIP/JSON/CSV exports, and authenticated read-only share links.
- Study-scoped distribution fitting, explicit Morris-derived model versions, promoted PCE/GPR surrogates, and reliability guidance.
- Deployment-selectable Groq or Cloudflare Workers AI explanations through the open-source Vercel AI SDK. Groq is the default; AI can query bounded persisted-result projections but cannot run calculations, see Python source, or mutate evidence.
- Weekly OpenTURNS release discovery plus full Python, TypeScript, browser, local-stack, and image CI on every push to `main`.

## Product workflow

Authenticated users land on a single project index. Creating a project opens its overview; models, direct analyses, dimensionality reduction, surrogate construction, distribution fitting, and previous runs then remain visibly scoped to that project. A new model combines the 23 reference models and the editable Python model into one authoring surface; the model name starts blank, and choosing a reference loads both its name and source unless the name was already edited manually. The Guided Builder compiles its named formulas and distributions into an OpenTURNS [`SymbolicFunction`](https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.SymbolicFunction.html).

`Validate & Assess` performs isolated sample evaluation and records a deterministic next-step route:

- direct analysis when the measured projection makes repeated model evaluation practical;
- Dimensionality Reduction Studio for models with at least 15 inputs, using OTMorris screening; or
- Surrogate Studio when 1,000 direct evaluations project above five seconds and baseline surrogate eligibility is established.

The threshold decision is versioned numerical metadata, not an AI decision. The selected AI provider gives a separate concise explanation from validated metadata. Validation status, deterministic facts, the AI brief, and the recommended route share one Model Understanding panel. Direct analysis intentionally excludes Morris, PCE, and GPR: those methods live in dedicated project studios. A reduced model or promoted surrogate then offers two explicit, provenance-preserving handoffs: start another analysis in the current project, or copy the model (and surrogate artifact when applicable) into a new project.

Surrogate Studio accepts either a saved model with a declared input distribution or empirical rows containing paired inputs and outputs. The latter path fits and validates an OpenTURNS Gaussian-process regression model, retains its XML artifact in R2, and makes clear that an input distribution must still be defined before uncertainty propagation or sensitivity analysis. Distribution Fitting includes a meaningful simply-supported-beam sample (`E`, `F`, `L`, `I`) for exploration.

The catalog calls generic random-sampling output analysis **Uncertainty Propagation**, not “Monte Carlo analysis.” Monte Carlo is its sampling design. OpenTURNS expectation convergence remains a separate method with a distinct stopping criterion and convergence evidence.

## Architecture

```text
React/Vite web  ->  Hono Worker API  ->  D1 metadata
                         |           ->  R2 sources and artifacts
                         |           ->  Queue task lifecycle
                         |           ->  Selected AI provider narrative
                         v
                 Cloudflare Sandbox  ->  uncertaintycat_core  ->  OpenTURNS
```

The stable numerical extension point is `uncertaintycat_core.plugins.base.AnalysisPlugin`. Plugins declare strict configuration, applicability, assumptions, resource class, implementation version, and a common JSON-safe result envelope. The Worker and UI consume the catalog rather than importing algorithm-specific code.

Read [Architecture](docs/ARCHITECTURE.md), [plugin development](docs/ANALYSIS_PLUGIN_GUIDE.md), [security](docs/SECURITY.md), [scientific validation](docs/SCIENTIFIC_VALIDATION.md), [testing](docs/TESTING.md), [deployment](docs/DEPLOYMENT.md), and the [dependency automation trust model](docs/DEPENDENCY_AUTOMATION.md). The upstream evaluation and scheduled-agent procedure is in the [OpenTURNS synchronization guide](docs/openturns-sync/README.md). Repository-wide agent rules are in [AGENTS.md](AGENTS.md).

## Quality gates

Prerequisites are Python 3.12, [uv](https://docs.astral.sh/uv/), Node.js 22, and npm. There is intentionally no repository launcher script; package commands and Playwright own the test processes they need.

```bash
uv sync --frozen --extra dev
npm ci
npm run check:examples
npm run typecheck
npm run test:ts
npm run build
uv run ruff format --check uncertaintycat_core services tests test_all_examples.py
uv run ruff check uncertaintycat_core services tests test_all_examples.py
uv run mypy uncertaintycat_core services
uv run pytest
uv run python test_all_examples.py
npm run test:e2e
npm run test:e2e:full-stack
```

The suites cover all reference models, strict serialization, multi-output behavior, dependency restrictions, sensitivity/reliability benchmarks, surrogate validation, the compute HTTP boundary, authenticated UI journeys, D1/R2/Queue persistence, report/export behavior, accessibility, and the deployed authentication boundary.

## Model contract

Executable source defines exactly the OpenTURNS objects `model` and `problem`:

```python
import openturns as ot

model = ot.SymbolicFunction(["x1", "x2"], ["x1 + x2^2"])
model.setOutputDescription(["response"])
problem = ot.JointDistribution([ot.Normal(), ot.Uniform(-1.0, 1.0)])
problem.setDescription(["normal_input", "uniform_input"])
```

AST preflight provides useful feedback; it is not a security boundary. User-authored Python is executed only after authentication and inside the isolation controls described in [docs/SECURITY.md](docs/SECURITY.md).

## Delivery

Every push to `main` starts the complete CI workflow. A successful CI run for that exact commit automatically applies forward-only D1 migrations, deploys the Worker, static assets, bindings, queue consumer, and Sandbox image to `uncertaintycat.com`, then runs production Playwright verification. There is no repository-wide pause variable or manual release script.

The original Streamlit source is preserved only as a historical reference in `Streamlit_Backup/`; it is excluded from the modern dependency graph and delivery path.

Projects can be permanently deleted only after typing the exact project name. Deletion removes the project's D1 graph and its private R2 model, dataset, and surrogate artifacts; it does not weaken account-wide quota records.

## License

MIT. See [LICENSE](LICENSE).
