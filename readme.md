# UncertaintyCat

UncertaintyCat is an open-source uncertainty-quantification workspace built on
[OpenTURNS](https://openturns.github.io/openturns/latest/). It turns a versioned Python model into
reproducible analysis runs, interactive reports, portable evidence bundles, and tool-grounded report
conversations.

This branch contains the successor to the original Streamlit application. The legacy entry point is
still present for comparison, while the new system is split into a React application, a Cloudflare
Worker-compatible control plane, and an isolated Python compute service.

## What is implemented

- Immutable model versions with input/output metadata, source hashes, and OpenTURNS provenance.
- Twelve schema-driven analysis plugins: Monte Carlo, EDA, correlation, Sobol, FAST, HSIC, Taylor,
  Morris, expectation convergence, reliability, polynomial chaos, and Gaussian-process regression.
- Multi-output propagation and exploration; scalar-output methods require an explicit output target.
- Durable projects, queued/idempotent runs, partial-failure reports, cancellation, quotas, and retries.
- Interactive native SVG/CSS results, browser-print PDF reports, ZIP JSON/CSV evidence bundles, and expiring,
  revocable share links.
- Cloudflare-hosted Workers AI chat, built on the open-source Vercel AI SDK and Cloudflare provider,
  constrained to five read-only persisted-report tools; it does not execute numerical calculations.
- Better Auth wiring for Cloudflare Access OIDC login, D1-backed cross-device history, persisted report
  conversations, and restricted guest access to server-approved examples.
- A weekly OpenTURNS release scout and CI gates for Python, TypeScript, browser smoke tests, and the
  compute image.

## Architecture

```text
React/Vite web  ->  Hono Worker API  ->  D1 metadata
                         |           ->  R2 model/artifact storage
                         |           ->  Queue task lifecycle
                         v
                 disposable Cloudflare Sandbox  ->  uncertaintycat_core  ->  OpenTURNS
```

The stable extension point is `uncertaintycat_core.plugins.base.AnalysisPlugin`. Each plugin owns a
strict configuration model and emits the same JSON-safe result envelope; the web UI consumes the
catalog instead of importing algorithm-specific Python.

See [Architecture](docs/ARCHITECTURE.md), [plugin guide](docs/ANALYSIS_PLUGIN_GUIDE.md),
[security model](docs/SECURITY.md), [scientific validation](docs/SCIENTIFIC_VALIDATION.md), and
[deployment runbook](docs/DEPLOYMENT.md). The complete upstream review and scheduled-agent operating
procedure is in the [OpenTURNS synchronization README](docs/openturns-sync/README.md).

## Local development

Prerequisites: Python 3.12, [uv](https://docs.astral.sh/uv/), Node 22, npm, and optionally Docker.

```bash
uv sync --frozen --extra dev
npm ci
cp apps/api/.dev.vars.example apps/api/.dev.vars
npm run dev:compute
```

In separate terminals:

```bash
npm run dev:api
npm run dev:web
```

Open `http://127.0.0.1:5173`. The local Wrangler configuration enables a development identity. To
exercise the hardened container boundary instead of the host compute process:

```bash
docker compose up --build compute
```

## Quality gates

```bash
npm run typecheck
npm run test:ts
npm run build
uv run ruff format --check uncertaintycat_core services tests test_all_examples.py
uv run ruff check uncertaintycat_core services tests test_all_examples.py
uv run pytest
npm run test:e2e       # first run: npx playwright install chromium
```

The Python suite validates every model under `examples/`, contract strictness, multi-output behavior,
dependent-input rejection, the known Ishigami Sobol structure, Gaussian-process validation on smooth and
dependent-input benchmarks, FORM on a standard-normal half-space, all catalog plugins, and the FastAPI
boundary.

## Model contract

A model source defines exactly the OpenTURNS objects `model` and `problem`:

```python
import openturns as ot

model = ot.SymbolicFunction(["x1", "x2"], ["x1 + x2^2"])
model.setOutputDescription(["response"])
problem = ot.JointDistribution([ot.Normal(), ot.Uniform(-1.0, 1.0)])
problem.setDescription(["normal_input", "uniform_input"])
```

The AST preflight is useful feedback, not a sandbox. Never expose the compute service to untrusted
custom source without the isolation controls described in the security document.

## Deployment status

The production Worker/static-assets configuration, D1 migration, R2/Queue bindings, Workers AI binding,
Sandbox image, custom-domain routes, and post-CI deployment workflow are implemented. The live Cloudflare
account has D1, R2, Queues, Workers Paid, the Cloudflare Access OIDC application, and a fully configured
GitHub production environment. Successful CI on `main` triggers the production deployment and live
non-mutating browser verification. The domain is never deliberately advanced to a build that cannot validate
or execute models.

## Legacy application

The original Streamlit application remains runnable during migration:

```bash
uv run streamlit run UncertaintyCat.py
```

New analysis work should target `uncertaintycat_core`; the legacy `modules/` package is retained as a
behavior reference and migration source.

Install the legacy-only visualization and Streamlit dependencies before running it:

```bash
uv sync --extra legacy
```

## License

MIT. See [LICENSE](LICENSE).
