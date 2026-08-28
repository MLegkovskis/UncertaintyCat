# UncertaintyCat agent instructions

## Product contract

- UncertaintyCat is a modern web interface and extension layer for OpenTURNS. OpenTURNS is the numerical authority; UncertaintyCat adds safe model handling, orchestration, provenance, persistence, visualization, and clearly labelled AI interpretation.
- Authentication is a hard product boundary. Only the static application shell/assets, `/`, `/health`, `/api/v1/session`, and `/api/auth/*` are public. Every model, example source, analysis catalog, dataset, study, run, report, shared report, export, surrogate, Model Understanding request, and chat request requires an authenticated Cloudflare identity. Enforce this in the Worker as well as the UI.
- The public homepage may contain static explanatory examples. It must not fetch private catalogs, expose executable source, create guest state, or offer anonymous computation.
- Keep numerical results deterministic and independently inspectable. AI may explain persisted results through bounded read-only tools; it must never become the source of numerical truth.

## Repository map and authority

- `uncertaintycat_core/`: framework-independent Python model validation, strict contracts, orchestration, and analysis plugins.
- `services/compute/`: private FastAPI/CLI adapter used inside the Cloudflare Sandbox boundary.
- `apps/api/`: Hono Cloudflare Worker, Better Auth, D1/R2/Queue ownership and lifecycle, exports, sharing, and the deployment-selectable Groq/Workers AI adapter.
- `apps/web/`: React/Vite UI, route-level auth gate, project dashboard, direct-analysis workspace, reduction/surrogate/distribution studios, reports, and browser tests.
- `packages/contracts/`: shared Zod schemas, API client, types, and generated reference-model catalog.
- `examples/`: canonical executable reference models used to generate the authenticated catalog.
- `apps/api/migrations/`: forward-only D1 migrations. Never rewrite an applied migration.
- `Streamlit_Backup/`: read-only historical reference imported from `main_bk_2026_Aug`. Do not build new functionality there or reintroduce it into the modern package graph.
- `docs/openturns-sync/README.md`: authoritative scheduled-agent process for evaluating upstream OpenTURNS changes.
- `docs/openturns-sync/evidence/`: machine-readable, pinned-source scientific and resource evidence required for every new or changed analysis plugin.

## Working rules

- Inspect `git status` and relevant diffs before editing. Preserve unrelated user changes and never reset the worktree.
- Extend numerical capability through `uncertaintycat_core.plugins.base.AnalysisPlugin`; do not put algorithms in React or the Worker.
- Prefer OpenTURNS APIs over parallel SciPy, scikit-learn, or handwritten implementations when OpenTURNS provides the relevant method.
- A plugin change requires strict JSON output, applicability checks, deterministic fixed-seed evidence, a scientific benchmark, catalog registration, safe UI defaults, documentation/version updates, and a refreshed `docs/openturns-sync/evidence/<plugin>.json` manifest. Complexity tests must use a test-side oracle independent of the production estimator and cover the first rejected resource boundary plus the UI default at the maximum supported dimension.
- Validation must produce a recommendation for every registered plugin. Model-level incompatibilities and resource-safe configuration belong in the deterministic assessment, must disable the UI with an exact reason, and must be enforced again by core/Worker boundaries rather than trusted from React.
- Long-running analyses must emit bounded, source-free progress phases. Persist progress per task; use indeterminate state for opaque OpenTURNS calls and never fabricate an exact percentage that the upstream algorithm does not expose.
- Keep Python source and private artifacts out of logs, client bundles, AI prompts, and public routes. Never commit secrets or real `.dev.vars` files.
- Use forward-only D1 migrations and immutable model/result provenance. Do not silently reinterpret an existing plugin key/version or stored result schema.
- Do not add convenience shell launchers or CI pause/skip controls. Standard package-manager test/dev commands and Playwright-managed test servers are allowed.
- Update the relevant README or `docs/` file whenever architecture, security boundaries, public behavior, operations, or scientific interpretation changes.

## Verification

Run focused tests while iterating, then the relevant gates before committing:

```bash
npm run check:examples
npm run check:scientific-change
npm run typecheck
npm run test:ts
npm run build
uv run ruff format --check uncertaintycat_core services tests scripts .github/scripts test_all_examples.py
uv run ruff check uncertaintycat_core services tests scripts .github/scripts test_all_examples.py
uv run mypy uncertaintycat_core services scripts/check_scientific_change.py .github/scripts/openturns_scout.py
uv run pytest
uv run python test_all_examples.py
npm run test:e2e
```

Run `npm run test:e2e:full-stack` for changes that cross the browser, Worker, D1/R2/Queue, compute protocol, or plugin execution boundary. Build both compute images when their Dockerfiles or runtime dependencies change.

## Delivery

- `main` is the delivery branch. For user-authorized implementation work, commit and push directly to `main`; do not open a pull request unless asked.
- Every push to `main` must run the complete CI workflow. Successful CI for that exact commit triggers production deployment and production Playwright verification. Never add path filters, repository-variable gates, commit-message skips, or manual-only release logic that weakens this chain.
- Dependabot updates are approved and squash-merged only by the post-CI workflow after all trust, signed-commit, ecosystem file-scope, merge-readiness, and exact-SHA checks in `docs/DEPENDENCY_AUTOMATION.md` pass. The bot-dispatched post-merge CI explicitly dispatches deployment for its exact successful SHA because `GITHUB_TOKEN` suppresses chained workflow events. Do not broaden this path to agent/source changes, weaken its stale-security exemptions, or remove that explicit release handoff.
- After pushing, verify the CI run is attached to the expected commit. If CI or deployment fails, inspect the logs, fix the cause, rerun local evidence, and push the repair.

## Code review rules

- Flag any route or handler that permits unauthenticated access outside the explicit public allowlist.
- Flag numerical behavior that bypasses OpenTURNS without a documented technical reason and benchmark evidence.
- Flag plugin changes without refreshed pinned-source evidence, an independently derived complexity oracle, maximum-bound rejection, and UI/core resource-budget alignment.
- Flag AI text presented as computed evidence, unbounded AI tools, or model source sent to any AI provider.
- Flag mutable provenance, destructive migration edits, non-finite JSON, missing ownership predicates, exposed secrets, or reduced CI coverage.
