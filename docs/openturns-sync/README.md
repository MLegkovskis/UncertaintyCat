# Autonomous OpenTURNS capability synchronization

This is the operating manual and copy-paste agent brief for keeping UncertaintyCat scientifically aligned
with relevant OpenTURNS development. It was written after completing the workflow once for real: the
2026-08-21 intake reviewed OpenTURNS `1.27.post1` and the then-current upstream default branch, selected the
newly stable Gaussian-process regression API, implemented a bounded GPR surrogate plugin, added scientific,
service, browser, and production tests, and released it through the normal delivery path.

The objective is **useful product parity**, not class-for-class API parity. OpenTURNS contains numerical
building blocks, internal infrastructure, experimental research APIs, process methods, plotting helpers, and
specialized workflows that do not all belong in a general `f(x)` web application. A successful scheduled run
may ship one carefully chosen capability, improve an existing plugin, record a justified no-op, or open a
well-evidenced follow-up. It must never add a feature merely to make the automation appear active.

## 1. Non-negotiable operating contract

Every autonomous run must obey these rules:

1. OpenTURNS remains the numerical authority. UncertaintyCat provides safe configuration, applicability
   checks, orchestration, provenance, serialization, visualization, and interpretation boundaries.
2. Start from the exact dependency pin in `pyproject.toml`, not from assumptions about “latest.” Compare it
   separately with the latest stable release and the current upstream development branch.
3. Inspect the existing plugin catalog before proposing work. Do not create a duplicate under a different
   label and do not reimplement an OpenTURNS algorithm in NumPy when a suitable stable upstream API exists.
4. Select at most one coherent scientific capability per scheduled run. A tightly coupled correction to an
   existing plugin counts as one capability.
5. Stable public OpenTURNS APIs are eligible by default. Experimental APIs are discovery signals, not
   production dependencies. Adopt one only after explicit human approval and a written fallback plan.
6. The method must be meaningful for the product: a user supplies an OpenTURNS `model` and input `problem`,
   runs an analysis, and receives understandable, reproducible evidence.
7. Every computation must be bounded. Reflect algorithmic complexity in config limits and in the UI default;
   do not merely rely on a task timeout.
8. Mathematical applicability is executable behavior. Reject invalid copulas, distributions, dimensions,
   outputs, or degenerate samples; do not return plausible-looking nonsense.
9. Persisted results must remain strict JSON: no NumPy values, `NaN`, infinity, opaque OpenTURNS objects, or
   unbounded inline samples.
10. Do not change an existing plugin's interpretation under the same version. Increment the plugin version for
    numerical/default/meaning changes and the schema version for incompatible payload changes.
11. Do not weaken a benchmark or widen a tolerance solely to make a proposed change pass.
12. Do not expose credentials in logs, commits, artifacts, issue bodies, or prompts.
13. Preserve unrelated working-tree changes. Never reset, clean, or overwrite user work.
14. Push `main` only after all applicable local gates pass. CI is the release authority; production deploys
    only after successful CI. Fix failures forward and never rewrite deployed history.
15. A clear, evidence-backed no-op is a first-class result when nothing crosses the admission threshold.

## 2. Current system map

Read these files before designing anything:

| Concern | Source of truth |
| --- | --- |
| OpenTURNS pin and Python dependencies | `pyproject.toml`, `uv.lock` |
| Model source contract and compilation | `uncertaintycat_core/model.py` |
| Request/result JSON contracts | `uncertaintycat_core/contracts.py` |
| Plugin protocol and catalog metadata | `uncertaintycat_core/plugins/base.py` |
| Registered analyses | `uncertaintycat_core/catalog.py` |
| Numerical implementations | `uncertaintycat_core/plugins/*.py` |
| Provenance and deterministic seeds | `uncertaintycat_core/runner.py` |
| Compute HTTP/process boundary | `services/compute/main.py`, `services/compute/cli.py` |
| Run composer and safe UI defaults | `apps/web/src/pages/Workspace.tsx` |
| Generic result rendering | `apps/web/src/pages/Report.tsx` and report components |
| API/queue/D1/R2 orchestration | `apps/api/src/` |
| Scientific regression tests | `tests/core/test_plugins.py` |
| Service boundary tests | `tests/integration/test_compute_service.py` |
| Mocked browser suite | `apps/web/e2e/ui-flows.spec.ts` |
| Real local full-stack suite | `apps/web/e2e/full-stack/journey.spec.ts` |
| Live non-mutating/mutating checks | `apps/web/e2e/production/smoke.spec.ts` |
| CI and post-CI production deployment | `.github/workflows/ci.yml`, `.github/workflows/deploy.yml` |
| Review checkpoint | `docs/openturns-sync/state.json` |

The extension path is deliberately narrow:

```text
OpenTURNS stable API
        |
        v
AnalysisPlugin + strict config  --> common AnalysisPayload
        |                               |
        v                               v
catalog JSON --> run composer      generic report renderer
        |                               |
        +---------- API / Queue / D1 / R2 / export / report chat
```

Most additions should require one Python plugin, catalog registration, a safe config mapping, and tests.
Queueing, persistence, exports, sharing, and report chat already consume the generic envelope.

### Current capability inventory

At this checkpoint the application has 12 plugins:

| Key | Capability | Important boundary |
| --- | --- | --- |
| `monte_carlo` | input propagation and output statistics | bounded inline samples |
| `eda` | summaries and input/output correlation matrices | descriptive, not causal |
| `correlation` | Pearson, Spearman, partial, and regression effects | association is not variance attribution |
| `sobol` | Saltelli first/total/second-order indices | independent inputs only |
| `fast` | Fourier amplitude sensitivity indices | independent inputs only |
| `hsic` | kernel dependence and permutation evidence | dependence is not causality |
| `taylor` | local derivative variance decomposition | local linear approximation |
| `morris` | elementary-effect screening | independent marginals; screening interpretation |
| `convergence` | running expectation evidence | finite-sample convergence evidence |
| `reliability` | FORM or Monte Carlo threshold probability | method-specific event geometry |
| `pce` | polynomial-chaos surrogate and hold-out validation | validation Q2 controls trust |
| `gpr` | GP surrogate, hold-out accuracy, and conditional intervals | continuous inputs; bounded exact GPR |

The legacy Streamlit `modules/` and `pages/` folders are migration references, not the target architecture.

## 3. Upstream authority hierarchy

Use primary upstream sources in this order and record every URL/commit consulted:

1. Installed/pinned API: introspection in the exact `uv.lock` environment. This is what production can run.
2. Release evidence: OpenTURNS `ChangeLog`, GitHub release/tag, and PyPI stable version.
3. Version-matched documentation and examples for the pinned or candidate tag.
4. Upstream declarations and Python/C++ tests for exact contracts and edge behavior.
5. The default branch for future discovery, deprecations, and changes to `openturns.experimental`.
6. Issues and pull requests for rationale and known limitations; discussion is not an API guarantee.
7. Secondary articles only as supplementary scientific context.

Canonical entry points:

- Documentation: <https://openturns.github.io/openturns/latest/>
- API index: <https://openturns.github.io/openturns/latest/genindex.html>
- Examples: <https://openturns.github.io/openturns/latest/examples/examples.html>
- Repository: <https://github.com/openturns/openturns>
- ChangeLog: <https://github.com/openturns/openturns/blob/master/ChangeLog>
- Releases: <https://github.com/openturns/openturns/releases>
- PyPI metadata: <https://pypi.org/pypi/openturns/json>

Review every major documentation family rather than searching only “sensitivity”: data analysis, statistical
tests, probabilistic modeling, distributions/copulas, design of experiments, surrogate modeling, reliability,
sensitivity, stochastic processes, functional modeling, calibration, optimization, and numerical methods.
Visualization is relevant only when it communicates material scientific evidence better than the generic
renderer. This is a systematic surface review, not a demand to read every source line. Narrow into code,
tests, theory, and examples after a candidate survives product screening.

## 4. Scheduled run procedure

### Phase A — establish a clean, exact baseline

```bash
git status --short
git branch --show-current
git rev-parse HEAD
git fetch origin main
git rev-parse origin/main
rg -n 'openturns==' pyproject.toml
uv sync --frozen --extra dev
uv run python -c 'import openturns as ot; print(ot.__version__)'
```

If the tree contains unrelated changes, preserve them and work around overlapping files. Stop before mutation
if the checkout cannot be reconciled safely. Record the UncertaintyCat commit and exact pin. Read the state
checkpoint and architecture:

```bash
sed -n '1,240p' docs/openturns-sync/state.json
sed -n '1,260p' docs/ARCHITECTURE.md
sed -n '1,260p' docs/ANALYSIS_PLUGIN_GUIDE.md
sed -n '1,280p' docs/SCIENTIFIC_VALIDATION.md
sed -n '1,220p' uncertaintycat_core/catalog.py
```

The state file is a navigation checkpoint and may lag the code. Derive the actual catalog and reconcile it.

### Phase B — capture upstream deltas reproducibly

Use a temporary checkout outside the repository; never vendor upstream into UncertaintyCat:

```bash
OPENTURNS_SCAN_DIR="$(mktemp -d)"
git clone --filter=blob:none https://github.com/openturns/openturns.git "$OPENTURNS_SCAN_DIR"
git -C "$OPENTURNS_SCAN_DIR" rev-parse HEAD
git -C "$OPENTURNS_SCAN_DIR" log -1 --format='%H%n%cI%n%s'
git -C "$OPENTURNS_SCAN_DIR" tag --sort=-v:refname | head -20
```

Read the stable PyPI version without adding dependencies:

```bash
uv run python .github/scripts/openturns_scout.py
uv run python - <<'PY'
import json
import urllib.request

with urllib.request.urlopen("https://pypi.org/pypi/openturns/json", timeout=30) as response:
    print(json.load(response)["info"]["version"])
PY
```

Compare three distinct ranges: last checkpoint to upstream `HEAD`; pinned to latest stable release; and the
installed pin to any proposed candidate pin. Useful queries:

```bash
git -C "$OPENTURNS_SCAN_DIR" diff --stat <last-reviewed>..HEAD
git -C "$OPENTURNS_SCAN_DIR" diff <last-reviewed>..HEAD -- ChangeLog python/doc python/test lib/test
rg -n 'New classes|New methods|left the experimental|Deprecated|Removed|Changes' \
  "$OPENTURNS_SCAN_DIR/ChangeLog"
find "$OPENTURNS_SCAN_DIR/python/doc/examples" -maxdepth 3 -type d | sort
```

For each candidate inspect its manual page, theory, example, declaration, and tests. Record whether it exists
in the installed pin:

```bash
uv run python - <<'PY'
import openturns as ot
import openturns.experimental as experimental

for name in ["CandidateClass"]:
    print(name, hasattr(ot, name), hasattr(experimental, name))
PY
```

Moving from `experimental` to the main namespace is a maturity signal. Entering `experimental` is normally a
watch-list event.

### Phase C — inventory the product, then build candidates

```bash
rg -o 'ot\.[A-Za-z_][A-Za-z0-9_]*' uncertaintycat_core tests examples \
  | sed 's/.*ot\./ot./' | sort -u
uv run python - <<'PY'
from uncertaintycat_core import analysis_catalog

for entry in analysis_catalog():
    print(entry.key, entry.version, entry.result_schema_version, entry.category)
PY
```

Candidate sources include a new stable analysis; a formerly experimental class; an estimator or variant of an
existing plugin; a correctness/deprecation/default change; a distribution/copula/model definition suitable
for the builder; a diagnostic, interval, convergence, or validation API; a stability/performance improvement;
a useful projection of a hidden result; or an upstream bug fix requiring a pin update and regression.

Do not turn documentation wording, test refactoring, build infrastructure, serialization internals, or a
specialized object unrelated to an arbitrary `f(x)` workflow into product work.

### Phase D — score before coding

Reject a candidate when any admission answer is “no”:

1. Is it meaningful for a user-defined OpenTURNS `model`/`problem`, or for defining that problem?
2. Is its scientific interpretation defensible and reportable?
3. Is there a stable API in the pin, or is a controlled pin upgrade justified?
4. Can compute/output size be bounded under the current sandbox?
5. Can it be validated against analytical, upstream, or published evidence?
6. Can invalid applicability be detected or clearly warned about?
7. Does it add capability rather than duplicate a plugin?

Score survivors from 0 to 5 and retain the table:

| Criterion | Weight | What earns 5 |
| --- | ---: | --- |
| User/scientific value | 25 | closes a common UQ decision gap |
| Product fit | 20 | maps naturally to generic result payloads |
| API maturity | 15 | stable, documented, tested, not pending removal |
| Validation strength | 15 | analytical/published benchmark with robust tolerances |
| Distinctiveness | 10 | materially expands the catalog |
| Operational safety | 10 | predictable complexity and hard bounds |
| Maintenance cost | 5 | small stable surface and clear upgrade path |

Use the weighted total to rank, but never bypass an admission gate. Normally ship only at 75/100 or above.
If nothing qualifies, update the checkpoint with reviewed candidates and finish with a no-op report.

### Phase E — write the scientific design before implementation

The design must answer: user question; exact stable APIs; dimensions and output policy; distribution/copula
assumptions; estimator/design/seed/convergence behavior; config bounds and complexity; trust diagnostics;
report labels and prohibited interpretations; evaluation count; errors/warnings; benchmark values/tolerances;
affected layers; and persisted-report compatibility.

Prototype against the installed pin before editing. Exercise a normal case, invalid/boundary case, and realistic
runtime at the proposed UI default.

## 5. Implementation playbook

### New analysis plugin

1. Add `uncertaintycat_core/plugins/<key>.py`.
2. Define a `StrictModel` config with hard, complexity-aware bounds. Include `seed` and `output_targets` where
   relevant.
3. Define permanent key, semantic version, schema version, name/category/description, durable assumptions,
   dependent/multi-output flags, and resource class.
4. In `applicability_warnings`, raise `IncompatibleAnalysisError` when interpretation is invalid; warn only
   when the result remains meaningful.
5. In `run`, use the parsed seed, validate targets, invoke stable OpenTURNS APIs, convert to JSON scalars, cap
   inline data, and return the exact evaluation count.
6. Translate upstream construction failures at the domain boundary, but never hide programmer defects inside
   a success payload.
7. Emit common fields: scalar `metrics`; bounded `tables` with true row count/truncation; paired `series`;
   labelled `matrices`; and grounded method/output `facts`.
8. Export `plugin` and register it in `uncertaintycat_core/catalog.py`.
9. Add scalar keys and an explicit safe config mapping in `Workspace.tsx`. Add custom controls only for a
   scientifically important choice not covered by generic budget/output controls.
10. Update all catalog counts and fixtures; find them rather than guessing:

```bash
rg -n '11|12|all .*plugins|analysis tasks|report sections|toHaveCount' \
  readme.md docs tests apps/web/e2e --glob '!**/node_modules/**'
```

### Existing analysis changed by upstream

- Reproduce old behavior with the old pin first.
- Add a regression that demonstrates the upstream change.
- Compare fixed-seed outputs across pins.
- Classify drift as intended improvement, changed default, bug fix, or regression.
- Set implicit defaults explicitly when old meaning must remain stable.
- Increment versions as required and document persisted-report compatibility.

### Dependency upgrade

Do not combine a broad pin upgrade with an unrelated feature unless required:

```bash
# Edit the exact pin in pyproject.toml with apply_patch, then:
uv lock --upgrade-package openturns
uv sync --frozen --extra dev
uv run python -c 'import openturns as ot; print(ot.__version__)'
```

Run all bundled model validations and scientific tests. Search ChangeLog for removed/deprecated names used in
core, tests, examples, and legacy code. Installation alone is not compatibility evidence.

## 6. Required evidence and tests

Core evidence: fixed-seed repeatability; an analytical/published/official benchmark; justified tolerances;
invalid applicability; output/dependence behavior; constant/non-finite behavior; config bounds; strict JSON;
evaluation accounting; table truncation; and realistic default/maximum runtime evidence where practical.

Product evidence: catalog schema; `/v1/execute`; mocked UI configuration/request; generic report rendering;
real local full-stack execution/report/export; production catalog; and, when authorized, one live mutation.

Mandatory local commands:

```bash
uv run ruff format --check uncertaintycat_core services tests test_all_examples.py
uv run ruff check uncertaintycat_core services tests test_all_examples.py
uv run mypy uncertaintycat_core services
uv run pytest --cov=uncertaintycat_core --cov=services --cov-report=term-missing
npm run typecheck
npm run test:ts
npm run build
npm run test:e2e
npm run test:e2e:full-stack
docker build -f services/compute/Dockerfile -t uncertaintycat-compute:verification .
docker build -f services/compute/Dockerfile.sandbox -t uncertaintycat-sandbox:verification .
```

Use Node 22. If system Node is older, invoke repository scripts through a Node 22 runtime. Do not skip a
failing layer because a lower layer passed. If a gate is unavailable, establish equivalent evidence or leave
the change unpushed and report the blocker.

## 7. Release, observation, and recovery

Before committing:

```bash
git status --short
git diff --check
git diff --stat
git diff
```

Verify every changed file belongs to the capability. Update `state.json` with the exact stable release,
upstream commit/date, catalog count, selected result, and deferred/rejected candidates. Commit explicit files
and push directly to `main` only after the complete local gate:

```bash
git add <explicit files>
git commit -m "feat: add <capability>"
git push origin main
```

The push triggers `.github/workflows/ci.yml`. Successful CI on `main` triggers `deploy.yml`, which builds the
assets, applies forward-only D1 migrations, deploys the Worker and compute Sandbox image, checks `/health`, and
runs production browser verification. The Worker can become healthy before the new Sandbox image has converged;
the production suite therefore polls for the release-specific catalog contract within a bounded window.
Observe both workflows to terminal success. Then verify:

```bash
curl --fail --silent --show-error https://uncertaintycat.com/health
curl --fail --silent --show-error https://uncertaintycat.com/api/v1/analyses/catalog
npm run test:e2e:production
```

Only run mutation-enabled production tests when disposable production data is authorized:

```bash
E2E_LIVE_MUTATIONS=true npm run test:e2e:production
```

If CI fails, inspect exact logs, reproduce, fix forward, rerun the affected complete gate, and push a new
commit. If deployment fails, never reverse a D1 migration or force-push. Diagnose and fix forward. For an app
regression without irreversible data change, redeploy a known-good Worker revision or create a tested revert
commit. Never use destructive Git commands against user work.

## 8. Required run report

Every run must return an auditable report containing:

1. UncertaintyCat starting/ending commit and OpenTURNS pin.
2. Latest stable version and exact upstream commit/date reviewed.
3. Release ranges, documentation categories, examples, tests, issues/PRs consulted.
4. Current catalog and any upstream deprecation affecting it.
5. Candidate table with admission decisions and weighted scores.
6. Selected capability or explicit no-op rationale.
7. Scientific design, assumptions, bounds, and prohibited interpretation.
8. Exact files changed and plugin/schema version decision.
9. Test commands/results, benchmark values, and runtime.
10. Commit/push, CI, deployment, and production verification status.
11. Deferred risks and best next candidate, without calling it pre-approved.

Scope claims to evidence. Prefer “systematically reviewed all documented categories, then deeply inspected
three candidates” over an unverifiable claim to have read every API source line.

## 9. Completed rehearsal: Gaussian Process Surrogate

### Discovery and choice

OpenTURNS 1.27 moved `GaussianProcessFitter`, `GaussianProcessRegression`, their result classes,
`GaussianProcessConditionalCovariance`, and `PosteriorDistribution` out of `openturns.experimental`. The older
`Kriging*` family was deprecated. OpenTURNS 1.26 had also improved GPR optimization normalization and bounds.
UncertaintyCat offered polynomial chaos but no GP surrogate, making this a clear release-linked gap.

Other candidates were recorded rather than lost:

- `ANCOVA`: strong dependent-input sensitivity candidate, but less directly tied to this release delta;
- `RankSobolSensitivityAlgorithm`: stable and interesting, but needs a distinct interpretation/benchmark;
- `GaussianProcessRegressionCrossValidation`: newly visible but still experimental, so not admitted.

### Shipped design

The `gpr` plugin uses only stable 1.27 APIs:

- `GaussianProcessFitter` for covariance/trend fitting;
- `GaussianProcessRegression` for the conditional metamodel;
- `GaussianProcessConditionalCovariance` for model-based conditional variance;
- `MetaModelValidation` on an independently sampled hold-out set;
- Matérn 3/2, Matérn 5/2, or squared-exponential covariance;
- constant or linear trend bases.

Version 1 is scalar-output and continuous-input only. It supports dependent continuous distributions because
it samples the declared joint distribution and does not interpret dimensions as independent variance
components. Exact GPR is cubic in training size, so training is capped at 512 points (default 128); validation
and inline prediction rows have separate caps.

The report contains hold-out R2, RMSE, MAE, interpolation RMSE, nominal 95% conditional-interval coverage,
optimized likelihood/amplitude/nugget, covariance scales, trend coefficients, a bounded prediction table, and
observed-versus-predicted series. Assumptions prevent the conditional intervals from being described as
guaranteed frequentist confidence intervals. Independent hold-out validation deliberately replaces the still
experimental upstream cross-validation class.

The reusable evidence pattern is:

- smooth `sin(x1) + 0.5*x2^2` validation for accuracy and repeatability;
- correlated-normal linear response for dependent inputs and the linear trend;
- discrete and constant-output rejection;
- Ishigami strict serialization/all-catalog execution;
- FastAPI, mocked UI, local full stack, and production verification.

This rehearsal is a process pattern, not a reason to prefer GPR-shaped features later.

## 10. Copy-paste scheduled-agent prompt

```text
You are the autonomous OpenTURNS capability-maintenance agent for UncertaintyCat.

Objective:
Keep UncertaintyCat meaningfully aligned with stable, relevant OpenTURNS development. UncertaintyCat is an
unofficial modern UI and reproducible entry point for applying uncertainty quantification to an arbitrary
OpenTURNS model f(x). Seek useful product parity, never mechanical class-for-class parity.

Before acting, read completely:
- docs/openturns-sync/README.md
- docs/openturns-sync/state.json
- docs/ARCHITECTURE.md
- docs/ANALYSIS_PLUGIN_GUIDE.md
- docs/SCIENTIFIC_VALIDATION.md
- pyproject.toml
- uncertaintycat_core/catalog.py
- uncertaintycat_core/plugins/base.py
- uncertaintycat_core/contracts.py
- uncertaintycat_core/runner.py
- apps/web/src/pages/Workspace.tsx
- .github/workflows/ci.yml
- .github/workflows/deploy.yml

Perform one complete intake cycle:

1. Inspect git status, branch, local HEAD, origin/main, and exact OpenTURNS pin. Preserve unrelated changes.
   Never reset, clean, force-push, or overwrite user work.
2. Fetch primary upstream evidence from OpenTURNS docs, API index, examples, ChangeLog, repository/default
   branch, releases/tags, PyPI, declarations, and tests. Clone upstream into a temporary directory. Record the
   exact commit and date.
3. Compare (a) prior checkpoint to current upstream HEAD, (b) pinned to latest stable release, and (c) current
   UncertaintyCat plugin/API use to new, stabilized, deprecated, changed, or removed upstream capability.
4. Review every major OpenTURNS documentation category, then deeply inspect serious candidates. Consider new
   analyses, improvements/corrections, problem definitions, diagnostics, and safe performance work. Ignore
   work irrelevant to an arbitrary f(x) web workflow.
5. Build and score candidates using the README admission gates and 100-point rubric. Do not use experimental
   production APIs without explicit human approval. Do not duplicate a plugin or force activity.
6. Select at most one capability. Before editing, document the user question, stable APIs, assumptions,
   copula/distribution/output applicability, complexity, bounds, evidence, prohibited interpretation,
   evaluation count, benchmark, and versioning.
7. Prototype the exact installed API. If a pin upgrade is required, isolate it, regenerate uv.lock, inspect
   deprecations/removals, and do not combine it with unrelated work.
8. Implement through AnalysisPlugin. Prefer stable OpenTURNS algorithms over custom numerical work. Emit
   strict bounded generic payloads. Add safe UI/catalog wiring and preserve persistence/export/chat contracts.
9. Add fixed-seed, benchmark, applicability, dependence/output, degeneracy, strict JSON, accounting, service,
   mocked browser, real full-stack, and production catalog evidence as applicable.
10. Update readme.md, scientific/method docs, every catalog count, and state.json. Record deferred/rejected
    candidates so the next run does not rediscover them without new evidence.
11. Run every Python, TypeScript, build, Playwright, full-stack, and container gate in the README. Inspect the
    full diff. Never weaken evidence merely to pass.
12. Only when all gates pass, commit explicit files and push directly to main. Observe GitHub CI and the
    post-CI Cloudflare deployment to terminal success. Verify health, catalog, public UI, and new capability.
13. Diagnose failures and fix forward. Never rewrite deployed history or reverse D1 migrations. If completion
    is unsafe, leave the repository recoverable, do not push unverified code, and report the exact blocker.
14. Return the section 8 evidence report with direct primary-source links, files, commits, workflow results,
    benchmark values, and remaining risks.

A justified no-op is success. Scientific meaning, stable APIs, bounded execution, regression evidence, and
production integrity matter more than feature count.
```

## 11. Scheduling recommendation

Run the lightweight release scout weekly and this deeper cycle weekly or fortnightly. Daily deep scans tend to
rediscover development-branch churn unless state-based no-op exits are cheap. Serialize scheduled agents so
two runs cannot select or deploy from one baseline. Release concurrency should not cancel an in-progress
deployment.

Even for a no-op, update the checkpoint only when new upstream commits or candidates were genuinely reviewed.
That keeps future diffs precise without fabricating repository activity.
