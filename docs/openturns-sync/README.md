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
14. Use the delivery mode explicitly authorized for the run. Direct user-authorized implementation may follow
    `AGENTS.md`; the scheduled ChatGPT workflow must use a feature branch and pull request, must not self-merge,
    and must not deploy production. In every mode CI remains authoritative and failures are fixed forward.
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
| Generic result rendering | `apps/web/src/pages/ReportPage.tsx` and report components |
| API/queue/D1/R2 orchestration | `apps/api/src/` |
| Scientific regression tests | `tests/core/test_plugins.py` |
| Pinned per-plugin evidence and resource audits | `docs/openturns-sync/evidence/*.json` |
| Diff-aware scientific policy gate | `scripts/check_scientific_change.py` |
| Service boundary tests | `tests/integration/test_compute_service.py` |
| Mocked browser suite | `apps/web/e2e/ui-flows.spec.ts` |
| Real local full-stack suite | `apps/web/e2e/full-stack/journey.spec.ts` |
| Live non-mutating auth-boundary checks | `apps/web/e2e/production/smoke.spec.ts` |
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

At this checkpoint the application has 15 plugins:

| Key | Capability | Important boundary |
| --- | --- | --- |
| `monte_carlo` | input propagation and output statistics | bounded inline samples |
| `eda` | summaries and input/output correlation matrices | descriptive, not causal |
| `correlation` | Pearson, Spearman, partial, and regression effects | association is not variance attribution |
| `ancova` | physical and correlation-driven first-order variance contributions | dependent continuous inputs; validated PCE approximation |
| `sobol` | Saltelli first/total/second-order indices | independent inputs only |
| `fast` | Fourier amplitude sensitivity indices | independent inputs only |
| `hsic` | kernel dependence and permutation evidence | continuous inputs; quadratic work cap; dependence is not causality |
| `target_hsic` | target-domain kernel association and permutation evidence | target association is not probability or causality |
| `taylor` | local derivative variance decomposition | local linear approximation |
| `morris` | elementary-effect screening | independent marginals; screening interpretation |
| `convergence` | running expectation evidence | finite-sample convergence evidence |
| `reliability` | FORM or Monte Carlo threshold probability | method-specific event geometry |
| `pce` | polynomial-chaos surrogate and hold-out validation | validation Q2 controls trust |
| `gpr` | GP surrogate, hold-out accuracy, and conditional intervals | continuous inputs; bounded exact GPR |
| `calibration_nlls` | nonlinear least-squares parameter calibration | named observations; local uncertainty approximation |

The legacy Streamlit source is isolated under `Streamlit_Backup/` as a read-only historical reference. It is
not part of the root dependency, test, package, or deployment graphs.

### 2026-08-28 global HSIC resource and progress hardening

The production damped-oscillator report exposed a resource-contract gap rather than a scientific applicability
failure. Its eight continuous independent inputs are valid for global HSIC, but the former shared standard budget
submitted 1,000 samples and 100 permutations. Inspection of the pinned OpenTURNS implementation shows nested
all-variable, permutation, and quadratic-kernel work. Local measurement quantified the operational consequence:
250 samples completed in about one second, 400 in about four seconds, and 1,000 used about 420 MB and about 70
seconds. The retained production failure record identifies a Cloudflare Sandbox runtime-update interruption—not
a scientific incompatibility or proven memory exhaustion—but the retry lifecycle exposed only a generic capacity
message and the unbounded request made every retry unnecessarily expensive.

Global `hsic` v2.1.0 now uses the same conservative resource unit as target HSIC,
`n^2 * (d + 1) * (B + 4)`, with a 150,000,000-unit ceiling checked before sampling. For `d=8` and `B=100`,
400 samples is the largest admitted request (149,760,000 units); 401 is rejected (150,509,736 units). Validation
publishes this safe limit, the composer explains and applies it, and core independently rejects a bypass. The
official fixed-seed Ishigami global-HSIC benchmark is retained in the evidence manifest.

The same slice makes compute state observable without inventing false precision. Every task persists its own D1
phase record from queue admission through model loading, applicability, OpenTURNS execution, evidence persistence,
completion, failure, cancellation, or retry. The Sandbox protocol streams only bounded phase metadata on stderr;
Python source and numerical samples remain excluded. HSIC publishes sampling, kernel construction, observed-index,
permutation-inference, and ranking boundaries. OpenTURNS calls without incremental callbacks are labelled and
animated as indeterminate until the next genuine boundary.

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

For any complexity or resource claim, inspect the pinned upstream implementation's loop nesting rather than
inferring cost from the public class name. Write the test-side work oracle independently—prefer explicit nested
loops over repeating production algebra—and prove all three cases: the UI default at the maximum supported
dimension is admissible, the first configuration over the cap is rejected before sampling, and the schema
maximum cannot bypass the cap. Record those calculations and exact upstream permalinks in
`docs/openturns-sync/evidence/<plugin>.json`.

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
9. Add or refresh `docs/openturns-sync/evidence/<key>.json`; it must identify exact pinned implementation and
   benchmark sources, declared benchmark/applicability/resource tests, independent resource oracle, boundary
   calculations, interpretation limits, and browser contracts.
10. Add scalar keys and an explicit safe config mapping in `Workspace.tsx`. Add custom controls only for a
   scientifically important choice not covered by generic budget/output controls.
11. Update all catalog counts and fixtures; find them rather than guessing:

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
the modern core, services, tests, and canonical examples. Consult `Streamlit_Backup/` only when historical
behavior is relevant. Installation alone is not compatibility evidence.

## 6. Required evidence and tests

Core evidence: fixed-seed repeatability; an analytical/published/official benchmark; justified tolerances;
invalid applicability; output/dependence behavior; constant/non-finite behavior; config bounds; strict JSON;
evaluation accounting; table truncation; and realistic default/maximum runtime evidence where practical.

Complexity evidence is adversarial, not a restatement of implementation. The manifest-declared resource tests
must independently reconstruct upstream nested work, exercise multiple dimensions/sample/permutation counts,
test the exact accept/reject boundary, and bind the browser default to the maximum supported model dimension.
The dedicated CI policy job reruns those tests and rejects a plugin diff that does not refresh its manifest,
declared Python/browser evidence, scientific validation, sync guide, and state record.

Product evidence: catalog schema; `/v1/execute`; mocked UI configuration/request; generic report rendering;
real full-stack execution/report/export; production health and authentication boundary; and, when a real
owner session is explicitly available, a focused authenticated live audit. Production APIs never expose the
analysis catalog to an unauthenticated scout.

Mandatory local commands:

```bash
npm run check:scientific-change
uv run ruff format --check uncertaintycat_core services tests scripts .github/scripts test_all_examples.py
uv run ruff check uncertaintycat_core services tests scripts .github/scripts test_all_examples.py
uv run mypy uncertaintycat_core services scripts/check_scientific_change.py .github/scripts/openturns_scout.py
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
upstream commit/date, catalog count, selected result, and deferred/rejected candidates. The canonical scheduled
ChatGPT workflow in [`CHATGPT_SCHEDULED_FEATURE_PROMPT.md`](CHATGPT_SCHEDULED_FEATURE_PROMPT.md) creates a
feature branch and pull request only after the complete local gate; it never pushes or merges `main`.

For a separate, explicitly user-authorized direct-delivery session, follow `AGENTS.md` and stage explicit files:

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

### 2026-08-28 calibration intake

This cycle started from UncertaintyCat `a83608c208d2426030164b25363ff80c99b1ef31` after the ANCOVA merge.
The frozen pin and latest stable PyPI release both remained `1.27.post1`; the installed build reports source
revision `772da39`. The upstream default branch also remained exactly
`2301120b56f5d879d31c7bdaf73219835e8a118a` (2026-08-23), so there was no checkpoint-to-HEAD source delta.
The systematic documentation-category review and pinned-API prototype produced this admission record:

| Candidate | Admission | Weighted score | Decision |
| --- | --- | ---: | --- |
| deterministic nonlinear least-squares calibration | pass | 97/100 | implemented as `calibration_nlls` |
| `RankSobolSensitivityAlgorithm` | pass | 76/100 | deferred because it overlaps the existing first-order Sobol question |
| `GaussianProcessRegressionCrossValidation` | reject | — | installed API remains experimental |
| `LineSampling` / `QuantileConfidence` | reject | — | stable only on the unreleased 1.28 development line |

The admitted slice uses stable
[`ParametricFunction`](https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.ParametricFunction.html),
[`NonLinearLeastSquaresCalibration`](https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.NonLinearLeastSquaresCalibration.html),
and [`CalibrationResult`](https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.CalibrationResult.html)
APIs. The review included the official
[`y = a + b exp(c x)` example](https://openturns.github.io/openturns/latest/auto_calibration/least_squares_and_gaussian_calibration/plot_calibration_quickstart.html),
the [calibration theory](https://openturns.github.io/openturns/latest/theory/data_analysis/code_calibration.html),
the [pinned implementation](https://github.com/openturns/openturns/blob/772da39d3324517acedd6068da1bff3bec9b0345/lib/src/Uncertainty/Bayesian/NonLinearLeastSquaresCalibration.cxx),
and its [upstream Python test](https://github.com/openturns/openturns/blob/772da39d3324517acedd6068da1bff3bec9b0345/python/test/t_NonLinearLeastSquaresCalibration_std.py).
The pinned constructor contains a row-count guard defect later corrected upstream; the plugin therefore
checks all row counts, dimensions, and names before construction instead of relying on that guard.

With fixed seed 0, 10 noisy observations, truth `[2.8, 1.2, 0.5]`, and start `[1, 1, 1]`, the pinned API
returned `[2.7731136593401917, 1.2035076055520555, 0.49974911285083384]` twice, with local-linear approximate
SDs `[0.037160718790851095, 0.006244761064512535, 0.0005431118055327854]`, 33 optimizer calls, 19
iterations, and 360 exact atomic model evaluations. Bootstrap is fixed to zero; the parameter distribution is
labelled as a local linear Gaussian approximation rather than an exact confidence guarantee. The serialized
result is capped at 1 MB in addition to 250 rows, 32 model inputs, eight parameters, and 500 optimizer calls. Gaussian-prior
calibration, weighted/correlated residuals, multi-output calibration, MCMC, bootstrap uncertainty, and
competing optimizer selection remain explicit follow-ups rather than hidden scope expansion.
An exact-response run at the 250-row storage bound produced 250 complete prediction rows in a 50,904-byte
payload, used 3,250 atomic model evaluations, and completed in 11.77 ms on the review host.

### 2026-08-28 target-domain HSIC intake

This cycle restarted from UncertaintyCat
`d9a835164f9533a5b73a1f00b086c82714aeeccc`, including the merged calibration work and maintainer
amendments. The installed and PyPI package versions remained `1.27.post1`; the installed binary reports
source revision `772da39d3324517acedd6068da1bff3bec9b0345`. The newest upstream stable tag reviewed was
`v1.27.3` at `26a63963fb71851b2d3d397a53ec1a5286ff8d62` (2026-07-28). Its changes after the installed
revision are SWIG, symbolic-parser, XML/HDF5, CI, and version fixes, with no target-HSIC algorithm change,
so this slice does not introduce dependency churn. The upstream default branch remained
`2301120b56f5d879d31c7bdaf73219835e8a118a` (2026-08-23), exactly the previous checkpoint.

The systematic category review and pinned-API prototypes produced this admission record:

| Candidate | Admission | Weighted score | Decision |
| --- | --- | ---: | --- |
| `HSICEstimatorTargetSensitivity` | pass | 91/100 | implemented as the new `target_hsic` key |
| standard-space cross-entropy importance sampling | pass | 83/100 | deferred; valuable rare-event extension, but broader reliability-method and stopping diagnostics are needed |
| bounded LHS / low-discrepancy design generation | pass | 78/100 | deferred; requires a retained-design journey rather than an analysis-only wrapper |
| `RankSobolSensitivityAlgorithm` | pass | 74/100 | below threshold because it duplicates the existing first-order Sobol question |
| `HSICEstimatorConditionalSensitivity` | pass | 73/100 | below threshold and intentionally deferred behind the first target-HSIC slice |
| Shapley effects | reject | — | no stable Shapley API exists in the installed pin |
| `LineSampling` / `QuantileConfidence` | reject | — | stable only on the unreleased 1.28 development line |

The admitted plugin uses stable
[`HSICEstimatorTargetSensitivity`](https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.HSICEstimatorTargetSensitivity.html),
[`DistanceToDomainFunction`](https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.DistanceToDomainFunction.html),
`ParametricFunction`, `SquaredExponential`, and `HSICUStat` APIs. Review included the official
[HSIC theory](https://openturns.github.io/openturns/1.25/theory/reliability_sensitivity/sensitivity_hsic.html),
[Ishigami example](https://openturns.github.io/openturns/latest/auto_sensitivity_analysis/plot_hsic_estimators_ishigami.html),
[pinned C++ implementation](https://github.com/openturns/openturns/blob/772da39d3324517acedd6068da1bff3bec9b0345/lib/src/Uncertainty/Algorithm/Sensitivity/HSICEstimatorTargetSensitivity.cxx),
and [pinned upstream test](https://github.com/openturns/openturns/blob/772da39d3324517acedd6068da1bff3bec9b0345/python/test/t_HSICEstimatorTargetSensitivity_std.py).

The vertical slice is scalar-output and continuous-input only. It transforms output through
`exp(-distance_to_critical_domain / s)`, with `s = 0.1` times the sampled output standard deviation by
default, and asks which inputs are associated with that smoothed target score. It caps inputs at 20,
samples at 500, permutations at 200, and estimated quadratic work at 30 million units; requires at least
five sampled observations inside and outside the target; stores only one bounded input-index table; and
accounts for exactly the sampled model evaluations. Dependent inputs remain allowed with an explicit
confounding warning. Reports prohibit interpreting the result as failure probability, variance allocation,
causal influence, or out-of-domain prediction.

Against the upstream fixed-seed Ishigami test (`a=5`, `b=0.1`, 100 observations, target `Y >= 5`,
100 permutations), the plugin obtained target R2-HSIC
`[0.26863688209966674, 0.004684228098984393, 0.0033996249931746553]`, matching the upstream expected
`[0.26863688, 0.00468423, 0.00339962]` within `1e-8`. Raw HSIC, asymptotic p-values, and permutation
p-values also match their upstream vectors within `1e-8`. The run was exactly repeatable, used 100 model
evaluations, took about 33 ms in the local prototype, and serialized to 2,784 bytes.

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
   mocked browser, real full-stack, and production auth-boundary evidence as applicable.
10. Update readme.md, scientific/method docs, every catalog count, and state.json. Record deferred/rejected
    candidates so the next run does not rediscover them without new evidence.
11. Run every Python, TypeScript, build, Playwright, full-stack, and container gate in the README. Inspect the
    full diff. Never weaken evidence merely to pass.
12. Only when all gates pass, use the delivery mode authorized for the run. The canonical scheduled ChatGPT
    workflow must commit to a feature branch, open a pull request, observe exact-head PR CI, and stop without
    self-merging or deploying. Direct `main` delivery and post-CI production observation apply only to a
    separate session explicitly authorized for that mode.
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

For ChatGPT sessions connected to a Codex cloud environment and GitHub, use the canonical, PR-based prompt in
[`CHATGPT_SCHEDULED_FEATURE_PROMPT.md`](CHATGPT_SCHEDULED_FEATURE_PROMPT.md). It includes collision handling,
untrusted-web-input boundaries, scientific admission gates, complete evidence, pull-request structure, CI
shepherding, and explicit no-op behavior. The abbreviated prompt above remains a process synopsis.

Even for a no-op, update the checkpoint only when new upstream commits or candidates were genuinely reviewed.
That keeps future diffs precise without fabricating repository activity.
