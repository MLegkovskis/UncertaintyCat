# 2026-09-03 synchronization: bounded subset sampling

## Baseline and fresh scan

- UncertaintyCat: `a6d855dd435e640db84c69cb1c0771e380edbb25` (clean `origin/main`).
- No open `agent/openturns-sync-*` PR at intake. Unrelated draft #35 was left alone.
- Installed/latest PyPI OpenTURNS: `1.27.post1`; installed build revision
  `772da39d3324517acedd6068da1bff3bec9b0345`. No pin change.
- Latest upstream stable tag: `v1.27.3`, `26a63963fb71851b2d3d397a53ec1a5286ff8d62`.
  GitHub's Releases page still labels v1.26 latest; tags and PyPI are recorded separately.
- Fresh upstream default-branch clone: `763b0ac2e812809c92227f1a0cfd3f87ec9a612e`,
  2026-09-03T16:25:03+02:00; 25 commits since the checkpoint's `2301120b...`.
- `uv sync --frozen --extra dev` and the repository upstream scout completed successfully.
  The entire lock was parsed and its artifact metadata checked; SHA256
  `4b04dc20d511c17e1ebb5c12a4b7ede67388005c89f4659cd623bf8680c2c272`.

The review covered the pinned documentation indexes for data analysis/statistical tests,
distributions/copulas/probabilistic modeling, experimental design, surrogate modeling,
reliability, sensitivity, stochastic processes, functional modeling, calibration,
optimization and numerical methods. This is a category scan followed by focused source
inspection, not a claim to have audited every OpenTURNS implementation.

Current main has 15 plugins: Monte Carlo, EDA, correlation, ANCOVA, Sobol, FAST, global
HSIC, target HSIC, Taylor, Morris, convergence, reliability, PCE, GPR and nonlinear
calibration. Reliability already includes FORM, SORM, Monte Carlo, directional and subset
sampling. Project workflows already include model/builder, dimension reduction,
Calibration Studio, model/empirical Surrogate Studio and distribution fitting (including
normal/Bernstein copulas). Generic retained reports, semantic charts, exports, surrogate
promotion and source-hashed reference examples are already implemented. Reintroducing
ANCOVA (#75), calibration (#76), target HSIC (#77), or chart-only wrappers is not additive.

Maintainer amendments reviewed: #75's assessment-before-submission fix `500431fd...`,
#76's source-hash/name/bundle boundary fix `1d777b9c...`, and #77's multiplicative
kernel-work correction `a2969b7c...`. Their lessons apply below; their implementations
were not used as feature templates.

## Candidate decision

Scores are 0–5 in order: scientific value (25), product fit (20), maturity (15),
validation (15), distinctiveness/correction (10), safety (10), maintenance (5).
Totals cannot override a failed admission gate.

| Candidate | Scores | Total | Admission and decision |
| --- | --- | ---: | --- |
| Correct subset-sampling work/stopping/report semantics | 5/5/5/5/5/4/4 | 97 | Pass; selected. Reproduced a fivefold budget overrun and contradictory stopping claims, directly illuminated by a new upstream change. |
| Retained LHS / low-discrepancy experimental design | 4/4/5/5/4/5/3 | 87 | Pass for a scoped independent-input retained design; deferred. Needs deliberate design persistence and consumption in later studies, not another sampling checkbox. |
| Standard-space cross-entropy importance sampling | 4/5/5/4/4/3/3 | 84 | Pass for bounded continuous-input events; deferred behind repairing existing reliability controls. Requires adaptive-population budget and weight/degeneracy evidence. |
| Rank-based first-order Sobol | 3/4/5/4/1/5/4 | 74 | Reject current direct-analysis proposal: same first-order question as Sobol/PCE, below threshold. A future retained paired-data journey could change distinctiveness. |
| Bootstrap uncertainty for existing NLLS | 4/4/5/3/3/2/3 | 73 | Below threshold. Current implementation explicitly disables bootstrap and correctly labels linearization; new upstream documentation does not expose a current default bug. Requires bounded repeated fits and empirical coverage evidence. |
| Conditional HSIC | 3/3/5/4/3/3/3 | 69 | Below threshold. Weighted within-domain question is distinct but less urgent; effective-weight coverage and clear separation from target association remain necessary. |
| Tail fitting / return-level workflow | 4/3/5/3/4/2/2 | 70 | Deferred at workflow/applicability gate: threshold/block selection, independence, tail diagnostics and likelihood optimization bounds need a coherent design beyond generic marginal fitting. |
| Field/process/Karhunen–Loeve workflow | 3/1/4/3/5/1/1 | 53 | Reject current slice: requires mesh/process provenance and contracts absent from scalar/vector `f(x)` journey. |
| LineSampling, QuantileConfidence, new Kent/factories | — | — | Reject stability gate: absent from stable installed namespace; stabilization/new experimental APIs are in unreleased 1.28 only. |

Other scanned changes concern comparison/copy-on-write correctness, internal linear algebra,
characteristic functions, task-arena overhead and experimental distribution factories.
No demonstrated current application regression or narrowly justified pin upgrade was found.
Existing GPR uses the stable replacement APIs, existing calibration explicitly selects
bootstrap zero, and existing reports already expose the relevant retained visual evidence.

## Primary sources

- [PyPI](https://pypi.org/project/openturns/),
  [stable tag](https://github.com/openturns/openturns/tree/26a63963fb71851b2d3d397a53ec1a5286ff8d62),
  [reviewed ChangeLog](https://github.com/openturns/openturns/blob/763b0ac2e812809c92227f1a0cfd3f87ec9a612e/ChangeLog).
- [New upstream stopping-control correction](https://github.com/openturns/openturns/commit/d5448af72dbadf0cffff6194cad0feeec1861530):
  forbids coefficient-of-variation stopping controls for SubsetSampling/CEIS/NAIS.
- [Pinned subset implementation](https://github.com/openturns/openturns/blob/772da39d3324517acedd6068da1bff3bec9b0345/lib/src/Uncertainty/Algorithm/Simulation/SubsetSampling.cxx),
  [declaration](https://github.com/openturns/openturns/blob/772da39d3324517acedd6068da1bff3bec9b0345/lib/src/Uncertainty/Algorithm/Simulation/openturns/SubsetSampling.hxx),
  [API](https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.SubsetSampling.html),
  [theory](https://github.com/openturns/openturns/blob/772da39d3324517acedd6068da1bff3bec9b0345/python/doc/theory/reliability_sensitivity/subset_sampling.rst),
  [official example](https://github.com/openturns/openturns/blob/772da39d3324517acedd6068da1bff3bec9b0345/python/doc/examples/reliability/reliability_analysis/plot_subset_sampling.py),
  [R–S upstream test](https://github.com/openturns/openturns/blob/772da39d3324517acedd6068da1bff3bec9b0345/python/test/t_SubsetSampling_R-S.py).
- [Probability-result Normal approximation](https://github.com/openturns/openturns/blob/772da39d3324517acedd6068da1bff3bec9b0345/lib/src/Uncertainty/Algorithm/Simulation/ProbabilitySimulationResult.cxx).
- [CEIS implementation](https://github.com/openturns/openturns/blob/772da39d3324517acedd6068da1bff3bec9b0345/lib/src/Uncertainty/Algorithm/Simulation/CrossEntropyImportanceSampling.cxx),
  [CEIS API/theory/example](https://github.com/openturns/openturns/blob/772da39d3324517acedd6068da1bff3bec9b0345/python/src/StandardSpaceCrossEntropyImportanceSampling_doc.i),
  [CEIS test](https://github.com/openturns/openturns/blob/772da39d3324517acedd6068da1bff3bec9b0345/python/test/t_StandardSpaceCrossEntropyImportanceSampling_std.py).
- [LHS API/example](https://github.com/openturns/openturns/blob/772da39d3324517acedd6068da1bff3bec9b0345/python/src/LHSExperiment_doc.i),
  [LHS implementation](https://github.com/openturns/openturns/blob/772da39d3324517acedd6068da1bff3bec9b0345/lib/src/Uncertainty/Algorithm/WeightedExperiments/LHSExperiment.cxx),
  [LHS test](https://github.com/openturns/openturns/blob/772da39d3324517acedd6068da1bff3bec9b0345/python/test/t_LHSExperiment_std.py).
- [Rank Sobol definition/limitations](https://github.com/openturns/openturns/blob/772da39d3324517acedd6068da1bff3bec9b0345/python/src/RankSobolSensitivityAlgorithm_doc.i),
  [conditional HSIC test](https://github.com/openturns/openturns/blob/772da39d3324517acedd6068da1bff3bec9b0345/python/test/t_HSICEstimatorConditionalSensitivity_std.py),
  [conditional HSIC implementation](https://github.com/openturns/openturns/blob/772da39d3324517acedd6068da1bff3bec9b0345/lib/src/Uncertainty/Algorithm/Sensitivity/HSICEstimatorConditionalSensitivity.cxx),
  [NLLS documentation clarification](https://github.com/openturns/openturns/commit/716e4ba18826a4a7b3d3efb6f5b74ca42422ceac).

## Design recorded before implementation

Question: what is the probability of the selected scalar output crossing the user's
threshold, within a genuinely enforced model-evaluation budget? Keep this within the
existing project direct-analysis reliability composer; do not add a studio or estimator.

Use stable `SubsetSampling(event, 2.0, 0.1)` and `SubsetSamplingResult`. The estimate is
the product of adaptively chosen conditional-event estimates; later populations use
modified Metropolis–Hastings chains. It is neither an independent Monte Carlo sample
nor an exact/unbiased finite-sample probability guarantee. Within-level autocorrelation
is approximated by OpenTURNS; between-level dependence can make its reported uncertainty
optimistic. Reaching the requested threshold is not proof of mixing, precision, causal
validity, or predictive validity outside the specified input distribution.

Scope: continuous distributions admitting OpenTURNS' standard-space transformation,
1–20 inputs, one selected varying finite scalar output; dependent continuous inputs are
allowed. Reject invalid targets, constant pilots, non-finite model values, and incomplete
or degenerate results. Do not infer event impossibility from a failed/constant sample.

Controls: 100–5,000 samples per level (multiple of 10), default 2,000; total requested
model budget at most 50,000, default 20,000; at most ten populations including the initial
one. Effective cap is `N * min(10, floor(budget/N))`; reject `N > budget`. Fixed block
size 1 preserves the pinned chain update ordering. Legacy `sample_size` aliases and
non-unit subset blocks fail with an explicit correction message. CoV is a diagnostic,
never a subset stopping target. Explicit legacy precision values are retained but warned
as unused. A point-call wrapper checks the cap before every original-model invocation;
there is no batch fallback or model-count inference from population counts. Count excludes
model compilation/validation and means top-level model point invocations, not arbitrary
internal subcomputations. Timeout checks use 60 seconds; the existing Sandbox deadline is
the hard backstop for a single opaque user evaluation.

Pinned loops allocate `N*(d+1)` working sample values and evaluate N model points in each
population. Conditional levels also have d-coordinate proposals and the chain-correlation
triple loop: with `Nc=N/10`, chain length 10, it performs `Nc * (9+8+...+1)=4.5*N`
indicator-pair products per conditional population. Hard caps bound those loops without
pretending to predict user-function runtime. Disable retained raw samples, use compact
internal history, retain at most ten level rows and a bounded strict JSON payload.

Reject budget/time interruption or a final threshold different from the requested one;
never publish an intermediate-domain estimate as the requested event probability. Persist
actual point calls, effective/requested caps, samples/levels, final probability, OpenTURNS
standard error/CoV and its nominal 95% Normal interval (clipped to [0,1], explicitly approximate),
level thresholds/probabilities and truthful termination facts. Do not expose the old
resetting within-level convergence trace or its misleading independent confidence band.
The level table is the accessible/exportable scientific report view.

Version the changed reliability behavior/configuration as plugin `3.0.0`; keep generic
result schema `1.0.0`, old payloads/readability intact, other reliability methods unchanged.
Version-pinned old reruns explicitly reject an unavailable plugin version; unpinned subset
reruns use the new safety contract and may reject unsafe historical settings. Historical
evidence is never rewritten. No migration, pin,
public route, source/AI exposure, owner predicate or deployment-chain change is needed.
Method-specific core assessment guidance is honored by UI and Worker and enforced again
by the plugin.

Prototype evidence (installed pin, seed 42): baseline N=1,000/advertised budget=1,000
actually made 5,000 calls for `P(Normal()>4)`, with identical probability
`3.54609981e-5` for requested CoV 0.8 and 0.001 but conflicting stopping reasons.
The guarded prototype stops at exactly 1,000 calls. Default N=2,000 at d=20 reached the
four-sigma threshold in five levels / 10,000 calls, approximately 0.044 seconds for a
simple sum. The official R–S model has independent R~Normal(7,1), S~Normal(2,1), event
R−S<0; analytical probability is `0.5*erfc(2.5)=0.00020347600872247946`.
Default prototype obtained `0.000222`, four levels / 8,000 calls, CoV 0.19221,
approximately 0.134 seconds. Planned analytical tolerance: absolute `8e-5` (finite-sample
sampling variability, not a coverage claim), alongside exact repeatability/upstream
cross-checks. Installed CEIS prototype used nine 1,000-point populations, confirming
its own adaptive budget obligation. Independent-input LHS prototype populated every
one of 20 strata in all three coordinates exactly once.

## Verification and residual risks

The implemented R–S benchmark returns probability `0.00022199999999999987`, standard
error `4.2670587059472225e-5`, CoV `0.19220985161924437`, and nominal Normal interval
`[0.00013836718616425342, 0.0003056328138357463]`. Its four threshold rows are
`3.2104054654614087`, `1.7583606142649777`, `0.6573466245247539`, `0`, with cumulative
probabilities `0.1`, `0.01`, `0.001`, `0.000222` (rounded here only). The full retained
payload is 1,718 UTF-8 JSON bytes with ordinary `json.dumps` spacing. Actual calls are
8,000, compared with an effective cap of 20,000. Analytical absolute error is about
`1.8524e-5`, below the justified `8e-5` sampling tolerance. Exact pinned-API construction
and repeated seeded runs agree within `1e-12`; this is not an interval-coverage claim.

Measured single-run numerical durations on this host: 131.44 ms in local Python 3.12.3,
128.75 ms in the HTTP compute image and 122.97 ms in the Sandbox image (both Python
3.14.7). All three produce identical reported probability, uncertainty, level rows and
8,000 calls. Timings exclude compilation/validation/container startup and are observations,
not a performance guarantee. The independent callback test stops at exactly 1,000 calls;
the maximum-budget test stops at exactly 50,000 calls without publishing a result.

Local commands and final outcomes (no CI/test configuration was weakened):

| Command | Result |
| --- | --- |
| `npm run check:examples` | Passed; 24 canonical models remain unchanged. |
| `npm run check:scientific-change` | Passed; 3 manifests, 27 declared Python test nodes, 54 expanded tests. |
| `npm run typecheck` | Passed in all three workspaces. |
| `npm run test:ts` | Passed: 41 Worker tests and 17 web tests. |
| `npm run build` | Passed: Worker dry-run, Vite build and public-source exclusion check. |
| `npm audit --omit=dev --audit-level=high` | Passed; zero vulnerabilities. |
| `uv run ruff format --check uncertaintycat_core services tests scripts .github/scripts test_all_examples.py` | Passed; 49 files. |
| `uv run ruff check uncertaintycat_core services tests scripts .github/scripts test_all_examples.py` | Passed. |
| `uv run mypy uncertaintycat_core services scripts/check_scientific_change.py .github/scripts/openturns_scout.py` | Passed; 35 source files. |
| `uv run pytest` | Passed; 156 tests. |
| `uv run pytest --cov=uncertaintycat_core --cov=services --cov-report=term-missing` | Passed; 156 tests, 89% total coverage, 97% reliability-module coverage. CLI subprocess execution is separately tested and not captured by the in-process coverage counter. |
| `uv run python test_all_examples.py` | Passed; 24/24 reference models. |
| `npm run test:e2e` | Passed; 68 browser tests including light/dark authentication and accessibility coverage. |
| `npm run test:e2e:full-stack` | Passed; 2 real journeys, including existing all-plugin coverage and the new R–S run/reload/export. |
| `docker build -f services/compute/Dockerfile -t uncertaintycat-compute:verification .` | Passed; image `8060dbc7fe4e89571494d4dfd07ee87afb70787b5c0c60b88d4b29933cedb07e`. |
| `docker build -f services/compute/Dockerfile.sandbox -t uncertaintycat-sandbox:verification .` | Passed; image `46b2ffbff49d59c7be6ee71181219506d770aa7913765f639570743969555310`. |
| Image import smoke tests and the same R–S `run_analysis` benchmark inside each image | Passed; pinned OpenTURNS `1.27.post1`, reliability `3.0.0`, exactly 8,000 calls. |

Focused iterations also exercised the core/service files, the existing reliability tests,
and the three new browser tests. They exposed and fixed a duplicate pytest basename,
new-test selector mismatches, and a keyboard-inaccessible CodeMirror scroll region.
Rendered subset composer/report screenshots were inspected; the composer received scoped
design-system styling and an explicit non-overlap test. Earlier failed iterations are not
presented as successful evidence. Existing non-failing Starlette deprecation, Vite chunk
size and local no-AI/Cobyla diagnostic warnings remain. The standard npm audit completed
with zero vulnerabilities; a supplementary retry with a 30-second fetch timeout failed
at npm's audit endpoint (external network timeout, not a vulnerability finding).
No production test or deployment
was initiated; the full-stack test uses the repository's isolated local auth boundary,
not a claim to have exercised Cloudflare's production identity provider.

The PR records the immutable head SHA and exact GitHub CI run URL after push, avoiding
a self-referential commit hash in this file. CI must complete for that head before handoff.

Remaining review focus: finite-sample bias/mixing, conservative rejection of incomplete
levels and constant pilots, performance of the point guard for expensive Python models,
and explicit compatibility of historical subset requests. FORM/SORM/directional method
evaluation accounting is outside this narrowly scoped subset correction and remains a
separate audit candidate; this record does not certify those paths as hard-bounded.
