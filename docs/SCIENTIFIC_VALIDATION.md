# Scientific validation

## Numerical authority

OpenTURNS is the numerical foundation. UncertaintyCat adds configuration bounds, applicability checks,
serialization, orchestration, provenance, and presentation; it does not replace the upstream algorithms.
Every result records the model hash, seed, OpenTURNS version, core version, plugin version, result schema
version, timestamps, warnings, assumptions, runtime, and model-evaluation count.

The migration baseline is OpenTURNS `1.27.post1`. Moving from the inherited `1.25` pin exposed the
removal of `ComposedDistribution`; legacy construction and validation paths were migrated to
`JointDistribution`, then both the scientific suite and all 24 example smoke runs were repeated.

## Current automated evidence

The test suite covers:

- compilation and sample evaluation for all 24 bundled Python reference models;
- consistent input/output dimensions, descriptions, finite values, and batch behavior;
- strict contracts that reject extra fields and serialize without NaN;
- repeatable multi-output Monte Carlo and EDA at fixed seeds;
- Ishigami Sobol first-order structure (`S1` near 0.314, `S2` near 0.442, `S3` near zero) with tolerances
  appropriate to a finite Saltelli design;
- dependent-copula rejection for classical Sobol;
- stable OpenTURNS global HSIC on the official fixed-seed Ishigami case: normalized indices
  `[0.29807297, 0.00344498, 0.07726572]` and permutation p-values
  `[0.0, 0.29670330, 0.00199800]` match the pinned upstream benchmark within `1e-8`;
- an independent loop-count oracle for global HSIC and the measured eight-input damped-oscillator boundary:
  400 samples at 100 permutations is admitted at 149,760,000 estimated units, while 401 samples is rejected
  before model sampling at 150,509,736 units; the browser request contract enforces the same model-specific cap;
- stable OpenTURNS target-domain HSIC on the official fixed-seed Ishigami case: target R2-HSIC
  `[0.26863688, 0.00468423, 0.00339962]`, raw indices and both p-value diagnostics match the upstream
  test within `1e-8`, repeat exactly, and account for 100 model evaluations;
- an independent loop-count oracle for target-domain HSIC resource work across zero/default/maximum
  permutation cases, the exact 150,000,000-unit accept/reject boundary, rejection of the schema maximum,
  and a browser contract proving the safe default remains admissible at 20 inputs;
- stable OpenTURNS ANCOVA physical/correlation decomposition for a correlated-normal linear benchmark,
  including dependent-holdout PCE validation and exact evaluation accounting;
- stable OpenTURNS nonlinear least-squares calibration of the official exponential family
  `y = a + b exp(c x)`: fixed-seed observations recover `[2.7731136593401917,
  1.2035076055520555, 0.49974911285083384]` from truth `[2.8, 1.2, 0.5]` within absolute
  tolerances `[0.05, 0.02, 0.005]`, repeat exactly, and report 360 atomic model evaluations;
- OpenTURNS-authoritative execution and strict serialization for correlation, FAST, HSIC, Taylor, OTMorris, convergence,
  reliability, PCE, and Gaussian-process regression;
- CSV/XLSX inspection, OpenTURNS marginal ranking/copula composition, and promoted PCE/GPR `ot.Study` XML round trips;
- fixed-seed Gaussian-process regression accuracy on a smooth nonlinear response, plus a correlated-input
  linear-trend benchmark and discrete/constant-output rejection;
- FORM probability near 0.5 for a standard normal response with threshold zero, plus stable Monte Carlo, SORM, directional, and subset method contracts;
- bounded subset sampling against the official independent resistance–stress case:
  `R~Normal(7,1)`, `S~Normal(2,1)`, `P(R-S<0)=0.00020347600872247946` analytically;
  seed 42 with 2,000 samples per level obtains `0.000222` within absolute `8e-5`,
  repeats exactly, matches direct OpenTURNS construction and uses exactly 8,000 point calls;
- independent subset population/chain-correlation loop oracles, admission at the default
  20-input maximum, rejection of budgets 50,001 and 2,000,000 before sampling, and a
  callback-side count proving exhaustion at exactly 1,000 original-model invocations;
- FastAPI health, catalog, validation, execution, Data Lab, and promoted-surrogate contracts;
- worker ZIP export structure and CSV quoting;
- frontend symbolic-model generation and a Chromium navigation smoke test.

Run the evidence locally with:

```bash
npm run check:scientific-change
uv run pytest
uv run pytest -m scientific
npm run test:ts
npm run test:e2e
```

## Method-specific interpretation controls

Subset sampling (`reliability` plugin `3.0.0`) uses stable OpenTURNS adaptive conditional
populations with continuous inputs and one selected varying finite output. Population
size is 100–5,000 in multiples of ten, default 2,000; total requested budget is at most
50,000, default 20,000; at most ten populations are attempted with block size one.
The effective cap reserves only complete populations. CoV is a diagnostic, not a stopping
target. Budget/time interruption, missing final threshold or degenerate evidence fails
without publishing an intermediate-domain probability. Point counts include rejected
MCMC proposals and exclude source construction/validation and user-model internal work.
The 60-second cooperative bound has the Sandbox deadline as backstop for an opaque call.

Only the final retained level row estimates the requested event. The nominal 95% Normal
interval comes from OpenTURNS' result distribution and is clipped to [0,1]; within-chain
correlation is estimated but between-level dependence/finite-sample bias can make it
optimistic. It is not an exact confidence guarantee, a mixing diagnostic, or causal or
out-of-domain predictive validation. No resetting within-level confidence trace is shown.
The generic result schema remains `1.0.0`; old results are not rewritten. Version-pinned
old reruns reject an unavailable plugin version; unpinned old subset configurations may
need correction (non-unit blocks and the ambiguous `sample_size` alias are rejected).
Other reliability methods are unchanged, and this subset envelope does not certify their
evaluation accounting. Full provenance, limitations and sources are in the
[cycle record](openturns-sync/2026-09-03-subset-sampling.md) and
[machine-readable evidence](openturns-sync/evidence/reliability.json).

- Sobol and FAST: independent inputs and non-zero selected-output variance.
- ANCOVA: two to ten continuous dependent inputs, a bounded polynomial basis, and dependent-holdout Q2 of
  at least 0.8; physical and correlation contributions are first-order and may include negative correlation
  contributions.
- Morris: the pinned official `otmorris==0.20.post1` module is authoritative; independent marginals and trajectories operate in probability space so unbounded marginals remain
  finite without inventing physical bounds.
- HSIC: normalized empirical Gaussian-kernel dependence; permutation p-values are finite-sample evidence,
  not a causal claim. Continuous inputs and a variable scalar output are required. The validation assessment
  publishes a model-specific maximum sample size at the fixed default permutation count, and core repeats the
  quadratic-work check before any analysis sampling.
- Target-domain HSIC: one scalar output and at most 20 continuous inputs; a threshold must leave at least
  five sampled observations inside and outside the target domain. Sample size, permutations, quadratic
  work, and report rows are hard-capped. The exponential distance filter and empirical kernel bandwidths
  make this a sample-dependent association screen—not event probability, variance allocation, causal
  influence, or evidence beyond the sampled input distribution. Slightly negative finite-sample unbiased
  U-statistic estimates are retained rather than clipped. Dependent-input results retain an explicit
  confounding warning. Plugin version `1.1.0` leaves this numerical and result-schema contract unchanged,
  but emits monotonic sampling, target-coverage, kernel, observed-index, permutation, and ranking phases.
  Opaque observed/permutation calls remain indeterminate rather than claiming fabricated completion
  percentages. Reports plot the exact retained target R2-HSIC rows and retain the underlying table.
- Taylor: local gradients at the mean; the independently sampled linear-surrogate Q2 exposes when a local
  approximation is poor globally.
- PCE: independent validation Q2/RMSE is always reported; a fitted surrogate is not automatically an
  acceptable surrogate.
- GPR: independently sampled hold-out R2/RMSE/MAE and conditional-interval coverage are always reported;
  exact fitting is capped at 512 training points, inputs must be continuous, and the model-based conditional
  intervals are not guaranteed frequentist confidence intervals.
- Calibration: one scalar output, 1–8 selected continuous parameters, at most 250 finite named observations,
  at least two residual degrees of freedom, full-rank start/optimum Jacobians, and at most 500 optimizer calls.
  The serialized analysis payload has a final hard cap of 1 MB in addition to its structural row/dimension caps.
  Bootstrap is fixed to zero. The parameter SDs, intervals, and correlation are OpenTURNS' local linear
  Gaussian approximation at the optimum—not exact confidence guarantees. Fit does not establish global
  identifiability, causality, or predictive validity outside the observed domain.
- FORM: local design-point approximation; Monte Carlo is available when nonlinear event geometry makes
  FORM unsuitable.
- Correlation: linear and monotonic coefficients are reported side by side rather than conflated.

## Acceptance policy for dependency or algorithm changes

1. Pin the new dependency in a branch and regenerate `uv.lock`.
2. Run all unit, service, and 24-model tests in the current and candidate environments.
3. Add upstream benchmark cases for changed/new methods.
4. Compare indices, moments, probabilities, convergence, warnings, failures, and runtime.
5. Explain every material drift; do not widen tolerances solely to make CI pass.
6. Version affected plugins and schemas, add migration notes, and retain old report readability.
7. Have a domain reviewer approve changes to reliability or sensitivity interpretation.
8. Refresh the plugin's pinned-source evidence manifest. CI rejects plugin diffs without an independent
   complexity oracle, exact resource boundary, changed Python/browser evidence, and updated sync records.

## Known validation gaps before a scientific 1.0 release

- broader published benchmark corpus for each plugin and multiple dimensions/distribution families;
- confidence-interval coverage tests across repeated randomized designs;
- broader dependent-input benchmarks, non-Gaussian copulas, and Shapley-effect methods beyond ANCOVA;
- broader target-HSIC threshold/filter robustness and conditional-HSIC benchmarks;
- rare-event reliability benchmarks and FORM failure/fallback cases;
- PCE basis/sparsity/degree sweeps and correlated-input transformations;
- repeated-design GPR calibration/coverage studies, anisotropic kernel diagnostics, and high-dimensional
  scaling evidence beyond the current exact-GPR cap;
- calibration benchmarks with correlated/heteroscedastic observation errors, multi-output responses,
  independently designed identifiability diagnostics, and out-of-domain predictive validation;
- cross-platform reproducibility and numerical-drift envelopes;
- performance/load budgets at preview, standard, and high profiles;
- independent review of formulas, labels, and assumptions by UQ practitioners.

These gaps are explicit launch work. Passing the current suite establishes a strong migration baseline,
not universal certification of every model and method.
