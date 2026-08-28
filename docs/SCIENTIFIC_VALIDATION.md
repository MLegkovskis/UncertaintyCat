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
- stable OpenTURNS target-domain HSIC on the official fixed-seed Ishigami case: target R2-HSIC
  `[0.26863688, 0.00468423, 0.00339962]`, raw indices and both p-value diagnostics match the upstream
  test within `1e-8`, repeat exactly, and account for 100 model evaluations;
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
- FastAPI health, catalog, validation, execution, Data Lab, and promoted-surrogate contracts;
- worker ZIP export structure and CSV quoting;
- frontend symbolic-model generation and a Chromium navigation smoke test.

Run the evidence locally with:

```bash
uv run pytest
uv run pytest -m scientific
npm run test:ts
npm run test:e2e
```

## Method-specific interpretation controls

- Sobol and FAST: independent inputs and non-zero selected-output variance.
- ANCOVA: two to ten continuous dependent inputs, a bounded polynomial basis, and dependent-holdout Q2 of
  at least 0.8; physical and correlation contributions are first-order and may include negative correlation
  contributions.
- Morris: the pinned official `otmorris==0.20.post1` module is authoritative; independent marginals and trajectories operate in probability space so unbounded marginals remain
  finite without inventing physical bounds.
- HSIC: normalized empirical Gaussian-kernel dependence; permutation p-values are finite-sample evidence,
  not a causal claim.
- Target-domain HSIC: one scalar output and at most 20 continuous inputs; a threshold must leave at least
  five sampled observations inside and outside the target domain. Sample size, permutations, quadratic
  work, and report rows are hard-capped. The exponential distance filter and empirical kernel bandwidths
  make this a sample-dependent association screen—not event probability, variance allocation, causal
  influence, or evidence beyond the sampled input distribution. Slightly negative finite-sample unbiased
  U-statistic estimates are retained rather than clipped. Dependent-input results retain an explicit
  confounding warning.
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
