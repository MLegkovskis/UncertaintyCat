# Browser and end-to-end testing

UncertaintyCat uses three complementary Playwright suites. Fast fixture-driven UI evidence is separated from real numerical infrastructure and deployed read-only checks.

## Scientific change policy

```bash
npm run check:scientific-change
```

This validates every retained `docs/openturns-sync/evidence/*.json` manifest against the installed OpenTURNS
pin and plugin catalog, checks exact upstream source permalinks and declared test references, then reruns the
manifest's benchmark, applicability, independent complexity-oracle, and resource-boundary tests. In CI it is
also diff-aware: a new or changed plugin cannot pass unless the same change refreshes its manifest, declared
Python and browser evidence, scientific validation, sync guide, and sync state. The required aggregate CI job
depends on this gate, so deployment cannot proceed when it fails.

The Python integration suite also executes the one-shot Sandbox CLI directly and asserts that HSIC emits the
expected monotonic, source-free phase protocol on stderr while stdout remains one valid strict result envelope.

## UI contract suite

```bash
npm run test:e2e
```

This suite uses stateful HTTP fixtures with the real React application. It covers:

- the static public overview, absence of private navigation, private-route login wall, and Cloudflare sign-in initiation;
- a production-bundle gate proving canonical authenticated example source and hashes are absent from public JavaScript and source maps;
- authenticated desktop/mobile navigation, account identity, immediate sign-out without a hard refresh, and theme persistence;
- discovery of all 24 reference models in the unified, immediately editable Python authoring view;
- blank/manual model naming, resizable Python authoring, project creation from the project index, project-scoped studio navigation, and the multi-output symbolic OpenTURNS builder;
- model validation success/failure, immediate deterministic equation fallback, Groq-compatible display-math interpretation for arbitrary authenticated Python, in-flight Model Understanding feedback, analysis controls locked until the complete AI brief succeeds, deterministic direct/reduction/surrogate routing, and the direct-only analysis catalog;
- integrated validation/Model Understanding, primary/fallback model policy, single-flight polling, explicit uncharged timeout/failure states, named-observation Calibration Studio, model- and data-driven surrogate fitting, dedicated Morris screening/reduction, and Distribution Fitting composition with beam sample data;
- desktop bounding-box and screenshot evidence guarding the validated two-column layout against header overlap;
- queued/running/terminal run states, separate accessible progress bars and named phase messages for every
  task, indeterminate OpenTURNS phases, retry state, cancellation, exact reruns, and project-oriented history;
- catalog-wide post-validation applicability, including explicit dependent-copula disablement and the
  model-specific damped-oscillator HSIC sample cap in both displayed guidance and submitted request JSON;
- metrics, tables, truncation notices, series, heatmaps, facts, equations, assumptions, provenance, partial failures, sharing, bundle export, and direct PDF-download wiring;
- catalog-wide visualization strategy coverage: table-only FAST, Sobol, HSIC, target-HSIC, Taylor, Morris,
  ANCOVA, PCE, reliability, and calibration results receive semantic charts; Monte Carlo samples receive
  histograms; convergence/validation series and correlation matrices retain line/scatter/heatmap views;
  every chart keeps its exact table, series, or matrix evidence available;
- stored report chat, streaming Markdown, human-readable evidence badges, section-aware sensitivity discovery, suggestions, and quota failures;
- equal-height scroll-bounded Model Understanding, unclipped exact-data disclosure controls, and syntax-highlighted immutable Python source;
- authenticated read-only shared reports; and
- automated WCAG A/AA scans in light/dark themes, private and public states, expanded controls, and mobile navigation.

The key regression contract is explicit: an unauthenticated browser cannot mount a model, data, run, report, or shared-report page.

## Real Cloudflare-compatible stack suite

```bash
npm run test:e2e:full-stack
```

Playwright owns every test process and creates isolated local D1/R2/Queue state. It starts the Hono Worker, FastAPI/OpenTURNS compute adapter, and Vite; applies forward-only migrations; creates an authenticated project; validates the Ishigami model; executes the ten independent-input direct analyses—including global and target-domain HSIC with their real OpenTURNS permutation paths—through the direct workspace; asserts terminal per-task progress persistence; calibrates the official nonlinear exponential model through the named-data studio and real compute boundary; builds a dependent symbolic model, proves a direct API attempt cannot bypass the Sobol applicability rejection, and executes ANCOVA through compute, persistence, and the generic report; exercises Morris and model-based PCE/GPR through their project studios; fits a data-driven GPR from pre-filled paired observations; downloads real report ZIPs; creates and opens an authenticated share link; reloads project history; and verifies persisted D1 records through the API. Successful execution also proves immutable source, dataset, and surrogate-artifact round-tripping through R2.

The test Wrangler configurations use `DEV_AUTH_BYPASS=true` and the browser supplies a synthetic Better Auth session. This is intentionally test-only: CI must not hold a human Cloudflare password, session cookie, or MFA recovery material. Production configuration omits the bypass.

## Deployed production suite

```bash
npm run test:e2e:production
```

This suite runs automatically after deployment and is deliberately read-only. It verifies:

- health and security headers;
- unauthenticated session discovery and the configured Cloudflare provider;
- no generated guest-identity cookie;
- HTTP 401 and `authentication_required` across representative catalog, example, project, run, and shared-report endpoints;
- the static method/model overview without protected API data;
- the login wall on direct private-route navigation;
- WCAG A/AA results for public and gated pages; and
- the real Cloudflare OIDC authorization origin, callback, and PKCE challenge.

Production mutation is not automated because analysis requires a real authenticated account. Authenticated application behavior is exercised by the full-stack suite; an owner can perform the focused manual audit in [DEPLOYMENT.md](DEPLOYMENT.md) when needed.

## Failure evidence

All suites retain trace, screenshot, and video evidence on failure. GitHub Actions uploads the relevant result and HTML-report directories for 14 days.

```bash
npx playwright show-trace path/to/trace.zip
```

Tests use roles and labels instead of screen coordinates except where the coordinate itself is the behavior under test, such as dismissing the mobile drawer. Update tests with observable product behavior whenever the UI or auth contract changes; do not weaken assertions merely to match a regression.
