# Browser and end-to-end testing

UncertaintyCat uses three complementary Playwright suites. Fast fixture-driven UI evidence is separated from real numerical infrastructure and deployed read-only checks.

## UI contract suite

```bash
npm run test:e2e
```

This suite uses stateful HTTP fixtures with the real React application. It covers:

- the static public overview, absence of private navigation, private-route login wall, and Cloudflare sign-in initiation;
- authenticated desktop/mobile navigation, account identity, immediate sign-out without a hard refresh, and theme persistence;
- discovery of all 23 reference models in the unified, immediately editable Python authoring view;
- blank/manual model naming, resizable Python authoring, project creation from the project index, project-scoped studio navigation, and the multi-output symbolic OpenTURNS builder;
- model validation success/failure, deterministic direct/reduction/surrogate routing, and the direct-only analysis catalog;
- integrated validation/Model Understanding, primary/fallback model policy, single-flight polling, explicit uncharged timeout/failure states, model- and data-driven surrogate fitting, dedicated Morris screening/reduction, and Distribution Fitting composition with beam sample data;
- desktop bounding-box and screenshot evidence guarding the validated two-column layout against header overlap;
- queued/running/terminal run states, cancellation, exact reruns, and project-oriented history;
- metrics, tables, truncation notices, series, heatmaps, facts, equations, assumptions, provenance, partial failures, sharing, bundle export, and direct PDF-download wiring;
- stored report chat, streaming Markdown, suggestions, citations, and quota failures;
- authenticated read-only shared reports; and
- automated WCAG A/AA scans in light/dark themes, private and public states, expanded controls, and mobile navigation.

The key regression contract is explicit: an unauthenticated browser cannot mount a model, data, run, report, or shared-report page.

## Real Cloudflare-compatible stack suite

```bash
npm run test:e2e:full-stack
```

Playwright owns every test process and creates isolated local D1/R2/Queue state. It starts the Hono Worker, FastAPI/OpenTURNS compute adapter, and Vite; applies forward-only migrations; creates an authenticated project; validates the Ishigami model; executes the nine direct-analysis plugins through the direct workspace; exercises Morris and model-based PCE/GPR through their project studios; fits a data-driven GPR from pre-filled paired observations; downloads the real ZIP; creates and opens an authenticated share link; reloads project history; and verifies persisted D1 records through the API. Successful execution also proves immutable source, dataset, and surrogate-artifact round-tripping through R2.

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
