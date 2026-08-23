# Browser and end-to-end testing

UncertaintyCat uses three complementary Playwright suites. Fast fixture-driven UI evidence is separated from real numerical infrastructure and deployed read-only checks.

## UI contract suite

```bash
npm run test:e2e
```

This suite uses stateful HTTP fixtures with the real React application. It covers:

- the static public overview, absence of private navigation, private-route login wall, and Cloudflare sign-in initiation;
- authenticated desktop/mobile navigation, account identity, sign-out, and theme persistence;
- discovery and retained selection of all 23 reference models;
- Python authoring and the multi-output symbolic OpenTURNS builder;
- model validation success/failure and all 12 plugin configurations;
- deterministic triage, streamed Model Understanding, surrogate promotion, Morris reduction, and Distribution Data Lab composition;
- queued/running/terminal run states, cancellation, exact reruns, and retained study chronology;
- metrics, tables, truncation notices, series, heatmaps, facts, equations, assumptions, provenance, partial failures, sharing, export, and print wiring;
- stored report chat, streaming Markdown, suggestions, citations, and quota failures;
- authenticated read-only shared reports; and
- automated WCAG A/AA scans in light/dark themes, private and public states, expanded controls, and mobile navigation.

The key regression contract is explicit: an unauthenticated browser cannot mount a model, data, run, report, or shared-report page.

## Real Cloudflare-compatible stack suite

```bash
npm run test:e2e:full-stack
```

Playwright owns every test process and creates isolated local D1/R2/Queue state. It starts the Hono Worker, FastAPI/OpenTURNS compute adapter, and Vite; applies forward-only migrations; creates an authenticated study; validates the Ishigami model; executes all 12 plugins; requires all task/report sections to succeed; downloads the real ZIP; creates and opens an authenticated share link; reloads study history; and verifies persisted D1 records through the API. Successful execution also proves immutable source round-tripping through R2.

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
