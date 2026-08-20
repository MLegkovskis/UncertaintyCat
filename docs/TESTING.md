# Browser and end-to-end testing

UncertaintyCat uses three complementary Playwright suites. They deliberately
separate fast UI feedback from real numerical and production infrastructure
checks.

## 1. UI contract suite

Run with:

```bash
npm run test:e2e
```

This suite uses stateful HTTP fixtures but the real React application. It runs
on every pull request and every push to `main`, and covers:

- desktop and mobile navigation;
- guest, Cloudflare sign-in initiation, retained-user account, and sign-out;
- project onboarding, Python authoring, and guided-builder authoring;
- validation success and failure;
- selection and method-specific configuration of all 11 plugins;
- run progress, task persistence, terminal state, and cancellation;
- activity empty/history states;
- report metrics, tables, truncated-data notices, series, heatmaps, facts,
  assumptions, partial failures, contents links, sharing, export wiring, and
  print-to-PDF wiring;
- authenticated stored chat history, suggestions, keyboard submission,
  streamed answers, citations, quota errors, and guest denial at both UI and
  API boundaries;
- read-only shared reports; and
- automated WCAG A/AA scans for every routed screen, expanded builder/account
  states, and the mobile drawer.

## 2. Real local Cloudflare-stack suite

Run with:

```bash
npm run test:e2e:full-stack
```

Playwright starts a fresh isolated local D1/R2/Queues state, the Hono Worker,
the FastAPI/OpenTURNS compute service, and Vite. The browser creates a retained
project, validates the curated Ishigami model, selects all 11 analysis plugins,
and requires all 11 tasks and report sections to succeed. It then downloads the
real ZIP, creates and opens a share link, reloads activity history, and verifies
the persisted D1 records through the API. Successful task execution also proves
the immutable source was written to and reread from R2.

The local Worker uses `DEV_AUTH_BYPASS=true` and the browser supplies a clearly
labelled synthetic retained-user session. This is intentional: CI never holds a
human Cloudflare password, cookie, or MFA recovery material.

## 3. Deployed production suite

Run the read-only checks with:

```bash
npm run test:e2e:production
```

These checks run automatically after deployment. They verify production health
and security headers, the live 11-plugin Sandbox catalog, guest session policy,
all public screens, deployed WCAG results, and the real Cloudflare Access OIDC
authorization request including PKCE and the exact callback URI.

A mutation test is available for release verification:

```bash
E2E_LIVE_MUTATIONS=true npm run test:e2e:production -- --grep "optional live mutation"
```

It performs one low-cost curated guest analysis through production D1, R2,
Queues, Cloudflare Sandbox, and report/export/share endpoints. It also proves
the report-chat UI is absent and a direct chat API request receives HTTP 401.
It is opt-in because it writes real production data and consumes paid services.
Any test project should be deleted from D1 and its exact model object deleted
from R2 after verification.

## Failure evidence

All suites retain a trace, screenshot, and video on failure. GitHub Actions
uploads those artifacts for 14 days. Open a trace locally with:

```bash
npx playwright show-trace path/to/trace.zip
```

The suites use role- and label-based locators instead of CSS coordinates except
where the coordinate itself is under test (clicking outside the mobile drawer).
This makes failures correspond to user-observable regressions.
