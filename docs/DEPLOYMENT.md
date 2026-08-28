# Deployment runbook

This is the production runbook for [uncertaintycat.com](https://uncertaintycat.com). The application is delivered as one Cloudflare Worker origin with React assets, Hono APIs, Better Auth, D1, R2, Queues, a selectable Groq/Workers AI adapter, and Cloudflare Sandbox compute.

## Production topology

- `uncertaintycat.com`: canonical Worker origin for static assets, API, auth cookies, and OAuth callbacks.
- `www.uncertaintycat.com`: permanent redirect to the apex.
- D1: Better Auth, studies, model versions, assessments, datasets, surrogates, runs, tasks, reports, conversations, usage, and hashed share links.
- R2: immutable Python source, original private datasets, and promoted OpenTURNS XML artifacts.
- Queue and dead-letter queue: analysis-task delivery and retry exhaustion.
- Per-run Cloudflare Sandbox container: Python/OpenTURNS compute with egress disabled.
- AI adapter: Groq by default for Model Understanding and report explanations, with the Workers AI binding retained as a deployment-selectable fallback.

Keeping one origin is intentional. If that changes, review `PUBLIC_WEB_ORIGIN`, `BETTER_AUTH_URL`, trusted origins, cookie policy, CORS, and the OIDC callback as one security change.

## Authentication boundary

Cloudflare Access is the OIDC provider presented by Better Auth. The production callback is:

```text
https://uncertaintycat.com/api/auth/callback/cloudflare
```

The authorization-code flow uses PKCE and the `openid`, `email`, and `profile` scopes. The site itself is not wrapped in a Cloudflare Access self-hosted application because the static overview remains public. Users explicitly select **Continue with Cloudflare**; after authentication, Better Auth stores the session in D1.

The Worker permits anonymous access only to static assets and SPA HTML, `/health`, `/api/auth/*`, and `/api/v1/session`. Middleware rejects every other `/api/v1/*` request with HTTP 401, including catalogs, example source, shared reports, and exports. Production must never define `DEV_AUTH_BYPASS`.

## GitHub configuration

Required production secrets:

- `CLOUDFLARE_API_TOKEN`;
- `BETTER_AUTH_SECRET` with at least 32 random bytes;
- `CLOUDFLARE_ACCESS_CLIENT_ID`;
- `CLOUDFLARE_ACCESS_CLIENT_SECRET`;
- `GROQ_API_KEY` when `AI_PROVIDER=groq` (the current default).

Required repository variables:

- `CLOUDFLARE_ACCOUNT_ID`;
- `CLOUDFLARE_D1_DATABASE_ID`;
- `AI_PROVIDER`, exactly `groq` or `cloudflare` (unset also defaults to `groq`).

Non-secret production origins, the default provider, and the Access issuer are checked into `apps/api/wrangler.production.jsonc`. `.github/scripts/prepare-cloudflare-config.mjs` creates an ignored deployment config containing the repository-provided D1 ID and validates the provider variable. Secrets are passed to Wrangler through an ephemeral permission-restricted file and removed in an `always()` step.

Configure Groq without printing the key:

```bash
gh secret set GROQ_API_KEY < /secure/path/to/groq.txt
gh variable set AI_PROVIDER --body groq
```

To return to the Cloudflare models, set `gh variable set AI_PROVIDER --body cloudflare` and redeploy an exact successful `main` SHA. Switching back requires no code change; the Workers AI binding remains deployed. Model Understanding cache keys include provider and model ID, so a switch cannot reuse prose produced by the previous provider.

The deployment token should remain scoped to the UncertaintyCat Cloudflare account/zone and only the Workers Scripts, Containers, D1, R2, Queues, Workers AI, and Workers Routes capabilities needed by the workflow. Never store the global API key in GitHub or the Worker.

Workers Logs are enabled at a 100% head sample rate. AI generation events contain request/record IDs, provider, model ID,
outcome, wall time, and output length only; they deliberately exclude prompts, model source, and persisted numerical
evidence. Groq uses `openai/gpt-oss-20b` for concise Model Understanding and `openai/gpt-oss-120b` for grounded report tool use; see Groq's [model catalog](https://console.groq.com/docs/models), [OpenAI-compatible endpoint](https://console.groq.com/docs/openai), and [local tool-calling guide](https://console.groq.com/docs/tool-use/local-tool-calling). Model Understanding requests should normally complete in a few seconds. Its primary and fallback attempts
have 12- and 15-second deadlines; exhausting both becomes an explicit retryable 504 and is not charged to the
successful-regeneration quota.

## Automatic delivery

`.github/workflows/ci.yml` runs on every push to `main`, every pull request, and manual dispatch. It has no repository-variable gate or skip path. The required jobs are:

1. Python formatting, lint, typing, unit/scientific/integration tests, and all 23 reference models;
2. TypeScript typing, unit tests, and production build;
3. mocked-browser navigation and accessibility;
4. real local Worker/D1/R2/Queue/compute browser journey;
5. local compute and Cloudflare Sandbox image builds.

The branch ruleset requires the aggregate `required` job from GitHub Actions and requires the pull-request
head to be current with `main`. That aggregate explicitly checks every result above, so GitHub's treatment
of a skipped job as successful cannot hide a missing gate. Pull requests additionally run GitHub dependency
review and reject newly introduced high/critical vulnerabilities; the TypeScript job audits production npm
dependencies. Every external action is pinned to an immutable commit SHA, and repository Actions policy
rejects floating action tags.

`.github/workflows/deploy.yml` listens for successful `CI` completion on `main`. It checks out the exact successful commit, builds web assets, generates the Wrangler config, validates secrets, applies forward-only D1 migrations, deploys the Worker/assets/bindings/queue consumer/Sandbox image, checks production health, and runs the read-only production Playwright suite.

There is deliberately no `AUTOMATION_ENABLED` variable, pause script, manual-only release path, or commit-message bypass. GitHub workflow concurrency may cancel an older in-progress CI run for the same ref; production deployments are serialized and are not cancelled mid-flight.

Dependabot pull requests follow the same full CI path. The privileged post-CI workflow additionally verifies
the trusted CI workflow/run identity, same-repository branch, signed Dependabot-only commits, ecosystem file
allowlist, bounded change size, merge readiness, and exact current head before exact-head approval and squash
merge. It explicitly dispatches `CI` with the resulting merge SHA. Because GitHub suppresses chained workflow
events created by `GITHUB_TOKEN`, that bot-triggered CI dispatches `deploy.yml` directly for its exact tested
SHA after the aggregate gate passes. Deployment revalidates that manual/bot inputs belong to `main` and have
successful CI for that exact revision. Direct pushes and human dispatches retain the normal `workflow_run`
release path.

Failed current-head dependency runs receive a label and an idempotent audit comment. A daily lifecycle pass
closes an unchanged non-security failure after 21 days, unless `dependencies:keep-open` is present; it never
adds a permanent ignore rule. Security-alert lookup failure disables stale closure conservatively. The full
rationale, update grouping, failure classification procedure, historical audit, and separate trust design for
future agent-authored code are in [DEPENDENCY_AUTOMATION.md](DEPENDENCY_AUTOMATION.md).

## Migration rules

- Add a new numbered SQL file under `apps/api/migrations/` for every schema change.
- Never edit or delete a migration already applied to production.
- Keep migrations forward-compatible with the currently deployed Worker when a staged rollout could overlap versions.
- Back up or export material production data before a destructive schema/data migration.
- A code rollback must not require reversing a D1 migration.

## Release verification

The automatic production suite verifies health, security headers, unauthenticated session discovery, absence of guest cookies, HTTP 401 across representative private APIs, the static overview, private-route login wall, accessibility, and the exact Cloudflare OIDC authorization request/callback/PKCE contract.

For an authenticated manual release audit:

- sign in through Cloudflare and confirm the account identity is visible;
- verify a separate unauthenticated browser cannot read catalogs, examples, studies, runs, reports, share tokens, or exports;
- create a study, validate a reference model and custom model, and run a small multi-analysis suite;
- observe queued/running/terminal states and a partial-failure report;
- upload a small dataset, rank distributions, and retain a generated model draft;
- build and promote a surrogate, then test both the current-project and new-project handoffs;
- create a disposable project and verify exact-name confirmation deletes its D1 records and R2 artifacts;
- download and inspect the evidence bundle and exact source;
- create and open a share link while authenticated, then expire or revoke it;
- ask report chat for a numerical value and confirm the persisted-result citation;
- inspect logs for request/run/task identifiers without source or secrets;
- confirm the Sandbox has no egress/secrets and is destroyed after execution.

## Incident and rollback notes

If CI fails, no deployment should start. If deployment or production verification fails, inspect the exact Actions run and Cloudflare Worker/Sandbox logs, fix forward on `main`, and let the normal chain redeploy. Restore a prior Worker/static revision only when it remains compatible with every applied D1 migration. Rotate a suspected secret in Cloudflare/GitHub first, then redeploy; never commit replacement credentials.
