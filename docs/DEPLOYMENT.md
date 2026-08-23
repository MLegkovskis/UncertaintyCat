# Deployment runbook

This is the production runbook for `uncertaintycat.com`. The backing resources and Cloudflare Access OIDC
application were created on 2026-08-19. The Worker is attached to the live domain only by the verified
production workflow.

## Target topology

- `uncertaintycat.com`: React static assets and Hono API in one Cloudflare Worker.
- D1: auth, projects, versions, runs, tasks, reports, chat, quotas, share links.
- R2: immutable model sources and large artifacts.
- Queues + DLQ: analysis task delivery and terminal failures.
- Per-run [Cloudflare Sandbox SDK](https://developers.cloudflare.com/sandbox/) container (Workers Paid):
  Python/OpenTURNS compute protocol with public egress disabled.

Keeping web and API on one origin simplifies cookies and CORS. If separate origins are used, set
`PUBLIC_WEB_ORIGIN`, `BETTER_AUTH_URL`, cookie policy, trusted origins, and OAuth callbacks explicitly.

## Local release rehearsal

```bash
./start_local.sh
uv run pytest
npm run typecheck
npm run test:ts
npm run build
npm run test:e2e
npm run test:e2e:full-stack
docker build -f services/compute/Dockerfile -t uncertaintycat-compute:local .
```

## Cloudflare resource state and bootstrap

Already created in account `062711e4730b4f1bfc21801a71cfe589`:

- private R2 bucket `uncertaintycat-artifacts-production`;
- queue `uncertaintycat-runs-production`;
- dead-letter queue `uncertaintycat-runs-dlq-production`;
- D1 database `uncertaintycat-production` (`ececff7c-67ee-4f74-bd96-3a8f10784fbc`) in Western Europe, with
  `0001_initial.sql` applied;
- active full Cloudflare zone `uncertaintycat.com`;
- Workers Paid with Containers/Sandbox enabled;
- Zero Trust organization `uncertaintycat.cloudflareaccess.com`;
- Cloudflare identity provider and OIDC SaaS application for the Better Auth callback;
- GitHub `production` environment with the account/D1 variables and Access OIDC secrets;
- scoped GitHub deployment token and generated Better Auth session secret.

Still required:

1. Run `Deploy production` once and complete the live smoke test.

`.github/scripts/prepare-cloudflare-config.mjs` replaces the checked-in all-zero D1 placeholder in an ignored
generated config. Resource IDs and secrets are never committed.

## Secrets and variables

Secrets:

- `BETTER_AUTH_SECRET` (at least 32 random bytes);
- `CLOUDFLARE_API_TOKEN` for GitHub deployment;
- `CLOUDFLARE_ACCESS_CLIENT_ID` and `CLOUDFLARE_ACCESS_CLIENT_SECRET`.

Non-secret environment-specific values:

- `BETTER_AUTH_URL` and `PUBLIC_WEB_ORIGIN` are fixed to the apex in the production config;
- `CLOUDFLARE_ACCOUNT_ID=062711e4730b4f1bfc21801a71cfe589`;
- `CLOUDFLARE_D1_DATABASE_ID=ececff7c-67ee-4f74-bd96-3a8f10784fbc`;
- `CLOUDFLARE_ACCESS_ISSUER` is fixed to the OIDC application issuer in the production config;
- exact SHA-256 allowlist generated from the canonical `examples/*.py` catalog.

Workers AI uses the `AI` binding; there is no model API key in the Worker. The local legacy Streamlit
utility can use the Workers AI REST API, but it is not part of production.

## Cloudflare sign-in

Cloudflare Access is the OIDC provider presented to Better Auth. Its only upstream identity method is
Cloudflare account authentication, and the allow policy accepts any user who successfully authenticates
through that method. It is configured with:

- team domain: `uncertaintycat.cloudflareaccess.com`;
- callback: `https://uncertaintycat.com/api/auth/callback/cloudflare`;
- authorization-code flow with PKCE;
- scopes: `openid`, `email`, and `profile`;
- one-hour Access token lifetime.

The public site is not placed behind an Access self-hosted application: guests can still browse and execute
approved examples. Access is used only as the OIDC broker for an explicit **Continue with Cloudflare** flow.
Better Auth stores users, accounts, and sessions in D1. Projects and runs use the authenticated Better Auth
user ID as owner, so the Activity screen restores prior executions after a user signs back in.

Production must not define `DEV_AUTH_BYPASS`. Rotate secrets through Cloudflare secret management rather
than committing `.dev.vars` or editing `wrangler.jsonc` with real values.

## GitHub delivery

CI tests Python, TypeScript, browser navigation, the real local stack, and both container images. Every automatic job is gated by the repository variable `AUTOMATION_ENABLED`. A manual CI dispatch always runs the full suite, and `.github/workflows/deploy.yml` runs after successful eligible CI on `main`, then:

1. builds the React assets;
2. generates the production Wrangler file from the D1 repository variable;
3. checks all required secrets;
4. applies forward-only D1 migrations;
5. deploys the Worker, assets, bindings, queue consumer, and Sandbox image;
6. smoke-tests `https://uncertaintycat.com/health`.

Use `scripts/automation.sh pause|resume|status`. For a release while automatic automation is paused, commit and push a clean `main` that exactly equals `origin/main`, then run `scripts/automation.sh release`; it dispatches full CI for that commit and prints the authoritative workflow URL.

The deployment token is scoped to this account and zone with Account permissions for Workers Scripts,
Containers, D1, R2, Queues, and Workers AI plus Zone Workers Routes. The account-wide global API key is not
stored in GitHub or exposed to the Worker.

## Release smoke test

- sign in with Cloudflare and verify guest isolation in another browser profile;
- save the curated example, reject modified guest source, and save authenticated custom source;
- run a small multi-analysis suite and observe queued/running/terminal states;
- force one invalid task and verify `partially_succeeded` report behavior;
- download and inspect the ZIP manifest/JSON/CSV files;
- create, open, expire, and revoke a share link;
- ask chat for a number and verify its result-field citation;
- validate security headers, CORS, cookies, logs, quotas, and no secret/source leakage;
- confirm compute has no egress/secrets and is destroyed after execution;
- verify the previous Worker/static deployment can be restored without reversing a D1 migration.

## Domain cutover

The Worker declares custom domains for the apex and `www`; the Worker permanently redirects `www` to the
apex. Wrangler creates the required custom-domain DNS records on the first successful deployment. Do not
run that deployment until the release smoke test can exercise D1 and Sandbox end to end.
