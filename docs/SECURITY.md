# Security model

## The primary rule

User-authored Python is untrusted code. `preflight_source` improves error messages and blocks obvious
imports/calls, but AST filtering is not a security boundary and must never be described as one.

## Trust boundaries

| Boundary          | Current control                                                               | Production requirement                                |
| ----------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------- |
| Browser to Worker | authenticated API middleware, strict Zod inputs, ownership checks, secure headers, quota, cookies | abuse controls and CSP review                         |
| Public surface    | static homepage, health, auth handlers, and session discovery only            | regression tests for the explicit allowlist           |
| Worker to compute | private Sandbox Durable Object binding or test-only local bearer token        | keep the Sandbox binding private                      |
| Python execution  | fresh non-root Sandbox, no secrets, egress disabled, timeout + forced destroy | adversarial verification                              |
| Results/chat      | structured stored facts and tool-only numerical access                        | prompt-injection evals, rate limits, retention policy |
| Sharing           | authentication plus 256-bit token, SHA-256 stored, expiry/revocation          | audit log and owner-visible link management           |

## Production compute launch gate

The production adapter now implements the core isolation boundary, but do not enable custom Python until
the Workers Paid deployment and adversarial verification confirm these properties:

- a fresh isolated boundary per model validation/run
  ([Cloudflare Sandbox](https://developers.cloudflare.com/sandbox/) or an equivalent container job);
- no cloud credentials, internal API token, OAuth secret, or other user data inside that boundary;
- outbound networking disabled by default;
- read-only base filesystem and a disposable size-limited scratch directory;
- non-root UID, no Linux capabilities, no privilege escalation, and a restrictive seccomp profile;
- wall-clock, CPU, memory, process-count, output-size, and source-size limits;
- termination that is authoritative even for native-code hangs;
- signed/authenticated input and result exchange;
- structured logs containing run/task IDs but no source or secrets;
- image provenance, vulnerability scanning, and prompt incident response.

`IsolatedComputeSandbox` disables internet egress, receives no Worker secrets, runs as an unprivileged user,
has a fixed instance class, executes a constant command with a three-minute timeout, and is scoped to one
run. Finalization/cancellation destroys it; failures destroy it immediately. There is no repository Compose
launcher; container hardening is verified through the production Sandbox image and CI builds.

## Authentication and authorization

Better Auth is wired to D1 with Cloudflare Access as its OIDC provider. Every project/model/run/report/chat
operation resolves an authenticated owner and applies ownership in SQL. Middleware covers every
`/api/v1/*` route except `/api/v1/session`, so catalog/example reads, shared reports, exports, and all
compute-adjacent operations fail with HTTP 401 before route-specific validation or storage access. The
browser repeats the boundary with an authenticated route wrapper, but server enforcement is authoritative.

Unauthenticated session discovery returns `authenticated: false` with an empty owner ID. It does not create
a guest cookie, project, quota ledger, or any other durable identity. The public homepage uses hard-coded
descriptive examples and never receives executable source from the API.

`DEV_AUTH_BYPASS=true` exists only in Wrangler configurations used by the isolated full-stack CI suite,
where no human Cloudflare credential is available. Production configuration must omit it and use a strong
`BETTER_AUTH_SECRET`.

## AI boundary

The deployment-selected Groq or Cloudflare model receives the question, up to 20 prior conversation messages, and outputs from five read-only tools
covering the outline, scalar summary, bounded tables, series, and matrices. The system prompt forbids
inventing or recalculating numbers, and numerical claims must cite an analysis key, field kind, and name.
The model cannot run Python or mutate a report. This reduces, but does not eliminate, prompt-injection and
misinterpretation risk; AI prose must remain visibly separate from numerical output.

Model equations are not inferred by an AI from private code. The Model Understanding response prepends only
curated bundled-reference equations or formulas from an authenticated user's validated guided-builder
definition, without sending those private formulas to the selected AI provider. Report chat must resolve a
stored field to its actual value before answering; citation tokens support that value rather than replacing it.
The non-numerical report-section inventory is supplied to chat so it cannot overlook a completed analysis,
but all numerical and ranking claims still require a bounded read-only tool result.

The orchestration is provided by the MIT-licensed Vercel AI SDK with Zod schemas. Groq is the default through
the official `@ai-sdk/groq` provider and `https://api.groq.com/openai/v1`; Cloudflare remains selectable through
the pinned `workers-ai-provider` and an account binding. `AI_PROVIDER` is validated at deployment, only the
selected adapter executes, and provider/model identity is included in the Model Understanding cache key.
Neither path receives Python source or private R2 objects. Groq's key is a Worker secret and is never returned
by session discovery, written to logs, or included in browser assets.

## Secrets

Real `.dev.vars` and environment files are ignored. The full-stack test configuration contains only
non-secret synthetic values. In production use Cloudflare secrets for auth, OAuth, and `GROQ_API_KEY`. Never put secrets in
checked-in Wrangler configuration, model source, R2 custom metadata, logs, report bundles, or client-side
environment variables.

## Remaining launch work

- export-account and formal retention endpoints (project deletion is implemented);
- share-link list/audit UI and security-event audit records;
- CSRF/provider callback deployment tests;
- content-security-policy tuning for CodeMirror and application assets;
- WAF/rate-limit rules by identity/IP and cost class;
- dependency/image scanning with a documented vulnerability policy;
- adversarial sandbox escape and report-chat evaluations.
