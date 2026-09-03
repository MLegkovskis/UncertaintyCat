# Operations and application monitoring

UncertaintyCat has two complementary observability planes. The authenticated owner dashboard at `/operator`
answers **what users and scientific work are retained right now**. Cloudflare's native observability answers
**what happened inside the runtime and infrastructure**. Neither substitutes for the other.

## Owner dashboard

The React dashboard calls `GET /api/v1/operator/overview?hours=24|168|720`. The Worker requires a valid
Cloudflare/Better Auth session and then matches the normalized identity email against `OPERATOR_EMAILS`.
Anonymous requests receive 401; authenticated non-operators receive 403. The UI hides the Operations link for
non-operators, but backend enforcement is authoritative.

The endpoint reads D1 only and returns:

- all-time user, project, and retained-model counts;
- run and analysis-task outcomes for the selected time window;
- per-method volume, completion rate, and average recorded duration;
- the 100 most recent runs, users, and projects;
- the 100 newest failed analysis tasks, distribution fits, Model Understanding generations, and tasks whose
  queued/running state has been unchanged for more than 15 minutes.

Stored errors are parsed into a bounded code and whitespace-normalized 240-character message. Source code,
input/output samples, task configuration/result envelopes, dataset contents, report/chat text, AI prompts, and
R2 keys are not selected or serialized. The response has `Cache-Control: private, no-store`. The browser
refreshes every 30 seconds only while the view is open and offers a manual refresh; this is near-real-time D1
state rather than a streaming event system.

Set operators in checked-in non-secret deployment configuration:

```json
"OPERATOR_EMAILS": "owner@example.com,second-operator@example.com"
```

Treat every change as an authorization change and release it through the full `main` CI/deployment chain.
Never use `DEV_AUTH_BYPASS` in production. The local full-stack profile deliberately grants
`developer@localhost` operator access so Playwright can exercise this boundary without human credentials.

## Cloudflare runtime plane

- [Workers Logs](https://developers.cloudflare.com/workers/observability/logs/workers-logs/) captures invocation
  logs, structured application events, errors, and uncaught exceptions. Search by the request ID returned in
  the `X-Request-Id` header, then correlate run/task IDs where present.
- [Workers Metrics](https://developers.cloudflare.com/workers/observability/metrics-and-analytics/) provides
  requests, errors, CPU time, wall time, and duration for the Worker.
- [D1 metrics and query insights](https://developers.cloudflare.com/d1/observability/metrics-analytics/) expose
  database reads/writes, storage, latency, and expensive queries in the dashboard, GraphQL API, and Wrangler.
- Queue, dead-letter queue, Sandbox/container, R2, and Workers AI dashboards remain the infrastructure sources
  for delivery backlog, retry exhaustion, compute capacity, artifacts, and provider usage.

The Worker already enables observability at a 100% head sample rate with uploaded source maps. Structured logs
must remain source-free and payload-free. Do not add a Cloudflare account analytics token to the application
Worker merely to reproduce native charts inside `/operator`; that would expand privilege and secret surface
without improving the exact D1 state view.

## Incident workflow

1. Open `/operator`, choose the relevant window, and identify the affected user, project, run, method, error
   code, and time. Check whether work is failed or merely active; a task is flagged stale after 15 minutes.
2. Open the retained run when available. Preserve its immutable numerical evidence and record its request/run/
   task identifiers.
3. Query Workers Logs around the recorded time/request identifier, then inspect Queue/DLQ and Sandbox metrics
   for compute failures or capacity symptoms. Use D1 metrics for database latency or query regressions.
4. Reproduce locally through the normal test commands. Never edit production results or applied migrations to
   hide a symptom; fix forward on `main`.
5. Verify the exact repair SHA through full CI, automatic migration/deployment, production Playwright, health,
   and a refreshed operator snapshot.

## Future time-series expansion

If the product needs long-term funnels, cohorts, latency percentiles, alerts, or high-cardinality event series,
add a source-free [Workers Analytics Engine](https://developers.cloudflare.com/analytics/analytics-engine/)
dataset with documented retention and dimensions. Its writes are non-blocking and it can feed external Grafana
views later. Do not use it as mutable numerical provenance, and do not replace D1's exact retained state with
eventual analytics aggregates.
