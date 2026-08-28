# Dependency and autonomous pull-request lifecycle

This document is the operational contract for dependency updates. It separates deterministic merge
authority from future agentic analysis: an agent may diagnose or propose a repair, but only repository
rules and the complete CI evidence can authorize a merge and release.

## Trust boundaries

Dependabot pull requests are untrusted code during `pull_request` CI. They receive no production secrets
and a read-only token. The privileged `workflow_run` automation executes the workflow definition from
`main`, never checks out or executes the pull-request revision, never downloads its artifacts or caches,
and uses the minimum write permissions only in its merge job.

Before approval, `.github/workflows/dependabot-automerge.yml` independently verifies all of the following:

1. the completed run is the repository's `.github/workflows/ci.yml`, was triggered by a Dependabot pull
   request, passed, and names the exact head SHA;
2. the pull request is open, non-draft, authored by `dependabot[bot]`, based on `main`, hosted in this
   repository, and still points at that SHA;
3. every commit is authored by Dependabot and has a valid GitHub verification signature;
4. the branch identifies a configured ecosystem and changes only its dependency surfaces:
   npm manifests/lockfiles, the root uv manifest/lockfile, GitHub workflow/action definitions, or the
   two compute Dockerfiles;
5. the change contains no more than 30 files and GitHub reports it as merge-ready; and
6. the approval and squash merge are bound to the tested head SHA.

Any source-code repair intentionally makes a Dependabot pull request ineligible for this narrow trust
path. Land the compatibility repair through the normal reviewed/agent workflow, then ask Dependabot to
rebase its dependency-only branch so the complete suite tests the combined result.

## Update policy

`.github/dependabot.yml` scans npm, uv, GitHub Actions, and compute-container bases every Monday in
staggered UTC slots.
Security updates are grouped per ecosystem; routine patch/minor updates are grouped to reduce churn;
major updates remain isolated because they often need migration work. Dependabot rebases automatically,
targets the default `main` branch, and may keep up to five version-update pull requests open per ecosystem.
The redundant explicit `target-branch` setting is deliberately omitted so these same group rules also
govern security updates. GitHub security updates are enabled at repository level and are not subject to
the version-update limit.

All external GitHub Actions are pinned to immutable commit SHAs. The readable version comment remains so
Dependabot can raise signed SHA updates. Repository Actions policy also requires SHA pinning, making a
floating tag a platform-level rejection rather than a review convention.

The Python, uv, and Cloudflare Sandbox image references are likewise pinned to registry digests while
retaining readable tags. Docker updates may change only `services/compute/Dockerfile` and
`services/compute/Dockerfile.sandbox`; CI rebuilds both images, imports OpenTURNS and the compute adapter
inside each resulting runtime, and exercises the full-stack compute boundary before the privileged
workflow may merge them.

## Test and release state machine

```text
Dependabot PR
  -> read-only complete CI
     -> failure: label and exact-run audit comment
        -> new version/rebase: fresh complete CI
        -> unchanged non-security failure for 21 days: close, but do not ignore future releases
     -> success: privileged identity/file/signature/SHA verification
        -> clear any failure label belonging to an earlier run of the same revision
        -> exact-head approval + squash merge
           -> explicit complete CI dispatch for the merge SHA
              -> exact successful-main authorization
                 -> production deploy + read-only production Playwright verification
```

The `required` CI job is the single branch-ruleset check. It succeeds only when dispatch integrity and all
five application gates succeed, plus dependency review on pull requests. A skipped underlying test cannot
accidentally satisfy it. The pull-request dependency review rejects newly introduced high/critical known
vulnerabilities, and the TypeScript gate also audits production npm dependencies.

Bot merges use `GITHUB_TOKEN`, which suppresses a normal chained push workflow. The merge workflow
therefore dispatches CI with `expected_sha`; CI rejects the dispatch if GitHub resolved `main` to another
revision. Any authorized maintainer or bot may initiate that exact-SHA dispatch, but an empty dispatch can
never release. Successful exact-SHA CI dispatches deployment explicitly. Deployment independently accepts
only a 40-character commit that belongs to `main` and has successful CI for that exact SHA.

The initial shell integrity check rejects malformed or mismatched dispatch inputs before any test can run.
After all tests pass, a workflow-dispatch-only job revalidates the event, 40-character input, and exact SHA
inside fail-closed shell code before dispatching deployment. Keeping input comparisons out of the job-level
expression prevents GitHub expression coercion from silently skipping an otherwise valid bot release.

## Failed-update lifecycle

`.github/workflows/dependabot-lifecycle.yml` records only a failure belonging to the pull request's current
head. It does not infer compatibility from a red job: infrastructure, test harness, lockfile topology, an
upstream regression, and a real API break are different diagnoses.

The daily stale pass closes an update only when its exact current head has remained on failed required CI
for at least 21 days. It conservatively refuses closure if Dependabot security-alert lookup fails, exempts
updates matching an open security alert, and honours `dependencies:keep-open`. Closure applies
`dependencies:blocked` and leaves an audit comment. It does not issue a Dependabot ignore command, so a
later fixed release can open a new pull request and be tested normally.

Use this diagnosis order for any failure:

1. Identify the exact PR head, workflow run, failed job, and first causal log line.
2. Reproduce with the lockfile from that head in an isolated worktree; do not mutate `main` merely to test.
3. Classify the result as test/infrastructure coupling, transient external failure, lockfile/install
   failure, real API/behavior incompatibility, vulnerability/policy rejection, or scientific drift.
4. Fix repository-owned infrastructure coupling on `main`. For a breaking update, implement and test the
   migration separately, preserving numerical and persisted-schema contracts.
5. Rebase/recreate the Dependabot PR, then require the entire suite again. Never waive or skip a gate.
6. Confirm the exact merge SHA receives post-merge CI, deployment, health, and production UI success.

## 2026-08-28 audit baseline

| Pull requests      | Finding                                                                                                                                                                           | Resolution                                                                                                                                                                                                                                     |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| #51, #52, #53, #55 | Individual GitHub Action updates passed; grouping later superseded them.                                                                                                          | Grouped #61 passed all gates and auto-merged.                                                                                                                                                                                                  |
| #54, #59           | Python proposals became obsolete after dependency/configuration cleanup.                                                                                                          | Dependabot closed them as superseded; neither merged.                                                                                                                                                                                          |
| #56                | Major frontend upgrades exposed genuine TypeScript/Vite/Lucide API breaks and the UI could not mount.                                                                             | Superseded by #60, where the required compatibility migration was made and tested.                                                                                                                                                             |
| #60                | Major npm migration plus explicit source/test repairs.                                                                                                                            | All five gates passed; exact-head workflow approved and auto-merged it.                                                                                                                                                                        |
| #61                | Dependency-only GitHub Actions group.                                                                                                                                             | All five gates passed; exact-head workflow approved and auto-merged it.                                                                                                                                                                        |
| #62, #63           | Four gates passed; full-stack startup assumed Wrangler existed under `apps/api/node_modules`, while npm legitimately hoisted it.                                                  | The harness now resolves Wrangler through the npm workspace contract. Dependabot replaced them with #64, which passed every strengthened gate, auto-merged, and entered exact-SHA post-merge CI.                                               |
| #64                | Grouped routine npm update replacing the two false-negative proposals.                                                                                                            | Passed the strengthened Python, TypeScript, browser, full-stack, image, dependency-review, and aggregate gates; auto-merged.                                                                                                                   |
| #65, #68           | uv updates passed CI, but the first merger revision did not yet recognize native `dependabot/uv/*` branches.                                                                      | Replaced by #69 after the ecosystem/file allowlist was corrected. This was automation-policy coupling, not dependency incompatibility.                                                                                                         |
| #66                | Pygments security update passed CI on a stale branch created before the pip-to-uv configuration correction.                                                                       | Pygments was upgraded in the root uv lock under the complete main-branch suite; the obsolete branch was closed without adding an ignore rule.                                                                                                  |
| #67                | Black security update targeted a development formatter the repository did not run or enforce.                                                                                     | Removed the unused Black dependency; Ruff remains the single enforced Python formatter.                                                                                                                                                        |
| #69                | Grouped routine Pydantic, Pytest, and Ruff uv update.                                                                                                                             | Passed every gate and auto-merged. Its exact-SHA post-merge CI exposed a release-dispatch expression bug, which is corrected by a validated job output.                                                                                        |
| #70, #72           | Security proposals affected only the read-only `Streamlit_Backup` archive, outside the package graph and deployment.                                                              | Closed as not used; corresponding archive alerts were dismissed with that explicit scope rather than changing the historical snapshot.                                                                                                         |
| #71, #73           | The first grouped idna security branch conflicted after #69 merged; replacement #73 was generated from current `main`.                                                            | #73 passed every gate and auto-merged; no compatibility repair or waiver was required.                                                                                                                                                         |
| #74                | The first Docker update moved the compute runtime from Python 3.12 to 3.14. Its first full-stack run lost the unrelated local Wrangler process, while both changed images passed. | The unchanged SHA passed the full-stack rerun, both images built and imported OpenTURNS/the compute app, and Python 3.14 independently passed all 23 examples and 66 Python tests. It then auto-merged and dispatched exact-SHA post-merge CI. |

The alert audit found 196 findings attached only to the excluded Streamlit archive (including its old
path before the directory rename). Those findings were dismissed as `not_used` with an auditable reason.
Active root-runtime findings were resolved through the dependency lifecycle or an exact lock update; the
historical archive itself remains unchanged.

This baseline is evidence for why failed CI must be diagnosed rather than automatically labelled an
incompatible library. It is not an allowlist for future versions.

## Future agent-authored code pull requests

Do not broaden the Dependabot merger to accept source changes or another author. A future autonomous code
agent needs a separate GitHub App identity, explicit allowed scopes, signed/attested commits, its own
policy workflow, adversarial prompt/input controls, and the same strict `required` CI check. Start with
proposal-only operation, then graduate narrow plugin/docs paths after measured evidence. Numerical plugin
changes must still satisfy `AGENTS.md`, `docs/ANALYSIS_PLUGIN_GUIDE.md`, and the OpenTURNS synchronization
contract; no agent label can override scientific validation, authentication, migration, or provenance
rules.
