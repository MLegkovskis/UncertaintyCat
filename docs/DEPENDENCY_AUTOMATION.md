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
   npm manifests/lockfiles, the root Python manifest/lockfile, or GitHub workflow/action definitions;
5. the change contains no more than 30 files and GitHub reports it as merge-ready; and
6. the approval and squash merge are bound to the tested head SHA.

Any source-code repair intentionally makes a Dependabot pull request ineligible for this narrow trust
path. Land the compatibility repair through the normal reviewed/agent workflow, then ask Dependabot to
rebase its dependency-only branch so the complete suite tests the combined result.

## Update policy

`.github/dependabot.yml` scans npm, Python, and GitHub Actions every Monday in staggered UTC slots.
Security updates are grouped per ecosystem; routine patch/minor updates are grouped to reduce churn;
major updates remain isolated because they often need migration work. Dependabot rebases automatically,
targets `main`, and may keep up to five version-update pull requests open per ecosystem. GitHub security
updates are enabled at repository level and are not subject to that version-update limit.

All external GitHub Actions are pinned to immutable commit SHAs. The readable version comment remains so
Dependabot can raise signed SHA updates. Repository Actions policy also requires SHA pinning, making a
floating tag a platform-level rejection rather than a review convention.

## Test and release state machine

```text
Dependabot PR
  -> read-only complete CI
     -> failure: label and exact-run audit comment
        -> new version/rebase: fresh complete CI
        -> unchanged non-security failure for 21 days: close, but do not ignore future releases
     -> success: privileged identity/file/signature/SHA verification
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
revision. Successful bot CI dispatches deployment explicitly. Deployment independently accepts only a
40-character commit that belongs to `main` and has successful CI for that exact SHA.

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

| Pull requests | Finding | Resolution |
| --- | --- | --- |
| #51, #52, #53, #55 | Individual GitHub Action updates passed; grouping later superseded them. | Grouped #61 passed all gates and auto-merged. |
| #54, #59 | Python proposals became obsolete after dependency/configuration cleanup. | Dependabot closed them as superseded; neither merged. |
| #56 | Major frontend upgrades exposed genuine TypeScript/Vite/Lucide API breaks and the UI could not mount. | Superseded by #60, where the required compatibility migration was made and tested. |
| #60 | Major npm migration plus explicit source/test repairs. | All five gates passed; exact-head workflow approved and auto-merged it. |
| #61 | Dependency-only GitHub Actions group. | All five gates passed; exact-head workflow approved and auto-merged it. |
| #62, #63 | Four gates passed; full-stack startup assumed Wrangler existed under `apps/api/node_modules`, while npm legitimately hoisted it. | The harness now resolves Wrangler through the npm workspace contract; the dependency update must rebase and pass the strengthened suite. |

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
