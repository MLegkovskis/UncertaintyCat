# Analysis plugin guide

An analysis plugin is the only supported way to add new numerical capability to the new application.

## Contract

Create `uncertaintycat_core/plugins/<key>.py` with:

1. A `StrictModel` configuration class. Include `seed` and `output_targets` where relevant and put hard,
   defensible bounds on computational parameters.
2. An `AnalysisPlugin[Config]` implementation with a stable key, semantic implementation version,
   user-facing metadata, assumptions, applicability flags, resource class, and `run` method.
3. A module-level `plugin` instance.
4. Registration in `uncertaintycat_core/catalog.py`.

`run` returns `(AnalysisPayload, model_evaluations)`. Payloads may contain scalar metrics, tables, series,
matrices, grounded facts, warnings, and artifact references. All values must serialize as strict JSON:
never emit NumPy objects, NaN, or infinity.

## Applicability is part of correctness

Reject an analysis with `IncompatibleAnalysisError` when its mathematical interpretation is invalid.
Examples already enforced:

- classical Sobol, FAST, and Morris reject dependent copulas;
- scalar algorithms reject missing output indices;
- Sobol rejects constant selected outputs;
- PCE reports construction incompatibilities instead of returning a misleading surrogate.

Use warnings only for conditions where the result remains meaningful. Put durable method assumptions in
the plugin metadata so every result and report retains them.

## Versioning

- Increment `version` for numerical behavior, defaults, algorithm, or interpretation changes.
- Increment `result_schema_version` for a breaking payload-shape change.
- Never reinterpret a persisted key/version combination in place.
- If an old config can be migrated without ambiguity, implement an explicit migration and test it.

## Tests required for merge

- deterministic repeatability at a fixed seed;
- a known analytical or published benchmark with statistical tolerances;
- invalid dimension/copula/output/config behavior;
- strict JSON serialization;
- resource bounds and model-evaluation accounting;
- service execution through `/v1/execute` for any new contract behavior;
- catalog and browser rendering for new payload shapes.

Prefer convergence assertions or known index structure over snapshotting random floating-point output.
Record OpenTURNS and plugin versions with any accepted numerical drift.

## Frontend defaults

The catalog exposes JSON Schema, descriptions, assumptions, output support, and resource class. The
current run composer has a small default-config mapping in `apps/web/src/pages/Workspace.tsx`; add a safe
default there for a new key. The next evolution is a constrained JSON-schema form renderer, after which
most plugins will require no frontend code.

## OpenTURNS release intake

The weekly scout opens an issue when the exact pin lags PyPI. Treat that as discovery, never as an
automatic upgrade. Review upstream release notes, isolate the dependency change, run the complete benchmark
suite, measure drift, and introduce newly useful algorithms as separate plugins. Do not expose every upstream
class simply because it exists.

The complete discovery rubric, stable/experimental API policy, state checkpoint, implementation procedure,
required evidence, deployment observation, and copy-paste scheduled-agent prompt live in
[`docs/openturns-sync/README.md`](openturns-sync/README.md). That workflow was rehearsed end to end when the
OpenTURNS 1.27 stable Gaussian-process regression API became the `gpr` plugin.
