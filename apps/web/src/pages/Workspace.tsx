import { python } from "@codemirror/lang-python";
import { useMutation, useQuery } from "@tanstack/react-query";
import CodeMirror from "@uiw/react-codemirror";
import type {
  AnalysisCatalogEntry,
  ExampleCatalogEntry,
  ModelUnderstanding,
  ModelVersion,
} from "@uncertaintycat/contracts";
import {
  Beaker,
  ArrowRight,
  ArrowDown,
  ArrowUp,
  Check,
  Code2,
  FlaskConical,
  Gauge,
  Play,
  Plus,
  Save,
  SlidersHorizontal,
  Trash2,
  Search,
  ScanSearch,
  Waves,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Link,
  useNavigate,
  useParams,
  useSearchParams,
} from "react-router-dom";

import { api, readTextStream } from "../api";
import { Markdown } from "../components/Markdown";
import { ProjectNav } from "../components/ProjectNav";
import {
  buildSymbolicModel,
  createBuilderVariable,
  distributionDefinition,
  DISTRIBUTION_REGISTRY,
  identityCorrelation,
  validateBuilder,
  type BuilderSpec,
  type BuilderVariable,
} from "../examples";
import { EmptyState } from "../components/Status";
import { useTheme } from "../components/Theme";

type AuthorMode = "source" | "builder";

const MODEL_UNDERSTANDING_CLIENT_TIMEOUT_MS = 35_000;
const MODEL_UNDERSTANDING_POLL_MS = 750;

function abortableDelay(milliseconds: number, signal: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    if (signal.aborted) {
      reject(new DOMException("The request was aborted.", "AbortError"));
      return;
    }
    const onAbort = () => {
      window.clearTimeout(timeout);
      reject(new DOMException("The request was aborted.", "AbortError"));
    };
    const timeout = window.setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, milliseconds);
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

function modelUnderstandingError(caught: unknown) {
  if (caught instanceof DOMException && caught.name === "AbortError") {
    return "The AI provider did not answer within 35 seconds. Please retry; failed requests are not charged.";
  }
  return caught instanceof Error
    ? caught.message
    : "Model Understanding failed. Please retry; failed requests are not charged.";
}

const SCALAR_ANALYSES = new Set([
  "ancova",
  "sobol",
  "fast",
  "hsic",
  "target_hsic",
  "taylor",
  "convergence",
  "reliability",
]);

type ReliabilityMethod =
  | "FORM"
  | "SORM"
  | "MONTE_CARLO"
  | "DIRECTIONAL_SAMPLING"
  | "SUBSET_SAMPLING";

function analysisConfig(
  key: string,
  sampleSize: number,
  model: ModelVersion,
  reliability: {
    method: ReliabilityMethod;
    threshold: number;
    operator: ">" | ">=" | "<" | "<=";
    maximum_evaluations: number;
    target_coefficient_of_variation: number;
  },
  targetHsic: {
    threshold: number;
    operator: "<=" | ">=";
    permutations: number;
  },
): Record<string, unknown> {
  switch (key) {
    case "ancova":
      return {
        training_size: Math.max(64, Math.min(sampleSize, 10_000)),
        validation_size: Math.max(
          64,
          Math.min(Math.ceil(sampleSize / 2), 2_000),
        ),
        ancova_sample_size: Math.max(128, Math.min(sampleSize * 2, 20_000)),
      };
    case "sobol":
      return { base_sample_size: Math.max(64, sampleSize) };
    case "taylor":
      return { validation_size: Math.max(64, Math.min(sampleSize, 5_000)) };
    case "reliability":
      return reliability;
    case "target_hsic": {
      const maximum = Number(
        model.assessment?.recommendations.find(
          (recommendation) => recommendation.capability === "target_hsic",
        )?.safe_config?.maximum_sample_size ?? 250,
      );
      return {
        sample_size: Math.max(50, Math.min(sampleSize, maximum)),
        ...targetHsic,
      };
    }
    case "hsic": {
      const recommendation = model.assessment?.recommendations.find(
        (candidate) => candidate.capability === "hsic",
      );
      const maximum = Number(
        recommendation?.safe_config?.maximum_sample_size ?? 250,
      );
      return {
        sample_size: Math.max(30, Math.min(sampleSize, maximum)),
        permutations: Number(recommendation?.safe_config?.permutations ?? 100),
      };
    }
    case "fast":
      return { sample_size: Math.max(65, sampleSize) };
    default:
      return { sample_size: sampleSize };
  }
}

function analysisIncompatibility(
  analysis: AnalysisCatalogEntry,
  model: ModelVersion | undefined,
) {
  if (!model) return undefined;
  const assessed = model.assessment?.recommendations.find(
    (recommendation) => recommendation.capability === analysis.key,
  );
  if (assessed?.status === "incompatible") {
    return (
      assessed.compatibility_warnings[0] ??
      "Incompatible with this validated model"
    );
  }
  const dependent = Boolean(model.metadata.dependent_inputs);
  if (analysis.requires_dependent_inputs && !dependent) {
    return "Requires a dependent input copula";
  }
  if (!analysis.supports_dependent_inputs && dependent) {
    return "Requires independent inputs";
  }
  return undefined;
}

function GuidedBuilder({
  spec,
  setSpec,
}: {
  spec: BuilderSpec;
  setSpec: (spec: BuilderSpec) => void;
}) {
  const errors = validateBuilder(spec);
  const updateVariables = (variables: BuilderVariable[]) => {
    const old = spec.copula.correlation;
    const correlation = identityCorrelation(variables.length).map(
      (row, rowIndex) =>
        row.map((value, columnIndex) => old[rowIndex]?.[columnIndex] ?? value),
    );
    setSpec({ ...spec, variables, copula: { ...spec.copula, correlation } });
  };
  const reorder = <T,>(items: T[], index: number, direction: -1 | 1) => {
    const target = index + direction;
    if (target < 0 || target >= items.length) return items;
    const copy = [...items];
    [copy[index], copy[target]] = [copy[target]!, copy[index]!];
    return copy;
  };
  return (
    <div className="builder">
      <div className="builder-header">
        <div>
          <h3>Guided model builder</h3>
          <p>
            Define formulas and input distributions that compile into an
            OpenTURNS{" "}
            <a
              href="https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.SymbolicFunction.html"
              target="_blank"
              rel="noreferrer"
            >
              SymbolicFunction
            </a>
            . OpenTURNS evaluates named formulas and can provide analytical
            derivatives for symbolic models.
          </p>
        </div>
        <button
          className="button secondary small"
          onClick={() =>
            updateVariables([
              ...spec.variables,
              createBuilderVariable(spec.variables.length),
            ])
          }
        >
          <Plus /> Add variable
        </button>
      </div>
      <div className="variable-list">
        {spec.variables.map((variable, index) => {
          const distribution = distributionDefinition(variable.distribution);
          return (
            <div className="variable-row expanded" key={variable.id}>
              <div className="reorder-controls">
                <button
                  aria-label={`Move variable ${index + 1} up`}
                  disabled={index === 0}
                  onClick={() =>
                    updateVariables(reorder(spec.variables, index, -1))
                  }
                >
                  <ArrowUp />
                </button>
                <button
                  aria-label={`Move variable ${index + 1} down`}
                  disabled={index === spec.variables.length - 1}
                  onClick={() =>
                    updateVariables(reorder(spec.variables, index, 1))
                  }
                >
                  <ArrowDown />
                </button>
              </div>
              <input
                aria-label={`Variable ${index + 1} name`}
                value={variable.name}
                onChange={(event) =>
                  updateVariables(
                    spec.variables.map((item, itemIndex) =>
                      itemIndex === index
                        ? { ...item, name: event.target.value }
                        : item,
                    ),
                  )
                }
              />
              <select
                aria-label={`Variable ${index + 1} distribution`}
                value={variable.distribution}
                onChange={(event) =>
                  updateVariables(
                    spec.variables.map((item, itemIndex) =>
                      itemIndex === index
                        ? {
                            ...item,
                            distribution: event.target
                              .value as BuilderVariable["distribution"],
                            parameters: distributionDefinition(
                              event.target
                                .value as BuilderVariable["distribution"],
                            ).parameters.map(
                              (parameter) => parameter.defaultValue,
                            ),
                          }
                        : item,
                    ),
                  )
                }
              >
                {DISTRIBUTION_REGISTRY.map((item) => (
                  <option value={item.key} key={item.key}>
                    {item.label}
                  </option>
                ))}
              </select>
              {distribution.parameters.map((parameter, parameterIndex) => (
                <label key={parameter.key}>
                  <span>{parameter.label}</span>
                  <input
                    aria-label={`Variable ${index + 1} ${parameter.label}`}
                    type="number"
                    step="any"
                    value={variable.parameters[parameterIndex] ?? ""}
                    onChange={(event) =>
                      updateVariables(
                        spec.variables.map((item, itemIndex) =>
                          itemIndex === index
                            ? {
                                ...item,
                                parameters: item.parameters.map(
                                  (value, valueIndex) =>
                                    valueIndex === parameterIndex
                                      ? Number(event.target.value)
                                      : value,
                                ),
                              }
                            : item,
                        ),
                      )
                    }
                  />
                </label>
              ))}
              <button
                className="icon-button danger-icon"
                aria-label={`Remove variable ${index + 1}`}
                disabled={spec.variables.length === 1}
                onClick={() =>
                  updateVariables(
                    spec.variables.filter((item) => item.id !== variable.id),
                  )
                }
              >
                <Trash2 />
              </button>
            </div>
          );
        })}
      </div>
      <div className="builder-subsection">
        <div className="builder-subheading">
          <div>
            <strong>Outputs</strong>
            <small>
              Each formula is evaluated by OpenTURNS SymbolicFunction.
            </small>
          </div>
          <button
            className="button secondary small"
            onClick={() =>
              setSpec({
                ...spec,
                outputs: [
                  ...spec.outputs,
                  {
                    id: crypto.randomUUID(),
                    name: `response_${spec.outputs.length + 1}`,
                    formula: spec.variables[0]?.name ?? "0",
                  },
                ],
              })
            }
          >
            <Plus /> Add output
          </button>
        </div>
        {spec.outputs.map((output, index) => (
          <div className="output-row" key={output.id}>
            <div className="reorder-controls">
              <button
                aria-label={`Move output ${index + 1} up`}
                disabled={index === 0}
                onClick={() =>
                  setSpec({
                    ...spec,
                    outputs: reorder(spec.outputs, index, -1),
                  })
                }
              >
                <ArrowUp />
              </button>
              <button
                aria-label={`Move output ${index + 1} down`}
                disabled={index === spec.outputs.length - 1}
                onClick={() =>
                  setSpec({ ...spec, outputs: reorder(spec.outputs, index, 1) })
                }
              >
                <ArrowDown />
              </button>
            </div>
            <input
              aria-label={`Output ${index + 1} name`}
              value={output.name}
              onChange={(event) =>
                setSpec({
                  ...spec,
                  outputs: spec.outputs.map((item) =>
                    item.id === output.id
                      ? { ...item, name: event.target.value }
                      : item,
                  ),
                })
              }
            />
            <input
              aria-label={`Output ${index + 1} formula`}
              value={output.formula}
              onChange={(event) =>
                setSpec({
                  ...spec,
                  outputs: spec.outputs.map((item) =>
                    item.id === output.id
                      ? { ...item, formula: event.target.value }
                      : item,
                  ),
                })
              }
              placeholder="sin(x1) + x2^2"
            />
            <button
              className="icon-button danger-icon"
              aria-label={`Remove output ${index + 1}`}
              disabled={spec.outputs.length === 1}
              onClick={() =>
                setSpec({
                  ...spec,
                  outputs: spec.outputs.filter((item) => item.id !== output.id),
                })
              }
            >
              <Trash2 />
            </button>
          </div>
        ))}
      </div>
      <div className="builder-subsection">
        <label className="copula-select">
          <span>Input dependence</span>
          <select
            value={spec.copula.kind}
            onChange={(event) =>
              setSpec({
                ...spec,
                copula: {
                  ...spec.copula,
                  kind: event.target.value as BuilderSpec["copula"]["kind"],
                },
              })
            }
          >
            <option value="independent">Independent</option>
            <option value="normal">Normal copula</option>
          </select>
        </label>
        {spec.copula.kind === "normal" && (
          <div
            className="correlation-editor"
            role="group"
            aria-label="Normal copula correlation matrix"
            style={
              {
                "--correlation-size": spec.variables.length,
              } as React.CSSProperties
            }
          >
            {spec.copula.correlation.map((row, rowIndex) =>
              row.map((value, columnIndex) => (
                <input
                  key={`${rowIndex}-${columnIndex}`}
                  aria-label={`Correlation ${spec.variables[rowIndex]?.name} and ${spec.variables[columnIndex]?.name}`}
                  type="number"
                  min="-1"
                  max="1"
                  step="0.05"
                  value={value}
                  disabled={rowIndex === columnIndex || columnIndex > rowIndex}
                  onChange={(event) => {
                    const next = spec.copula.correlation.map((item) => [
                      ...item,
                    ]);
                    next[rowIndex]![columnIndex] = Number(event.target.value);
                    next[columnIndex]![rowIndex] = Number(event.target.value);
                    setSpec({
                      ...spec,
                      copula: { ...spec.copula, correlation: next },
                    });
                  }}
                />
              )),
            )}
          </div>
        )}
      </div>
      {errors.length > 0 && (
        <div className="builder-errors" role="status">
          {errors.map((error) => (
            <span key={error}>{error}</span>
          ))}
        </div>
      )}
      {errors.length === 0 && (
        <details className="source-preview">
          <summary>Generated OpenTURNS source preview</summary>
          <pre>
            <code>{buildSymbolicModel(spec)}</code>
          </pre>
          <p>
            SymbolicFunction · exact analytical gradient and Hessian ·{" "}
            {spec.copula.kind === "normal"
              ? "Normal copula dependence"
              : "independent inputs"}
          </p>
        </details>
      )}
    </div>
  );
}

function ReferenceExamples({
  examples,
  selectedId,
  onSelect,
}: {
  examples: readonly ExampleCatalogEntry[];
  selectedId: string;
  onSelect: (example: ExampleCatalogEntry) => void;
}) {
  const [search, setSearch] = useState("");
  const needle = search.trim().toLocaleLowerCase();
  const visible = examples.filter(
    (example) =>
      !needle ||
      `${example.title} ${example.domain} ${example.summary}`
        .toLocaleLowerCase()
        .includes(needle),
  );
  return (
    <div className="examples-browser">
      <div className="examples-toolbar">
        <div>
          <h3>Reference models</h3>
          <p>Select a model to load its editable Python source below.</p>
        </div>
        <label className="study-search">
          <Search />
          <input
            aria-label="Search reference models"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search 24 examples"
          />
        </label>
      </div>
      <div className="examples-grid">
        {visible.map((example) => (
          <button
            className={`example-card ${example.id === selectedId ? "selected" : ""}`}
            key={example.id}
            onClick={() => onSelect(example)}
          >
            <span>{example.domain}</span>
            <strong>{example.title}</strong>
            <p>{example.summary}</p>
            <small>{example.inputDimension} inputs</small>
          </button>
        ))}
      </div>
    </div>
  );
}

function ModelValidationPendingPane({
  aiModelLabel,
}: {
  aiModelLabel: string;
}) {
  return (
    <aside
      className="understanding-pane validation-pending-pane"
      aria-label="Model Understanding"
      aria-busy="true"
    >
      <header>
        <div>
          <span className="section-kicker">Validation &amp; explanation</span>
          <h2>Model Understanding</h2>
        </div>
        <small>{aiModelLabel}</small>
      </header>
      <section className="validation-pending" role="status" aria-live="polite">
        <div className="validation-loader" aria-hidden="true">
          <span />
          <span />
          <ScanSearch />
        </div>
        <div>
          <strong>Your model is being validated…</strong>
          <p>
            OpenTURNS is checking the executable model, input distribution,
            output shape, and bounded pilot behaviour.
          </p>
        </div>
        <ol aria-label="Validation progress">
          <li className="active">Isolated model checks</li>
          <li>Deterministic assessment</li>
          <li>Model brief</li>
        </ol>
        <small>
          Direct analyses stay locked until deterministic validation succeeds.
        </small>
      </section>
    </aside>
  );
}

function ModelUnderstandingPane({
  model,
  projectId,
  aiModelLabel,
}: {
  model: ModelVersion;
  projectId: string;
  aiModelLabel: string;
}) {
  const [content, setContent] = useState("");
  const [status, setStatus] = useState<
    "loading" | "streaming" | "waiting" | "ready" | "failed"
  >("loading");
  const [error, setError] = useState<string>();
  const activeRequest = useRef<AbortController | undefined>(undefined);
  const assessment = model.assessment;

  const fetchUnderstanding = useCallback(
    async (signal: AbortSignal) => {
      const response = await fetch(
        `/api/v1/model-versions/${model.id}/understanding`,
        { credentials: "include", signal },
      );
      if (!response.ok) {
        const body = (await response.json().catch(() => ({}))) as {
          error?: { message?: string };
        };
        throw new Error(
          body.error?.message ?? "Model Understanding is unavailable.",
        );
      }
      return (await response.json()) as {
        understanding: ModelUnderstanding | null;
      };
    },
    [model.id],
  );

  const waitForCompletion = useCallback(
    async (signal: AbortSignal) => {
      setStatus("waiting");
      while (!signal.aborted) {
        await abortableDelay(MODEL_UNDERSTANDING_POLL_MS, signal);
        const { understanding } = await fetchUnderstanding(signal);
        if (understanding?.status === "succeeded" && understanding.content) {
          setContent(understanding.content);
          setStatus("ready");
          return;
        }
        if (understanding?.status === "failed") {
          throw new Error(
            "The AI provider could not create the explanation. Please retry; failed requests are not charged.",
          );
        }
      }
    },
    [fetchUnderstanding],
  );

  const generate = useCallback(
    async (regenerate: boolean) => {
      activeRequest.current?.abort();
      const controller = new AbortController();
      activeRequest.current = controller;
      const timeout = window.setTimeout(
        () => controller.abort(),
        MODEL_UNDERSTANDING_CLIENT_TIMEOUT_MS,
      );
      setContent("");
      setError(undefined);
      setStatus("streaming");
      try {
        const response = await fetch(
          `/api/v1/model-versions/${model.id}/understanding`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            credentials: "include",
            body: JSON.stringify({ regenerate }),
            signal: controller.signal,
          },
        );
        if (response.status === 202) {
          await waitForCompletion(controller.signal);
          return;
        }
        if (!response.ok) {
          const body = (await response.json().catch(() => ({}))) as {
            error?: { message?: string };
          };
          throw new Error(
            body.error?.message ?? "Model Understanding is unavailable.",
          );
        }
        let receivedContent = false;
        await readTextStream(response, (chunk) => {
          receivedContent ||= Boolean(chunk);
          setContent((current) => current + chunk);
        });
        if (!receivedContent) {
          throw new Error("The AI provider returned an empty explanation.");
        }
        setStatus("ready");
      } catch (caught) {
        setError(modelUnderstandingError(caught));
        setStatus("failed");
      } finally {
        window.clearTimeout(timeout);
        if (activeRequest.current === controller) {
          activeRequest.current = undefined;
        }
      }
    },
    [model.id, waitForCompletion],
  );

  const pollActiveGeneration = useCallback(async () => {
    activeRequest.current?.abort();
    const controller = new AbortController();
    activeRequest.current = controller;
    const timeout = window.setTimeout(
      () => controller.abort(),
      MODEL_UNDERSTANDING_CLIENT_TIMEOUT_MS,
    );
    setError(undefined);
    try {
      await waitForCompletion(controller.signal);
    } catch (caught) {
      setError(modelUnderstandingError(caught));
      setStatus("failed");
    } finally {
      window.clearTimeout(timeout);
      if (activeRequest.current === controller) {
        activeRequest.current = undefined;
      }
    }
  }, [waitForCompletion]);

  useEffect(() => {
    let active = true;
    void api
      .getModelUnderstanding(model.id)
      .then(({ understanding }) => {
        if (!active) return;
        if (understanding?.status === "succeeded" && understanding.content) {
          setContent(understanding.content);
          setStatus("ready");
        } else if (understanding?.status === "generating") {
          void pollActiveGeneration();
        } else {
          void generate(false);
        }
      })
      .catch((caught) => {
        if (!active) return;
        setError(
          caught instanceof Error
            ? caught.message
            : "Model Understanding is unavailable.",
        );
        setStatus("failed");
      });
    return () => {
      active = false;
      activeRequest.current?.abort();
    };
  }, [generate, model.id, pollActiveGeneration]);

  return (
    <aside className="understanding-pane" aria-label="Model Understanding">
      <header>
        <div>
          <span className="section-kicker">Validation &amp; explanation</span>
          <h2>Model Understanding</h2>
        </div>
        <small>{aiModelLabel}</small>
      </header>
      <section className="validation-facts">
        <div className="validation-success">
          <Check />
          <div>
            <strong>Model validated</strong>
            <span>
              {model.metadata.input_dimension} inputs →{" "}
              {model.metadata.output_dimension} outputs · OpenTURNS{" "}
              {model.metadata.openturns_version}
            </span>
          </div>
        </div>
        <h3>Deterministic validation facts</h3>
        <dl>
          <div>
            <dt>Shape</dt>
            <dd>
              {model.metadata.input_dimension} inputs →{" "}
              {model.metadata.output_dimension} outputs
            </dd>
          </div>
          <div>
            <dt>Function</dt>
            <dd>{model.metadata.function_type ?? model.sourceKind}</dd>
          </div>
          <div>
            <dt>Dependence</dt>
            <dd>
              {model.metadata.dependent_inputs
                ? model.metadata.copula
                : "Independent"}
            </dd>
          </div>
          <div>
            <dt>Projected direct cost</dt>
            <dd>
              {Math.round(
                assessment?.profile.projected_1000_evaluation_runtime_ms ?? 0,
              ).toLocaleString()}{" "}
              ms / 1,000
            </dd>
          </div>
        </dl>
        {assessment?.profile.pilot_outputs.map((output) => (
          <div className="pilot-strip" key={output.output_index}>
            <strong>{output.output_name}</strong>
            <span>mean {output.mean.toPrecision(4)}</span>
            <span>σ {output.standard_deviation.toPrecision(4)}</span>
            <span>
              5–95% {output.quantile_05.toPrecision(4)} to{" "}
              {output.quantile_95.toPrecision(4)}
            </span>
          </div>
        ))}
      </section>
      <section
        className="understanding-narrative"
        aria-live="polite"
        aria-busy={status === "streaming" || status === "waiting"}
      >
        <h3>
          Model brief <small>generated by {aiModelLabel}</small>
        </h3>
        {(status === "streaming" || status === "waiting") && !content && (
          <div className="assistant-placeholder">
            <span /> <span /> <span />{" "}
            {status === "waiting"
              ? "An existing AI generation is finishing…"
              : "The AI provider is drafting a concise explanation…"}
          </div>
        )}
        {content && <Markdown>{content}</Markdown>}
        {error && <div className="inline-error">{error}</div>}
        <div className="understanding-actions">
          {status === "failed" && (
            <button
              className="button secondary small"
              onClick={() => void generate(false)}
            >
              Retry
            </button>
          )}
          {status === "ready" && (
            <button
              className="button secondary small"
              onClick={() => void generate(true)}
            >
              Regenerate
            </button>
          )}
        </div>
      </section>
      <WorkflowAssessment model={model} projectId={projectId} compact />
    </aside>
  );
}

function workflowPath(model: ModelVersion) {
  if (model.assessment?.workflow?.path) return model.assessment.workflow.path;
  const projected =
    model.assessment?.profile.projected_1000_evaluation_runtime_ms ?? 0;
  const surrogateEligible = model.assessment?.recommendations.some(
    (item) =>
      ["gpr", "pce"].includes(item.capability) &&
      item.status !== "incompatible",
  );
  if (model.metadata.input_dimension >= 15)
    return "dimensionality_reduction" as const;
  if (projected > 5_000 && surrogateEligible) return "surrogate" as const;
  return "direct" as const;
}

function WorkflowAssessment({
  model,
  projectId,
  compact = false,
}: {
  model: ModelVersion;
  projectId: string;
  compact?: boolean;
}) {
  const path = workflowPath(model);
  const projected = Math.round(
    model.assessment?.profile.projected_1000_evaluation_runtime_ms ?? 0,
  );
  const content =
    path === "dimensionality_reduction"
      ? {
          icon: <ScanSearch />,
          kicker: "Screen dimensions first",
          title: "Dimensionality reduction is the recommended next step.",
          body: `${model.metadata.input_dimension} inputs cross the high-dimensional screening threshold. Run Morris screening before committing a large budget to global analyses.`,
          href: `/studies/${projectId}/dimension-reduction?modelId=${model.id}`,
          action: "Open Dimension Reduction Studio",
        }
      : path === "surrogate"
        ? {
            icon: <Waves />,
            kicker: "Approximate before scaling",
            title: "A validated surrogate is the recommended next step.",
            body: `Measured validation projects ${projected.toLocaleString()} ms for 1,000 direct evaluations. Build and validate an approximation before larger studies.`,
            href: `/studies/${projectId}/surrogates?modelId=${model.id}`,
            action: "Open Surrogate Studio",
          }
        : {
            icon: <Gauge />,
            kicker: "Direct analysis recommended",
            title: "This model is practical to evaluate directly.",
            body: `Measured validation projects ${projected.toLocaleString()} ms for 1,000 evaluations. Continue with the direct OpenTURNS analyses below.`,
            href: "#direct-analyses",
            action: "Continue to direct analyses",
          };
  return (
    <section
      className={`workflow-assessment ${path} ${compact ? "compact" : ""}`}
      aria-label="Recommended analysis route"
    >
      <div className="workflow-icon">{content.icon}</div>
      <div>
        <span className="section-kicker">{content.kicker}</span>
        <h2>{content.title}</h2>
        <p>{content.body}</p>
      </div>
      {content.href.startsWith("/") ? (
        <Link className="button primary" to={content.href}>
          {content.action} <ArrowRight />
        </Link>
      ) : (
        <a className="button primary" href={content.href}>
          {content.action} <ArrowRight />
        </a>
      )}
      <div className="workflow-alternatives">
        <span>Other tools remain available by choice:</span>
        {path !== "dimensionality_reduction" && (
          <Link
            to={`/studies/${projectId}/dimension-reduction?modelId=${model.id}`}
          >
            screen dimensions
          </Link>
        )}
        {path !== "surrogate" && (
          <Link to={`/studies/${projectId}/surrogates?modelId=${model.id}`}>
            build a surrogate
          </Link>
        )}
      </div>
    </section>
  );
}

export function Workspace() {
  const { theme } = useTheme();
  const navigate = useNavigate();
  const { projectId: routeProjectId } = useParams();
  const [searchParams] = useSearchParams();
  const sourceModelId = searchParams.get("sourceModel") ?? "";
  const dataFitId = searchParams.get("dataFit") ?? "";
  const requestedExampleId = searchParams.get("example") ?? "";
  const requestedNext = searchParams.get("next") ?? "";
  const requestedSurrogateId = searchParams.get("surrogate") ?? "";
  const projectsQuery = useQuery({
    queryKey: ["projects"],
    queryFn: api.listProjects,
  });
  const sessionQuery = useQuery({
    queryKey: ["session-policy"],
    queryFn: api.session,
  });
  const catalogQuery = useQuery({
    queryKey: ["catalog"],
    queryFn: api.catalog,
  });
  const examplesQuery = useQuery({
    queryKey: ["examples"],
    queryFn: api.examples,
  });
  const [modelName, setModelName] = useState("");
  const [modelNameEdited, setModelNameEdited] = useState(false);
  const [parentVersionId, setParentVersionId] = useState<string>();
  const [dataFitProvenance, setDataFitProvenance] = useState<{
    fitRunId: string;
    datasetId: string;
    builderSpec?: Record<string, unknown>;
  }>();
  const [mode, setMode] = useState<AuthorMode>("source");
  const [selectedExampleId, setSelectedExampleId] = useState("");
  const [source, setSource] = useState<string>("");
  const [builderSpec, setBuilderSpec] = useState<BuilderSpec>(() => ({
    variables: [
      {
        id: crypto.randomUUID(),
        name: "x1",
        distribution: "Normal",
        parameters: [0, 1],
      },
      {
        id: crypto.randomUUID(),
        name: "x2",
        distribution: "Uniform",
        parameters: [-1, 1],
      },
    ],
    outputs: [
      { id: crypto.randomUUID(), name: "response", formula: "x1 + x2^2" },
    ],
    copula: { kind: "independent", correlation: identityCorrelation(2) },
  }));
  const [savedModel, setSavedModel] = useState<ModelVersion>();
  const [selected, setSelected] = useState<string[]>([
    "monte_carlo",
    "eda",
    "sobol",
  ]);
  const [sampleSize, setSampleSize] = useState(1000);
  const [outputTarget, setOutputTarget] = useState(0);
  const [reliabilityMethod, setReliabilityMethod] =
    useState<ReliabilityMethod>("FORM");
  const [reliabilityThreshold, setReliabilityThreshold] = useState(0);
  const [reliabilityOperator, setReliabilityOperator] = useState<
    ">" | ">=" | "<" | "<="
  >(">");
  const [reliabilityMaximumEvaluations, setReliabilityMaximumEvaluations] =
    useState(20_000);
  const [reliabilityTargetCov, setReliabilityTargetCov] = useState(0.05);
  const [targetHsicThreshold, setTargetHsicThreshold] = useState(0);
  const [targetHsicOperator, setTargetHsicOperator] = useState<"<=" | ">=">(
    ">=",
  );
  const [targetHsicPermutations, setTargetHsicPermutations] = useState(100);
  const [analysisSurrogateId, setAnalysisSurrogateId] =
    useState(requestedSurrogateId);
  const [error, setError] = useState<string>();
  const projects = projectsQuery.data?.projects ?? [];
  const activeProjectId = routeProjectId ?? "";
  const activeProject = projects.find((item) => item.id === activeProjectId);
  const examples = examplesQuery.data?.examples ?? [];
  const definitionQuery = useQuery({
    queryKey: ["model-definition", sourceModelId],
    queryFn: () => api.getModelDefinition(sourceModelId),
    enabled: Boolean(sourceModelId),
  });
  useEffect(() => {
    const pilot = savedModel?.assessment?.profile.pilot_outputs[outputTarget];
    if (pilot) setTargetHsicThreshold(pilot.mean);
  }, [outputTarget, savedModel?.id]);
  useEffect(() => {
    const definition = definitionQuery.data?.definition;
    if (!definition) return;
    setSource(definition.source);
    setMode("source");
    setModelName(
      requestedSurrogateId
        ? definition.modelVersion.displayName
        : `${definition.modelVersion.displayName} copy`,
    );
    setModelNameEdited(true);
    setParentVersionId(definition.modelVersion.id);
    setSavedModel(requestedSurrogateId ? definition.modelVersion : undefined);
    setAnalysisSurrogateId(requestedSurrogateId);
  }, [definitionQuery.data, requestedSurrogateId]);
  useEffect(() => {
    if (!dataFitId) return;
    try {
      const draft = JSON.parse(
        window.sessionStorage.getItem("uncertaintycat-data-lab-draft") ??
          "null",
      ) as {
        fitRunId?: string;
        datasetId?: string;
        source?: string;
        builderSpec?: Record<string, unknown>;
      } | null;
      if (!draft?.source || draft.fitRunId !== dataFitId || !draft.datasetId)
        return;
      setSource(draft.source);
      setMode("source");
      setModelName("Data-fit model draft");
      setModelNameEdited(true);
      setParentVersionId(undefined);
      setDataFitProvenance({
        fitRunId: draft.fitRunId,
        datasetId: draft.datasetId,
        ...(draft.builderSpec ? { builderSpec: draft.builderSpec } : {}),
      });
      setSavedModel(undefined);
    } catch {
      setError(
        "The Data Lab draft could not be restored. Reopen it from Data Lab.",
      );
    }
  }, [dataFitId]);
  useEffect(() => {
    if (!requestedExampleId || !examples.length || sourceModelId || dataFitId)
      return;
    const example = examples.find((item) => item.id === requestedExampleId);
    if (!example) return;
    setSelectedExampleId(example.id);
    setSource(example.source);
    setModelName(example.title);
    setModelNameEdited(false);
    setMode("source");
    setSavedModel(undefined);
  }, [dataFitId, examples, requestedExampleId, sourceModelId]);
  const generatedSource = useMemo(
    () =>
      validateBuilder(builderSpec).length === 0
        ? buildSymbolicModel(builderSpec)
        : "",
    [builderSpec],
  );
  const directAnalyses = useMemo(
    () =>
      (catalogQuery.data?.analyses ?? []).filter(
        (analysis) =>
          !["calibration_nlls", "morris", "pce", "gpr"].includes(analysis.key),
      ),
    [catalogQuery.data],
  );
  const analysisGroups = useMemo(
    () => [
      {
        title: "Uncertainty quantification",
        description:
          "Propagate the input uncertainty and characterize the model response.",
        analyses: directAnalyses.filter((analysis) =>
          ["monte_carlo", "eda", "convergence"].includes(analysis.key),
        ),
      },
      {
        title: "Sensitivity & reliability",
        description:
          "Rank influential inputs or estimate a clearly defined failure event.",
        analyses: directAnalyses.filter(
          (analysis) =>
            !["monte_carlo", "eda", "convergence"].includes(analysis.key),
        ),
      },
    ],
    [directAnalyses],
  );
  useEffect(() => {
    if (!savedModel) return;
    setSelected((current) =>
      current.filter((key) => {
        const analysis = directAnalyses.find(
          (candidate) => candidate.key === key,
        );
        return !analysis || !analysisIncompatibility(analysis, savedModel);
      }),
    );
  }, [directAnalyses, savedModel]);
  const selectExample = (example: ExampleCatalogEntry) => {
    setSelectedExampleId(example.id);
    setSource(example.source);
    if (!modelNameEdited) setModelName(example.title);
    setParentVersionId(undefined);
    setDataFitProvenance(undefined);
    setSavedModel(undefined);
    setAnalysisSurrogateId("");
    setMode("source");
    window.localStorage.setItem("uncertaintycat-last-example", example.id);
  };

  const saveModel = useMutation({
    mutationFn: async () => {
      if (!activeProjectId) throw new Error("Create a project first.");
      const modelSource = mode === "source" ? source : generatedSource;
      if (!modelSource)
        throw new Error("Complete the model definition before validation.");
      if (!modelName.trim())
        throw new Error("Enter a model name before validation.");
      return api.createModel(activeProjectId, {
        source: modelSource,
        displayName: modelName.trim(),
        sourceKind:
          mode === "builder"
            ? "builder"
            : examples.some(
                  (example) =>
                    example.id === selectedExampleId &&
                    example.source === source,
                )
              ? "example"
              : "python",
        ...(mode === "builder"
          ? { builderSpec: builderSpec as unknown as Record<string, unknown> }
          : {}),
        ...(dataFitProvenance?.builderSpec
          ? { builderSpec: dataFitProvenance.builderSpec }
          : {}),
        ...(parentVersionId ? { parentVersionId } : {}),
        ...(dataFitProvenance
          ? {
              derivation: {
                type: "distribution_fit",
                dataAnalysisRunId: dataFitProvenance.fitRunId,
                datasetId: dataFitProvenance.datasetId,
              },
            }
          : {}),
      });
    },
    onMutate: () => setError(undefined),
    onSuccess: ({ modelVersion }) => {
      setSavedModel(modelVersion);
      setError(undefined);
      if (
        ["calibration", "dimension-reduction", "surrogates"].includes(
          requestedNext,
        )
      ) {
        navigate(
          `/studies/${activeProjectId}/${requestedNext}?modelId=${modelVersion.id}`,
        );
      }
    },
    onError: (caught) =>
      setError(caught instanceof Error ? caught.message : "Validation failed."),
  });
  const analysisComposerLocked = !savedModel || saveModel.isPending;
  const createRun = useMutation({
    mutationFn: async () => {
      if (!savedModel) throw new Error("Validate and save the model first.");
      const reliability = {
        method: reliabilityMethod,
        threshold: reliabilityThreshold,
        operator: reliabilityOperator,
        maximum_evaluations: reliabilityMaximumEvaluations,
        target_coefficient_of_variation: reliabilityTargetCov,
      };
      const targetHsic = {
        threshold: targetHsicThreshold,
        operator: targetHsicOperator,
        permutations: targetHsicPermutations,
      };
      const analyses = selected.map((key) => ({
        analysisKey: key,
        config: analysisConfig(
          key,
          sampleSize,
          savedModel,
          reliability,
          targetHsic,
        ),
        outputTargets: SCALAR_ANALYSES.has(key) ? [outputTarget] : [],
      }));
      return api.createRun({
        modelVersionId: savedModel.id,
        ...(analysisSurrogateId
          ? { surrogateModelId: analysisSurrogateId }
          : {}),
        analyses,
        seed: 42,
        accuracyProfile: "standard",
        idempotencyKey: crypto.randomUUID(),
      });
    },
    onSuccess: ({ run }) => navigate(`/runs/${run.id}`),
    onError: (caught) =>
      setError(
        caught instanceof Error ? caught.message : "Run could not be created.",
      ),
  });

  return (
    <div className="page workspace-page">
      <ProjectNav
        projectId={activeProjectId}
        projectName={activeProject?.name}
      />
      <div className="page-heading split">
        <div>
          <span className="section-kicker">Model &amp; analyses</span>
          <h1>Define the model you want to study.</h1>
          <p>
            A model is a Python function <code>y = f(x)</code>. Pair it with an
            OpenTURNS input distribution, validate it, then choose
            uncertainty-quantification or sensitivity methods.
          </p>
        </div>
      </div>
      <section
        className={`studio-card ${savedModel || saveModel.isPending ? "validated-studio" : ""}`}
        aria-busy={saveModel.isPending}
      >
        <div className="studio-authoring">
          {dataFitProvenance && (
            <div className="provenance-note">
              Distribution draft from retained fit{" "}
              {dataFitProvenance.fitRunId.slice(0, 8)}. Define an OpenTURNS{" "}
              <code>model</code> before validation; the original dataset and fit
              remain unchanged.
            </div>
          )}
          {requestedNext && !dataFitProvenance && (
            <div className="provenance-note">
              This reference model was opened from{" "}
              {requestedNext === "surrogates"
                ? "Surrogate Studio"
                : requestedNext === "calibration"
                  ? "Calibration Studio"
                  : "Dimension Reduction Studio"}
              . Validate and assess it before continuing.
            </div>
          )}
          <label className="model-name-field">
            <span>Model name</span>
            <input
              value={modelName}
              placeholder="Enter your model name here"
              onChange={(event) => {
                setModelName(event.target.value);
                setModelNameEdited(true);
                setSavedModel(undefined);
                setAnalysisSurrogateId("");
              }}
            />
          </label>
          <div className="mode-tabs">
            <button
              className={mode === "source" ? "active" : ""}
              onClick={() => setMode("source")}
            >
              <Code2 /> Examples &amp; Python model
            </button>
            <button
              className={mode === "builder" ? "active" : ""}
              onClick={() => setMode("builder")}
            >
              <SlidersHorizontal /> Guided builder
            </button>
          </div>
          {mode === "source" ? (
            <>
              <ReferenceExamples
                examples={examples}
                selectedId={selectedExampleId}
                onSelect={selectExample}
              />
              <div className="editor-shell resizable-editor">
                <div className="editor-title">
                  <span>model.py</span>
                  <small>
                    Editable Python · define <code>model</code> and{" "}
                    <code>problem</code>
                  </small>
                </div>
                <CodeMirror
                  onCreateEditor={(view) =>
                    view.contentDOM.setAttribute(
                      "aria-label",
                      "Python model source",
                    )
                  }
                  height="100%"
                  theme={theme}
                  value={source}
                  extensions={[python()]}
                  onChange={(value) => {
                    setSource(value);
                    setSavedModel(undefined);
                    setAnalysisSurrogateId("");
                  }}
                  basicSetup={{
                    foldGutter: true,
                    highlightActiveLine: true,
                    autocompletion: true,
                    bracketMatching: true,
                  }}
                />
              </div>
            </>
          ) : (
            <GuidedBuilder
              spec={builderSpec}
              setSpec={(spec) => {
                setBuilderSpec(spec);
                setSavedModel(undefined);
                setAnalysisSurrogateId("");
              }}
            />
          )}
          <div className="studio-footer">
            <div className="model-status">
              {saveModel.isPending ? (
                <>
                  <span className="validation-dot" />
                  <div>
                    <strong>Validation in progress</strong>
                    <small>
                      OpenTURNS checks are running in the isolated compute
                      boundary.
                    </small>
                  </div>
                </>
              ) : !savedModel ? (
                <>
                  <span className="idle-dot" />
                  <div>
                    <strong>Not yet validated</strong>
                    <small>
                      Validation executes sample evaluations in the isolated
                      compute boundary.
                    </small>
                  </div>
                </>
              ) : (
                <small>
                  Edit the definition to validate another saved model.
                </small>
              )}
            </div>
            <button
              className="button primary"
              onClick={() => saveModel.mutate()}
              disabled={
                saveModel.isPending ||
                !modelName.trim() ||
                (mode === "source" ? !source.trim() : !generatedSource)
              }
            >
              <Save />{" "}
              {saveModel.isPending
                ? "Validating and assessing…"
                : "Validate & Assess"}
            </button>
          </div>
        </div>
        {saveModel.isPending ? (
          <ModelValidationPendingPane
            aiModelLabel={
              sessionQuery.data?.ai?.modelUnderstanding.label ??
              "Configured AI provider"
            }
          />
        ) : savedModel ? (
          <ModelUnderstandingPane
            model={savedModel}
            projectId={activeProjectId}
            aiModelLabel={
              sessionQuery.data?.ai?.modelUnderstanding.label ??
              "Configured AI provider"
            }
          />
        ) : null}
      </section>
      {error && <div className="error-banner">{error}</div>}
      {savedModel && analysisSurrogateId && (
        <div className="surrogate-source-banner">
          <Waves />
          <div>
            <strong>Promoted surrogate selected in Surrogate Studio</strong>
            <small>
              This run will use that explicit approximation as its evidence
              source. Edit the model to return to direct evaluation.
            </small>
          </div>
        </div>
      )}
      <section
        id="direct-analyses"
        className={`analysis-composer ${analysisComposerLocked ? "disabled-panel" : ""}`}
        aria-disabled={analysisComposerLocked}
        aria-busy={saveModel.isPending}
      >
        <fieldset
          className="analysis-composer-fields"
          disabled={analysisComposerLocked}
        >
          <legend className="sr-only">Direct analysis configuration</legend>
          <div className="composer-heading">
            <div>
              <span className="section-kicker">Run composer</span>
              <h2>Choose direct OpenTURNS analyses.</h2>
            </div>
            <div className="composer-controls">
              <label className="sample-budget">
                <span>Standard sample budget</span>
                <input
                  type="number"
                  min="64"
                  max="20000"
                  step="64"
                  value={sampleSize}
                  onChange={(event) =>
                    setSampleSize(Number(event.target.value))
                  }
                />
              </label>
              {savedModel && savedModel.metadata.output_dimension > 1 && (
                <label>
                  <span>Scalar analysis output</span>
                  <select
                    value={outputTarget}
                    onChange={(event) =>
                      setOutputTarget(Number(event.target.value))
                    }
                  >
                    {savedModel.metadata.outputs.map((output) => (
                      <option value={output.index} key={output.index}>
                        {output.name}
                      </option>
                    ))}
                  </select>
                </label>
              )}
            </div>
          </div>
          {analysisComposerLocked && (
            <div className="analysis-lock-note" role="status">
              <ScanSearch />
              <div>
                <strong>
                  {saveModel.isPending
                    ? "Analyses will unlock after validation"
                    : "Validate the model to unlock analyses"}
                </strong>
                <small>
                  Analysis choices are enabled only for a deterministic,
                  successfully validated model.
                </small>
              </div>
            </div>
          )}
          {selected.includes("reliability") && (
            <div className="analysis-settings">
              {selected.includes("reliability") && (
                <div className="reliability-studio">
                  <div className="reliability-step">
                    <i>1</i>
                    <div>
                      <strong>Define failure event</strong>
                      <small>
                        Output{" "}
                        {savedModel?.metadata.outputs[outputTarget]?.name ??
                          outputTarget}
                      </small>
                    </div>
                  </div>
                  <label>
                    <span>Failure event</span>
                    <select
                      value={reliabilityOperator}
                      onChange={(event) =>
                        setReliabilityOperator(
                          event.target.value as ">" | ">=" | "<" | "<=",
                        )
                      }
                    >
                      <option value=">">Output &gt; threshold</option>
                      <option value=">=">Output ≥ threshold</option>
                      <option value="<">Output &lt; threshold</option>
                      <option value="<=">Output ≤ threshold</option>
                    </select>
                  </label>
                  <label>
                    <span>Threshold</span>
                    <input
                      type="number"
                      value={reliabilityThreshold}
                      onChange={(event) =>
                        setReliabilityThreshold(Number(event.target.value))
                      }
                    />
                  </label>
                  {savedModel?.assessment?.profile.pilot_outputs[
                    outputTarget
                  ] &&
                    (() => {
                      const pilot =
                        savedModel.assessment.profile.pilot_outputs[
                          outputTarget
                        ]!;
                      const intersects =
                        reliabilityThreshold >= pilot.minimum &&
                        reliabilityThreshold <= pilot.maximum;
                      return (
                        <div
                          className={`reliability-preview ${intersects ? "intersects" : "outside"}`}
                        >
                          <div className="reliability-step">
                            <i>2</i>
                            <div>
                              <strong>Bounded pilot preview</strong>
                              <small>
                                {pilot.minimum.toPrecision(4)} to{" "}
                                {pilot.maximum.toPrecision(4)} observed · q05{" "}
                                {pilot.quantile_05.toPrecision(4)} · q95{" "}
                                {pilot.quantile_95.toPrecision(4)}
                              </small>
                            </div>
                          </div>
                          <p>
                            {intersects
                              ? "Threshold intersects the bounded validation sample."
                              : "Threshold is outside the bounded validation sample; rare-event evidence may be weak or zero."}
                          </p>
                        </div>
                      );
                    })()}
                  <div className="reliability-step">
                    <i>3</i>
                    <div>
                      <strong>Select method and stopping controls</strong>
                      <small>
                        FORM/SORM are local approximations; simulation methods
                        are sampling estimates.
                      </small>
                    </div>
                  </div>
                  <label>
                    <span>Reliability method</span>
                    <select
                      value={reliabilityMethod}
                      onChange={(event) =>
                        setReliabilityMethod(
                          event.target.value as ReliabilityMethod,
                        )
                      }
                    >
                      <option value="FORM">FORM · local approximation</option>
                      <option value="SORM">
                        SORM · local curvature approximation
                      </option>
                      <option value="MONTE_CARLO">
                        Monte Carlo simulation
                      </option>
                      <option value="DIRECTIONAL_SAMPLING">
                        Directional sampling
                      </option>
                      <option value="SUBSET_SAMPLING">Subset sampling</option>
                    </select>
                  </label>
                  <label>
                    <span>Maximum evaluations</span>
                    <input
                      type="number"
                      min="100"
                      max="2000000"
                      value={reliabilityMaximumEvaluations}
                      onChange={(event) =>
                        setReliabilityMaximumEvaluations(
                          Number(event.target.value),
                        )
                      }
                    />
                  </label>
                  <label>
                    <span>Target coefficient of variation</span>
                    <input
                      type="number"
                      min="0.001"
                      max="1"
                      step="0.01"
                      value={reliabilityTargetCov}
                      onChange={(event) =>
                        setReliabilityTargetCov(Number(event.target.value))
                      }
                    />
                  </label>
                </div>
              )}
            </div>
          )}
          {selected.includes("target_hsic") && (
            <div className="analysis-settings">
              <div className="reliability-studio">
                <div className="reliability-step">
                  <i>1</i>
                  <div>
                    <strong>Define the critical target domain</strong>
                    <small>
                      Output{" "}
                      {savedModel?.metadata.outputs[outputTarget]?.name ??
                        outputTarget}
                    </small>
                  </div>
                </div>
                <label>
                  <span>Target domain</span>
                  <select
                    value={targetHsicOperator}
                    onChange={(event) =>
                      setTargetHsicOperator(event.target.value as "<=" | ">=")
                    }
                  >
                    <option value=">=">Output ≥ threshold</option>
                    <option value="<=">Output ≤ threshold</option>
                  </select>
                </label>
                <label>
                  <span>Target HSIC threshold</span>
                  <input
                    type="number"
                    value={targetHsicThreshold}
                    onChange={(event) =>
                      setTargetHsicThreshold(Number(event.target.value))
                    }
                  />
                </label>
                {savedModel?.assessment?.profile.pilot_outputs[outputTarget] &&
                  (() => {
                    const pilot =
                      savedModel.assessment.profile.pilot_outputs[
                        outputTarget
                      ]!;
                    const intersects =
                      targetHsicThreshold >= pilot.minimum &&
                      targetHsicThreshold <= pilot.maximum;
                    return (
                      <div
                        className={`reliability-preview ${intersects ? "intersects" : "outside"}`}
                      >
                        <div className="reliability-step">
                          <i>2</i>
                          <div>
                            <strong>Bounded pilot preview</strong>
                            <small>
                              {pilot.minimum.toPrecision(4)} to{" "}
                              {pilot.maximum.toPrecision(4)} observed · q05{" "}
                              {pilot.quantile_05.toPrecision(4)} · q95{" "}
                              {pilot.quantile_95.toPrecision(4)}
                            </small>
                          </div>
                        </div>
                        <p>
                          {intersects
                            ? "The threshold intersects the validation sample. Execution still requires at least five sampled points on each side."
                            : "The threshold is outside the validation sample; the target screen may be rejected for insufficient coverage."}
                        </p>
                      </div>
                    );
                  })()}
                <div className="reliability-step">
                  <i>3</i>
                  <div>
                    <strong>Set bounded permutation evidence</strong>
                    <small>
                      This screens association with a smoothed target score; it
                      does not estimate event probability or causal influence.
                    </small>
                  </div>
                </div>
                <label>
                  <span>Target HSIC permutations</span>
                  <input
                    type="number"
                    min="0"
                    max="200"
                    value={targetHsicPermutations}
                    onChange={(event) =>
                      setTargetHsicPermutations(Number(event.target.value))
                    }
                  />
                </label>
              </div>
            </div>
          )}
          {directAnalyses.length ? (
            <div className="analysis-group-list">
              {analysisGroups.map((group) => (
                <section className="analysis-group" key={group.title}>
                  <header>
                    <h3>{group.title}</h3>
                    <p>{group.description}</p>
                  </header>
                  <div className="analysis-options">
                    {group.analyses.map((analysis: AnalysisCatalogEntry) => {
                      const incompatibility = analysisIncompatibility(
                        analysis,
                        savedModel,
                      );
                      const recommendation =
                        savedModel?.assessment?.recommendations.find(
                          (candidate) => candidate.capability === analysis.key,
                        );
                      const resourceGuidance =
                        analysis.key === "hsic" &&
                        recommendation?.safe_config?.maximum_sample_size
                          ? `Resource-safe limit: at most ${Number(recommendation.safe_config.maximum_sample_size).toLocaleString()} samples at ${Number(recommendation.safe_config.permutations ?? 100).toLocaleString()} permutations.`
                          : undefined;
                      return (
                        <label
                          className={`analysis-option ${selected.includes(analysis.key) ? "selected" : ""} ${incompatibility ? "incompatible" : ""}`}
                          key={analysis.key}
                        >
                          <input
                            type="checkbox"
                            disabled={
                              analysisComposerLocked || Boolean(incompatibility)
                            }
                            checked={selected.includes(analysis.key)}
                            onChange={() =>
                              setSelected((current) =>
                                current.includes(analysis.key)
                                  ? current.filter(
                                      (key) => key !== analysis.key,
                                    )
                                  : [...current, analysis.key],
                              )
                            }
                          />
                          <span className="option-icon">
                            {analysis.category === "Sensitivity" ? (
                              <Beaker />
                            ) : analysis.category === "Exploration" ? (
                              <FlaskConical />
                            ) : (
                              <Code2 />
                            )}
                          </span>
                          <span>
                            <strong>{analysis.name}</strong>
                            <small>{analysis.description}</small>
                            <em>
                              {incompatibility ??
                                resourceGuidance ??
                                `${analysis.resource_class} · OpenTURNS plugin v${analysis.version}`}
                            </em>
                          </span>
                        </label>
                      );
                    })}
                  </div>
                </section>
              ))}
            </div>
          ) : (
            <EmptyState
              title="Catalog unavailable"
              body="Start the local compute service to load analysis plugins."
            />
          )}
          <div className="composer-footer">
            <p>
              {selected.length} analysis task{selected.length === 1 ? "" : "s"}{" "}
              · seed 42 · deterministic provenance
            </p>
            <button
              className="button primary run-button"
              disabled={
                !savedModel || selected.length === 0 || createRun.isPending
              }
              onClick={() => createRun.mutate()}
            >
              <Play /> {createRun.isPending ? "Queuing…" : "Run analyses"}
            </button>
          </div>
        </fieldset>
      </section>
    </div>
  );
}
