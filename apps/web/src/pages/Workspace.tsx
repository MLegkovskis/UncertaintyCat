import { python } from "@codemirror/lang-python";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import CodeMirror from "@uiw/react-codemirror";
import type {
  AnalysisCatalogEntry,
  ExampleCatalogEntry,
  ModelVersion,
  Project,
  SurrogateModel,
} from "@uncertaintycat/contracts";
import { AI_MODEL_LABEL } from "@uncertaintycat/contracts";
import {
  Beaker,
  ArrowDown,
  ArrowUp,
  Check,
  ChevronRight,
  Code2,
  FlaskConical,
  Play,
  Plus,
  Save,
  SlidersHorizontal,
  Trash2,
  Search,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";

import { api, readTextStream } from "../api";
import { Markdown } from "../components/Markdown";
import { ResultView } from "../components/ResultView";
import {
  buildSymbolicModel,
  createBuilderVariable,
  distributionDefinition,
  DISTRIBUTION_REGISTRY,
  identityCorrelation,
  ISHIGAMI_SOURCE,
  validateBuilder,
  type BuilderSpec,
  type BuilderVariable,
} from "../examples";
import { EmptyState } from "../components/Status";
import { useTheme } from "../components/Theme";

type AuthorMode = "example" | "builder" | "code";

const SCALAR_ANALYSES = new Set([
  "sobol",
  "fast",
  "hsic",
  "taylor",
  "convergence",
  "morris",
  "reliability",
  "pce",
  "gpr",
]);

type GprKernel = "MATERN_1_5" | "MATERN_2_5" | "SQUARED_EXPONENTIAL";
type GprTrend = "CONSTANT" | "LINEAR";
type ReliabilityMethod =
  | "FORM"
  | "SORM"
  | "MONTE_CARLO"
  | "DIRECTIONAL_SAMPLING"
  | "SUBSET_SAMPLING";

function analysisConfig(
  key: string,
  sampleSize: number,
  reliability: {
    method: ReliabilityMethod;
    threshold: number;
    operator: ">" | ">=" | "<" | "<=";
    maximum_evaluations: number;
    target_coefficient_of_variation: number;
  },
  pceDegree: number,
  gpr: { kernel: GprKernel; trend: GprTrend },
): Record<string, unknown> {
  switch (key) {
    case "sobol":
      return { base_sample_size: Math.max(64, sampleSize) };
    case "taylor":
      return { validation_size: Math.max(64, Math.min(sampleSize, 5_000)) };
    case "morris":
      return { trajectories: 10, levels: 6 };
    case "reliability":
      return reliability;
    case "pce":
      return {
        training_size: Math.max(64, sampleSize),
        validation_size: Math.max(64, Math.min(sampleSize, 1_000)),
        degree: pceDegree,
      };
    case "gpr":
      return {
        training_size: Math.max(16, Math.min(sampleSize, 512)),
        validation_size: Math.max(64, Math.min(sampleSize, 2_000)),
        ...gpr,
      };
    case "fast":
      return { sample_size: Math.max(65, sampleSize) };
    default:
      return { sample_size: sampleSize };
  }
}

function SurrogateStudio({
  model,
  projectId,
  outputTarget,
  sampleSize,
  pceDegree,
  gprKernel,
  gprTrend,
  selectedSurrogateId,
  setSelectedSurrogateId,
}: {
  model: ModelVersion;
  projectId: string;
  outputTarget: number;
  sampleSize: number;
  pceDegree: number;
  gprKernel: GprKernel;
  gprTrend: GprTrend;
  selectedSurrogateId: string;
  setSelectedSurrogateId: (id: string) => void;
}) {
  const client = useQueryClient();
  const [method, setMethod] = useState<"pce" | "gpr">("gpr");
  const [current, setCurrent] = useState<SurrogateModel>();
  const [acknowledge, setAcknowledge] = useState(false);
  const [reason, setReason] = useState("");
  const [error, setError] = useState<string>();
  const query = useQuery({
    queryKey: ["surrogates", projectId],
    queryFn: () => api.listSurrogates(projectId),
  });
  const exact = (query.data?.surrogates ?? []).filter(
    (item) => item.sourceModelVersionId === model.id,
  );
  const promoted = exact.filter((item) => item.status === "promoted");
  const recommendation = model.assessment?.recommendations.find(
    (item) => item.capability === method,
  );
  const build = useMutation({
    mutationFn: () =>
      api.createSurrogate(model.id, {
        method,
        config:
          method === "pce"
            ? {
                degree: pceDegree,
                training_size: Math.max(30, Math.min(sampleSize, 10_000)),
                validation_size: Math.max(20, Math.min(sampleSize, 2_000)),
                sparse: true,
              }
            : {
                training_size: Math.max(16, Math.min(sampleSize, 512)),
                validation_size: Math.max(20, Math.min(sampleSize, 2_000)),
                kernel: gprKernel,
                trend: gprTrend,
              },
        outputTarget,
        seed: 42,
      }),
    onSuccess: async ({ surrogate }) => {
      setCurrent(surrogate);
      setAcknowledge(false);
      setReason("");
      setError(undefined);
      await client.invalidateQueries({ queryKey: ["surrogates", projectId] });
    },
    onError: (caught) => setError(caught instanceof Error ? caught.message : "Surrogate build failed."),
  });
  const promote = useMutation({
    mutationFn: () =>
      api.promoteSurrogate(current?.id ?? "", {
        acknowledgeOverride: acknowledge,
        reason,
      }),
    onSuccess: async ({ surrogate }) => {
      setCurrent(surrogate);
      setSelectedSurrogateId(surrogate.id);
      setError(undefined);
      await client.invalidateQueries({ queryKey: ["surrogates", projectId] });
    },
    onError: (caught) => setError(caught instanceof Error ? caught.message : "Promotion failed."),
  });
  const guidance = current?.validation.guidance;
  return (
    <section className="surrogate-studio">
      <div className="section-copy split">
        <div><span className="section-kicker">Optional Surrogate Studio</span><h3>Approximate an expensive direct model</h3><p>{recommendation?.status ?? "available"} · {recommendation?.rationale_codes.join(", ") ?? "independent validation required"}</p></div>
        <label><span>Method</span><select value={method} onChange={(event) => setMethod(event.target.value as "pce" | "gpr")}><option value="pce">Polynomial chaos</option><option value="gpr">Gaussian process regression</option></select></label>
      </div>
      <div className="surrogate-guidance">
        <span>Measured direct projection <strong>{Math.round(model.assessment?.profile.projected_1000_evaluation_runtime_ms ?? 0)} ms / 1,000 evaluations</strong></span>
        <span>Promotion defaults <strong>Q²/R² ≥ 0.95</strong> and <strong>normalized RMSE ≤ 0.10</strong></span>
      </div>
      <button className="button secondary" onClick={() => build.mutate()} disabled={build.isPending}>{build.isPending ? "Building and validating…" : `Build ${method.toUpperCase()} candidate`}</button>
      {current && guidance && (
        <div className={`surrogate-validation ${guidance.meetsDefault ? "accepted" : "review"}`}>
          <div><span>{method === "pce" ? "Hold-out Q²" : "Hold-out R²"}</span><strong>{guidance.score.toPrecision(5)}</strong></div>
          <div><span>Normalized RMSE</span><strong>{guidance.normalizedRmse.toPrecision(5)}</strong></div>
          <div><span>Guidance</span><strong>{guidance.meetsDefault ? "Meets default" : "Override required"}</strong></div>
          <details className="surrogate-evidence" open>
            <summary>Independent hold-out evidence and assumptions</summary>
            <ResultView result={current.validation.result} />
          </details>
          {!guidance.meetsDefault && <><label className="confirmation-check"><input type="checkbox" checked={acknowledge} onChange={(event) => setAcknowledge(event.target.checked)} /><span>I acknowledge the validation is below the default promotion guidance.</span></label><label><span>Recorded reason</span><input value={reason} onChange={(event) => setReason(event.target.value)} placeholder="Why this approximation is acceptable…" /></label></>}
          <button className="button primary" onClick={() => promote.mutate()} disabled={promote.isPending || (!guidance.meetsDefault && (!acknowledge || reason.trim().length < 10))}>{promote.isPending ? "Serializing OpenTURNS XML…" : "Promote exact result"}</button>
        </div>
      )}
      <label className="surrogate-selection"><span>Evidence source for downstream run</span><select value={selectedSurrogateId} onChange={(event) => setSelectedSurrogateId(event.target.value)}><option value="">Direct model</option>{promoted.map((item) => <option key={item.id} value={item.id}>Promoted {item.method.toUpperCase()} · {item.id.slice(0, 8)}</option>)}</select><small>A promoted surrogate is used only when selected here explicitly.</small></label>
      {error && <div className="inline-error" role="alert">{error}</div>}
    </section>
  );
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
    const correlation = identityCorrelation(variables.length).map((row, rowIndex) =>
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
            Define named symbolic outputs, stable OpenTURNS marginals, and
            independent or correlated inputs. The registry generates the exact source.
          </p>
        </div>
        <button
          className="button secondary small"
          onClick={() =>
            updateVariables([...spec.variables, createBuilderVariable(spec.variables.length)])
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
              <button aria-label={`Move variable ${index + 1} up`} disabled={index === 0} onClick={() => updateVariables(reorder(spec.variables, index, -1))}><ArrowUp /></button>
              <button aria-label={`Move variable ${index + 1} down`} disabled={index === spec.variables.length - 1} onClick={() => updateVariables(reorder(spec.variables, index, 1))}><ArrowDown /></button>
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
                            event.target.value as BuilderVariable["distribution"],
                          ).parameters.map((parameter) => parameter.defaultValue),
                        }
                      : item,
                  ),
                )
              }
            >
              {DISTRIBUTION_REGISTRY.map((item) => <option value={item.key} key={item.key}>{item.label}</option>)}
            </select>
            {distribution.parameters.map((parameter, parameterIndex) => (
              <label key={parameter.key}>
                <span>{parameter.label}</span>
                <input
                  aria-label={`Variable ${index + 1} ${parameter.label}`}
                  type="number"
                  step="any"
                  value={variable.parameters[parameterIndex] ?? ""}
                  onChange={(event) => updateVariables(spec.variables.map((item, itemIndex) =>
                    itemIndex === index
                      ? { ...item, parameters: item.parameters.map((value, valueIndex) => valueIndex === parameterIndex ? Number(event.target.value) : value) }
                      : item,
                  ))}
                />
              </label>
            ))}
            <button
              className="icon-button danger-icon"
              aria-label={`Remove variable ${index + 1}`}
              disabled={spec.variables.length === 1}
              onClick={() => updateVariables(spec.variables.filter((item) => item.id !== variable.id))}
            ><Trash2 /></button>
          </div>
        )})}
      </div>
      <div className="builder-subsection">
        <div className="builder-subheading">
          <div><strong>Outputs</strong><small>Each formula is evaluated by OpenTURNS SymbolicFunction.</small></div>
          <button className="button secondary small" onClick={() => setSpec({ ...spec, outputs: [...spec.outputs, { id: crypto.randomUUID(), name: `response_${spec.outputs.length + 1}`, formula: spec.variables[0]?.name ?? "0" }] })}><Plus /> Add output</button>
        </div>
        {spec.outputs.map((output, index) => (
          <div className="output-row" key={output.id}>
            <div className="reorder-controls">
              <button aria-label={`Move output ${index + 1} up`} disabled={index === 0} onClick={() => setSpec({ ...spec, outputs: reorder(spec.outputs, index, -1) })}><ArrowUp /></button>
              <button aria-label={`Move output ${index + 1} down`} disabled={index === spec.outputs.length - 1} onClick={() => setSpec({ ...spec, outputs: reorder(spec.outputs, index, 1) })}><ArrowDown /></button>
            </div>
            <input aria-label={`Output ${index + 1} name`} value={output.name} onChange={(event) => setSpec({ ...spec, outputs: spec.outputs.map((item) => item.id === output.id ? { ...item, name: event.target.value } : item) })} />
            <input aria-label={`Output ${index + 1} formula`} value={output.formula} onChange={(event) => setSpec({ ...spec, outputs: spec.outputs.map((item) => item.id === output.id ? { ...item, formula: event.target.value } : item) })} placeholder="sin(x1) + x2^2" />
            <button className="icon-button danger-icon" aria-label={`Remove output ${index + 1}`} disabled={spec.outputs.length === 1} onClick={() => setSpec({ ...spec, outputs: spec.outputs.filter((item) => item.id !== output.id) })}><Trash2 /></button>
          </div>
        ))}
      </div>
      <div className="builder-subsection">
        <label className="copula-select"><span>Input dependence</span><select value={spec.copula.kind} onChange={(event) => setSpec({ ...spec, copula: { ...spec.copula, kind: event.target.value as BuilderSpec["copula"]["kind"] } })}><option value="independent">Independent</option><option value="normal">Normal copula</option></select></label>
        {spec.copula.kind === "normal" && (
          <div
            className="correlation-editor"
            role="group"
            aria-label="Normal copula correlation matrix"
            style={{ "--correlation-size": spec.variables.length } as React.CSSProperties}
          >
            {spec.copula.correlation.map((row, rowIndex) => row.map((value, columnIndex) => (
              <input
                key={`${rowIndex}-${columnIndex}`}
                aria-label={`Correlation ${spec.variables[rowIndex]?.name} and ${spec.variables[columnIndex]?.name}`}
                type="number" min="-1" max="1" step="0.05" value={value}
                disabled={rowIndex === columnIndex || columnIndex > rowIndex}
                onChange={(event) => {
                  const next = spec.copula.correlation.map((item) => [...item]);
                  next[rowIndex]![columnIndex] = Number(event.target.value);
                  next[columnIndex]![rowIndex] = Number(event.target.value);
                  setSpec({ ...spec, copula: { ...spec.copula, correlation: next } });
                }}
              />
            ))) }
          </div>
        )}
      </div>
      {errors.length > 0 && <div className="builder-errors" role="status">{errors.map((error) => <span key={error}>{error}</span>)}</div>}
      {errors.length === 0 && (
        <details className="source-preview"><summary>Generated OpenTURNS source preview</summary><pre><code>{buildSymbolicModel(spec)}</code></pre><p>SymbolicFunction · exact analytical gradient and Hessian · {spec.copula.kind === "normal" ? "Normal copula dependence" : "independent inputs"}</p></details>
      )}
    </div>
  );
}

function ReferenceExamples({
  examples,
  selectedId,
  setSelectedId,
}: {
  examples: readonly ExampleCatalogEntry[];
  selectedId: string;
  setSelectedId: (id: string) => void;
}) {
  const [search, setSearch] = useState("");
  const needle = search.trim().toLocaleLowerCase();
  const visible = examples.filter((example) =>
    !needle || `${example.title} ${example.domain} ${example.summary}`.toLocaleLowerCase().includes(needle),
  );
  return (
    <div className="examples-browser">
      <div className="examples-toolbar">
        <div><h3>Reference-model catalog</h3><p>Canonical, hash-checked OpenTURNS models for authenticated workspaces.</p></div>
        <label className="study-search"><Search /><input aria-label="Search reference models" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search 23 examples" /></label>
      </div>
      <div className="examples-grid">
        {visible.map((example) => (
          <button className={`example-card ${example.id === selectedId ? "selected" : ""}`} key={example.id} onClick={() => { setSelectedId(example.id); window.localStorage.setItem("uncertaintycat-last-example", example.id); }}>
            <span>{example.domain}</span><strong>{example.title}</strong><p>{example.summary}</p>
            <small>{example.inputDimension} inputs · {example.outputDimension} output · {example.difficulty}</small>
            <em>{example.suggestedAnalyses.map((analysis) => analysis.replaceAll("_", " ")).join(" · ")}</em>
          </button>
        ))}
      </div>
    </div>
  );
}

function ModelUnderstandingPane({
  model,
}: {
  model: ModelVersion;
}) {
  const [content, setContent] = useState("");
  const [status, setStatus] = useState<
    "loading" | "streaming" | "ready" | "failed"
  >("loading");
  const [error, setError] = useState<string>();
  const startedModel = useRef<string | undefined>(undefined);
  const assessment = model.assessment;

  const generate = useCallback(
    async (regenerate: boolean) => {
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
          },
        );
        if (!response.ok) {
          const body = (await response.json().catch(() => ({}))) as {
            error?: { message?: string };
          };
          throw new Error(
            body.error?.message ?? "Model Understanding is unavailable.",
          );
        }
        await readTextStream(response, (chunk) =>
          setContent((current) => current + chunk),
        );
        setStatus("ready");
      } catch (caught) {
        setError(
          caught instanceof Error
            ? caught.message
            : "Model Understanding failed.",
        );
        setStatus("failed");
      }
    },
    [model.id],
  );

  useEffect(() => {
    if (startedModel.current === model.id) return;
    startedModel.current = model.id;
    let active = true;
    void api
      .getModelUnderstanding(model.id)
      .then(({ understanding }) => {
        if (!active) return;
        if (understanding?.status === "succeeded" && understanding.content) {
          setContent(understanding.content);
          setStatus("ready");
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
    };
  }, [generate, model.id]);

  return (
    <aside className="understanding-pane" aria-label="Model Understanding">
      <header>
        <div>
          <span className="section-kicker">Validate and understand</span>
          <h2>Model Understanding</h2>
        </div>
        <small>{AI_MODEL_LABEL}</small>
      </header>
      <section className="validation-facts">
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
            <dt>Derivatives</dt>
            <dd>
              {model.metadata.exact_gradient_available
                ? "Exact gradient and Hessian"
                : "Not declared exact"}
            </dd>
          </div>
          <div>
            <dt>Batch evaluation</dt>
            <dd>
              {model.metadata.batch_evaluation_supported
                ? "Supported"
                : "Pointwise fallback"}
            </dd>
          </div>
          <div>
            <dt>Validation</dt>
            <dd>{model.metadata.validation_runtime_ms.toFixed(1)} ms</dd>
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
      {assessment && (
        <section className="triage-list">
          <h3>Deterministic triage</h3>
          <p>Recommendations do not select analyses or alter variables.</p>
          {assessment.recommendations.map((item) => (
            <div
              className={`triage-row ${item.status}`}
              key={item.capability}
            >
              <strong>{item.capability.replaceAll("_", " ")}</strong>
              <span>{item.status}</span>
              <small>
                {item.rationale_codes
                  .map((code) =>
                    code.replaceAll("_", " ").toLocaleLowerCase(),
                  )
                  .join(" · ")}
              </small>
            </div>
          ))}
        </section>
      )}
      <section
        className="understanding-narrative"
        aria-live="polite"
        aria-busy={status === "streaming"}
      >
        {status === "streaming" && !content && (
          <div className="assistant-placeholder">
            <span /> <span /> <span /> Building a source-grounded explanation…
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
    </aside>
  );
}

export function Workspace() {
  const { theme } = useTheme();
  const client = useQueryClient();
  const navigate = useNavigate();
  const { projectId: routeProjectId } = useParams();
  const [searchParams] = useSearchParams();
  const sourceModelId = searchParams.get("sourceModel") ?? "";
  const dataFitId = searchParams.get("dataFit") ?? "";
  const projectsQuery = useQuery({
    queryKey: ["projects"],
    queryFn: api.listProjects,
  });
  const catalogQuery = useQuery({
    queryKey: ["catalog"],
    queryFn: api.catalog,
  });
  const examplesQuery = useQuery({ queryKey: ["examples"], queryFn: api.examples });
  const [projectId, setProjectId] = useState<string | undefined>(routeProjectId);
  const [projectName, setProjectName] = useState("My UQ study");
  const [modelName, setModelName] = useState("Ishigami reference model");
  const [parentVersionId, setParentVersionId] = useState<string>();
  const [dataFitProvenance, setDataFitProvenance] = useState<{
    fitRunId: string;
    datasetId: string;
    builderSpec?: Record<string, unknown>;
  }>();
  const [mode, setMode] = useState<AuthorMode>("example");
  const [selectedExampleId, setSelectedExampleId] = useState(
    () => window.localStorage.getItem("uncertaintycat-last-example") ?? "ishigami",
  );
  const [source, setSource] = useState<string>(ISHIGAMI_SOURCE);
  const [builderSpec, setBuilderSpec] = useState<BuilderSpec>(() => ({
    variables: [
      { id: crypto.randomUUID(), name: "x1", distribution: "Normal", parameters: [0, 1] },
      { id: crypto.randomUUID(), name: "x2", distribution: "Uniform", parameters: [-1, 1] },
    ],
    outputs: [{ id: crypto.randomUUID(), name: "response", formula: "x1 + x2^2" }],
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
  const [pceDegree, setPceDegree] = useState(3);
  const [gprKernel, setGprKernel] = useState<GprKernel>("MATERN_2_5");
  const [gprTrend, setGprTrend] = useState<GprTrend>("CONSTANT");
  const [selectedSurrogateId, setSelectedSurrogateId] = useState("");
  const [error, setError] = useState<string>();
  const projects = projectsQuery.data?.projects ?? [];
  const activeProjectId = projectId ?? projects[0]?.id;
  const activeProject = projects.find((item) => item.id === activeProjectId);
  const examples = examplesQuery.data?.examples ?? [];
  const selectedExample = examples.find((example) => example.id === selectedExampleId);
  const definitionQuery = useQuery({
    queryKey: ["model-definition", sourceModelId],
    queryFn: () => api.getModelDefinition(sourceModelId),
    enabled: Boolean(sourceModelId),
  });
  useEffect(() => {
    const definition = definitionQuery.data?.definition;
    if (!definition) return;
    setProjectId(definition.project.id);
    setSource(definition.source);
    setMode("code");
    setModelName(`${definition.modelVersion.displayName} copy`);
    setParentVersionId(definition.modelVersion.id);
    setSavedModel(undefined);
  }, [definitionQuery.data]);
  useEffect(() => {
    if (!dataFitId) return;
    try {
      const draft = JSON.parse(
        window.sessionStorage.getItem("uncertaintycat-data-lab-draft") ?? "null",
      ) as {
        fitRunId?: string;
        datasetId?: string;
        source?: string;
        builderSpec?: Record<string, unknown>;
      } | null;
      if (
        !draft?.source ||
        draft.fitRunId !== dataFitId ||
        !draft.datasetId
      )
        return;
      setSource(draft.source);
      setMode("code");
      setModelName("Data-fit model draft");
      setParentVersionId(undefined);
      setDataFitProvenance({
        fitRunId: draft.fitRunId,
        datasetId: draft.datasetId,
        ...(draft.builderSpec ? { builderSpec: draft.builderSpec } : {}),
      });
      setSavedModel(undefined);
    } catch {
      setError("The Data Lab draft could not be restored. Reopen it from Data Lab.");
    }
  }, [dataFitId]);
  const generatedSource = useMemo(
    () => validateBuilder(builderSpec).length === 0 ? buildSymbolicModel(builderSpec) : "",
    [builderSpec],
  );

  const createProject = useMutation({
    mutationFn: () =>
      api.createProject({
        name: projectName,
        description: "Created in the UncertaintyCat workspace",
      }),
    onSuccess: async ({ project }) => {
      await client.invalidateQueries({ queryKey: ["projects"] });
      setProjectId(project.id);
    },
  });
  const saveModel = useMutation({
    mutationFn: async () => {
      if (!activeProjectId) throw new Error("Create a project first.");
      const modelSource = mode === "code" ? source : mode === "example" ? selectedExample?.source ?? "" : generatedSource;
      if (!modelSource) throw new Error("Complete the model definition before validation.");
      return api.createModel(activeProjectId, {
        source: modelSource,
        displayName: mode === "example" ? selectedExample?.title ?? modelName : modelName,
        sourceKind:
          mode === "builder"
            ? "builder"
            : mode === "example"
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
    onSuccess: ({ modelVersion }) => {
      setSavedModel(modelVersion);
      setSelectedSurrogateId("");
      setError(undefined);
    },
    onError: (caught) =>
      setError(caught instanceof Error ? caught.message : "Validation failed."),
  });
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
      const analyses = selected.map((key) => ({
        analysisKey: key,
        config: analysisConfig(key, sampleSize, reliability, pceDegree, {
          kernel: gprKernel,
          trend: gprTrend,
        }),
        outputTargets: SCALAR_ANALYSES.has(key) ? [outputTarget] : [],
      }));
      return api.createRun({
        modelVersionId: savedModel.id,
        ...(selectedSurrogateId
          ? { surrogateModelId: selectedSurrogateId }
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

  if (
    !projectsQuery.isLoading &&
    projects.length === 0 &&
    !createProject.isPending
  ) {
    return (
      <div className="page narrow-page">
        <div className="page-heading">
          <span className="section-kicker">New workspace</span>
          <h1>Start with a durable project.</h1>
          <p>
            Models, runs, reports, exports, and conversations are versioned
            inside a project.
          </p>
        </div>
        <div className="onboarding-card">
          <label>
            <span>Project name</span>
            <input
              value={projectName}
              onChange={(event) => setProjectName(event.target.value)}
            />
          </label>
          <button
            className="button primary"
            onClick={() => createProject.mutate()}
          >
            Create project <ChevronRight />
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="page workspace-page">
      <div className="page-heading split">
        <div>
          <span className="section-kicker">Model studio</span>
          <h1>{activeProject?.name ?? "Loading project…"}</h1>
          <p>Author, validate, and run an immutable OpenTURNS model.</p>
        </div>
        <div className="project-select">
          <label>
            Project
            <select
              value={activeProjectId}
              onChange={(event) => {
                setProjectId(event.target.value);
                setSavedModel(undefined);
                setSelectedSurrogateId("");
              }}
            >
              {projects.map((project) => (
                <option key={project.id} value={project.id}>
                  {project.name}
                </option>
              ))}
            </select>
          </label>
        </div>
      </div>
      <div className="stepper">
        <span className={activeProject ? "complete" : "active"}>
          <i>{activeProject ? <Check /> : "1"}</i> Project
        </span>
        <b />
        <span className={savedModel ? "complete" : "active"}>
          <i>{savedModel ? <Check /> : "2"}</i> Model
        </span>
        <b />
        <span className={savedModel ? "active" : ""}>
          <i>3</i> Analysis
        </span>
      </div>
      <section className={`studio-card ${savedModel ? "validated-studio" : ""}`}>
        <div className="studio-authoring">
        {dataFitProvenance && (
          <div className="provenance-note">
            Distribution draft from retained fit {dataFitProvenance.fitRunId.slice(0, 8)}.
            Define an OpenTURNS <code>model</code> before validation; the original dataset and fit remain unchanged.
          </div>
        )}
        <label className="model-name-field">
          <span>Model name</span>
          <input value={modelName} onChange={(event) => setModelName(event.target.value)} />
        </label>
        <div className="mode-tabs">
          <button
            className={mode === "example" ? "active" : ""}
            onClick={() => setMode("example")}
          >
            <FlaskConical /> Examples
          </button>
          <button
            className={mode === "code" ? "active" : ""}
            onClick={() => setMode("code")}
          >
            <Code2 /> Python model
          </button>
          <button
            className={mode === "builder" ? "active" : ""}
            onClick={() => setMode("builder")}
          >
            <SlidersHorizontal /> Guided builder
          </button>
        </div>
        {mode === "example" ? (
          <>
            <ReferenceExamples examples={examples} selectedId={selectedExampleId} setSelectedId={setSelectedExampleId} />
            {selectedExample && (
              <div className="example-copy-actions">
                <button
                  className="button secondary"
                  onClick={() => {
                    setSource(selectedExample.source);
                    setModelName(`${selectedExample.title} copy`);
                    setParentVersionId(undefined);
                    setSavedModel(undefined);
                    setMode("code");
                  }}
                >
                  <Code2 /> Copy selected example to editable Python
                </button>
                <small>The catalog source stays immutable; saving creates a study-scoped model version.</small>
              </div>
            )}
          </>
        ) : mode === "code" ? (
          <div className="editor-shell">
            <div className="editor-title">
              <span>model.py</span>
              <small>OpenTURNS · NumPy · SciPy</small>
            </div>
            <CodeMirror
              onCreateEditor={(view) =>
                view.contentDOM.setAttribute("aria-label", "Python model source")
              }
              height="520px"
              theme={theme}
              value={source}
              extensions={[python()]}
              onChange={setSource}
              basicSetup={{
                foldGutter: true,
                highlightActiveLine: true,
                autocompletion: true,
                bracketMatching: true,
              }}
            />
          </div>
        ) : (
          <GuidedBuilder
            spec={builderSpec}
            setSpec={setBuilderSpec}
          />
        )}
        <div className="studio-footer">
          <div className="model-status">
            {savedModel ? (
              <>
                <span className="success-dot" />
                <div>
                  <strong>Validated as version {savedModel.version}</strong>
                  <small>
                    {savedModel.metadata.input_dimension} inputs ·{" "}
                    {savedModel.metadata.output_dimension} outputs · OpenTURNS{" "}
                    {savedModel.metadata.openturns_version}
                  </small>
                </div>
              </>
            ) : (
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
            )}
          </div>
          <button
            className="button primary"
            onClick={() => saveModel.mutate()}
            disabled={saveModel.isPending}
          >
            <Save /> {saveModel.isPending ? "Validating…" : "Validate & save"}
          </button>
        </div>
        </div>
        {savedModel && (
          <ModelUnderstandingPane model={savedModel} />
        )}
      </section>
      {error && <div className="error-banner">{error}</div>}
      {savedModel && activeProjectId && (
        <SurrogateStudio
          model={savedModel}
          projectId={activeProjectId}
          outputTarget={outputTarget}
          sampleSize={sampleSize}
          pceDegree={pceDegree}
          gprKernel={gprKernel}
          gprTrend={gprTrend}
          selectedSurrogateId={selectedSurrogateId}
          setSelectedSurrogateId={setSelectedSurrogateId}
        />
      )}
      <section
        className={`analysis-composer ${savedModel ? "" : "disabled-panel"}`}
      >
        <div className="composer-heading">
          <div>
            <span className="section-kicker">Run composer</span>
            <h2>Choose the evidence you need.</h2>
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
                onChange={(event) => setSampleSize(Number(event.target.value))}
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
        {(selected.includes("reliability") ||
          selected.includes("pce") ||
          selected.includes("gpr")) && (
          <div className="analysis-settings">
            {selected.includes("reliability") && (
              <div className="reliability-studio">
                <div className="reliability-step">
                  <i>1</i><div><strong>Define failure event</strong><small>Output {savedModel?.metadata.outputs[outputTarget]?.name ?? outputTarget}</small></div>
                </div>
                <label>
                  <span>Failure event</span>
                  <select
                    value={reliabilityOperator}
                    onChange={(event) =>
                      setReliabilityOperator(event.target.value as ">" | ">=" | "<" | "<=")
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
                {savedModel?.assessment?.profile.pilot_outputs[outputTarget] && (() => {
                  const pilot = savedModel.assessment.profile.pilot_outputs[outputTarget]!;
                  const intersects = reliabilityThreshold >= pilot.minimum && reliabilityThreshold <= pilot.maximum;
                  return <div className={`reliability-preview ${intersects ? "intersects" : "outside"}`}>
                    <div className="reliability-step"><i>2</i><div><strong>Bounded pilot preview</strong><small>{pilot.minimum.toPrecision(4)} to {pilot.maximum.toPrecision(4)} observed · q05 {pilot.quantile_05.toPrecision(4)} · q95 {pilot.quantile_95.toPrecision(4)}</small></div></div>
                    <p>{intersects ? "Threshold intersects the bounded validation sample." : "Threshold is outside the bounded validation sample; rare-event evidence may be weak or zero."}</p>
                  </div>;
                })()}
                <div className="reliability-step"><i>3</i><div><strong>Select method and stopping controls</strong><small>FORM/SORM are local approximations; simulation methods are sampling estimates.</small></div></div>
                <label>
                  <span>Reliability method</span>
                  <select value={reliabilityMethod} onChange={(event) => setReliabilityMethod(event.target.value as ReliabilityMethod)}>
                    <option value="FORM">FORM · local approximation</option>
                    <option value="SORM">SORM · local curvature approximation</option>
                    <option value="MONTE_CARLO">Monte Carlo simulation</option>
                    <option value="DIRECTIONAL_SAMPLING">Directional sampling</option>
                    <option value="SUBSET_SAMPLING">Subset sampling</option>
                  </select>
                </label>
                <label><span>Maximum evaluations</span><input type="number" min="100" max="2000000" value={reliabilityMaximumEvaluations} onChange={(event) => setReliabilityMaximumEvaluations(Number(event.target.value))} /></label>
                <label><span>Target coefficient of variation</span><input type="number" min="0.001" max="1" step="0.01" value={reliabilityTargetCov} onChange={(event) => setReliabilityTargetCov(Number(event.target.value))} /></label>
              </div>
            )}
            {selected.includes("pce") && (
              <label>
                <span>PCE total degree</span>
                <input
                  type="number"
                  min="1"
                  max="12"
                  value={pceDegree}
                  onChange={(event) => setPceDegree(Number(event.target.value))}
                />
              </label>
            )}
            {selected.includes("gpr") && (
              <>
                <label>
                  <span>GPR covariance kernel</span>
                  <select
                    value={gprKernel}
                    onChange={(event) =>
                      setGprKernel(event.target.value as GprKernel)
                    }
                  >
                    <option value="MATERN_1_5">Matérn 3/2</option>
                    <option value="MATERN_2_5">Matérn 5/2</option>
                    <option value="SQUARED_EXPONENTIAL">
                      Squared exponential
                    </option>
                  </select>
                </label>
                <label>
                  <span>GPR trend</span>
                  <select
                    value={gprTrend}
                    onChange={(event) =>
                      setGprTrend(event.target.value as GprTrend)
                    }
                  >
                    <option value="CONSTANT">Constant</option>
                    <option value="LINEAR">Linear</option>
                  </select>
                </label>
              </>
            )}
          </div>
        )}
        {catalogQuery.data?.analyses.length ? (
          <div className="analysis-options">
            {catalogQuery.data.analyses.map(
              (analysis: AnalysisCatalogEntry) => (
                <label
                  className={`analysis-option ${selected.includes(analysis.key) ? "selected" : ""}`}
                  key={analysis.key}
                >
                  <input
                    type="checkbox"
                    checked={selected.includes(analysis.key)}
                    onChange={() =>
                      setSelected((current) =>
                        current.includes(analysis.key)
                          ? current.filter((key) => key !== analysis.key)
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
                      {analysis.resource_class} · v{analysis.version}
                    </em>
                  </span>
                </label>
              ),
            )}
          </div>
        ) : (
          <EmptyState
            title="Catalog unavailable"
            body="Start the local compute service to load analysis plugins."
          />
        )}
        <div className="composer-footer">
          <p>
            {selected.length} analysis task{selected.length === 1 ? "" : "s"} ·
            seed 42 · deterministic provenance
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
      </section>
    </div>
  );
}
