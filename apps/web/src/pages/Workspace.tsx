import { python } from "@codemirror/lang-python";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import CodeMirror from "@uiw/react-codemirror";
import type {
  AnalysisCatalogEntry,
  ModelVersion,
  Project,
} from "@uncertaintycat/contracts";
import {
  Beaker,
  Check,
  ChevronRight,
  Code2,
  FlaskConical,
  Play,
  Plus,
  Save,
  SlidersHorizontal,
} from "lucide-react";
import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import { api } from "../api";
import {
  buildSymbolicModel,
  ISHIGAMI_SOURCE,
  type BuilderVariable,
} from "../examples";
import { EmptyState } from "../components/Status";

type AuthorMode = "code" | "builder";

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

function analysisConfig(
  key: string,
  sampleSize: number,
  reliability: {
    method: "FORM" | "MONTE_CARLO";
    threshold: number;
    operator: ">" | "<";
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
      return { ...reliability, sample_size: Math.max(100, sampleSize) };
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

function GuidedBuilder({
  variables,
  setVariables,
  formula,
  setFormula,
}: {
  variables: BuilderVariable[];
  setVariables: (items: BuilderVariable[]) => void;
  formula: string;
  setFormula: (value: string) => void;
}) {
  return (
    <div className="builder">
      <div className="builder-header">
        <div>
          <h3>Guided model builder</h3>
          <p>
            Define independent marginals and a symbolic response. The generated
            model uses the same immutable Python contract.
          </p>
        </div>
        <button
          className="button secondary small"
          onClick={() =>
            setVariables([
              ...variables,
              {
                name: `x${variables.length + 1}`,
                distribution: "Normal",
                first: 0,
                second: 1,
              },
            ])
          }
        >
          <Plus /> Add variable
        </button>
      </div>
      <div className="variable-list">
        {variables.map((variable, index) => (
          <div className="variable-row" key={index}>
            <input
              aria-label={`Variable ${index + 1} name`}
              value={variable.name}
              onChange={(event) =>
                setVariables(
                  variables.map((item, itemIndex) =>
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
                setVariables(
                  variables.map((item, itemIndex) =>
                    itemIndex === index
                      ? {
                          ...item,
                          distribution: event.target
                            .value as BuilderVariable["distribution"],
                        }
                      : item,
                  ),
                )
              }
            >
              <option>Normal</option>
              <option>Uniform</option>
            </select>
            <label>
              <span>
                {variable.distribution === "Normal" ? "Mean" : "Lower"}
              </span>
              <input
                type="number"
                value={variable.first}
                onChange={(event) =>
                  setVariables(
                    variables.map((item, itemIndex) =>
                      itemIndex === index
                        ? { ...item, first: Number(event.target.value) }
                        : item,
                    ),
                  )
                }
              />
            </label>
            <label>
              <span>
                {variable.distribution === "Normal" ? "Std dev" : "Upper"}
              </span>
              <input
                type="number"
                value={variable.second}
                onChange={(event) =>
                  setVariables(
                    variables.map((item, itemIndex) =>
                      itemIndex === index
                        ? { ...item, second: Number(event.target.value) }
                        : item,
                    ),
                  )
                }
              />
            </label>
          </div>
        ))}
      </div>
      <label className="formula-field">
        <span>Response formula</span>
        <input
          value={formula}
          onChange={(event) => setFormula(event.target.value)}
          placeholder="sin(x1) + 7 * sin(x2)^2"
        />
      </label>
    </div>
  );
}

export function Workspace() {
  const client = useQueryClient();
  const navigate = useNavigate();
  const projectsQuery = useQuery({
    queryKey: ["projects"],
    queryFn: api.listProjects,
  });
  const catalogQuery = useQuery({
    queryKey: ["catalog"],
    queryFn: api.catalog,
  });
  const [projectId, setProjectId] = useState<string>();
  const [projectName, setProjectName] = useState("My UQ study");
  const [mode, setMode] = useState<AuthorMode>("code");
  const [source, setSource] = useState(ISHIGAMI_SOURCE);
  const [variables, setVariables] = useState<BuilderVariable[]>([
    { name: "x1", distribution: "Normal", first: 0, second: 1 },
    { name: "x2", distribution: "Uniform", first: -1, second: 1 },
  ]);
  const [formula, setFormula] = useState("x1 + x2^2");
  const [savedModel, setSavedModel] = useState<ModelVersion>();
  const [selected, setSelected] = useState<string[]>([
    "monte_carlo",
    "eda",
    "sobol",
  ]);
  const [sampleSize, setSampleSize] = useState(1000);
  const [outputTarget, setOutputTarget] = useState(0);
  const [reliabilityMethod, setReliabilityMethod] = useState<
    "FORM" | "MONTE_CARLO"
  >("FORM");
  const [reliabilityThreshold, setReliabilityThreshold] = useState(0);
  const [reliabilityOperator, setReliabilityOperator] = useState<">" | "<">(
    ">",
  );
  const [pceDegree, setPceDegree] = useState(3);
  const [gprKernel, setGprKernel] = useState<GprKernel>("MATERN_2_5");
  const [gprTrend, setGprTrend] = useState<GprTrend>("CONSTANT");
  const [error, setError] = useState<string>();
  const projects = projectsQuery.data?.projects ?? [];
  const activeProjectId = projectId ?? projects[0]?.id;
  const activeProject = projects.find((item) => item.id === activeProjectId);
  const generatedSource = useMemo(
    () => buildSymbolicModel(variables, formula),
    [variables, formula],
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
      const modelSource = mode === "code" ? source : generatedSource;
      return api.createModel(activeProjectId, {
        source: modelSource,
        sourceKind:
          mode === "builder"
            ? "builder"
            : modelSource === ISHIGAMI_SOURCE
              ? "example"
              : "python",
        ...(mode === "builder" ? { builderSpec: { variables, formula } } : {}),
      });
    },
    onSuccess: ({ modelVersion }) => {
      setSavedModel(modelVersion);
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
      <section className="studio-card">
        <div className="mode-tabs">
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
        {mode === "code" ? (
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
              theme="dark"
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
            variables={variables}
            setVariables={setVariables}
            formula={formula}
            setFormula={setFormula}
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
      </section>
      {error && <div className="error-banner">{error}</div>}
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
              <>
                <label>
                  <span>Reliability method</span>
                  <select
                    value={reliabilityMethod}
                    onChange={(event) =>
                      setReliabilityMethod(
                        event.target.value as "FORM" | "MONTE_CARLO",
                      )
                    }
                  >
                    <option value="FORM">FORM</option>
                    <option value="MONTE_CARLO">Monte Carlo</option>
                  </select>
                </label>
                <label>
                  <span>Failure event</span>
                  <select
                    value={reliabilityOperator}
                    onChange={(event) =>
                      setReliabilityOperator(event.target.value as ">" | "<")
                    }
                  >
                    <option value=">">Output &gt; threshold</option>
                    <option value="<">Output &lt; threshold</option>
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
              </>
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
