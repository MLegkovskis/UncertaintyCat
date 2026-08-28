import type {
  AnalysisCatalogEntry,
  AnalysisResult,
  Dataset,
  ModelAssessment,
  ModelMetadata,
  ModelVersion,
  Project,
  Report,
  Run,
} from "@uncertaintycat/contracts";
import { EXAMPLE_CATALOG } from "@uncertaintycat/contracts";
import type { Page, Route } from "@playwright/test";

export const catalog: AnalysisCatalogEntry[] = [
  ["monte_carlo", "Uncertainty Propagation", "Propagation", "lite"],
  ["eda", "Exploratory Data Analysis", "Exploration", "lite"],
  ["correlation", "Correlation Analysis", "Sensitivity", "lite"],
  ["ancova", "ANCOVA Dependent-Input Sensitivity", "Sensitivity", "heavy"],
  ["sobol", "Sobol Sensitivity", "Sensitivity", "standard"],
  ["fast", "FAST Sensitivity", "Sensitivity", "standard"],
  ["hsic", "HSIC Dependence", "Sensitivity", "standard"],
  ["target_hsic", "Target-Domain HSIC Sensitivity", "Sensitivity", "standard"],
  ["taylor", "Taylor Decomposition", "Sensitivity", "standard"],
  ["convergence", "Expectation Convergence", "Propagation", "lite"],
  ["morris", "Morris Screening", "Sensitivity", "standard"],
  ["reliability", "Reliability Analysis", "Reliability", "heavy"],
  ["pce", "Polynomial Chaos Expansion", "Metamodel", "heavy"],
  ["gpr", "Gaussian Process Surrogate", "Surrogate", "heavy"],
  ["calibration_nlls", "Nonlinear Least-Squares Calibration", "Calibration", "heavy"],
].map(([key, name, category, resourceClass]) => ({
  key,
  version: ["ancova", "calibration_nlls", "target_hsic"].includes(key) ? "1.0.0" : "2.0.0",
  name,
  category,
  description: `${name} produces versioned numerical evidence.`,
  assumptions: [`${name} test assumption`],
  supports_dependent_inputs: !["sobol", "fast", "morris", "pce"].includes(key),
  requires_dependent_inputs: key === "ancova",
  supports_multi_output: !["calibration_nlls", "target_hsic"].includes(key),
  resource_class: resourceClass as AnalysisCatalogEntry["resource_class"],
  config_schema: {},
}));

const modelMetadata: ModelMetadata = {
  source_hash: "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
  input_dimension: 3,
  output_dimension: 1,
  inputs: [
    { index: 0, name: "x1", distribution: "Uniform", parameters: [-3.14, 3.14], kind: "continuous", mean: 0, standard_deviation: 1.81 },
    { index: 1, name: "x2", distribution: "Uniform", parameters: [-3.14, 3.14], kind: "continuous", mean: 0, standard_deviation: 1.81 },
    { index: 2, name: "x3", distribution: "Uniform", parameters: [-3.14, 3.14], kind: "continuous", mean: 0, standard_deviation: 1.81 },
  ],
  outputs: [{ index: 0, name: "Y" }],
  openturns_version: "1.25",
  batch_evaluation_supported: true,
  validation_runtime_ms: 12,
  warnings: [],
};

const modelAssessment: ModelAssessment = {
  version: "1.3.0",
  profile: {
    input_dimension: 3,
    output_dimension: 1,
    continuous_marginals: 3,
    discrete_marginals: 0,
    copula: "IndependentCopula",
    dependent_inputs: false,
    function_type: "SymbolicFunction",
    batch_support: true,
    validation_evaluation_runtime_ms: 1,
    projected_1000_evaluation_runtime_ms: 125,
    pilot_sample_size: 8,
    pilot_outputs: [{ output_index: 0, output_name: "Y", minimum: -6, maximum: 8, mean: 1.2, standard_deviation: 2.1, quantile_05: -4.5, quantile_95: 6.8, variable: true }],
  },
  workflow: { path: "direct", rationale_codes: ["DIRECT_EVALUATION_PRACTICAL"] },
  recommendations: [
    { capability: "ancova", status: "incompatible", priority: 2, rationale_codes: ["INDEPENDENT_INPUTS_USE_SOBOL"], compatibility_warnings: ["ANCOVA requires two to ten continuous inputs with a dependent copula."] },
    { capability: "gpr", status: "available", priority: 3, rationale_codes: ["DIRECT_MODEL_RUNTIME_WITHIN_FIVE_SECONDS"], compatibility_warnings: [] },
    { capability: "pce", status: "available", priority: 3, rationale_codes: ["SYMBOLIC_SMOOTH_CONTINUOUS_MODEL"], compatibility_warnings: [] },
    { capability: "target_hsic", status: "available", priority: 4, rationale_codes: ["USER_DEFINED_CRITICAL_DOMAIN_REQUIRED"], projected_evaluations: 250, compatibility_warnings: ["Define a scalar critical output domain before target-HSIC execution."] },
  ],
};

const savedModel = {
  id: "model-1",
  projectId: "project-1",
  version: 1,
  sourceKind: "example" as const,
  displayName: "Ishigami reference model",
  sourceHash: modelMetadata.source_hash,
  metadata: modelMetadata,
  assessment: modelAssessment,
  createdAt: "2026-08-19T12:00:00Z",
};

export const calibrationSavedModel: ModelVersion = {
  ...savedModel,
  id: "model-calibration",
  displayName: "Exponential calibration benchmark",
  sourceHash: EXAMPLE_CATALOG.find((example) => example.id === "calibration_exponential")!.sha256,
  metadata: {
    ...modelMetadata,
    source_hash: EXAMPLE_CATALOG.find((example) => example.id === "calibration_exponential")!.sha256,
    input_dimension: 4,
    inputs: [
      { index: 0, name: "a", distribution: "Uniform", parameters: [0, 5], kind: "continuous", mean: 2.5, standard_deviation: 1.44 },
      { index: 1, name: "b", distribution: "Uniform", parameters: [0.5, 2], kind: "continuous", mean: 1.25, standard_deviation: 0.43 },
      { index: 2, name: "c", distribution: "Uniform", parameters: [0.1, 0.6], kind: "continuous", mean: 0.35, standard_deviation: 0.14 },
      { index: 3, name: "x", distribution: "Uniform", parameters: [0.5, 9.5], kind: "continuous", mean: 5, standard_deviation: 2.6 },
    ],
    outputs: [{ index: 0, name: "y" }],
    openturns_version: "1.27.post1",
  },
  assessment: {
    ...modelAssessment,
    profile: {
      ...modelAssessment.profile,
      input_dimension: 4,
      continuous_marginals: 4,
      pilot_outputs: [{ ...modelAssessment.profile.pilot_outputs[0]!, output_name: "y" }],
    },
  },
};

const dataset: Dataset = {
  id: "dataset-1",
  projectId: "project-1",
  name: "Fixture observations.csv",
  sourceKind: "csv",
  sha256: "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
  rowCount: 6,
  columns: [
    { name: "temperature", type: "numeric", missingCount: 0, invalidNumericCount: 0, nonFiniteCount: 0, finiteCount: 6, uniqueCount: 6, minimum: 18, maximum: 23, mean: 20.5 },
    { name: "pressure", type: "numeric", missingCount: 0, invalidNumericCount: 0, nonFiniteCount: 0, finiteCount: 6, uniqueCount: 6, minimum: 1, maximum: 1.5, mean: 1.25 },
  ],
  preview: [
    { temperature: 18, pressure: 1 }, { temperature: 19, pressure: 1.1 }, { temperature: 20, pressure: 1.2 },
    { temperature: 21, pressure: 1.3 }, { temperature: 22, pressure: 1.4 }, { temperature: 23, pressure: 1.5 },
  ],
  warnings: ["temperature: fewer than 20 finite observations; fit evidence is weak."],
  createdAt: "2026-08-19T12:00:00Z",
};

export const project: Project = {
  id: "project-1",
  name: "Browser verification study",
  description: "Stateful browser fixture",
  createdAt: "2026-08-19T12:00:00Z",
  updatedAt: "2026-08-19T12:00:00Z",
};

export function analysisResult(key = "monte_carlo"): AnalysisResult {
  return {
    analysis_key: key,
    plugin_version: "2.0.0",
    result_schema_version: "1.0.0",
    model_hash: modelMetadata.source_hash,
    seed: 42,
    uq_core_version: "0.2.0",
    openturns_version: "1.25",
    status: "succeeded",
    started_at: "2026-08-19T12:00:00Z",
    completed_at: "2026-08-19T12:00:01Z",
    runtime: { duration_ms: 18, model_evaluations: 128, sample_size: 128 },
    warnings: ["Synthetic browser fixture; do not interpret scientifically."],
    assumptions: ["Inputs follow the declared independent marginals."],
    payload: {
      metrics: { mean: 3.5, standard_deviation: 1.25 },
      tables: {
        descriptive_statistics: {
          columns: ["Statistic", "Y"],
          rows: [["Mean", 3.5], ["Standard deviation", 1.25]],
          row_count: 500,
          truncated: true,
        },
      },
      series: {
        running_mean: {
          name: "Running mean",
          x: [1, 2, 3, 4],
          y: [2.8, 3.2, 3.4, 3.5],
          x_label: "Sample size",
          y_label: "Mean",
        },
      },
      matrices: {
        correlation: {
          row_labels: ["Y"],
          column_labels: ["x1", "x2", "x3"],
          values: [[0.8, -0.2, 0.05]],
        },
      },
      facts: { strongest_input: "x1", sample_count: 128 },
    },
  };
}

export function makeRun(status: Run["status"] = "succeeded"): Run {
  return {
    id: "run-1",
    projectId: project.id,
    modelVersionId: "model-1",
    projectName: project.name,
    modelDisplayName: "Ishigami reference model",
    modelVersion: 1,
    sourceKind: "example",
    status,
    seed: 42,
    accuracyProfile: "standard",
    evidenceSource: "direct",
    createdAt: "2026-08-19T12:00:00Z",
    completedAt: status === "running" || status === "queued" ? null : "2026-08-19T12:00:02Z",
    tasks: catalog.slice(0, 3).map((entry, index) => ({
      id: `task-${index + 1}`,
      analysisKey: entry.key,
      pluginVersion: entry.version,
      config: { sample_size: 128 },
      outputTargets: [],
      status:
        status === "cancelled"
          ? "cancelled"
          : status === "running" && index > 0
            ? "queued"
            : "succeeded",
      ...(status !== "cancelled" && !(status === "running" && index > 0)
        ? { result: analysisResult(entry.key) }
        : {}),
    })),
  };
}

export function makeReport(): Report {
  return {
    id: "report-1",
    runId: "run-1",
    title: "Verification report",
    status: "partially_succeeded",
    generatedAt: "2026-08-19T12:00:02Z",
    project: { id: project.id, name: project.name },
    modelVersion: {
      id: "model-1",
      version: 1,
      displayName: "Ishigami reference model",
      sourceKind: "example",
      createdAt: "2026-08-19T12:00:00Z",
    },
    seed: 42,
    accuracyProfile: "standard",
    evidenceSource: "direct",
    surrogate: null,
    model: modelMetadata,
    sections: [
      { key: "monte_carlo", status: "succeeded", result: analysisResult() },
      {
        key: "sobol",
        status: "failed",
        error: { code: "fixture_failure", message: "Deliberate partial-failure evidence." },
      },
    ],
  };
}

async function json(route: Route, body: unknown, status = 200) {
  await route.fulfill({
    status,
    contentType: "application/json",
    body: JSON.stringify(body),
  });
}

export interface MockApiOptions {
  authenticated?: boolean;
  projects?: Project[];
  runs?: Run[];
  report?: Report;
  modelUnderstanding?: (route: Route) => Promise<void>;
  models?: ModelVersion[];
}

export async function installMockApi(page: Page, options: MockApiOptions = {}) {
  let projects = options.projects ?? [];
  let runs = options.runs ?? [];
  let models = options.models ?? (options.authenticated && projects.length ? [savedModel] : []);
  let surrogates: Array<Record<string, unknown>> = [];
  let dataSurrogates: Array<Record<string, unknown>> = [];
  const report = options.report ?? makeReport();

  await page.route("**/api/auth/get-session", (route) =>
    json(
      route,
      options.authenticated
        ? {
            session: { id: "session-1", expiresAt: "2099-01-01T00:00:00Z" },
            user: { id: "user-1", name: "Mark Legkovskis", email: "mlegkovskis@gmail.com" },
          }
        : null,
    ),
  );
  await page.route("**/api/auth/sign-in/social", (route) =>
    json(route, {
      redirect: true,
      url: "https://uncertaintycat.cloudflareaccess.com/cdn-cgi/access/sso/oidc/authorize?client_id=test",
    }),
  );
  await page.route("**/api/auth/sign-out", (route) => json(route, { success: true }));
  await page.route("**/api/v1/session", (route) =>
    json(route, {
      identity: options.authenticated
        ? { ownerId: "user-1", authenticated: true, name: "Mark Legkovskis", email: "mlegkovskis@gmail.com" }
        : { ownerId: "", authenticated: false },
      providers: ["cloudflare"],
      ai: {
        provider: "groq",
        configured: true,
        modelUnderstanding: { modelId: "openai/gpt-oss-20b", label: "Groq · GPT-OSS 20B" },
        reportChat: { modelId: "openai/gpt-oss-120b", label: "Groq · GPT-OSS 120B" },
      },
    }),
  );
  await page.route("**/api/v1/analyses/catalog", (route) =>
    json(route, { analyses: catalog }),
  );
  await page.route("**/api/v1/examples", (route) =>
    json(route, { examples: EXAMPLE_CATALOG }),
  );
  await page.route("**/api/v1/projects", async (route) => {
    if (route.request().method() === "POST") {
      const input = route.request().postDataJSON() as { name: string; description: string };
      const created = {
        ...project,
        id: `project-created-${projects.length + 1}`,
        name: input.name,
        description: input.description,
      };
      projects = [created, ...projects];
      await json(route, { project: created }, 201);
      return;
    }
    await json(route, { projects });
  });
  await page.route(/\/api\/v1\/projects\/[^/]+$/, async (route) => {
    if (route.request().method() !== "DELETE") {
      await route.fallback();
      return;
    }
    const projectId = new URL(route.request().url()).pathname.split("/").at(-1)!;
    projects = projects.filter((candidate) => candidate.id !== projectId);
    await json(route, { deletedProjectId: projectId, deletedArtifactCount: 4 });
  });
  await page.route("**/api/v1/projects/*/models", async (route) => {
    if (route.request().method() === "POST") {
      const input = route.request().postDataJSON() as { displayName?: string; sourceKind?: "python" | "builder" | "example" };
      const createdModel = { ...savedModel, sourceKind: input.sourceKind ?? "builder", displayName: input.displayName ?? "Browser model" };
      models = [createdModel, ...models];
      await json(
        route,
        {
          modelVersion: createdModel,
        },
        201,
      );
      return;
    }
    await json(route, { modelVersions: models });
  });
  await page.route("**/api/v1/runs", async (route) => {
    if (route.request().method() === "POST") {
      const created = makeRun("queued");
      runs = [created, ...runs];
      await json(route, { run: created }, 202);
      return;
    }
    await json(route, { runs });
  });
  await page.route("**/api/v1/projects/*/datasets", (route) =>
    json(route, { datasets: options.authenticated ? [dataset] : [] }),
  );
  await page.route("**/api/v1/projects/*/surrogates", (route) =>
    json(route, { surrogates }),
  );
  await page.route("**/api/v1/projects/*/data-surrogates", (route) =>
    json(route, { surrogates: dataSurrogates }),
  );
  await page.route("**/api/v1/datasets", async (route) =>
    json(route, { dataset }, 201),
  );
  await page.route(/\/api\/v1\/datasets\/[^/]+\/fits$/, async (route) => {
    if (route.request().method() === "GET") {
      await json(route, { fitRuns: [] });
      return;
    }
    const input = route.request().postDataJSON() as { selectedMarginals?: Record<string, string>; copula?: string };
    const selected = input.selectedMarginals ?? {};
    const makeColumn = (name: string, values: number[]) => ({
      column: name,
      sampleSize: values.length,
      warnings: ["Fewer than 20 observations."],
      rankings: [
        { candidate: "Normal", distribution: "Normal", parameters: [20, 2], parameterDescription: ["mu", "sigma"], bic: 2.1, aic: 2, aicc: 2.2, test: { name: "Lilliefors", statistic: 0.1, pValue: 0.6, significanceLevel: 0.05, rejected: false } },
        { candidate: "Uniform", distribution: "Uniform", parameters: [17.5, 23.5], parameterDescription: ["a", "b"], bic: 2.4, aic: 2.3, aicc: 2.5, test: { name: "Lilliefors", statistic: 0.12, pValue: 0.4, significanceLevel: 0.05, rejected: false } },
      ],
      rejectedCandidates: [],
      selectedMarginal: selected[name] ?? null,
      plot: { sample: values, pdf: { x: values, y: values.map(() => 0.2) }, cdf: { empiricalX: values, empiricalY: values.map((_value, index) => (index + 1) / values.length), fittedX: values, fittedY: values.map((_value, index) => (index + 0.5) / values.length) }, qq: { theoretical: values, observed: values } },
    });
    const generated = Object.keys(selected).length > 0;
    await json(route, { fitRun: { id: generated ? "fit-2" : "fit-1", datasetId: dataset.id, status: "succeeded", config: input, result: { openturnsVersion: "1.27.post1", columns: [makeColumn("temperature", [18, 19, 20, 21, 22, 23]), makeColumn("pressure", [1, 1.1, 1.2, 1.3, 1.4, 1.5])], copula: generated ? { kind: input.copula ?? "independent", className: "IndependentCopula" } : null, generatedSource: generated ? "import openturns as ot\nproblem = ot.JointDistribution([ot.Normal(), ot.Normal()])\n" : null, builderSpec: generated ? { inputs: [] } : null, assumptions: ["OpenTURNS authority"] }, generatedSource: generated ? "import openturns as ot" : null, openturnsVersion: "1.27.post1", createdAt: "2026-08-19T12:00:00Z", completedAt: "2026-08-19T12:00:01Z" } }, 201);
  });
  await page.route(/\/api\/v1\/datasets\/[^/]+\/surrogates$/, async (route) => {
    const input = route.request().postDataJSON() as {
      inputColumns: string[];
      outputColumn: string;
      kernel: string;
      trend: string;
      validationFraction: number;
      seed: number;
    };
    const surrogate = {
      id: "data-surrogate-1",
      projectId: project.id,
      datasetId: dataset.id,
      method: "gpr",
      pluginVersion: "1.0.0",
      openturnsVersion: "1.27.post1",
      inputColumns: input.inputColumns,
      outputColumn: input.outputColumn,
      config: { kernel: input.kernel, trend: input.trend, validationFraction: input.validationFraction, seed: input.seed },
      validation: { trainingSize: 24, validationSize: 6, r2: 0.982, rmse: 0.12, normalizedRmse: 0.06, meetsDefault: true, observed: [18, 19, 20, 21, 22, 23], predicted: [18.1, 18.9, 20.2, 20.9, 22.1, 22.8] },
      artifact: { sha256: "dataabcdef", sizeBytes: 2048, resultType: "GaussianProcessRegressionResult" },
      createdAt: "2026-08-23T12:00:00Z",
    };
    dataSurrogates = [surrogate];
    await json(route, { surrogate }, 201);
  });
  await page.route(/\/api\/v1\/runs\/[^/]+\/rerun$/, (route) =>
    json(route, { run: makeRun("queued") }, 202),
  );
  await page.route(/\/api\/v1\/runs\/[^/]+\/cancel$/, (route) =>
    json(route, { status: "cancelled" }),
  );
  await page.route(/\/api\/v1\/runs\/[^/]+$/, (route) =>
    json(route, { run: runs[0] ?? makeRun("succeeded") }),
  );
  await page.route("**/api/v1/reports/*", (route) =>
    json(route, { report }),
  );
  await page.route(/\/api\/v1\/model-versions\/[^/]+\/definition$/, (route) =>
    json(route, {
      definition: {
        modelVersion: {
          id: "model-1",
          projectId: project.id,
          version: 1,
          sourceKind: "example",
          displayName: "Ishigami reference model",
          sourceHash: modelMetadata.source_hash,
          metadata: modelMetadata,
          assessment: modelAssessment,
          createdAt: "2026-08-19T12:00:00Z",
        },
        project,
        source: "import openturns as ot\n# fixture source",
        builderSpec: {
          variables: [
            { name: "x1", distribution: "Normal", parameters: [0, 1] },
            { name: "x2", distribution: "Uniform", parameters: [-1, 1] },
          ],
          outputs: [{ name: "response", formula: "x1 + x2^2" }],
          copula: { kind: "independent", correlation: [[1, 0], [0, 1]] },
        },
        visibility: "owner",
      },
    }),
  );
  await page.route(/\/api\/v1\/model-versions\/[^/]+\/understanding$/, async (route) => {
    if (options.modelUnderstanding) {
      await options.modelUnderstanding(route);
      return;
    }
    if (route.request().method() === "GET") {
      await json(route, { understanding: null });
      return;
    }
    await route.fulfill({
      contentType: "text/markdown; charset=utf-8",
      body: "### Model equation\n\n$$y = x_1 + x_2^2$$\n\n### Model overview\n\nThe validated fixture has **three inputs**.\n\n### Questions to confirm\n\n- Which units apply?",
    });
  });
  await page.route(/\/api\/v1\/model-versions\/[^/]+\/surrogates$/, async (route) => {
    const validation = analysisResult("gpr");
    validation.payload.metrics = { validation_r2: 0.98, validation_normalized_rmse: 0.06 };
    const surrogate = {
      id: "surrogate-1", projectId: project.id, sourceModelVersionId: "model-1",
      sourceModelHash: modelMetadata.source_hash, method: "gpr", pluginVersion: "2.0.0",
      openturnsVersion: "1.27.post1", status: "validated",
      validation: { config: { training_size: 128, validation_size: 128 }, outputTargets: [0], seed: 42, result: validation, guidance: { score: 0.98, normalizedRmse: 0.06, scoreThreshold: 0.95, normalizedRmseThreshold: 0.1, meetsDefault: true } },
      acknowledgement: null, artifact: null, createdAt: "2026-08-19T12:00:00Z", promotedAt: null,
    };
    surrogates = [surrogate];
    await json(route, { surrogate }, 201);
  });
  await page.route(/\/api\/v1\/surrogates\/[^/]+\/promote$/, async (route) => {
    const promoted = { ...surrogates[0], status: "promoted", artifact: { sha256: "abcdef", sizeBytes: 1024, resultType: "GaussianProcessRegressionResult" }, promotedAt: "2026-08-19T12:00:02Z" };
    surrogates = [promoted];
    await json(route, { surrogate: promoted });
  });
  await page.route(/\/api\/v1\/surrogates\/[^/]+\/copy$/, async (route) => {
    const input = route.request().postDataJSON() as {
      targetProjectId: string;
      targetModelVersionId: string;
    };
    const copied = {
      ...surrogates[0],
      id: "surrogate-copied",
      projectId: input.targetProjectId,
      sourceModelVersionId: input.targetModelVersionId,
      status: "promoted",
    };
    await json(route, { surrogate: copied }, 201);
  });
  await page.route(/\/api\/v1\/model-versions\/[^/]+\/derived-reduction$/, async (route) =>
    json(route, { modelVersion: { id: "model-reduced", projectId: project.id, version: 2, sourceKind: "python", displayName: "Morris-screened model", sourceHash: "fedcba", metadata: { ...modelMetadata, input_dimension: 2, inputs: modelMetadata.inputs.slice(0, 2) }, assessment: modelAssessment, parentVersionId: "model-1", derivation: { type: "morris_parametric_reduction" }, createdAt: "2026-08-19T12:00:03Z" } }, 201),
  );
  await page.route("**/api/v1/shared-reports/*", (route) =>
    json(route, { report }),
  );
  await page.route(/\/api\/v1\/reports\/[^/]+\/share-links$/, (route) =>
    json(
      route,
      {
        shareLink: {
          id: "link-1",
          url: "http://127.0.0.1:4173/shared/share-token",
          expiresAt: "2026-09-18T12:00:00Z",
          createdAt: "2026-08-19T12:00:00Z",
        },
      },
      201,
    ),
  );
  await page.route(/\/api\/v1\/reports\/[^/]+\/chat$/, async (route) => {
    if (route.request().method() === "POST") {
      await route.fulfill({
        status: 200,
        contentType: "text/plain; charset=utf-8",
        body: "x1 is greatest [analysis.fact:monte_carlo.strongest_input]",
      });
      return;
    }
    await json(route, { messages: [] });
  });
  await page.route("**/api/v1/reports/*/export", (route) =>
    route.fulfill({
      contentType: "application/zip",
      headers: { "Content-Disposition": "attachment; filename=uncertaintycat-run-1.zip" },
      body: "PK synthetic archive",
    }),
  );

  return {
    projects: () => projects,
    runs: () => runs,
  };
}
