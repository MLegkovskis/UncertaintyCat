import type {
  AnalysisCatalogEntry,
  AnalysisResult,
  Dataset,
  ModelAssessment,
  ModelMetadata,
  ModelVersion,
  OperatorOverview,
  OperatorProjectDetail,
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
  [
    "calibration_nlls",
    "Nonlinear Least-Squares Calibration",
    "Calibration",
    "heavy",
  ],
].map(([key, name, category, resourceClass]) => ({
  key,
  version: ["ancova", "calibration_nlls"].includes(key)
    ? "1.0.0"
    : key === "target_hsic"
      ? "1.1.0"
      : ["hsic", "morris"].includes(key)
        ? "2.1.0"
        : "2.0.0",
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
  source_hash:
    "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
  input_dimension: 3,
  output_dimension: 1,
  inputs: [
    {
      index: 0,
      name: "x1",
      distribution: "Uniform",
      parameters: [-3.14, 3.14],
      kind: "continuous",
      mean: 0,
      standard_deviation: 1.81,
    },
    {
      index: 1,
      name: "x2",
      distribution: "Uniform",
      parameters: [-3.14, 3.14],
      kind: "continuous",
      mean: 0,
      standard_deviation: 1.81,
    },
    {
      index: 2,
      name: "x3",
      distribution: "Uniform",
      parameters: [-3.14, 3.14],
      kind: "continuous",
      mean: 0,
      standard_deviation: 1.81,
    },
  ],
  outputs: [{ index: 0, name: "Y" }],
  equations: [
    {
      output_name: "Ishigami response",
      latex: String.raw`Y=\sin(x_1)+7\sin^2(x_2)+0.1x_3^4\sin(x_1)`,
      representation: "closed_form",
    },
  ],
  openturns_version: "1.25",
  batch_evaluation_supported: true,
  validation_runtime_ms: 12,
  warnings: [],
};

const modelAssessment: ModelAssessment = {
  version: "1.4.0",
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
    pilot_outputs: [
      {
        output_index: 0,
        output_name: "Y",
        minimum: -6,
        maximum: 8,
        mean: 1.2,
        standard_deviation: 2.1,
        quantile_05: -4.5,
        quantile_95: 6.8,
        variable: true,
      },
    ],
  },
  workflow: {
    path: "direct",
    rationale_codes: ["DIRECT_EVALUATION_PRACTICAL"],
  },
  recommendations: [
    {
      capability: "ancova",
      status: "incompatible",
      priority: 2,
      rationale_codes: ["INDEPENDENT_INPUTS_USE_SOBOL"],
      compatibility_warnings: [
        "ANCOVA requires two to ten continuous inputs with a dependent copula.",
      ],
    },
    {
      capability: "gpr",
      status: "available",
      priority: 3,
      rationale_codes: ["DIRECT_MODEL_RUNTIME_WITHIN_FIVE_SECONDS"],
      compatibility_warnings: [],
    },
    {
      capability: "hsic",
      status: "available",
      priority: 3,
      rationale_codes: ["PLUGIN_MODEL_CONTRACT_SATISFIED"],
      projected_evaluations: 250,
      compatibility_warnings: [],
      safe_config: { maximum_sample_size: 600, permutations: 100 },
    },
    {
      capability: "pce",
      status: "available",
      priority: 3,
      rationale_codes: ["SYMBOLIC_SMOOTH_CONTINUOUS_MODEL"],
      compatibility_warnings: [],
    },
    {
      capability: "target_hsic",
      status: "available",
      priority: 4,
      rationale_codes: ["USER_DEFINED_CRITICAL_DOMAIN_REQUIRED"],
      projected_evaluations: 250,
      compatibility_warnings: [
        "Define a scalar critical output domain before target-HSIC execution.",
      ],
      safe_config: { maximum_sample_size: 250, permutations: 100 },
    },
  ],
};

export const savedModel: ModelVersion = {
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
  sourceHash: EXAMPLE_CATALOG.find(
    (example) => example.id === "calibration_exponential",
  )!.sha256,
  metadata: {
    ...modelMetadata,
    source_hash: EXAMPLE_CATALOG.find(
      (example) => example.id === "calibration_exponential",
    )!.sha256,
    input_dimension: 4,
    inputs: [
      {
        index: 0,
        name: "a",
        distribution: "Uniform",
        parameters: [0, 5],
        kind: "continuous",
        mean: 2.5,
        standard_deviation: 1.44,
      },
      {
        index: 1,
        name: "b",
        distribution: "Uniform",
        parameters: [0.5, 2],
        kind: "continuous",
        mean: 1.25,
        standard_deviation: 0.43,
      },
      {
        index: 2,
        name: "c",
        distribution: "Uniform",
        parameters: [0.1, 0.6],
        kind: "continuous",
        mean: 0.35,
        standard_deviation: 0.14,
      },
      {
        index: 3,
        name: "x",
        distribution: "Uniform",
        parameters: [0.5, 9.5],
        kind: "continuous",
        mean: 5,
        standard_deviation: 2.6,
      },
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
      pilot_outputs: [
        { ...modelAssessment.profile.pilot_outputs[0]!, output_name: "y" },
      ],
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
    {
      name: "temperature",
      type: "numeric",
      missingCount: 0,
      invalidNumericCount: 0,
      nonFiniteCount: 0,
      finiteCount: 6,
      uniqueCount: 6,
      minimum: 18,
      maximum: 23,
      mean: 20.5,
    },
    {
      name: "pressure",
      type: "numeric",
      missingCount: 0,
      invalidNumericCount: 0,
      nonFiniteCount: 0,
      finiteCount: 6,
      uniqueCount: 6,
      minimum: 1,
      maximum: 1.5,
      mean: 1.25,
    },
  ],
  preview: [
    { temperature: 18, pressure: 1 },
    { temperature: 19, pressure: 1.1 },
    { temperature: 20, pressure: 1.2 },
    { temperature: 21, pressure: 1.3 },
    { temperature: 22, pressure: 1.4 },
    { temperature: 23, pressure: 1.5 },
  ],
  warnings: [
    "temperature: fewer than 20 finite observations; fit evidence is weak.",
  ],
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
          rows: [
            ["Mean", 3.5],
            ["Standard deviation", 1.25],
          ],
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

const FLOOD_INPUTS = [
  "Q (Flow Rate)",
  "Ks (Strickler)",
  "Zv (Downstream)",
  "Zm (Upstream)",
  "B (Width)",
  "L (Length)",
  "Zb (Bank Alt)",
  "Hd (Dyke Height)",
];

export function visualizationAuditResult(key: string): AnalysisResult {
  const result = analysisResult(key);
  result.payload = {
    metrics: {},
    tables: {},
    series: {},
    matrices: {},
    facts: {},
  };
  const rankedRows = FLOOD_INPUTS.map((name, index) => [
    name,
    Number((0.58 - index * 0.07).toFixed(3)),
    Number((0.68 - index * 0.065).toFixed(3)),
  ]);
  switch (key) {
    case "ancova":
      result.payload.tables.indices = {
        columns: [
          "Input",
          "ANCOVA Contribution",
          "Physical Contribution",
          "Correlation Contribution",
        ],
        rows: FLOOD_INPUTS.map((name, index) => [
          name,
          0.48 - index * 0.04,
          0.39 - index * 0.03,
          0.09 - index * 0.01,
        ]),
        row_count: FLOOD_INPUTS.length,
        truncated: false,
      };
      break;
    case "calibration_nlls":
      result.payload.tables.calibrated_parameters = {
        columns: ["Parameter", "Starting Value", "Calibrated Value"],
        rows: [
          ["Flow coefficient", 30, 31.2],
          ["River width", 300, 298.7],
          ["Dyke height", 3, 3.15],
        ],
        row_count: 3,
        truncated: false,
      };
      result.payload.series.observed_vs_predicted = {
        name: "Observed versus predicted",
        x: [1, 2, 3, 4, 5, 6],
        y: [1.1, 1.9, 3.15, 3.85, 5.1, 5.95],
        x_label: "Observed",
        y_label: "Predicted",
      };
      result.payload.matrices.parameter_correlation = {
        row_labels: ["Flow coefficient", "River width", "Dyke height"],
        column_labels: ["Flow coefficient", "River width", "Dyke height"],
        values: [
          [1, -0.32, 0.1],
          [-0.32, 1, 0.26],
          [0.1, 0.26, 1],
        ],
      };
      break;
    case "convergence": {
      const x = Array.from({ length: 180 }, (_, index) => index + 20);
      const mean = x.map((value) => -5.9 + 0.25 * Math.exp(-(value - 20) / 42));
      const halfWidth = x.map((value) => 0.7 / Math.sqrt(value));
      result.payload.series.running_mean = {
        name: "Running expectation estimate",
        x,
        y: mean,
        x_label: "Model evaluations",
        y_label: "y0",
      };
      result.payload.series.confidence_lower = {
        name: "95% confidence lower",
        x,
        y: mean.map((value, index) => value - halfWidth[index]!),
        x_label: "Model evaluations",
        y_label: "y0",
      };
      result.payload.series.confidence_upper = {
        name: "95% confidence upper",
        x,
        y: mean.map((value, index) => value + halfWidth[index]!),
        x_label: "Model evaluations",
        y_label: "y0",
      };
      break;
    }
    case "correlation":
    case "eda":
      result.payload.matrices.pearson = {
        row_labels: ["y0"],
        column_labels: FLOOD_INPUTS,
        values: [[0.584, -0.3, 0.516, -0.026, -0.026, -0.005, -0.174, -0.439]],
      };
      break;
    case "fast":
      result.payload.tables.indices = {
        columns: ["Variable", "First Order", "Total Order", "Interaction"],
        rows: rankedRows.map(([name, first, total]) => [
          name,
          first,
          total,
          Number(total) - Number(first),
        ]),
        row_count: FLOOD_INPUTS.length,
        truncated: false,
      };
      break;
    case "gpr":
      result.payload.series.observed_vs_predicted = {
        name: "Validation predictions",
        x: Array.from({ length: 36 }, (_, index) => index / 3),
        y: Array.from(
          { length: 36 },
          (_, index) => index / 3 + Math.sin(index) * 0.12,
        ),
        x_label: "Observed output",
        y_label: "Predicted output",
      };
      break;
    case "hsic":
      result.payload.tables.indices = {
        columns: ["Variable", "Normalized HSIC"],
        rows: rankedRows.map(([name, value]) => [name, value]),
        row_count: FLOOD_INPUTS.length,
        truncated: false,
      };
      break;
    case "monte_carlo":
      result.payload.series.output_y0 = {
        name: "y0",
        x: Array.from({ length: 256 }, (_, index) => index + 1),
        y: Array.from(
          { length: 256 },
          (_, index) => 42 + Math.sin(index / 9) * 8 + (index % 7) * 0.4,
        ),
        x_label: "Sample",
        y_label: "y0",
      };
      break;
    case "morris":
      result.payload.tables.effects = {
        columns: ["Variable", "Mean Absolute Effect", "Effect Dispersion"],
        rows: FLOOD_INPUTS.map((name, index) => [
          name,
          0.9 - index * 0.08,
          0.12 + index * 0.035,
        ]),
        row_count: FLOOD_INPUTS.length,
        truncated: false,
      };
      break;
    case "pce":
      result.payload.tables.pce_sobol_indices = {
        columns: ["Input", "First Order", "Total Order"],
        rows: rankedRows,
        row_count: FLOOD_INPUTS.length,
        truncated: false,
      };
      result.payload.series.observed_vs_predicted = {
        name: "PCE validation",
        x: [1, 2, 3, 4, 5, 6],
        y: [1.02, 2.06, 2.94, 4.03, 5.05, 5.97],
        x_label: "Observed",
        y_label: "Predicted",
      };
      break;
    case "reliability":
      result.payload.tables.design_point = {
        columns: ["Variable", "Importance Factor"],
        rows: FLOOD_INPUTS.map((name, index) => [name, 0.42 - index * 0.04]),
        row_count: FLOOD_INPUTS.length,
        truncated: false,
      };
      result.payload.series.probability_convergence = {
        name: "Failure probability estimate",
        x: Array.from({ length: 160 }, (_, index) => index + 1),
        y: Array.from(
          { length: 160 },
          (_, index) => 0.025 + 0.02 * Math.exp(-index / 28),
        ),
        x_label: "Model evaluations",
        y_label: "Probability",
      };
      break;
    case "sobol":
      result.payload.tables.indices = {
        columns: ["Variable", "First Order", "Total Order"],
        rows: rankedRows,
        row_count: FLOOD_INPUTS.length,
        truncated: false,
      };
      result.payload.matrices.second_order = {
        row_labels: FLOOD_INPUTS,
        column_labels: FLOOD_INPUTS,
        values: FLOOD_INPUTS.map((_, row) =>
          FLOOD_INPUTS.map((__, column) =>
            row === column
              ? 0
              : Number((0.08 / (1 + Math.abs(row - column))).toFixed(3)),
          ),
        ),
      };
      break;
    case "target_hsic":
      result.payload.tables.target_indices = {
        columns: ["Input", "Target R2-HSIC"],
        rows: rankedRows.map(([name, value]) => [name, value]),
        row_count: FLOOD_INPUTS.length,
        truncated: false,
      };
      break;
    case "taylor":
      result.payload.tables.indices = {
        columns: ["Variable", "Taylor Importance Factor"],
        rows: rankedRows.map(([name, value]) => [name, value]),
        row_count: FLOOD_INPUTS.length,
        truncated: false,
      };
      break;
  }
  result.payload.metrics.sample_size = 1_000;
  return result;
}

export function makeVisualizationAuditReport(): Report {
  const report = makeReport();
  report.title = "Visualization hardening audit";
  report.status = "succeeded";
  report.sections = catalog.map((entry) => ({
    key: entry.key,
    status: "succeeded" as const,
    result: visualizationAuditResult(entry.key),
  }));
  return report;
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
    completedAt:
      status === "running" || status === "queued"
        ? null
        : "2026-08-19T12:00:02Z",
    tasks: [catalog[0]!, catalog[6]!, catalog[2]!].map((entry, index) => {
      const taskStatus =
        status === "cancelled"
          ? "cancelled"
          : status === "running"
            ? index === 0
              ? "succeeded"
              : index === 1
                ? "running"
                : "queued"
            : "succeeded";
      return {
        id: `task-${index + 1}`,
        analysisKey: entry.key,
        pluginVersion: entry.version,
        config: { sample_size: 128 },
        outputTargets: [],
        status: taskStatus,
        ...(taskStatus === "succeeded"
          ? { result: analysisResult(entry.key) }
          : {}),
        ...(taskStatus === "running"
          ? {
              progress: {
                phase: "permutation_inference",
                percent: 58,
                message: "OpenTURNS is evaluating 100 permutation replicates.",
                indeterminate: true,
                attempt: 0,
                updatedAt: "2026-08-19T12:00:01Z",
              },
            }
          : taskStatus === "queued"
            ? {
                progress: {
                  phase: "queued",
                  percent: 0,
                  message: "Waiting for isolated compute capacity.",
                  indeterminate: true,
                  attempt: 0,
                  updatedAt: "2026-08-19T12:00:00Z",
                },
              }
            : {}),
      };
    }),
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
        error: {
          code: "fixture_failure",
          message: "Deliberate partial-failure evidence.",
        },
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
  operator?: boolean;
  projects?: Project[];
  runs?: Run[];
  report?: Report;
  modelUnderstanding?: (route: Route) => Promise<void>;
  models?: ModelVersion[];
  operatorOverview?: OperatorOverview;
  operatorProject?: OperatorProjectDetail;
}

export function makeOperatorOverview(): OperatorOverview {
  return {
    generatedAt: "2026-09-03T09:30:00.000Z",
    windowHours: 168,
    refreshAfterSeconds: 30,
    summary: {
      users: 3,
      projects: 9,
      models: 12,
      runs: 12,
      successfulRuns: 11,
      failedRuns: 1,
      activeRuns: 0,
      tasks: 41,
      successfulTasks: 40,
      failedTasks: 1,
      activeTasks: 0,
      runSuccessRate: 11 / 12,
    },
    runStatus: [
      { status: "succeeded", count: 11 },
      { status: "partially_succeeded", count: 1 },
    ],
    analyses: [
      {
        key: "hsic",
        total: 3,
        succeeded: 2,
        failed: 1,
        active: 0,
        successRate: 2 / 3,
        averageDurationMs: 12_400,
      },
    ],
    issues: [
      {
        id: "task-failed",
        kind: "analysis",
        code: "compute_failed",
        message: "The bounded compute task did not complete.",
        status: "failed",
        analysisKey: "hsic",
        runId: "run-1",
        projectId: "project-1",
        projectName: "Beam study",
        ownerEmail: "mlegkovskis@gmail.com",
        occurredAt: "2026-09-03T09:20:00.000Z",
      },
    ],
    recentRuns: [
      {
        id: "run-1",
        projectId: "project-1",
        projectName: "Beam study",
        modelName: "Simply supported beam",
        ownerEmail: "mlegkovskis@gmail.com",
        status: "partially_succeeded",
        createdAt: "2026-09-03T09:10:00.000Z",
        completedAt: "2026-09-03T09:20:00.000Z",
        durationMs: 600_000,
        tasks: 4,
        failedTasks: 1,
      },
    ],
    users: [
      {
        id: "user-1",
        name: "Mark Legkovskis",
        email: "mlegkovskis@gmail.com",
        registeredAt: "2026-08-01T09:00:00.000Z",
        projectCount: 3,
        periodRunCount: 5,
        periodFailedRunCount: 1,
        lastActivityAt: "2026-09-03T09:20:00.000Z",
      },
    ],
    projects: [
      {
        id: "project-1",
        name: "Beam study",
        ownerName: "Mark Legkovskis",
        ownerEmail: "mlegkovskis@gmail.com",
        modelCount: 2,
        periodRunCount: 5,
        periodFailedRunCount: 1,
        lastActivityAt: "2026-09-03T09:20:00.000Z",
      },
    ],
  };
}

export function makeOperatorProject(): OperatorProjectDetail {
  return {
    generatedAt: "2026-09-03T09:30:00.000Z",
    refreshAfterSeconds: 30,
    runPage: {
      page: 1,
      pageSize: 50,
      totalPages: 1,
      totalRuns: 1,
      start: 1,
      end: 1,
    },
    project: {
      id: "project-1",
      name: "Beam study",
      ownerName: "Another Engineer",
      ownerEmail: "engineer@example.com",
      createdAt: "2026-09-01T12:00:00.000Z",
      updatedAt: "2026-09-03T09:20:00.000Z",
      modelCount: 1,
      runCount: 1,
      taskCount: 2,
      failedTaskCount: 1,
      activeTaskCount: 0,
    },
    models: [
      {
        id: "model-1",
        displayName: "Simply supported beam",
        version: 1,
        sourceKind: "example",
        inputDimension: 4,
        outputDimension: 1,
        createdAt: "2026-09-01T12:00:00.000Z",
      },
    ],
    runs: [
      {
        id: "run-1",
        modelName: "Simply supported beam",
        modelVersion: 1,
        status: "partially_succeeded",
        createdAt: "2026-09-03T09:10:00.000Z",
        startedAt: "2026-09-03T09:10:01.000Z",
        completedAt: "2026-09-03T09:20:00.000Z",
        durationMs: 599_000,
        tasks: [
          {
            id: "task-ok",
            analysisKey: "monte_carlo",
            pluginVersion: "2.0.0",
            status: "succeeded",
            createdAt: "2026-09-03T09:10:00.000Z",
            startedAt: "2026-09-03T09:10:01.000Z",
            completedAt: "2026-09-03T09:10:02.000Z",
            durationMs: 1000,
          },
          {
            id: "task-failed",
            analysisKey: "hsic",
            pluginVersion: "2.1.0",
            status: "failed",
            createdAt: "2026-09-03T09:10:00.000Z",
            startedAt: "2026-09-03T09:10:02.000Z",
            completedAt: "2026-09-03T09:20:00.000Z",
            durationMs: 598_000,
            error: {
              code: "compute_failed",
              message: "The bounded compute task did not complete.",
            },
          },
        ],
      },
    ],
  };
}

export async function installMockApi(page: Page, options: MockApiOptions = {}) {
  let projects = options.projects ?? [];
  let runs = options.runs ?? [];
  let models =
    options.models ??
    (options.authenticated && projects.length ? [savedModel] : []);
  let surrogates: Array<Record<string, unknown>> = [];
  let dataSurrogates: Array<Record<string, unknown>> = [];
  const report = options.report ?? makeReport();

  await page.route("**/api/auth/get-session", (route) =>
    json(
      route,
      options.authenticated
        ? {
            session: { id: "session-1", expiresAt: "2099-01-01T00:00:00Z" },
            user: {
              id: "user-1",
              name: "Mark Legkovskis",
              email: "mlegkovskis@gmail.com",
            },
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
  await page.route("**/api/auth/sign-out", (route) =>
    json(route, { success: true }),
  );
  await page.route("**/api/v1/session", (route) =>
    json(route, {
      identity: options.authenticated
        ? {
            ownerId: "user-1",
            authenticated: true,
            operator: options.operator ?? false,
            name: "Mark Legkovskis",
            email: "mlegkovskis@gmail.com",
          }
        : { ownerId: "", authenticated: false, operator: false },
      providers: ["cloudflare"],
      ai: {
        provider: "groq",
        configured: true,
        modelUnderstanding: {
          modelId: "openai/gpt-oss-20b",
          label: "Groq · GPT-OSS 20B + 120B equation review",
        },
        reportChat: {
          modelId: "openai/gpt-oss-120b",
          label: "Groq · GPT-OSS 120B",
        },
      },
    }),
  );
  await page.route("**/api/v1/operator/overview?*", (route) =>
    options.operator
      ? json(route, options.operatorOverview ?? makeOperatorOverview())
      : json(
          route,
          {
            error: {
              code: "operator_access_required",
              message: "Operator access required.",
            },
          },
          403,
        ),
  );
  await page.route("**/api/v1/operator/projects/*", (route) => {
    if (!options.operator) {
      return json(
        route,
        {
          error: {
            code: "operator_access_required",
            message: "Operator access required.",
          },
        },
        403,
      );
    }
    const detail = structuredClone(
      options.operatorProject ?? makeOperatorProject(),
    );
    const requestedPage = Number(
      new URL(route.request().url()).searchParams.get("page") ?? 1,
    );
    detail.runPage.page = Math.min(
      detail.runPage.totalPages,
      Math.max(1, requestedPage),
    );
    detail.runPage.start =
      detail.runPage.totalRuns === 0
        ? 0
        : (detail.runPage.page - 1) * detail.runPage.pageSize + 1;
    detail.runPage.end = Math.min(
      detail.runPage.page * detail.runPage.pageSize,
      detail.runPage.totalRuns,
    );
    return json(route, detail);
  });
  await page.route("**/api/v1/operator/reports/*", (route) =>
    options.operator
      ? json(route, { report })
      : json(
          route,
          {
            error: {
              code: "operator_access_required",
              message: "Operator access required.",
            },
          },
          403,
        ),
  );
  await page.route("**/api/v1/analyses/catalog", (route) =>
    json(route, { analyses: catalog }),
  );
  await page.route("**/api/v1/examples", (route) =>
    json(route, { examples: EXAMPLE_CATALOG }),
  );
  await page.route("**/api/v1/projects", async (route) => {
    if (route.request().method() === "POST") {
      const input = route.request().postDataJSON() as {
        name: string;
        description: string;
      };
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
    const projectId = new URL(route.request().url()).pathname
      .split("/")
      .at(-1)!;
    projects = projects.filter((candidate) => candidate.id !== projectId);
    await json(route, { deletedProjectId: projectId, deletedArtifactCount: 4 });
  });
  await page.route("**/api/v1/projects/*/models", async (route) => {
    if (route.request().method() === "POST") {
      const input = route.request().postDataJSON() as {
        displayName?: string;
        sourceKind?: "python" | "builder" | "example";
      };
      const createdModel = {
        ...(options.models?.[0] ?? savedModel),
        sourceKind: input.sourceKind ?? "builder",
        displayName: input.displayName ?? "Browser model",
      };
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
    const input = route.request().postDataJSON() as {
      selectedMarginals?: Record<string, string>;
      copula?: string;
    };
    const selected = input.selectedMarginals ?? {};
    const makeColumn = (name: string, values: number[]) => ({
      column: name,
      sampleSize: values.length,
      warnings: ["Fewer than 20 observations."],
      rankings: [
        {
          candidate: "Normal",
          distribution: "Normal",
          parameters: [20, 2],
          parameterDescription: ["mu", "sigma"],
          bic: 2.1,
          aic: 2,
          aicc: 2.2,
          test: {
            name: "Lilliefors",
            statistic: 0.1,
            pValue: 0.6,
            significanceLevel: 0.05,
            rejected: false,
          },
        },
        {
          candidate: "Uniform",
          distribution: "Uniform",
          parameters: [17.5, 23.5],
          parameterDescription: ["a", "b"],
          bic: 2.4,
          aic: 2.3,
          aicc: 2.5,
          test: {
            name: "Lilliefors",
            statistic: 0.12,
            pValue: 0.4,
            significanceLevel: 0.05,
            rejected: false,
          },
        },
      ],
      rejectedCandidates: [],
      selectedMarginal: selected[name] ?? null,
      plot: {
        sample: values,
        pdf: { x: values, y: values.map(() => 0.2) },
        cdf: {
          empiricalX: values,
          empiricalY: values.map(
            (_value, index) => (index + 1) / values.length,
          ),
          fittedX: values,
          fittedY: values.map((_value, index) => (index + 0.5) / values.length),
        },
        qq: { theoretical: values, observed: values },
      },
    });
    const generated = Object.keys(selected).length > 0;
    await json(
      route,
      {
        fitRun: {
          id: generated ? "fit-2" : "fit-1",
          datasetId: dataset.id,
          status: "succeeded",
          config: input,
          result: {
            openturnsVersion: "1.27.post1",
            columns: [
              makeColumn("temperature", [18, 19, 20, 21, 22, 23]),
              makeColumn("pressure", [1, 1.1, 1.2, 1.3, 1.4, 1.5]),
            ],
            copula: generated
              ? {
                  kind: input.copula ?? "independent",
                  className: "IndependentCopula",
                }
              : null,
            generatedSource: generated
              ? "import openturns as ot\nproblem = ot.JointDistribution([ot.Normal(), ot.Normal()])\n"
              : null,
            builderSpec: generated ? { inputs: [] } : null,
            assumptions: ["OpenTURNS authority"],
          },
          generatedSource: generated ? "import openturns as ot" : null,
          openturnsVersion: "1.27.post1",
          createdAt: "2026-08-19T12:00:00Z",
          completedAt: "2026-08-19T12:00:01Z",
        },
      },
      201,
    );
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
      config: {
        kernel: input.kernel,
        trend: input.trend,
        validationFraction: input.validationFraction,
        seed: input.seed,
      },
      validation: {
        trainingSize: 24,
        validationSize: 6,
        r2: 0.982,
        rmse: 0.12,
        normalizedRmse: 0.06,
        meetsDefault: true,
        observed: [18, 19, 20, 21, 22, 23],
        predicted: [18.1, 18.9, 20.2, 20.9, 22.1, 22.8],
      },
      artifact: {
        sha256: "dataabcdef",
        sizeBytes: 2048,
        resultType: "GaussianProcessRegressionResult",
      },
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
  await page.route("**/api/v1/reports/*", (route) => json(route, { report }));
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
          copula: {
            kind: "independent",
            correlation: [
              [1, 0],
              [0, 1],
            ],
          },
        },
        visibility: "owner",
      },
    }),
  );
  await page.route(
    /\/api\/v1\/model-versions\/[^/]+\/understanding$/,
    async (route) => {
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
        body: "### Interpreted model equation\n\n$$Y = \\sin(x_1) + 7\\sin^2(x_2) + 0.1x_3^4\\sin(x_1)$$\n\n_AI-interpreted from the authenticated Python definition; verify against the source before engineering use._\n\n### Model overview\n\nThe validated fixture has **three inputs**.\n\n### Input uncertainty\n\nThree bounded inputs.\n\n### Dependence and propagation\n\nThe inputs are independent.\n\n### Validated pilot behaviour\n\nThe bounded pilot executed successfully.\n\n### Questions to confirm\n\n- Which units apply?",
      });
    },
  );
  await page.route(
    /\/api\/v1\/model-versions\/[^/]+\/surrogates$/,
    async (route) => {
      const validation = analysisResult("gpr");
      validation.payload.metrics = {
        validation_r2: 0.98,
        validation_normalized_rmse: 0.06,
      };
      const surrogate = {
        id: "surrogate-1",
        projectId: project.id,
        sourceModelVersionId: "model-1",
        sourceModelHash: modelMetadata.source_hash,
        method: "gpr",
        pluginVersion: "2.0.0",
        openturnsVersion: "1.27.post1",
        status: "validated",
        validation: {
          config: { training_size: 128, validation_size: 128 },
          outputTargets: [0],
          seed: 42,
          result: validation,
          guidance: {
            score: 0.98,
            normalizedRmse: 0.06,
            scoreThreshold: 0.95,
            normalizedRmseThreshold: 0.1,
            meetsDefault: true,
          },
        },
        acknowledgement: null,
        artifact: null,
        createdAt: "2026-08-19T12:00:00Z",
        promotedAt: null,
      };
      surrogates = [surrogate];
      await json(route, { surrogate }, 201);
    },
  );
  await page.route(/\/api\/v1\/surrogates\/[^/]+\/promote$/, async (route) => {
    const promoted = {
      ...surrogates[0],
      status: "promoted",
      artifact: {
        sha256: "abcdef",
        sizeBytes: 1024,
        resultType: "GaussianProcessRegressionResult",
      },
      promotedAt: "2026-08-19T12:00:02Z",
    };
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
  await page.route(
    /\/api\/v1\/model-versions\/[^/]+\/derived-reduction$/,
    async (route) =>
      json(
        route,
        {
          modelVersion: {
            id: "model-reduced",
            projectId: project.id,
            version: 2,
            sourceKind: "python",
            displayName: "Morris-screened model",
            sourceHash: "fedcba",
            metadata: {
              ...modelMetadata,
              input_dimension: 2,
              inputs: modelMetadata.inputs.slice(0, 2),
            },
            assessment: modelAssessment,
            parentVersionId: "model-1",
            derivation: { type: "morris_parametric_reduction" },
            createdAt: "2026-08-19T12:00:03Z",
          },
        },
        201,
      ),
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
      headers: {
        "Content-Disposition": "attachment; filename=uncertaintycat-run-1.zip",
      },
      body: "PK synthetic archive",
    }),
  );

  return {
    projects: () => projects,
    runs: () => runs,
  };
}
