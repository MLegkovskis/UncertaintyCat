import type {
  AnalysisCatalogEntry,
  AnalysisResult,
  ModelMetadata,
  Project,
  Report,
  Run,
} from "@uncertaintycat/contracts";
import type { Page, Route } from "@playwright/test";

export const catalog: AnalysisCatalogEntry[] = [
  ["monte_carlo", "Monte Carlo Propagation", "Propagation", "lite"],
  ["eda", "Exploratory Data Analysis", "Exploration", "lite"],
  ["correlation", "Correlation Analysis", "Sensitivity", "lite"],
  ["sobol", "Sobol Sensitivity", "Sensitivity", "standard"],
  ["fast", "FAST Sensitivity", "Sensitivity", "standard"],
  ["hsic", "HSIC Dependence", "Sensitivity", "standard"],
  ["taylor", "Taylor Decomposition", "Sensitivity", "standard"],
  ["convergence", "Expectation Convergence", "Propagation", "lite"],
  ["morris", "Morris Screening", "Sensitivity", "standard"],
  ["reliability", "Reliability Analysis", "Reliability", "heavy"],
  ["pce", "Polynomial Chaos Expansion", "Metamodel", "heavy"],
  ["gpr", "Gaussian Process Surrogate", "Surrogate", "heavy"],
].map(([key, name, category, resourceClass]) => ({
  key,
  version: "1.0.0",
  name,
  category,
  description: `${name} produces versioned numerical evidence.`,
  assumptions: [`${name} test assumption`],
  supports_dependent_inputs: true,
  supports_multi_output: true,
  resource_class: resourceClass as AnalysisCatalogEntry["resource_class"],
  config_schema: {},
}));

export const modelMetadata: ModelMetadata = {
  source_hash: "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
  input_dimension: 3,
  output_dimension: 1,
  inputs: [
    { index: 0, name: "x1", distribution: "Uniform", parameters: [-3.14, 3.14] },
    { index: 1, name: "x2", distribution: "Uniform", parameters: [-3.14, 3.14] },
    { index: 2, name: "x3", distribution: "Uniform", parameters: [-3.14, 3.14] },
  ],
  outputs: [{ index: 0, name: "Y" }],
  openturns_version: "1.25",
  batch_evaluation_supported: true,
  validation_runtime_ms: 12,
  warnings: [],
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
    plugin_version: "1.0.0",
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
    status,
    seed: 42,
    accuracyProfile: "standard",
    createdAt: "2026-08-19T12:00:00Z",
    completedAt: status === "running" || status === "queued" ? null : "2026-08-19T12:00:02Z",
    tasks: catalog.slice(0, 3).map((entry, index) => ({
      id: `task-${index + 1}`,
      analysisKey: entry.key,
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
}

export async function installMockApi(page: Page, options: MockApiOptions = {}) {
  let projects = options.projects ?? [];
  let runs = options.runs ?? [];
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
        ? { ownerId: "user-1", authenticated: true, email: "mlegkovskis@gmail.com" }
        : { ownerId: "guest:test", authenticated: false },
      providers: ["cloudflare"],
    }),
  );
  await page.route("**/api/v1/analyses/catalog", (route) =>
    json(route, { analyses: catalog }),
  );
  await page.route("**/api/v1/projects", async (route) => {
    if (route.request().method() === "POST") {
      const input = route.request().postDataJSON() as { name: string; description: string };
      const created = { ...project, name: input.name, description: input.description };
      projects = [created, ...projects];
      await json(route, { project: created }, 201);
      return;
    }
    await json(route, { projects });
  });
  await page.route("**/api/v1/projects/*/models", async (route) => {
    if (route.request().method() === "POST") {
      await json(
        route,
        {
          modelVersion: {
            id: "model-1",
            projectId: project.id,
            version: 1,
            sourceKind: "builder",
            sourceHash: modelMetadata.source_hash,
            metadata: modelMetadata,
            createdAt: "2026-08-19T12:00:00Z",
          },
        },
        201,
      );
      return;
    }
    await json(route, { modelVersions: [] });
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
  await page.route(/\/api\/v1\/runs\/[^/]+\/cancel$/, (route) =>
    json(route, { status: "cancelled" }),
  );
  await page.route(/\/api\/v1\/runs\/[^/]+$/, (route) =>
    json(route, { run: runs[0] ?? makeRun("succeeded") }),
  );
  await page.route("**/api/v1/reports/*", (route) =>
    json(route, { report }),
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
        body: "x1 is greatest [monte_carlo.fact:strongest_input]",
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
