import { zValidator } from "@hono/zod-validator";
import {
  boundedSubsetConfigSchema,
  copySurrogateSchema,
  createModelVersionSchema,
  createDataSurrogateSchema,
  createProjectSchema,
  createReducedModelSchema,
  createRunSchema,
  createSurrogateSchema,
  distributionFitSchema,
  EXAMPLE_CATALOG,
  type AnalysisCatalogEntry,
  type Dataset,
  type DataSurrogateModel,
  type DistributionFitInput,
  type DistributionFitResult,
  type DistributionFitRun,
  type ModelAssessment,
  type ModelMetadata,
  type Report,
  type SurrogateModel,
  promoteSurrogateSchema,
  subsetSamplingIncompatibility,
  uploadDatasetSchema,
} from "@uncertaintycat/contracts";
import { generateObject, generateText, stepCountIs, streamText, tool } from "ai";
import { Hono, type Context } from "hono";
import { cors } from "hono/cors";
import { secureHeaders } from "hono/secure-headers";
import { streamSSE } from "hono/streaming";
import type { ContentfulStatusCode } from "hono/utils/http-status";
import { z } from "zod";

import { createAuth, identityFor } from "./auth";
import {
  generationFailure,
  generationLeaseIsActive,
  MODEL_UNDERSTANDING_FALLBACK_TIMEOUT_MS,
  MODEL_UNDERSTANDING_LEASE_MS,
  MODEL_UNDERSTANDING_PRIMARY_TIMEOUT_MS,
  MODEL_UNDERSTANDING_PROMPT_VERSION,
  MODEL_UNDERSTANDING_REVIEW_TIMEOUT_MS,
  REPORT_CHAT_TIMEOUT_MS,
  runSequentialFallback,
} from "./ai-config";
import {
  aiProviderOptions,
  aiRuntime,
  createAiLanguageModel,
  modelUnderstandingCacheVersion,
} from "./ai-provider";
import {
  MODEL_UNDERSTANDING_REVIEW_SYSTEM_PROMPT,
  MODEL_UNDERSTANDING_SYSTEM_PROMPT,
  MODEL_UNDERSTANDING_STRUCTURED_REVIEW_SYSTEM_PROMPT,
  MODEL_UNDERSTANDING_STRUCTURED_SYSTEM_PROMPT,
  modelUnderstandingPrompt,
  modelUnderstandingReviewPrompt,
  modelUnderstandingSectionsSchema,
  modelUnderstandingValidationIssues,
  renderStructuredModelUnderstanding,
  reportChatSystemPrompt,
  selectValidatedModelUnderstanding,
  validModelUnderstanding,
} from "./ai-prompts";
import { computeFetch, destroyRunSandbox } from "./compute-client";
import {
  ComputeRequestError,
  failRunTask,
  processRunTask,
  requeueRunTask,
} from "./compute";
import {
  loadModelDefinition,
  loadOwnedRun,
  modelMetadata,
  now,
  parseJson,
  withDerivedEquations,
} from "./db";
import type { Env, Identity, RunTaskMessage } from "./env";
import { createReportBundle } from "./exports";
import {
  loadOperatorOverview,
  loadOperatorProject,
  operatorWindow,
} from "./operator";

type Variables = { requestId: string; identity?: Identity };
type AppContext = Context<{ Bindings: Env; Variables: Variables }>;
const app = new Hono<{ Bindings: Env; Variables: Variables }>();

function jsonError(
  c: AppContext,
  status: ContentfulStatusCode,
  code: string,
  message: string,
) {
  return c.json(
    { error: { code, message }, requestId: c.get("requestId") },
    status,
  );
}

function authenticatedIdentity(c: AppContext): Identity {
  const identity = c.get("identity");
  if (!identity?.authenticated) {
    throw new Error(
      "Authenticated API middleware did not resolve an identity.",
    );
  }
  return identity;
}

async function sha256Hex(value: string): Promise<string> {
  const digest = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(value),
  );
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

async function sha256Bytes(value: Uint8Array<ArrayBuffer>): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", value);
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

function decodeBase64(value: string): Uint8Array<ArrayBuffer> {
  const binary = atob(value);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1)
    bytes[index] = binary.charCodeAt(index);
  return bytes;
}

function encodeBase64(value: ArrayBuffer): string {
  const bytes = new Uint8Array(value);
  const chunks: string[] = [];
  for (let offset = 0; offset < bytes.length; offset += 0x8000) {
    chunks.push(
      String.fromCharCode(...bytes.subarray(offset, offset + 0x8000)),
    );
  }
  return btoa(chunks.join(""));
}

function forwardedJsonRequest(
  c: AppContext,
  path: string,
  body: unknown,
): Request {
  const headers = new Headers(c.req.raw.headers);
  headers.delete("Content-Length");
  headers.set("Content-Type", "application/json");
  return new Request(new URL(path, c.req.url), {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });
}

function token(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(32));
  return btoa(String.fromCharCode(...bytes))
    .replaceAll("+", "-")
    .replaceAll("/", "_")
    .replaceAll("=", "");
}

async function reportModelContext(env: Env, modelVersionId: string) {
  return env.DB.prepare(
    `SELECT m.version, m.display_name, m.source_kind, m.created_at,
            m.parent_version_id, p.id AS project_id, p.name AS project_name
     FROM model_versions m JOIN projects p ON p.id = m.project_id
     WHERE m.id = ?`,
  )
    .bind(modelVersionId)
    .first<{
      version: number;
      display_name: string;
      source_kind: "python" | "builder" | "example";
      created_at: string;
      parent_version_id: string | null;
      project_id: string;
      project_name: string;
    }>();
}

async function reportSurrogateContext(env: Env, surrogateId?: string | null) {
  if (!surrogateId) return null;
  return env.DB.prepare(
    `SELECT id, method, plugin_version, openturns_version
     FROM surrogate_models WHERE id = ? AND status = 'promoted'`,
  )
    .bind(surrogateId)
    .first<{
      id: string;
      method: "pce" | "gpr";
      plugin_version: string;
      openturns_version: string;
    }>();
}

app.use("*", secureHeaders());
app.use("*", async (c, next) => {
  const url = new URL(c.req.url);
  if (url.hostname === "www.uncertaintycat.com") {
    url.hostname = "uncertaintycat.com";
    return c.redirect(url.toString(), 308);
  }
  await next();
});

app.use(
  "*",
  cors({
    origin: (origin, c) => {
      const allowed = [
        c.env.PUBLIC_WEB_ORIGIN,
        "http://127.0.0.1:5173",
        "http://localhost:5173",
      ];
      return allowed.includes(origin) ? origin : null;
    },
    credentials: true,
  }),
);
app.use("*", async (c, next) => {
  const requestId = c.req.header("cf-ray") ?? crypto.randomUUID();
  c.set("requestId", requestId);
  c.header("X-Request-Id", requestId);
  await next();
});

app.use("/api/v1/*", async (c, next) => {
  if (c.req.path === "/api/v1/session") {
    await next();
    return;
  }
  const identity = await identityFor(c);
  if (!identity.authenticated) {
    return jsonError(
      c,
      401,
      "authentication_required",
      "Sign in with Cloudflare to access UncertaintyCat analyses.",
    );
  }
  c.set("identity", identity);
  await next();
});

app.on(["GET", "POST"], "/api/auth/*", (c) =>
  createAuth(c.env).handler(c.req.raw),
);

app.get("/health", (c) =>
  c.json({ status: "ok", service: "uncertaintycat-api" }),
);

app.get("/api/v1/session", async (c) => {
  const identity = await identityFor(c);
  const ai = aiRuntime(c.env);
  const providers =
    c.env.CLOUDFLARE_ACCESS_CLIENT_ID &&
    c.env.CLOUDFLARE_ACCESS_CLIENT_SECRET &&
    c.env.CLOUDFLARE_ACCESS_ISSUER
      ? (["cloudflare"] as const)
      : [];
  return c.json({
    identity,
    providers,
    ai: {
      provider: ai.provider,
      configured: ai.configured,
      modelUnderstanding: {
        modelId: ai.models.modelUnderstanding.modelId,
        label: ai.models.modelUnderstanding.label,
      },
      reportChat: {
        modelId: ai.models.reportChat.modelId,
        label: ai.models.reportChat.label,
      },
    },
  });
});

app.get("/api/v1/operator/overview", async (c) => {
  const identity = authenticatedIdentity(c);
  if (!identity.operator) {
    return jsonError(
      c,
      403,
      "operator_access_required",
      "This operational view is restricted to configured UncertaintyCat operators.",
    );
  }
  const windowHours = operatorWindow(c.req.query("hours"));
  const overview = await loadOperatorOverview(c.env, windowHours);
  c.header("Cache-Control", "private, no-store");
  console.log(
    JSON.stringify({
      event: "operator_overview_read",
      requestId: c.get("requestId"),
      operatorId: identity.ownerId,
      windowHours,
    }),
  );
  return c.json(overview);
});

app.get("/api/v1/operator/projects/:projectId", async (c) => {
  const identity = authenticatedIdentity(c);
  if (!identity.operator) {
    return jsonError(
      c,
      403,
      "operator_access_required",
      "This operational view is restricted to configured UncertaintyCat operators.",
    );
  }
  const pageQuery = c.req.query("page");
  const requestedPage = pageQuery === undefined ? undefined : Number(pageQuery);
  const focusedRunId = c.req.query("run");
  const project = await loadOperatorProject(
    c.env,
    c.req.param("projectId"),
    requestedPage,
    focusedRunId,
  );
  if (!project) {
    return jsonError(
      c,
      404,
      "operator_project_not_found",
      "The project no longer exists.",
    );
  }
  c.header("Cache-Control", "private, no-store");
  console.log(
    JSON.stringify({
      event: "operator_project_read",
      requestId: c.get("requestId"),
      operatorId: identity.ownerId,
      projectId: project.project.id,
      page: project.runPage.page,
      focusedRunId: focusedRunId ?? null,
    }),
  );
  return c.json(project);
});

app.get("/api/v1/operator/reports/:reportId", async (c) => {
  const identity = authenticatedIdentity(c);
  if (!identity.operator) {
    return jsonError(
      c,
      403,
      "operator_access_required",
      "This operational view is restricted to configured UncertaintyCat operators.",
    );
  }
  const report = await loadRetainedReport(c.env, c.req.param("reportId"));
  if (!report) {
    return jsonError(
      c,
      404,
      "operator_report_not_found",
      "The retained numerical report is not available.",
    );
  }
  c.header("Cache-Control", "private, no-store");
  console.log(
    JSON.stringify({
      event: "operator_report_read",
      requestId: c.get("requestId"),
      operatorId: identity.ownerId,
      runId: report.runId,
    }),
  );
  return c.json({ report });
});

app.get("/api/v1/analyses/catalog", async (c) => {
  const response = await computeFetch(c.env, "/v1/catalog").catch(() => null);
  if (!response)
    return jsonError(
      c,
      503,
      "catalog_unavailable",
      "The analysis catalog is unavailable.",
    );
  if (!response.ok)
    return jsonError(
      c,
      503,
      "catalog_unavailable",
      "The analysis catalog is unavailable.",
    );
  return c.json({
    analyses: (await response.json()) as AnalysisCatalogEntry[],
  });
});

app.get("/api/v1/examples", (c) =>
  c.json({
    examples: EXAMPLE_CATALOG.map((example) => ({ ...example })),
  }),
);

app.get("/api/v1/examples/:exampleId", (c) => {
  const example = EXAMPLE_CATALOG.find(
    (entry) => entry.id === c.req.param("exampleId"),
  );
  if (!example)
    return jsonError(c, 404, "example_not_found", "Reference model not found.");
  return c.json({ example });
});

interface DatasetRow {
  id: string;
  project_id: string;
  name: string;
  source_kind: Dataset["sourceKind"];
  sha256: string;
  row_count: number;
  column_metadata_json: string;
  created_at: string;
}

function datasetPayload(row: DatasetRow): Dataset {
  const metadata = parseJson<Pick<Dataset, "columns" | "preview" | "warnings">>(
    row.column_metadata_json,
    { columns: [], preview: [], warnings: [] },
  );
  return {
    id: row.id,
    projectId: row.project_id,
    name: row.name,
    sourceKind: row.source_kind,
    sha256: row.sha256,
    rowCount: row.row_count,
    columns: metadata.columns,
    preview: metadata.preview,
    warnings: metadata.warnings,
    createdAt: row.created_at,
  };
}

interface DistributionFitRow {
  id: string;
  dataset_id: string;
  status: DistributionFitRun["status"];
  config_json: string;
  result_json: string | null;
  generated_source: string | null;
  error_json: string | null;
  openturns_version: string | null;
  created_at: string;
  completed_at: string | null;
}

function distributionFitPayload(row: DistributionFitRow): DistributionFitRun {
  return {
    id: row.id,
    datasetId: row.dataset_id,
    status: row.status,
    config: parseJson<DistributionFitInput>(row.config_json, {
      selectedColumns: [],
      candidates: [],
      selectedMarginals: {},
      copula: "independent",
      significanceLevel: 0.05,
    }),
    result: parseJson<DistributionFitResult | null>(row.result_json, null),
    generatedSource: row.generated_source,
    error: parseJson<{ code: string; message: string } | null>(
      row.error_json,
      null,
    ),
    openturnsVersion: row.openturns_version,
    createdAt: row.created_at,
    completedAt: row.completed_at,
  };
}

interface DataSurrogateRow {
  id: string;
  project_id: string;
  dataset_id: string;
  method: "gpr";
  plugin_version: string;
  openturns_version: string;
  input_columns_json: string;
  output_column: string;
  config_json: string;
  validation_json: string;
  artifact_json: string;
  created_at: string;
}

function dataSurrogatePayload(row: DataSurrogateRow): DataSurrogateModel {
  return {
    id: row.id,
    projectId: row.project_id,
    datasetId: row.dataset_id,
    method: row.method,
    pluginVersion: row.plugin_version,
    openturnsVersion: row.openturns_version,
    inputColumns: parseJson<string[]>(row.input_columns_json, []),
    outputColumn: row.output_column,
    config: parseJson<DataSurrogateModel["config"]>(row.config_json, {
      kernel: "MATERN_2_5",
      trend: "CONSTANT",
      seed: 42,
      validationFraction: 0.2,
    }),
    validation: parseJson<DataSurrogateModel["validation"]>(
      row.validation_json,
      {} as DataSurrogateModel["validation"],
    ),
    artifact: parseJson<DataSurrogateModel["artifact"]>(
      row.artifact_json,
      {} as DataSurrogateModel["artifact"],
    ),
    createdAt: row.created_at,
  };
}

type StoredSurrogateValidation = SurrogateModel["validation"] & {
  artifact?: { sha256: string; sizeBytes: number; resultType: string } | null;
};

interface SurrogateRow {
  id: string;
  project_id: string;
  source_model_version_id: string;
  source_model_hash: string;
  method: SurrogateModel["method"];
  plugin_version: string;
  openturns_version: string;
  status: SurrogateModel["status"];
  validation_json: string;
  acknowledgement_json: string | null;
  object_key: string | null;
  created_at: string;
  promoted_at: string | null;
}

function surrogatePayload(row: SurrogateRow): SurrogateModel {
  const validation = parseJson<StoredSurrogateValidation>(
    row.validation_json,
    {} as StoredSurrogateValidation,
  );
  return {
    id: row.id,
    projectId: row.project_id,
    sourceModelVersionId: row.source_model_version_id,
    sourceModelHash: row.source_model_hash,
    method: row.method,
    pluginVersion: row.plugin_version,
    openturnsVersion: row.openturns_version,
    status: row.status,
    validation,
    acknowledgement: parseJson<{
      acknowledgeOverride: boolean;
      reason: string;
    } | null>(row.acknowledgement_json, null),
    artifact: validation.artifact ?? null,
    createdAt: row.created_at,
    promotedAt: row.promoted_at,
  };
}

const surrogateColumns = `id, project_id, source_model_version_id,
  source_model_hash, method, plugin_version, openturns_version, status,
  validation_json, acknowledgement_json, object_key, created_at, promoted_at`;

app.get("/api/v1/projects", async (c) => {
  const identity = authenticatedIdentity(c);
  const rows = await c.env.DB.prepare(
    "SELECT id, name, description, created_at, updated_at FROM projects WHERE owner_id = ? ORDER BY updated_at DESC",
  )
    .bind(identity.ownerId)
    .all<{
      id: string;
      name: string;
      description: string;
      created_at: string;
      updated_at: string;
    }>();
  return c.json({
    projects: rows.results.map((row) => ({
      id: row.id,
      name: row.name,
      description: row.description,
      createdAt: row.created_at,
      updatedAt: row.updated_at,
    })),
  });
});

app.post(
  "/api/v1/projects",
  zValidator("json", createProjectSchema),
  async (c) => {
    const identity = authenticatedIdentity(c);
    const input = c.req.valid("json");
    const id = crypto.randomUUID();
    const timestamp = now();
    await c.env.DB.prepare(
      "INSERT INTO projects (id, owner_id, name, description, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
    )
      .bind(
        id,
        identity.ownerId,
        input.name,
        input.description,
        timestamp,
        timestamp,
      )
      .run();
    return c.json(
      {
        project: {
          id,
          name: input.name,
          description: input.description,
          createdAt: timestamp,
          updatedAt: timestamp,
        },
      },
      201,
    );
  },
);

app.delete("/api/v1/projects/:projectId", async (c) => {
  const identity = authenticatedIdentity(c);
  const projectId = c.req.param("projectId");
  const project = await c.env.DB.prepare(
    "SELECT id FROM projects WHERE id = ? AND owner_id = ?",
  )
    .bind(projectId, identity.ownerId)
    .first<{ id: string }>();
  if (!project)
    return jsonError(c, 404, "project_not_found", "Project not found.");

  const [artifactRows, activeRunRows] = await Promise.all([
    c.env.DB.prepare(
      `SELECT source_key AS object_key FROM model_versions WHERE project_id = ?
       UNION ALL
       SELECT object_key FROM datasets WHERE project_id = ?
       UNION ALL
       SELECT object_key FROM surrogate_models
         WHERE project_id = ? AND object_key IS NOT NULL
       UNION ALL
       SELECT object_key FROM data_surrogate_models WHERE project_id = ?`,
    )
      .bind(projectId, projectId, projectId, projectId)
      .all<{ object_key: string }>(),
    c.env.DB.prepare(
      `SELECT id FROM runs WHERE project_id = ? AND owner_id = ?
       AND status IN ('queued', 'running')`,
    )
      .bind(projectId, identity.ownerId)
      .all<{ id: string }>(),
  ]);

  await c.env.DB.batch([
    c.env.DB.prepare(
      `DELETE FROM idempotency_keys
       WHERE owner_id = ? AND resource_type = 'run'
         AND resource_id IN (SELECT id FROM runs WHERE project_id = ?)`,
    ).bind(identity.ownerId, projectId),
    c.env.DB.prepare("DELETE FROM projects WHERE id = ? AND owner_id = ?").bind(
      projectId,
      identity.ownerId,
    ),
  ]);
  await Promise.all(
    activeRunRows.results.map((run) => destroyRunSandbox(c.env, run.id)),
  );
  const artifactKeys = [
    ...new Set(
      artifactRows.results
        .map((row) => row.object_key)
        .filter((key): key is string => Boolean(key)),
    ),
  ];
  let deletedArtifactCount = 0;
  if (artifactKeys.length > 0) {
    try {
      await c.env.ARTIFACTS.delete(artifactKeys);
      deletedArtifactCount = artifactKeys.length;
    } catch (error) {
      console.error(
        JSON.stringify({
          event: "project_artifact_cleanup_failed",
          requestId: c.get("requestId"),
          projectId,
          artifactCount: artifactKeys.length,
          error: error instanceof Error ? error.message : String(error),
        }),
      );
    }
  }
  c.header("Cache-Control", "private, no-store");
  return c.json({ deletedProjectId: projectId, deletedArtifactCount });
});

app.get("/api/v1/projects/:projectId/datasets", async (c) => {
  const identity = authenticatedIdentity(c);
  const projectId = c.req.param("projectId");
  const ownership = await c.env.DB.prepare(
    "SELECT id FROM projects WHERE id = ? AND owner_id = ?",
  )
    .bind(projectId, identity.ownerId)
    .first();
  if (!ownership)
    return jsonError(c, 404, "project_not_found", "Project not found.");
  const rows = await c.env.DB.prepare(
    `SELECT id, project_id, name, source_kind, sha256, row_count,
            column_metadata_json, created_at
     FROM datasets WHERE project_id = ? AND owner_id = ? ORDER BY created_at DESC`,
  )
    .bind(projectId, identity.ownerId)
    .all<DatasetRow>();
  c.header("Cache-Control", "private, no-store");
  return c.json({ datasets: rows.results.map(datasetPayload) });
});

app.post(
  "/api/v1/datasets",
  zValidator("json", uploadDatasetSchema),
  async (c) => {
    const identity = authenticatedIdentity(c);
    const input = c.req.valid("json");
    const project = await c.env.DB.prepare(
      "SELECT id FROM projects WHERE id = ? AND owner_id = ?",
    )
      .bind(input.projectId, identity.ownerId)
      .first();
    if (!project)
      return jsonError(c, 404, "project_not_found", "Project not found.");
    let bytes: Uint8Array<ArrayBuffer>;
    try {
      bytes = decodeBase64(input.contentBase64);
    } catch {
      return jsonError(
        c,
        422,
        "invalid_dataset",
        "The dataset is not valid base64 data.",
      );
    }
    if (!bytes.length || bytes.length > 10_000_000)
      return jsonError(
        c,
        413,
        "dataset_too_large",
        "Datasets must be between 1 byte and 10 MB.",
      );
    const inspection = await computeFetch(c.env, "/v1/data/inspect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        content_base64: input.contentBase64,
        source_kind: input.sourceKind,
      }),
    }).catch(() => null);
    if (!inspection)
      return jsonError(
        c,
        503,
        "compute_unavailable",
        "Dataset validation is unavailable.",
      );
    const inspectionBody = (await inspection.json()) as {
      dataset?: {
        rowCount: number;
        columns: Dataset["columns"];
        preview: Dataset["preview"];
        warnings: string[];
      };
      error?: { code?: string; message?: string };
    };
    if (!inspection.ok || !inspectionBody.dataset)
      return jsonError(
        c,
        422,
        inspectionBody.error?.code ?? "invalid_dataset",
        inspectionBody.error?.message ?? "Dataset validation failed.",
      );
    const id = crypto.randomUUID();
    const extension = input.sourceKind === "xlsx" ? "xlsx" : "csv";
    const objectKey = `datasets/${identity.ownerId}/${input.projectId}/${id}.${extension}`;
    const sha256 = await sha256Bytes(bytes);
    const timestamp = now();
    await c.env.ARTIFACTS.put(objectKey, bytes, {
      httpMetadata: {
        contentType:
          input.sourceKind === "xlsx"
            ? "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            : "text/csv; charset=utf-8",
      },
      customMetadata: { sha256, originalName: input.name },
    });
    try {
      await c.env.DB.batch([
        c.env.DB.prepare(
          `INSERT INTO datasets
           (id, project_id, owner_id, name, source_kind, object_key, sha256,
            row_count, column_metadata_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        ).bind(
          id,
          input.projectId,
          identity.ownerId,
          input.name,
          input.sourceKind,
          objectKey,
          sha256,
          inspectionBody.dataset.rowCount,
          JSON.stringify({
            columns: inspectionBody.dataset.columns,
            preview: inspectionBody.dataset.preview,
            warnings: inspectionBody.dataset.warnings,
          }),
          timestamp,
        ),
        c.env.DB.prepare(
          "UPDATE projects SET updated_at = ? WHERE id = ?",
        ).bind(timestamp, input.projectId),
      ]);
    } catch (error) {
      await c.env.ARTIFACTS.delete(objectKey);
      throw error;
    }
    return c.json(
      {
        dataset: {
          id,
          projectId: input.projectId,
          name: input.name,
          sourceKind: input.sourceKind,
          sha256,
          rowCount: inspectionBody.dataset.rowCount,
          columns: inspectionBody.dataset.columns,
          preview: inspectionBody.dataset.preview,
          warnings: inspectionBody.dataset.warnings,
          createdAt: timestamp,
        } satisfies Dataset,
      },
      201,
    );
  },
);

app.get("/api/v1/datasets/:datasetId/fits", async (c) => {
  const identity = authenticatedIdentity(c);
  const dataset = await c.env.DB.prepare(
    "SELECT id FROM datasets WHERE id = ? AND owner_id = ?",
  )
    .bind(c.req.param("datasetId"), identity.ownerId)
    .first();
  if (!dataset)
    return jsonError(c, 404, "dataset_not_found", "Dataset not found.");
  const rows = await c.env.DB.prepare(
    `SELECT id, dataset_id, status, config_json, result_json, generated_source,
            error_json, openturns_version, created_at, completed_at
     FROM data_analysis_runs WHERE dataset_id = ? AND owner_id = ?
     ORDER BY created_at DESC`,
  )
    .bind(c.req.param("datasetId"), identity.ownerId)
    .all<DistributionFitRow>();
  c.header("Cache-Control", "private, no-store");
  return c.json({ fitRuns: rows.results.map(distributionFitPayload) });
});

app.post(
  "/api/v1/datasets/:datasetId/fits",
  zValidator("json", distributionFitSchema),
  async (c) => {
    const identity = authenticatedIdentity(c);
    const dataset = await c.env.DB.prepare(
      `SELECT id, source_kind, object_key FROM datasets
       WHERE id = ? AND owner_id = ?`,
    )
      .bind(c.req.param("datasetId"), identity.ownerId)
      .first<{
        id: string;
        source_kind: Dataset["sourceKind"];
        object_key: string;
      }>();
    if (!dataset)
      return jsonError(c, 404, "dataset_not_found", "Dataset not found.");
    const object = await c.env.ARTIFACTS.get(dataset.object_key);
    if (!object)
      return jsonError(
        c,
        500,
        "dataset_artifact_missing",
        "The immutable dataset artifact is missing.",
      );
    const input = c.req.valid("json");
    const id = crypto.randomUUID();
    const timestamp = now();
    await c.env.DB.prepare(
      `INSERT INTO data_analysis_runs
       (id, dataset_id, owner_id, status, config_json, created_at)
       VALUES (?, ?, ?, 'running', ?, ?)`,
    )
      .bind(id, dataset.id, identity.ownerId, JSON.stringify(input), timestamp)
      .run();
    const response = await computeFetch(c.env, "/v1/data/fit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        content_base64: encodeBase64(await object.arrayBuffer()),
        source_kind: dataset.source_kind,
        selected_columns: input.selectedColumns,
        candidates: input.candidates,
        selected_marginals: input.selectedMarginals,
        copula: input.copula,
        significance_level: input.significanceLevel,
      }),
    }).catch(() => null);
    const completedAt = now();
    if (!response) {
      const error = {
        code: "compute_unavailable",
        message: "Distribution fitting is unavailable.",
      };
      await c.env.DB.prepare(
        `UPDATE data_analysis_runs SET status = 'failed', error_json = ?, completed_at = ?
         WHERE id = ?`,
      )
        .bind(JSON.stringify(error), completedAt, id)
        .run();
      return jsonError(c, 503, error.code, error.message);
    }
    const body = (await response.json()) as {
      fit?: DistributionFitResult;
      error?: { code?: string; message?: string };
    };
    if (!response.ok || !body.fit) {
      const error = {
        code: body.error?.code ?? "distribution_fit_failed",
        message: body.error?.message ?? "Distribution fitting failed.",
      };
      await c.env.DB.prepare(
        `UPDATE data_analysis_runs SET status = 'failed', error_json = ?, completed_at = ?
         WHERE id = ?`,
      )
        .bind(JSON.stringify(error), completedAt, id)
        .run();
      return jsonError(c, 422, error.code, error.message);
    }
    await c.env.DB.prepare(
      `UPDATE data_analysis_runs SET status = 'succeeded', result_json = ?,
              generated_source = ?, openturns_version = ?, completed_at = ?
       WHERE id = ?`,
    )
      .bind(
        JSON.stringify(body.fit),
        body.fit.generatedSource ?? null,
        body.fit.openturnsVersion,
        completedAt,
        id,
      )
      .run();
    return c.json(
      {
        fitRun: {
          id,
          datasetId: dataset.id,
          status: "succeeded",
          config: input,
          result: body.fit,
          generatedSource: body.fit.generatedSource ?? null,
          error: null,
          openturnsVersion: body.fit.openturnsVersion,
          createdAt: timestamp,
          completedAt,
        } satisfies DistributionFitRun,
      },
      201,
    );
  },
);

app.get("/api/v1/projects/:projectId/data-surrogates", async (c) => {
  const identity = authenticatedIdentity(c);
  const projectId = c.req.param("projectId");
  const project = await c.env.DB.prepare(
    "SELECT id FROM projects WHERE id = ? AND owner_id = ?",
  )
    .bind(projectId, identity.ownerId)
    .first();
  if (!project)
    return jsonError(c, 404, "project_not_found", "Project not found.");
  const rows = await c.env.DB.prepare(
    `SELECT id, project_id, dataset_id, method, plugin_version,
            openturns_version, input_columns_json, output_column, config_json,
            validation_json, artifact_json, created_at
     FROM data_surrogate_models
     WHERE project_id = ? AND owner_id = ? ORDER BY created_at DESC`,
  )
    .bind(projectId, identity.ownerId)
    .all<DataSurrogateRow>();
  c.header("Cache-Control", "private, no-store");
  return c.json({ surrogates: rows.results.map(dataSurrogatePayload) });
});

app.post(
  "/api/v1/datasets/:datasetId/surrogates",
  zValidator("json", createDataSurrogateSchema),
  async (c) => {
    const identity = authenticatedIdentity(c);
    const dataset = await c.env.DB.prepare(
      `SELECT id, project_id, source_kind, object_key FROM datasets
       WHERE id = ? AND owner_id = ?`,
    )
      .bind(c.req.param("datasetId"), identity.ownerId)
      .first<{
        id: string;
        project_id: string;
        source_kind: Dataset["sourceKind"];
        object_key: string;
      }>();
    if (!dataset)
      return jsonError(c, 404, "dataset_not_found", "Dataset not found.");
    const object = await c.env.ARTIFACTS.get(dataset.object_key);
    if (!object)
      return jsonError(
        c,
        500,
        "dataset_artifact_missing",
        "The immutable dataset artifact is missing.",
      );
    const input = c.req.valid("json");
    const response = await computeFetch(c.env, "/v1/data/surrogate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        content_base64: encodeBase64(await object.arrayBuffer()),
        source_kind: dataset.source_kind,
        input_columns: input.inputColumns,
        output_column: input.outputColumn,
        validation_fraction: input.validationFraction,
        kernel: input.kernel,
        trend: input.trend,
        seed: input.seed,
      }),
    }).catch(() => null);
    if (!response)
      return jsonError(
        c,
        503,
        "compute_unavailable",
        "Data-driven surrogate fitting is unavailable.",
      );
    const body = (await response.json()) as {
      surrogate?: Omit<
        DataSurrogateModel,
        "id" | "projectId" | "datasetId" | "createdAt"
      > & {
        artifact: DataSurrogateModel["artifact"] & { xmlBase64: string };
      };
      error?: { code?: string; message?: string };
    };
    if (!response.ok || !body.surrogate)
      return jsonError(
        c,
        422,
        body.error?.code ?? "data_surrogate_failed",
        body.error?.message ?? "The data-driven surrogate could not be fitted.",
      );
    let xml: Uint8Array<ArrayBuffer>;
    try {
      xml = decodeBase64(body.surrogate.artifact.xmlBase64);
    } catch {
      return jsonError(
        c,
        500,
        "surrogate_artifact_invalid",
        "The fitted surrogate artifact is invalid.",
      );
    }
    if ((await sha256Bytes(xml)) !== body.surrogate.artifact.sha256)
      return jsonError(
        c,
        500,
        "surrogate_artifact_checksum_mismatch",
        "The fitted surrogate checksum did not match.",
      );
    const id = crypto.randomUUID();
    const timestamp = now();
    const objectKey = `data-surrogates/${identity.ownerId}/${dataset.project_id}/${id}.xml`;
    const artifact = {
      sha256: body.surrogate.artifact.sha256,
      sizeBytes: body.surrogate.artifact.sizeBytes,
      resultType: body.surrogate.artifact.resultType,
    };
    await c.env.ARTIFACTS.put(objectKey, xml, {
      httpMetadata: { contentType: "application/xml" },
      customMetadata: { sha256: artifact.sha256, datasetId: dataset.id },
    });
    try {
      await c.env.DB.prepare(
        `INSERT INTO data_surrogate_models
         (id, project_id, dataset_id, owner_id, method, plugin_version,
          openturns_version, input_columns_json, output_column, config_json,
          validation_json, object_key, artifact_json, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      )
        .bind(
          id,
          dataset.project_id,
          dataset.id,
          identity.ownerId,
          body.surrogate.method,
          body.surrogate.pluginVersion,
          body.surrogate.openturnsVersion,
          JSON.stringify(body.surrogate.inputColumns),
          body.surrogate.outputColumn,
          JSON.stringify(body.surrogate.config),
          JSON.stringify(body.surrogate.validation),
          objectKey,
          JSON.stringify(artifact),
          timestamp,
        )
        .run();
    } catch (error) {
      await c.env.ARTIFACTS.delete(objectKey);
      throw error;
    }
    return c.json(
      {
        surrogate: {
          id,
          projectId: dataset.project_id,
          datasetId: dataset.id,
          method: body.surrogate.method,
          pluginVersion: body.surrogate.pluginVersion,
          openturnsVersion: body.surrogate.openturnsVersion,
          inputColumns: body.surrogate.inputColumns,
          outputColumn: body.surrogate.outputColumn,
          config: body.surrogate.config,
          validation: body.surrogate.validation,
          artifact,
          createdAt: timestamp,
        } satisfies DataSurrogateModel,
      },
      201,
    );
  },
);

app.get("/api/v1/projects/:projectId/models", async (c) => {
  const identity = authenticatedIdentity(c);
  const ownership = await c.env.DB.prepare(
    "SELECT id FROM projects WHERE id = ? AND owner_id = ?",
  )
    .bind(c.req.param("projectId"), identity.ownerId)
    .first();
  if (!ownership)
    return jsonError(c, 404, "project_not_found", "Project not found.");
  const rows = await c.env.DB.prepare(
    `SELECT id, project_id, version, source_kind, display_name, source_hash,
            metadata_json, equations_json, assessment_json, parent_version_id,
            derivation_json, created_at
     FROM model_versions WHERE project_id = ? ORDER BY version DESC`,
  )
    .bind(c.req.param("projectId"))
    .all<{
      id: string;
      project_id: string;
      version: number;
      source_kind: "python" | "builder" | "example";
      display_name: string;
      source_hash: string;
      metadata_json: string;
      equations_json: string | null;
      assessment_json: string | null;
      parent_version_id: string | null;
      derivation_json: string | null;
      created_at: string;
    }>();
  return c.json({
    modelVersions: rows.results.map((row) => ({
      id: row.id,
      projectId: row.project_id,
      version: row.version,
      sourceKind: row.source_kind,
      displayName: row.display_name,
      sourceHash: row.source_hash,
      metadata: withDerivedEquations(
        parseJson<ModelMetadata>(row.metadata_json, {} as ModelMetadata),
        row.equations_json,
      ),
      assessment: parseJson<ModelAssessment | null>(row.assessment_json, null),
      parentVersionId: row.parent_version_id,
      derivation: parseJson<Record<string, unknown> | null>(
        row.derivation_json,
        null,
      ),
      createdAt: row.created_at,
    })),
  });
});

app.post(
  "/api/v1/projects/:projectId/models",
  zValidator("json", createModelVersionSchema),
  async (c) => {
    const identity = authenticatedIdentity(c);
    const input = c.req.valid("json");
    const projectId = c.req.param("projectId");
    const project = await c.env.DB.prepare(
      "SELECT id FROM projects WHERE id = ? AND owner_id = ?",
    )
      .bind(projectId, identity.ownerId)
      .first();
    if (!project)
      return jsonError(c, 404, "project_not_found", "Project not found.");

    const validation = await computeFetch(c.env, "/v1/validate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ source: input.source, seed: 42 }),
    }).catch(() => null);
    if (!validation)
      return jsonError(
        c,
        503,
        "compute_unavailable",
        "The model validator is unavailable.",
      );
    const validationBody = (await validation.json()) as {
      metadata?: ModelMetadata;
      assessment?: ModelAssessment;
      error?: { code?: string; message?: string };
    };
    if (!validation.ok || !validationBody.metadata) {
      return jsonError(
        c,
        422,
        validationBody.error?.code ?? "invalid_model",
        validationBody.error?.message ?? "Model validation failed.",
      );
    }
    const referenceEquations =
      input.sourceKind === "example"
        ? EXAMPLE_CATALOG.find(
            (example) =>
              example.sha256 === validationBody.metadata?.source_hash,
          )?.equations
        : undefined;
    const metadata: ModelMetadata = referenceEquations?.length
      ? {
          ...validationBody.metadata,
          equations: referenceEquations.map((equation) => ({
            output_name: equation.outputName,
            latex: equation.latex,
            representation: "closed_form" as const,
          })),
        }
      : validationBody.metadata;
    const assessment = validationBody.assessment;
    if (assessment) {
      const attached = await c.env.DB.prepare(
        "SELECT COUNT(*) AS count FROM datasets WHERE project_id = ? AND owner_id = ?",
      )
        .bind(projectId, identity.ownerId)
        .first<{ count: number }>();
      if (Number(attached?.count ?? 0) > 0) {
        assessment.recommendations = assessment.recommendations.map(
          (recommendation) =>
            recommendation.capability === "distribution_fitting"
              ? {
                  ...recommendation,
                  status: "available",
                  rationale_codes: ["EMPIRICAL_DATA_ATTACHED"],
                  compatibility_warnings: [],
                }
              : recommendation,
        );
      }
    }
    if (input.parentVersionId) {
      const parent = await c.env.DB.prepare(
        "SELECT id FROM model_versions WHERE id = ? AND project_id = ?",
      )
        .bind(input.parentVersionId, projectId)
        .first();
      if (!parent)
        return jsonError(
          c,
          422,
          "invalid_parent_model",
          "The parent model version is not part of this study.",
        );
    }
    const existing = await c.env.DB.prepare(
      `SELECT id, version, source_kind, display_name, source_hash, metadata_json,
              equations_json, assessment_json, parent_version_id, derivation_json, created_at
       FROM model_versions WHERE project_id = ? AND source_hash = ?`,
    )
      .bind(projectId, metadata.source_hash)
      .first<{
        id: string;
        version: number;
        source_kind: "python" | "builder" | "example";
        display_name: string;
        source_hash: string;
        metadata_json: string;
        equations_json: string | null;
        assessment_json: string | null;
        parent_version_id: string | null;
        derivation_json: string | null;
        created_at: string;
      }>();
    if (existing) {
      await c.env.DB.prepare(
        "UPDATE model_versions SET assessment_json = ?, equations_json = ? WHERE id = ?",
      )
        .bind(
          assessment ? JSON.stringify(assessment) : existing.assessment_json,
          JSON.stringify(metadata.equations ?? []),
          existing.id,
        )
        .run();
      return c.json({
        modelVersion: {
          id: existing.id,
          projectId,
          version: existing.version,
          sourceKind: existing.source_kind,
          displayName: existing.display_name,
          sourceHash: existing.source_hash,
          metadata: withDerivedEquations(
            parseJson<ModelMetadata>(existing.metadata_json, metadata),
            JSON.stringify(metadata.equations ?? []),
          ),
          assessment:
            assessment ??
            parseJson<ModelAssessment | null>(existing.assessment_json, null),
          parentVersionId: existing.parent_version_id,
          derivation: parseJson<Record<string, unknown> | null>(
            existing.derivation_json,
            null,
          ),
          createdAt: existing.created_at,
        },
      });
    }
    const versionRow = await c.env.DB.prepare(
      "SELECT COALESCE(MAX(version), 0) + 1 AS next_version FROM model_versions WHERE project_id = ?",
    )
      .bind(projectId)
      .first<{ next_version: number }>();
    const version = Number(versionRow?.next_version ?? 1);
    const displayName = input.displayName ?? `Model v${version}`;
    const id = crypto.randomUUID();
    const sourceKey = `models/${identity.ownerId}/${projectId}/${id}.py`;
    await c.env.ARTIFACTS.put(sourceKey, input.source, {
      httpMetadata: { contentType: "text/x-python; charset=utf-8" },
      customMetadata: { sha256: metadata.source_hash },
    });
    const timestamp = now();
    try {
      await c.env.DB.batch([
        c.env.DB.prepare(
          `INSERT INTO model_versions
           (id, project_id, version, source_kind, display_name, source_key,
            source_hash, metadata_json, equations_json, assessment_json, builder_spec_json,
            parent_version_id, derivation_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        ).bind(
          id,
          projectId,
          version,
          input.sourceKind,
          displayName,
          sourceKey,
          metadata.source_hash,
          JSON.stringify(metadata),
          JSON.stringify(metadata.equations ?? []),
          assessment ? JSON.stringify(assessment) : null,
          input.builderSpec ? JSON.stringify(input.builderSpec) : null,
          input.parentVersionId ?? null,
          input.derivation ? JSON.stringify(input.derivation) : null,
          timestamp,
        ),
        c.env.DB.prepare(
          "UPDATE projects SET updated_at = ? WHERE id = ?",
        ).bind(timestamp, projectId),
      ]);
    } catch (error) {
      await c.env.ARTIFACTS.delete(sourceKey);
      throw error;
    }
    return c.json(
      {
        modelVersion: {
          id,
          projectId,
          version,
          sourceKind: input.sourceKind,
          displayName,
          sourceHash: metadata.source_hash,
          metadata,
          assessment: assessment ?? null,
          parentVersionId: input.parentVersionId ?? null,
          derivation: input.derivation ?? null,
          createdAt: timestamp,
        },
      },
      201,
    );
  },
);

app.get("/api/v1/model-versions/:modelVersionId/definition", async (c) => {
  const identity = authenticatedIdentity(c);
  const definition = await loadModelDefinition(
    c.env,
    c.req.param("modelVersionId"),
    identity.ownerId,
  );
  if (!definition)
    return jsonError(c, 404, "model_not_found", "Model version not found.");
  c.header("Cache-Control", "private, no-store");
  return c.json({ definition });
});

app.get("/api/v1/model-versions/:modelVersionId/source", async (c) => {
  const identity = authenticatedIdentity(c);
  const definition = await loadModelDefinition(
    c.env,
    c.req.param("modelVersionId"),
    identity.ownerId,
  );
  if (!definition)
    return jsonError(c, 404, "model_not_found", "Model version not found.");
  c.header(
    "Content-Disposition",
    `attachment; filename=model-v${definition.modelVersion.version}.py`,
  );
  c.header("Content-Type", "text/x-python; charset=utf-8");
  c.header("Cache-Control", "private, no-store");
  return c.body(definition.source);
});

app.post(
  "/api/v1/model-versions/:modelVersionId/derived-reduction",
  zValidator("json", createReducedModelSchema),
  async (c) => {
    const identity = authenticatedIdentity(c);
    const modelVersionId = c.req.param("modelVersionId");
    const definition = await loadModelDefinition(
      c.env,
      modelVersionId,
      identity.ownerId,
    );
    if (!definition)
      return jsonError(c, 404, "model_not_found", "Model version not found.");
    const input = c.req.valid("json");
    const morris = await c.env.DB.prepare(
      `SELECT t.result_json
       FROM analysis_tasks t JOIN runs r ON r.id = t.run_id
       WHERE r.id = ? AND r.owner_id = ? AND r.model_version_id = ?
         AND t.analysis_key = 'morris' AND t.status = 'succeeded'`,
    )
      .bind(input.morrisRunId, identity.ownerId, modelVersionId)
      .first<{ result_json: string }>();
    if (!morris?.result_json)
      return jsonError(
        c,
        422,
        "morris_evidence_required",
        "A successful Morris run for this exact model version is required.",
      );
    const result = parseJson<{
      plugin_version?: string;
      payload?: { tables?: { effects?: { rows?: unknown[][] } } };
    }>(morris.result_json, {});
    if (
      !new Set(["2.0.0", "2.1.0"]).has(result.plugin_version ?? "") ||
      !result.payload?.tables?.effects
    )
      return jsonError(
        c,
        422,
        "incompatible_morris_evidence",
        "The reduction requires compatible OTMorris plugin evidence.",
      );
    const dimension = definition.modelVersion.metadata.input_dimension;
    const fixed = [...input.fixedVariables].sort(
      (left, right) => left.index - right.index,
    );
    if (new Set(fixed.map((item) => item.index)).size !== fixed.length)
      return jsonError(
        c,
        422,
        "duplicate_fixed_variable",
        "Each fixed variable may be specified only once.",
      );
    if (
      fixed.some((item) => item.index >= dimension) ||
      fixed.length >= dimension
    )
      return jsonError(
        c,
        422,
        "invalid_reduction",
        "Fixed indices must exist and at least one input must remain active.",
      );
    const fixedIndices = fixed.map((item) => item.index);
    const fixedValues = fixed.map((item) => item.value);
    const retainedIndices = Array.from(
      { length: dimension },
      (_, index) => index,
    ).filter((index) => !fixedIndices.includes(index));
    const retainedVariables = retainedIndices.map(
      (index) =>
        definition.modelVersion.metadata.inputs[index]?.name ?? `X${index}`,
    );
    const fixedVariables = fixed.map((item) => ({
      index: item.index,
      name:
        definition.modelVersion.metadata.inputs[item.index]?.name ??
        `X${item.index}`,
      value: item.value,
    }));
    const derivedSource = `${definition.source.trimEnd()}

# UncertaintyCat derived reduction; parent source above remains immutable.
_uncertaintycat_parent_model = model
_uncertaintycat_parent_problem = problem
_uncertaintycat_fixed_indices = ${JSON.stringify(fixedIndices)}
_uncertaintycat_fixed_values = ${JSON.stringify(fixedValues)}
_uncertaintycat_retained_indices = ${JSON.stringify(retainedIndices)}
model = ot.ParametricFunction(
    _uncertaintycat_parent_model,
    _uncertaintycat_fixed_indices,
    _uncertaintycat_fixed_values,
)
problem = ot.JointDistribution([
    _uncertaintycat_parent_problem.getMarginal(index)
    for index in _uncertaintycat_retained_indices
])
problem.setDescription(${JSON.stringify(retainedVariables)})
`;
    return app.fetch(
      forwardedJsonRequest(
        c,
        `/api/v1/projects/${definition.project.id}/models`,
        {
          source: derivedSource,
          sourceKind: "python",
          displayName: input.displayName,
          parentVersionId: modelVersionId,
          derivation: {
            type: "morris_parametric_reduction",
            parentModelVersionId: modelVersionId,
            morrisRunId: input.morrisRunId,
            morrisPluginVersion: result.plugin_version,
            fixedVariables,
            retainedIndices,
            retainedVariables,
            userConfirmed: input.confirmed,
          },
        },
      ),
      c.env,
      c.executionCtx,
    );
  },
);

app.get("/api/v1/projects/:projectId/surrogates", async (c) => {
  const identity = authenticatedIdentity(c);
  const projectId = c.req.param("projectId");
  const project = await c.env.DB.prepare(
    "SELECT id FROM projects WHERE id = ? AND owner_id = ?",
  )
    .bind(projectId, identity.ownerId)
    .first();
  if (!project)
    return jsonError(c, 404, "project_not_found", "Project not found.");
  const rows = await c.env.DB.prepare(
    `SELECT ${surrogateColumns} FROM surrogate_models
     WHERE project_id = ? AND owner_id = ? ORDER BY created_at DESC`,
  )
    .bind(projectId, identity.ownerId)
    .all<SurrogateRow>();
  c.header("Cache-Control", "private, no-store");
  return c.json({ surrogates: rows.results.map(surrogatePayload) });
});

app.post(
  "/api/v1/model-versions/:modelVersionId/surrogates",
  zValidator("json", createSurrogateSchema),
  async (c) => {
    const identity = authenticatedIdentity(c);
    const definition = await loadModelDefinition(
      c.env,
      c.req.param("modelVersionId"),
      identity.ownerId,
    );
    if (!definition)
      return jsonError(c, 404, "model_not_found", "Model version not found.");
    const input = c.req.valid("json");
    if (input.outputTarget >= definition.modelVersion.metadata.output_dimension)
      return jsonError(
        c,
        422,
        "invalid_output_target",
        "The selected surrogate output does not exist.",
      );
    if (!definition.modelVersion.assessment) {
      const assessmentResponse = await computeFetch(c.env, "/v1/validate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source: definition.source, seed: input.seed }),
      }).catch(() => null);
      if (!assessmentResponse)
        return jsonError(
          c,
          503,
          "compute_unavailable",
          "The model assessment required for surrogate execution is unavailable.",
        );
      const assessed = (await assessmentResponse.json()) as {
        assessment?: ModelAssessment;
        error?: { code?: string; message?: string };
      };
      if (!assessmentResponse.ok || !assessed.assessment)
        return jsonError(
          c,
          422,
          assessed.error?.code ?? "model_assessment_failed",
          assessed.error?.message ?? "The source model could not be assessed.",
        );
      definition.modelVersion.assessment = assessed.assessment;
      await c.env.DB.prepare(
        "UPDATE model_versions SET assessment_json = ? WHERE id = ?",
      )
        .bind(JSON.stringify(assessed.assessment), definition.modelVersion.id)
        .run();
    }
    const response = await computeFetch(c.env, "/v1/execute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source: definition.source,
        seed: input.seed,
        analysis: {
          analysis_key: input.method,
          config: input.config,
          output_targets: [input.outputTarget],
        },
      }),
    }).catch(() => null);
    if (!response)
      return jsonError(
        c,
        503,
        "compute_unavailable",
        "Surrogate validation is unavailable.",
      );
    const body = (await response.json()) as {
      result?: SurrogateModel["validation"]["result"];
      error?: { code?: string; message?: string };
    };
    if (!response.ok || !body.result)
      return jsonError(
        c,
        422,
        body.error?.code ?? "surrogate_validation_failed",
        body.error?.message ?? "Surrogate validation failed.",
      );
    const metrics = body.result.payload.metrics;
    const score = Number(
      input.method === "pce" ? metrics.validation_q2 : metrics.validation_r2,
    );
    const normalizedRmse = Number(metrics.validation_normalized_rmse);
    if (!Number.isFinite(score) || !Number.isFinite(normalizedRmse))
      return jsonError(
        c,
        422,
        "surrogate_validation_incomplete",
        "The surrogate did not produce finite independent validation metrics.",
      );
    const guidance = {
      score,
      normalizedRmse,
      scoreThreshold: 0.95,
      normalizedRmseThreshold: 0.1,
      meetsDefault: score >= 0.95 && normalizedRmse <= 0.1,
    };
    const validation: StoredSurrogateValidation = {
      config: input.config,
      outputTargets: [input.outputTarget],
      seed: input.seed,
      result: body.result,
      guidance,
      artifact: null,
    };
    const id = crypto.randomUUID();
    const timestamp = now();
    await c.env.DB.prepare(
      `INSERT INTO surrogate_models
       (id, project_id, owner_id, source_model_version_id, source_model_hash,
        method, plugin_version, openturns_version, status, validation_json,
        created_at)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'validated', ?, ?)`,
    )
      .bind(
        id,
        definition.project.id,
        identity.ownerId,
        definition.modelVersion.id,
        definition.modelVersion.sourceHash,
        input.method,
        body.result.plugin_version,
        body.result.openturns_version,
        JSON.stringify(validation),
        timestamp,
      )
      .run();
    return c.json(
      {
        surrogate: surrogatePayload({
          id,
          project_id: definition.project.id,
          source_model_version_id: definition.modelVersion.id,
          source_model_hash: definition.modelVersion.sourceHash,
          method: input.method,
          plugin_version: body.result.plugin_version,
          openturns_version: body.result.openturns_version,
          status: "validated",
          validation_json: JSON.stringify(validation),
          acknowledgement_json: null,
          object_key: null,
          created_at: timestamp,
          promoted_at: null,
        }),
      },
      201,
    );
  },
);

app.post(
  "/api/v1/surrogates/:surrogateId/promote",
  zValidator("json", promoteSurrogateSchema),
  async (c) => {
    const identity = authenticatedIdentity(c);
    const row = await c.env.DB.prepare(
      `SELECT ${surrogateColumns} FROM surrogate_models
       WHERE id = ? AND owner_id = ?`,
    )
      .bind(c.req.param("surrogateId"), identity.ownerId)
      .first<SurrogateRow>();
    if (!row)
      return jsonError(c, 404, "surrogate_not_found", "Surrogate not found.");
    if (row.status === "promoted")
      return c.json({ surrogate: surrogatePayload(row) });
    const validation = parseJson<StoredSurrogateValidation>(
      row.validation_json,
      {} as StoredSurrogateValidation,
    );
    const acknowledgement = c.req.valid("json");
    if (
      !validation.guidance.meetsDefault &&
      (!acknowledgement.acknowledgeOverride ||
        acknowledgement.reason.length < 10)
    )
      return jsonError(
        c,
        422,
        "promotion_acknowledgement_required",
        "Validation is below the default threshold; record an explicit reason of at least 10 characters.",
      );
    const definition = await loadModelDefinition(
      c.env,
      row.source_model_version_id,
      identity.ownerId,
    );
    if (
      !definition ||
      definition.modelVersion.sourceHash !== row.source_model_hash
    )
      return jsonError(
        c,
        409,
        "source_model_mismatch",
        "The exact source model is unavailable or does not match validation provenance.",
      );
    const response = await computeFetch(c.env, "/v1/surrogates/serialize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source: definition.source,
        method: row.method,
        config: validation.config,
        output_targets: validation.outputTargets,
        seed: validation.seed,
      }),
    }).catch(() => null);
    if (!response)
      return jsonError(
        c,
        503,
        "compute_unavailable",
        "Surrogate serialization is unavailable.",
      );
    const body = (await response.json()) as {
      surrogate?: {
        xmlBase64: string;
        sha256: string;
        sizeBytes: number;
        resultType: string;
        pluginVersion: string;
        openturnsVersion: string;
        sourceModelHash: string;
      };
      error?: { code?: string; message?: string };
    };
    if (!response.ok || !body.surrogate)
      return jsonError(
        c,
        422,
        body.error?.code ?? "surrogate_serialization_failed",
        body.error?.message ?? "Surrogate serialization failed.",
      );
    if (
      body.surrogate.sourceModelHash !== row.source_model_hash ||
      body.surrogate.pluginVersion !== row.plugin_version ||
      body.surrogate.openturnsVersion !== row.openturns_version
    )
      return jsonError(
        c,
        409,
        "surrogate_provenance_mismatch",
        "Rebuilt surrogate provenance does not match its validated record.",
      );
    let xml: Uint8Array<ArrayBuffer>;
    try {
      xml = decodeBase64(body.surrogate.xmlBase64);
    } catch {
      return jsonError(
        c,
        500,
        "surrogate_artifact_invalid",
        "The serialized surrogate artifact is invalid.",
      );
    }
    const computedSha = await sha256Bytes(xml);
    if (computedSha !== body.surrogate.sha256)
      return jsonError(
        c,
        500,
        "surrogate_artifact_checksum_mismatch",
        "The serialized surrogate checksum did not match.",
      );
    const objectKey = `surrogates/${identity.ownerId}/${row.project_id}/${row.id}.xml`;
    await c.env.ARTIFACTS.put(objectKey, xml, {
      httpMetadata: { contentType: "application/xml" },
      customMetadata: {
        sha256: computedSha,
        sourceModelHash: row.source_model_hash,
        pluginVersion: row.plugin_version,
        openturnsVersion: row.openturns_version,
      },
    });
    const promotedAt = now();
    const artifact = {
      sha256: computedSha,
      sizeBytes: xml.byteLength,
      resultType: body.surrogate.resultType,
    };
    validation.artifact = artifact;
    try {
      await c.env.DB.prepare(
        `UPDATE surrogate_models SET status = 'promoted', validation_json = ?,
                acknowledgement_json = ?, object_key = ?, promoted_at = ?
         WHERE id = ? AND owner_id = ?`,
      )
        .bind(
          JSON.stringify(validation),
          JSON.stringify(acknowledgement),
          objectKey,
          promotedAt,
          row.id,
          identity.ownerId,
        )
        .run();
    } catch (error) {
      await c.env.ARTIFACTS.delete(objectKey);
      throw error;
    }
    return c.json({
      surrogate: surrogatePayload({
        ...row,
        status: "promoted",
        validation_json: JSON.stringify(validation),
        acknowledgement_json: JSON.stringify(acknowledgement),
        object_key: objectKey,
        promoted_at: promotedAt,
      }),
    });
  },
);

app.post(
  "/api/v1/surrogates/:surrogateId/copy",
  zValidator("json", copySurrogateSchema),
  async (c) => {
    const identity = authenticatedIdentity(c);
    const input = c.req.valid("json");
    const source = await c.env.DB.prepare(
      `SELECT ${surrogateColumns} FROM surrogate_models
       WHERE id = ? AND owner_id = ? AND status = 'promoted'`,
    )
      .bind(c.req.param("surrogateId"), identity.ownerId)
      .first<SurrogateRow>();
    if (!source?.object_key)
      return jsonError(
        c,
        404,
        "surrogate_not_found",
        "Promoted surrogate not found.",
      );
    const target = await c.env.DB.prepare(
      `SELECT m.id, m.project_id, m.source_hash
       FROM model_versions m JOIN projects p ON p.id = m.project_id
       WHERE m.id = ? AND m.project_id = ? AND p.owner_id = ?`,
    )
      .bind(input.targetModelVersionId, input.targetProjectId, identity.ownerId)
      .first<{ id: string; project_id: string; source_hash: string }>();
    if (!target)
      return jsonError(
        c,
        404,
        "target_model_not_found",
        "Target model version not found in the selected project.",
      );
    if (target.source_hash !== source.source_model_hash)
      return jsonError(
        c,
        409,
        "surrogate_source_mismatch",
        "The target model does not match the surrogate's validated source hash.",
      );
    const artifact = await c.env.ARTIFACTS.get(source.object_key);
    if (!artifact)
      return jsonError(
        c,
        500,
        "surrogate_artifact_missing",
        "The promoted surrogate artifact is missing.",
      );

    const id = crypto.randomUUID();
    const objectKey = `surrogates/${identity.ownerId}/${target.project_id}/${id}.xml`;
    const timestamp = now();
    await c.env.ARTIFACTS.put(objectKey, await artifact.arrayBuffer(), {
      httpMetadata: { contentType: "application/xml" },
      customMetadata: {
        ...artifact.customMetadata,
        copiedFromSurrogateId: source.id,
      },
    });
    try {
      await c.env.DB.batch([
        c.env.DB.prepare(
          `INSERT INTO surrogate_models
           (id, project_id, owner_id, source_model_version_id,
            source_model_hash, method, plugin_version, openturns_version,
            status, validation_json, acknowledgement_json, object_key,
            created_at, promoted_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'promoted', ?, ?, ?, ?, ?)`,
        ).bind(
          id,
          target.project_id,
          identity.ownerId,
          target.id,
          target.source_hash,
          source.method,
          source.plugin_version,
          source.openturns_version,
          source.validation_json,
          source.acknowledgement_json,
          objectKey,
          timestamp,
          timestamp,
        ),
        c.env.DB.prepare(
          "UPDATE projects SET updated_at = ? WHERE id = ? AND owner_id = ?",
        ).bind(timestamp, target.project_id, identity.ownerId),
      ]);
    } catch (error) {
      await c.env.ARTIFACTS.delete(objectKey);
      throw error;
    }
    return c.json(
      {
        surrogate: surrogatePayload({
          ...source,
          id,
          project_id: target.project_id,
          source_model_version_id: target.id,
          source_model_hash: target.source_hash,
          object_key: objectKey,
          created_at: timestamp,
          promoted_at: timestamp,
        }),
      },
      201,
    );
  },
);

app.get("/api/v1/surrogates/:surrogateId/artifact", async (c) => {
  const identity = authenticatedIdentity(c);
  const row = await c.env.DB.prepare(
    "SELECT object_key, method FROM surrogate_models WHERE id = ? AND owner_id = ? AND status = 'promoted'",
  )
    .bind(c.req.param("surrogateId"), identity.ownerId)
    .first<{ object_key: string; method: string }>();
  if (!row?.object_key)
    return jsonError(
      c,
      404,
      "surrogate_not_found",
      "Promoted surrogate not found.",
    );
  const object = await c.env.ARTIFACTS.get(row.object_key);
  if (!object)
    return jsonError(
      c,
      500,
      "surrogate_artifact_missing",
      "Surrogate artifact is missing.",
    );
  c.header("Content-Type", "application/xml");
  c.header(
    "Content-Disposition",
    `attachment; filename=${row.method}-surrogate.xml`,
  );
  c.header("Cache-Control", "private, no-store");
  return c.body(object.body);
});

interface UnderstandingRow {
  id: string;
  model_version_id: string;
  model_hash: string;
  prompt_version: string;
  ai_model_id: string;
  status: "pending" | "generating" | "succeeded" | "failed";
  content: string | null;
  error: string | null;
  created_at: string;
  updated_at: string;
}

function understandingPayload(row: UnderstandingRow) {
  return {
    id: row.id,
    modelVersionId: row.model_version_id,
    modelHash: row.model_hash,
    promptVersion: row.prompt_version,
    aiModelId: row.ai_model_id,
    status: row.status,
    content: row.content,
    error: row.error,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  };
}

app.get("/api/v1/model-versions/:modelVersionId/understanding", async (c) => {
  const identity = authenticatedIdentity(c);
  const promptVersion = modelUnderstandingCacheVersion(
    c.env,
    MODEL_UNDERSTANDING_PROMPT_VERSION,
  );
  const definition = await loadModelDefinition(
    c.env,
    c.req.param("modelVersionId"),
    identity.ownerId,
  );
  if (!definition)
    return jsonError(c, 404, "model_not_found", "Model version not found.");
  const row = await c.env.DB.prepare(
    `SELECT id, model_version_id, model_hash, prompt_version, ai_model_id,
            status, content, error, created_at, updated_at
     FROM model_understandings WHERE model_hash = ? AND prompt_version = ?`,
  )
    .bind(definition.modelVersion.sourceHash, promptVersion)
    .first<UnderstandingRow>();
  c.header("Cache-Control", "private, no-store");
  return c.json({ understanding: row ? understandingPayload(row) : null });
});

const understandingSchema = z.object({
  regenerate: z.boolean().default(false),
});
app.post(
  "/api/v1/model-versions/:modelVersionId/understanding",
  zValidator("json", understandingSchema),
  async (c) => {
    const identity = authenticatedIdentity(c);
    const runtime = aiRuntime(c.env);
    const promptVersion = modelUnderstandingCacheVersion(
      c.env,
      MODEL_UNDERSTANDING_PROMPT_VERSION,
    );
    if (!runtime.configured)
      return jsonError(
        c,
        503,
        "ai_unavailable",
        `${runtime.provider === "groq" ? "Groq" : "Cloudflare Workers AI"} is unavailable in this environment.`,
      );
    const definition = await loadModelDefinition(
      c.env,
      c.req.param("modelVersionId"),
      identity.ownerId,
    );
    if (!definition)
      return jsonError(c, 404, "model_not_found", "Model version not found.");
    const input = c.req.valid("json");
    const cached = await c.env.DB.prepare(
      `SELECT id, model_version_id, model_hash, prompt_version, ai_model_id,
              status, content, error, created_at, updated_at
       FROM model_understandings WHERE model_hash = ? AND prompt_version = ?`,
    )
      .bind(definition.modelVersion.sourceHash, promptVersion)
      .first<UnderstandingRow>();
    if (!input.regenerate && cached?.status === "succeeded" && cached.content) {
      c.header("Content-Type", "text/markdown; charset=utf-8");
      c.header("X-UncertaintyCat-Cache", "hit");
      return c.body(cached.content);
    }
    if (generationLeaseIsActive(cached?.status, cached?.updated_at)) {
      c.header("Cache-Control", "private, no-store");
      c.header("Retry-After", "1");
      return c.json(
        { understanding: cached ? understandingPayload(cached) : null },
        202,
      );
    }
    if (input.regenerate) {
      const midnight = new Date();
      midnight.setUTCHours(0, 0, 0, 0);
      const usage = await c.env.DB.prepare(
        `SELECT COUNT(*) AS count FROM usage_ledger
         WHERE owner_id = ? AND kind = 'model_understanding_regeneration' AND created_at >= ?`,
      )
        .bind(identity.ownerId, midnight.toISOString())
        .first<{ count: number }>();
      if (Number(usage?.count ?? 0) >= 3)
        return jsonError(
          c,
          429,
          "model_understanding_quota_exceeded",
          "Daily Model Understanding regeneration quota exceeded.",
        );
    }
    const timestamp = now();
    const understandingId = cached?.id ?? crypto.randomUUID();
    const staleBefore = new Date(
      Date.now() - MODEL_UNDERSTANDING_LEASE_MS,
    ).toISOString();
    const claim = await c.env.DB.prepare(
      `INSERT INTO model_understandings
       (id, model_version_id, model_hash, prompt_version, ai_model_id, status,
        content, error, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, 'generating', NULL, NULL, ?, ?)
       ON CONFLICT(model_hash, prompt_version) DO UPDATE SET
         model_version_id = excluded.model_version_id,
         ai_model_id = excluded.ai_model_id,
         status = 'generating', content = NULL, error = NULL,
         updated_at = excluded.updated_at
       WHERE model_understandings.status != 'generating'
          OR model_understandings.updated_at < ?`,
    )
      .bind(
        understandingId,
        definition.modelVersion.id,
        definition.modelVersion.sourceHash,
        promptVersion,
        runtime.models.modelUnderstanding.modelId,
        timestamp,
        timestamp,
        staleBefore,
      )
      .run();
    if (Number(claim.meta.changes ?? 0) === 0) {
      const active = await c.env.DB.prepare(
        `SELECT id, model_version_id, model_hash, prompt_version, ai_model_id,
                status, content, error, created_at, updated_at
         FROM model_understandings WHERE model_hash = ? AND prompt_version = ?`,
      )
        .bind(definition.modelVersion.sourceHash, promptVersion)
        .first<UnderstandingRow>();
      c.header("Cache-Control", "private, no-store");
      c.header("Retry-After", "1");
      return c.json(
        { understanding: active ? understandingPayload(active) : null },
        202,
      );
    }

    const generationStartedAt = Date.now();
    console.log(
      JSON.stringify({
        event: "model_understanding_generation_started",
        requestId: c.get("requestId"),
        understandingId,
        aiProvider: runtime.provider,
        aiModelId: runtime.models.modelUnderstanding.modelId,
        fallbackAiModelId: runtime.models.modelUnderstanding.fallbackModelId,
        regenerate: input.regenerate,
      }),
    );
    try {
      const attempts = [
        {
          modelId: runtime.models.modelUnderstanding.modelId,
          timeoutMs: MODEL_UNDERSTANDING_PRIMARY_TIMEOUT_MS,
        },
        {
          modelId: runtime.models.modelUnderstanding.fallbackModelId,
          timeoutMs: MODEL_UNDERSTANDING_FALLBACK_TIMEOUT_MS,
        },
      ] as const;
      const generation = await runSequentialFallback(
        attempts,
        async (attempt, index) => {
          const attemptStartedAt = Date.now();
          if (index > 0) {
            console.warn(
              JSON.stringify({
                event: "model_understanding_fallback_started",
                requestId: c.get("requestId"),
                understandingId,
                aiModelId: attempt.modelId,
              }),
            );
          }
          try {
            const providerOptions = aiProviderOptions(
              runtime.provider,
              "modelUnderstanding",
            );
            const model = createAiLanguageModel(
              c.env,
              attempt.modelId,
              `understanding:${definition.modelVersion.sourceHash.slice(0, 48)}`,
              "modelUnderstanding",
            );
            const narrative =
              runtime.provider === "groq"
                ? renderStructuredModelUnderstanding(
                    (
                      await generateObject({
                        model,
                        ...(providerOptions ? { providerOptions } : {}),
                        schema: modelUnderstandingSectionsSchema,
                        schemaName: "model_understanding",
                        schemaDescription:
                          "A bounded engineering model explanation with LaTeX equations and validated-fact narrative sections.",
                        maxOutputTokens: 1_400,
                        maxRetries: 1,
                        abortSignal: AbortSignal.timeout(attempt.timeoutMs),
                        temperature: 0.1,
                        system: MODEL_UNDERSTANDING_STRUCTURED_SYSTEM_PROMPT,
                        prompt: modelUnderstandingPrompt(definition),
                      })
                    ).object,
                  )
                : (
                    await generateText({
                      model,
                      ...(providerOptions ? { providerOptions } : {}),
                      maxOutputTokens: 1_400,
                      maxRetries: 1,
                      timeout: attempt.timeoutMs,
                      temperature: 0.1,
                      system: MODEL_UNDERSTANDING_SYSTEM_PROMPT,
                      prompt: modelUnderstandingPrompt(definition),
                    })
                  ).text.trim();
            if (!narrative)
              throw new Error("The AI provider returned an empty explanation.");
            const validationIssues =
              modelUnderstandingValidationIssues(narrative);
            if (validationIssues.length > 0) {
              console.warn(
                JSON.stringify({
                  event: "model_understanding_candidate_invalid",
                  requestId: c.get("requestId"),
                  understandingId,
                  aiModelId: attempt.modelId,
                  validationIssues,
                }),
              );
            }
            return {
              content: narrative,
              modelId: attempt.modelId,
              attemptDurationMs: Date.now() - attemptStartedAt,
            };
          } catch (error) {
            const failure = generationFailure(error);
            console.warn(
              JSON.stringify({
                event: "model_understanding_attempt_failed",
                requestId: c.get("requestId"),
                understandingId,
                aiModelId: attempt.modelId,
                durationMs: Date.now() - attemptStartedAt,
                diagnostic: failure.diagnostic,
                providerStatusCode: failure.providerStatusCode,
              }),
            );
            throw error;
          }
        },
      );
      const reviewStartedAt = Date.now();
      const generatedValidationIssues = modelUnderstandingValidationIssues(
        generation.result.content,
      );
      const allReviewerAttempts = [
        runtime.models.modelUnderstanding.reviewerModelId,
        generation.result.modelId,
        runtime.models.modelUnderstanding.modelId,
      ].filter((modelId, index, values) => values.indexOf(modelId) === index);
      const reviewerAttempts =
        generatedValidationIssues.length === 0
          ? allReviewerAttempts.slice(0, 1)
          : allReviewerAttempts;
      let reviewedContent: string | undefined;
      let reviewerModelId: string | undefined;
      let reviewerFallbackUsed = false;
      try {
        const review = await runSequentialFallback(
          reviewerAttempts,
          async (candidateReviewerModelId, index) => {
            if (index > 0) {
              console.warn(
                JSON.stringify({
                  event: "model_understanding_review_fallback_started",
                  requestId: c.get("requestId"),
                  understandingId,
                  aiModelId: candidateReviewerModelId,
                }),
              );
            }
            const providerOptions = aiProviderOptions(
              runtime.provider,
              "modelUnderstanding",
            );
            try {
              const model = createAiLanguageModel(
                c.env,
                candidateReviewerModelId,
                `understanding-review:${definition.modelVersion.sourceHash.slice(0, 48)}`,
                "modelUnderstanding",
              );
              const reviewed =
                runtime.provider === "groq"
                  ? renderStructuredModelUnderstanding(
                      (
                        await generateObject({
                          model,
                          ...(providerOptions ? { providerOptions } : {}),
                          schema: modelUnderstandingSectionsSchema,
                          schemaName: "reviewed_model_understanding",
                          schemaDescription:
                            "An independently reviewed engineering model explanation with corrected LaTeX equations.",
                          maxOutputTokens: 1_600,
                          maxRetries: 1,
                          abortSignal: AbortSignal.timeout(
                            MODEL_UNDERSTANDING_REVIEW_TIMEOUT_MS,
                          ),
                          temperature: 0,
                          system:
                            MODEL_UNDERSTANDING_STRUCTURED_REVIEW_SYSTEM_PROMPT,
                          prompt: modelUnderstandingReviewPrompt(
                            definition,
                            generation.result.content,
                          ),
                        })
                      ).object,
                    )
                  : (
                      await generateText({
                        model,
                        ...(providerOptions ? { providerOptions } : {}),
                        maxOutputTokens: 1_600,
                        maxRetries: 1,
                        timeout: MODEL_UNDERSTANDING_REVIEW_TIMEOUT_MS,
                        temperature: 0,
                        system: MODEL_UNDERSTANDING_REVIEW_SYSTEM_PROMPT,
                        prompt: modelUnderstandingReviewPrompt(
                          definition,
                          generation.result.content,
                        ),
                      })
                    ).text.trim();
              const validationIssues =
                modelUnderstandingValidationIssues(reviewed);
              if (validationIssues.length > 0)
                throw new Error(
                  `The equation reviewer returned an invalid brief: ${validationIssues.join(",")}`,
                );
              return {
                content: reviewed,
                modelId: candidateReviewerModelId,
              };
            } catch (error) {
              const failure = generationFailure(error);
              console.warn(
                JSON.stringify({
                  event: "model_understanding_review_attempt_failed",
                  requestId: c.get("requestId"),
                  understandingId,
                  aiModelId: candidateReviewerModelId,
                  diagnostic: failure.diagnostic,
                  providerStatusCode: failure.providerStatusCode,
                }),
              );
              throw error;
            }
          },
        );
        reviewedContent = review.result.content;
        reviewerModelId = review.result.modelId;
        reviewerFallbackUsed = review.index > 0;
      } catch (error) {
        if (!validModelUnderstanding(generation.result.content)) throw error;
        const failure = generationFailure(error);
        console.warn(
          JSON.stringify({
            event: "model_understanding_review_unavailable_using_valid_candidate",
            requestId: c.get("requestId"),
            understandingId,
            diagnostic: failure.diagnostic,
            providerStatusCode: failure.providerStatusCode,
          }),
        );
      }
      const selected = selectValidatedModelUnderstanding(
        generation.result.content,
        reviewedContent,
      );
      if (!selected)
        throw new Error(
          "The AI provider returned an invalid Model Understanding structure.",
        );
      const content = selected.content;
      const modelId =
        selected.source === "reviewed"
          ? (reviewerModelId ?? generation.result.modelId)
          : generation.result.modelId;
      const attemptDurationMs = generation.result.attemptDurationMs;
      const statements = [
        c.env.DB.prepare(
          `UPDATE model_understandings SET status = 'succeeded', content = ?,
                  ai_model_id = ?, error = NULL, updated_at = ? WHERE id = ?`,
        ).bind(content, modelId, now(), understandingId),
      ];
      if (input.regenerate) {
        statements.push(
          c.env.DB.prepare(
            `INSERT INTO usage_ledger
             (id, owner_id, kind, units, reference_id, created_at)
             VALUES (?, ?, 'model_understanding_regeneration', 1, ?, ?)`,
          ).bind(
            crypto.randomUUID(),
            identity.ownerId,
            definition.modelVersion.id,
            now(),
          ),
        );
      }
      await c.env.DB.batch(statements);
      const durationMs = Date.now() - generationStartedAt;
      console.log(
        JSON.stringify({
          event: "model_understanding_generation_succeeded",
          requestId: c.get("requestId"),
          understandingId,
          aiModelId: modelId,
          fallbackUsed: generation.index > 0,
          reviewerModelId,
          reviewerAccepted: selected.source === "reviewed",
          reviewerFallbackUsed,
          reviewDurationMs: Date.now() - reviewStartedAt,
          attemptDurationMs,
          durationMs,
          outputCharacters: content.length,
        }),
      );
      c.header("Content-Type", "text/markdown; charset=utf-8");
      c.header("Cache-Control", "private, no-store");
      c.header("X-UncertaintyCat-Cache", "miss");
      c.header("X-UncertaintyCat-AI-Duration-Ms", String(durationMs));
      return c.body(content);
    } catch (error) {
      const failure = generationFailure(error);
      await c.env.DB.prepare(
        `UPDATE model_understandings SET status = 'failed', error = ?,
                updated_at = ? WHERE id = ?`,
      )
        .bind(failure.diagnostic, now(), understandingId)
        .run();
      console.error(
        JSON.stringify({
          event: "model_understanding_generation_failed",
          requestId: c.get("requestId"),
          understandingId,
          aiProvider: runtime.provider,
          aiModelId: runtime.models.modelUnderstanding.modelId,
          fallbackAiModelId: runtime.models.modelUnderstanding.fallbackModelId,
          durationMs: Date.now() - generationStartedAt,
          code: failure.code,
          diagnostic: failure.diagnostic,
          providerStatusCode: failure.providerStatusCode,
        }),
      );
      return jsonError(c, failure.status, failure.code, failure.message);
    }
  },
);

app.post("/api/v1/runs", zValidator("json", createRunSchema), async (c) => {
  const identity = authenticatedIdentity(c);
  const input = c.req.valid("json");
  const model = await c.env.DB.prepare(
    `SELECT m.id, m.project_id, m.assessment_json FROM model_versions m
     JOIN projects p ON p.id = m.project_id WHERE m.id = ? AND p.owner_id = ?`,
  )
    .bind(input.modelVersionId, identity.ownerId)
    .first<{
      id: string;
      project_id: string;
      assessment_json: string | null;
    }>();
  if (!model)
    return jsonError(c, 404, "model_not_found", "Model version not found.");
  const assessment = parseJson<ModelAssessment | null>(
    model.assessment_json,
    null,
  );
  for (const analysis of input.analyses) {
    const recommendation = assessment?.recommendations.find(
      (candidate) => candidate.capability === analysis.analysisKey,
    );
    if (recommendation?.status === "incompatible") {
      return jsonError(
        c,
        422,
        "analysis_incompatible",
        recommendation.compatibility_warnings[0] ??
          `${analysis.analysisKey} is incompatible with this validated model.`,
      );
    }
    if (analysis.analysisKey === "reliability" && analysis.config.method === "SUBSET_SAMPLING") {
      if (analysis.outputTargets.length > 1)
        return jsonError(c, 422, "invalid_subset_config", "Subset sampling requires one scalar output target.");
      const parsed = boundedSubsetConfigSchema.safeParse(analysis.config);
      if (!parsed.success)
        return jsonError(c, 422, "invalid_subset_config", "Subset sampling requires 100–5,000 samples per level (multiples of 10), block size 1, and a sufficient total budget of at most 50,000 evaluations.");
      if (parsed.data.output_targets && analysis.outputTargets.length &&
          JSON.stringify(parsed.data.output_targets) !== JSON.stringify(analysis.outputTargets))
        return jsonError(c, 422, "invalid_subset_config", "Conflicting subset output targets are not allowed.");
      const target = analysis.outputTargets[0] ?? parsed.data.output_targets?.[0] ?? 0;
      const reason = subsetSamplingIncompatibility(assessment, target);
      if (reason) return jsonError(c, 422, "analysis_incompatible", reason);
    }
  }
  if (input.surrogateModelId) {
    const surrogate = await c.env.DB.prepare(
      `SELECT id, validation_json FROM surrogate_models
       WHERE id = ? AND owner_id = ? AND project_id = ?
         AND source_model_version_id = ? AND status = 'promoted'`,
    )
      .bind(
        input.surrogateModelId,
        identity.ownerId,
        model.project_id,
        model.id,
      )
      .first<{ id: string; validation_json: string }>();
    if (!surrogate)
      return jsonError(
        c,
        422,
        "promoted_surrogate_not_found",
        "The selected promoted surrogate does not belong to this exact model version.",
      );
    const validation = parseJson<StoredSurrogateValidation | null>(
      surrogate.validation_json,
      null,
    );
    const surrogateOutputTarget = validation?.outputTargets[0];
    if (surrogateOutputTarget === undefined)
      return jsonError(
        c,
        409,
        "surrogate_provenance_incomplete",
        "The promoted surrogate does not retain its source output target.",
      );
    if (
      input.analyses.some((analysis) =>
        analysis.outputTargets.some(
          (target) => target !== surrogateOutputTarget,
        ),
      )
    )
      return jsonError(
        c,
        422,
        "surrogate_output_mismatch",
        "Every explicit analysis output must match the output used to build the promoted surrogate.",
      );
  }

  if (input.idempotencyKey) {
    const existing = await c.env.DB.prepare(
      "SELECT resource_id FROM idempotency_keys WHERE owner_id = ? AND idempotency_key = ? AND resource_type = 'run'",
    )
      .bind(identity.ownerId, input.idempotencyKey)
      .first<{ resource_id: string }>();
    if (existing) {
      const run = await loadOwnedRun(
        c.env,
        existing.resource_id,
        identity.ownerId,
      );
      if (run) return c.json({ run });
    }
  }
  const midnight = new Date();
  midnight.setUTCHours(0, 0, 0, 0);
  const usage = await c.env.DB.prepare(
    "SELECT COALESCE(SUM(units), 0) AS units FROM usage_ledger WHERE owner_id = ? AND kind = 'compute_job' AND created_at >= ?",
  )
    .bind(identity.ownerId, midnight.toISOString())
    .first<{ units: number }>();
  const dailyLimit = 20;
  if (Number(usage?.units ?? 0) + input.analyses.length > dailyLimit) {
    return jsonError(
      c,
      429,
      "compute_quota_exceeded",
      `Daily compute quota of ${dailyLimit} tasks exceeded.`,
    );
  }
  const runId = crypto.randomUUID();
  const timestamp = now();
  const taskRows = input.analyses.map((analysis) => ({
    id: crypto.randomUUID(),
    analysis,
  }));
  const statements = [
    c.env.DB.prepare(
      `INSERT INTO runs
       (id, owner_id, project_id, model_version_id, surrogate_model_id,
        status, seed, accuracy_profile, created_at)
       VALUES (?, ?, ?, ?, ?, 'queued', ?, ?, ?)`,
    ).bind(
      runId,
      identity.ownerId,
      model.project_id,
      model.id,
      input.surrogateModelId ?? null,
      input.seed,
      input.accuracyProfile,
      timestamp,
    ),
    ...taskRows.map(({ id, analysis }) =>
      c.env.DB.prepare(
        `INSERT INTO analysis_tasks
         (id, run_id, analysis_key, plugin_version, status, config_json,
          output_targets_json, progress_json, created_at)
         VALUES (?, ?, ?, ?, 'queued', ?, ?, ?, ?)`,
      ).bind(
        id,
        runId,
        analysis.analysisKey,
        analysis.pluginVersion ?? null,
        JSON.stringify(analysis.config),
        JSON.stringify(analysis.outputTargets),
        JSON.stringify({
          phase: "queued",
          percent: 0,
          message: "Waiting for compute capacity.",
          indeterminate: true,
          attempt: 0,
          updatedAt: timestamp,
        }),
        timestamp,
      ),
    ),
    c.env.DB.prepare(
      "INSERT INTO usage_ledger (id, owner_id, kind, units, reference_id, created_at) VALUES (?, ?, 'compute_job', ?, ?, ?)",
    ).bind(
      crypto.randomUUID(),
      identity.ownerId,
      input.analyses.length,
      runId,
      timestamp,
    ),
  ];
  if (input.idempotencyKey) {
    statements.push(
      c.env.DB.prepare(
        "INSERT INTO idempotency_keys (owner_id, idempotency_key, resource_type, resource_id, created_at) VALUES (?, ?, 'run', ?, ?)",
      ).bind(identity.ownerId, input.idempotencyKey, runId, timestamp),
    );
  }
  await c.env.DB.batch(statements);
  await Promise.all(
    taskRows.map(({ id }) =>
      c.env.RUN_QUEUE.send({ taskId: id, runId, attempt: 0 }),
    ),
  );
  const run = await loadOwnedRun(c.env, runId, identity.ownerId);
  return c.json({ run }, 202);
});

app.get("/api/v1/runs", async (c) => {
  const identity = authenticatedIdentity(c);
  const rows = await c.env.DB.prepare(
    "SELECT id FROM runs WHERE owner_id = ? ORDER BY created_at DESC LIMIT 50",
  )
    .bind(identity.ownerId)
    .all<{ id: string }>();
  const runs = await Promise.all(
    rows.results.map((row) => loadOwnedRun(c.env, row.id, identity.ownerId)),
  );
  return c.json({ runs: runs.filter((run) => run !== null) });
});

app.get("/api/v1/runs/:runId", async (c) => {
  const identity = authenticatedIdentity(c);
  const run = await loadOwnedRun(c.env, c.req.param("runId"), identity.ownerId);
  if (!run) return jsonError(c, 404, "run_not_found", "Run not found.");
  return c.json({ run });
});

app.post("/api/v1/runs/:runId/rerun", async (c) => {
  const identity = authenticatedIdentity(c);
  const sourceRun = await loadOwnedRun(
    c.env,
    c.req.param("runId"),
    identity.ownerId,
  );
  if (!sourceRun) return jsonError(c, 404, "run_not_found", "Run not found.");
  const request = forwardedJsonRequest(c, "/api/v1/runs", {
    modelVersionId: sourceRun.modelVersionId,
    ...(sourceRun.surrogateModelId
      ? { surrogateModelId: sourceRun.surrogateModelId }
      : {}),
    analyses: sourceRun.tasks.map((task) => ({
      analysisKey: task.analysisKey,
      ...(task.pluginVersion ? { pluginVersion: task.pluginVersion } : {}),
      config: task.config,
      outputTargets: task.outputTargets,
    })),
    seed: sourceRun.seed,
    accuracyProfile: sourceRun.accuracyProfile,
    idempotencyKey: crypto.randomUUID(),
  });
  return app.fetch(request, c.env, c.executionCtx);
});

app.post("/api/v1/runs/:runId/cancel", async (c) => {
  const identity = authenticatedIdentity(c);
  const timestamp = now();
  const updated = await c.env.DB.prepare(
    `UPDATE runs SET status = 'cancelled', cancelled_at = ?, completed_at = ?
     WHERE id = ? AND owner_id = ? AND status IN ('queued', 'running')`,
  )
    .bind(timestamp, timestamp, c.req.param("runId"), identity.ownerId)
    .run();
  if (!updated.meta.changes)
    return jsonError(c, 409, "run_not_cancellable", "Run is not cancellable.");
  await c.env.DB.prepare(
    "UPDATE analysis_tasks SET status = 'cancelled', completed_at = ? WHERE run_id = ? AND status = 'queued'",
  )
    .bind(timestamp, c.req.param("runId"))
    .run();
  await destroyRunSandbox(c.env, c.req.param("runId"));
  return c.json({ status: "cancelled" });
});

app.get("/api/v1/runs/:runId/events", async (c) => {
  const identity = authenticatedIdentity(c);
  const runId = c.req.param("runId");
  if (!(await loadOwnedRun(c.env, runId, identity.ownerId))) {
    return jsonError(c, 404, "run_not_found", "Run not found.");
  }
  return streamSSE(c, async (stream) => {
    let last = "";
    for (let index = 0; index < 900; index += 1) {
      const run = await loadOwnedRun(c.env, runId, identity.ownerId);
      if (!run) break;
      const serialized = JSON.stringify(run);
      if (serialized !== last) {
        await stream.writeSSE({
          event: "run",
          data: serialized,
          id: String(index),
        });
        last = serialized;
      }
      if (
        ["succeeded", "partially_succeeded", "failed", "cancelled"].includes(
          run.status,
        )
      )
        break;
      await stream.sleep(1000);
    }
  });
});

async function loadRetainedReport(
  env: Env,
  reportId: string,
  ownerId?: string,
): Promise<Report | null> {
  const reportRow = await env.DB.prepare(
    `SELECT reports.id, reports.run_id, reports.title, reports.status,
            reports.updated_at, runs.owner_id
     FROM reports JOIN runs ON runs.id = reports.run_id
     WHERE (reports.id = ? OR reports.run_id = ?)
       AND (? IS NULL OR runs.owner_id = ?)`,
  )
    .bind(reportId, reportId, ownerId ?? null, ownerId ?? null)
    .first<{
      id: string;
      run_id: string;
      title: string;
      status: string;
      updated_at: string;
      owner_id: string;
    }>();
  if (!reportRow) return null;
  const run = await loadOwnedRun(env, reportRow.run_id, reportRow.owner_id);
  if (!run) return null;
  const [metadata, context, surrogate] = await Promise.all([
    modelMetadata(env, run.modelVersionId),
    reportModelContext(env, run.modelVersionId),
    reportSurrogateContext(env, run.surrogateModelId),
  ]);
  if (!context) return null;
  return {
    id: reportRow.id,
    runId: reportRow.run_id,
    title: reportRow.title,
    status: reportRow.status,
    generatedAt: reportRow.updated_at,
    project: { id: context.project_id, name: context.project_name },
    modelVersion: {
      id: run.modelVersionId,
      version: context.version,
      displayName: context.display_name,
      sourceKind: context.source_kind,
      createdAt: context.created_at,
      parentVersionId: context.parent_version_id,
    },
    seed: run.seed,
    accuracyProfile: run.accuracyProfile,
    evidenceSource: surrogate ? "surrogate" : "direct",
    surrogate: surrogate
      ? {
          id: surrogate.id,
          method: surrogate.method,
          pluginVersion: surrogate.plugin_version,
          openturnsVersion: surrogate.openturns_version,
        }
      : null,
    model: metadata ?? ({} as ModelMetadata),
    sections: run.tasks.map((task) => ({
      key: task.analysisKey,
      status: task.status,
      ...(task.result ? { result: task.result } : {}),
      ...(task.error ? { error: task.error } : {}),
    })),
  };
}

app.get("/api/v1/reports/:reportId", async (c) => {
  const identity = authenticatedIdentity(c);
  const report = await loadRetainedReport(
    c.env,
    c.req.param("reportId"),
    identity.ownerId,
  );
  if (!report)
    return jsonError(c, 404, "report_not_found", "Report is not ready.");
  return c.json({ report });
});

app.get("/api/v1/reports/:reportId/export", async (c) => {
  const identity = authenticatedIdentity(c);
  const reportRow = await c.env.DB.prepare(
    `SELECT reports.id, reports.run_id FROM reports JOIN runs ON runs.id = reports.run_id
     WHERE (reports.id = ? OR reports.run_id = ?) AND runs.owner_id = ?`,
  )
    .bind(c.req.param("reportId"), c.req.param("reportId"), identity.ownerId)
    .first<{ id: string; run_id: string }>();
  if (!reportRow)
    return jsonError(c, 404, "report_not_found", "Report not found.");
  const run = await loadOwnedRun(c.env, reportRow.run_id, identity.ownerId);
  if (!run) return jsonError(c, 404, "run_not_found", "Run not found.");
  const metadata = run ? await modelMetadata(c.env, run.modelVersionId) : null;
  if (c.req.query("format") === "json") {
    c.header(
      "Content-Disposition",
      `attachment; filename=uncertaintycat-${reportRow.run_id}.json`,
    );
    c.header("Content-Type", "application/json; charset=utf-8");
    return c.body(
      JSON.stringify(
        { manifestVersion: "1.0.0", generatedAt: now(), metadata, run },
        null,
        2,
      ),
    );
  }
  const archive = createReportBundle(run, metadata, now());
  const body = archive.buffer.slice(
    archive.byteOffset,
    archive.byteOffset + archive.byteLength,
  ) as ArrayBuffer;
  c.header(
    "Content-Disposition",
    `attachment; filename=uncertaintycat-${reportRow.run_id}.zip`,
  );
  c.header("Content-Type", "application/zip");
  return c.body(body);
});

const shareLinkSchema = z.object({
  expiresInDays: z.number().int().min(1).max(365).nullable().default(30),
  includeModelDefinition: z.boolean().default(false),
});
app.post(
  "/api/v1/reports/:reportId/share-links",
  zValidator("json", shareLinkSchema),
  async (c) => {
    const identity = authenticatedIdentity(c);
    const report = await c.env.DB.prepare(
      `SELECT reports.id FROM reports JOIN runs ON runs.id = reports.run_id
       WHERE (reports.id = ? OR reports.run_id = ?) AND runs.owner_id = ?`,
    )
      .bind(c.req.param("reportId"), c.req.param("reportId"), identity.ownerId)
      .first<{ id: string }>();
    if (!report)
      return jsonError(c, 404, "report_not_found", "Report not found.");
    const input = c.req.valid("json");
    const rawToken = token();
    const id = crypto.randomUUID();
    const createdAt = now();
    const expiresAt = input.expiresInDays
      ? new Date(Date.now() + input.expiresInDays * 86_400_000).toISOString()
      : null;
    await c.env.DB.prepare(
      `INSERT INTO report_share_links
       (id, report_id, token_hash, expires_at, include_model_definition, created_at)
       VALUES (?, ?, ?, ?, ?, ?)`,
    )
      .bind(
        id,
        report.id,
        await sha256Hex(rawToken),
        expiresAt,
        input.includeModelDefinition ? 1 : 0,
        createdAt,
      )
      .run();
    const origin = c.env.PUBLIC_WEB_ORIGIN ?? new URL(c.req.url).origin;
    return c.json(
      {
        shareLink: {
          id,
          url: `${origin}/shared/${rawToken}`,
          expiresAt,
          createdAt,
        },
      },
      201,
    );
  },
);

app.delete("/api/v1/reports/:reportId/share-links/:linkId", async (c) => {
  const identity = authenticatedIdentity(c);
  const updated = await c.env.DB.prepare(
    `UPDATE report_share_links SET revoked_at = ? WHERE id = ? AND report_id IN (
       SELECT reports.id FROM reports JOIN runs ON runs.id = reports.run_id
       WHERE (reports.id = ? OR reports.run_id = ?) AND runs.owner_id = ?
     ) AND revoked_at IS NULL`,
  )
    .bind(
      now(),
      c.req.param("linkId"),
      c.req.param("reportId"),
      c.req.param("reportId"),
      identity.ownerId,
    )
    .run();
  if (!updated.meta.changes)
    return jsonError(c, 404, "share_link_not_found", "Share link not found.");
  return c.body(null, 204);
});

app.get("/api/v1/shared-reports/:token", async (c) => {
  const record = await c.env.DB.prepare(
    `SELECT reports.id, reports.run_id, reports.title, reports.status,
            reports.updated_at, runs.owner_id, links.include_model_definition
     FROM report_share_links links
     JOIN reports ON reports.id = links.report_id
     JOIN runs ON runs.id = reports.run_id
     WHERE links.token_hash = ? AND links.revoked_at IS NULL
       AND (links.expires_at IS NULL OR links.expires_at > ?)`,
  )
    .bind(await sha256Hex(c.req.param("token")), now())
    .first<{
      id: string;
      run_id: string;
      title: string;
      status: string;
      updated_at: string;
      owner_id: string;
      include_model_definition: number;
    }>();
  if (!record)
    return jsonError(
      c,
      404,
      "share_link_not_found",
      "This share link is invalid or expired.",
    );
  const run = await loadOwnedRun(c.env, record.run_id, record.owner_id);
  if (!run) return jsonError(c, 404, "run_not_found", "Run not found.");
  const metadata = await modelMetadata(c.env, run.modelVersionId);
  const context = await reportModelContext(c.env, run.modelVersionId);
  const surrogate = await reportSurrogateContext(c.env, run.surrogateModelId);
  if (!context)
    return jsonError(c, 404, "model_not_found", "Model version not found.");
  const sharedDefinition = record.include_model_definition
    ? await loadModelDefinition(
        c.env,
        run.modelVersionId,
        record.owner_id,
        "shared",
      )
    : null;
  const report: Report = {
    id: record.id,
    runId: record.run_id,
    title: record.title,
    status: record.status,
    generatedAt: record.updated_at,
    project: { id: context.project_id, name: context.project_name },
    modelVersion: {
      id: run.modelVersionId,
      version: context.version,
      displayName: context.display_name,
      sourceKind: context.source_kind,
      createdAt: context.created_at,
      parentVersionId: context.parent_version_id,
    },
    seed: run.seed,
    accuracyProfile: run.accuracyProfile,
    evidenceSource: surrogate ? "surrogate" : "direct",
    surrogate: surrogate
      ? {
          id: surrogate.id,
          method: surrogate.method,
          pluginVersion: surrogate.plugin_version,
          openturnsVersion: surrogate.openturns_version,
        }
      : null,
    model: metadata ?? ({} as ModelMetadata),
    ...(sharedDefinition ? { modelDefinition: sharedDefinition } : {}),
    sections: run.tasks.map((task) => ({
      key: task.analysisKey,
      status: task.status,
      ...(task.result ? { result: task.result } : {}),
      ...(task.error ? { error: task.error } : {}),
    })),
  };
  c.header("Cache-Control", "private, no-store");
  return c.json({ report });
});

const chatSchema = z.object({ message: z.string().trim().min(1).max(4_000) });
app.get("/api/v1/reports/:reportId/chat", async (c) => {
  const identity = authenticatedIdentity(c);
  const report = await c.env.DB.prepare(
    `SELECT reports.id FROM reports JOIN runs ON runs.id = reports.run_id
     WHERE (reports.id = ? OR reports.run_id = ?) AND runs.owner_id = ?`,
  )
    .bind(c.req.param("reportId"), c.req.param("reportId"), identity.ownerId)
    .first<{ id: string }>();
  if (!report)
    return jsonError(c, 404, "report_not_found", "Report not found.");
  const rows = await c.env.DB.prepare(
    `SELECT id, role, content, created_at FROM chat_messages
     WHERE report_id = ? AND owner_id = ? ORDER BY created_at ASC LIMIT 100`,
  )
    .bind(report.id, identity.ownerId)
    .all<{
      id: string;
      role: "user" | "assistant";
      content: string;
      created_at: string;
    }>();
  return c.json({
    messages: rows.results.map((row) => ({
      id: row.id,
      role: row.role,
      content: row.content,
      createdAt: row.created_at,
    })),
  });
});

app.post(
  "/api/v1/reports/:reportId/chat",
  zValidator("json", chatSchema),
  async (c) => {
    const identity = authenticatedIdentity(c);
    const runtime = aiRuntime(c.env);
    if (!runtime.configured)
      return jsonError(
        c,
        503,
        "ai_unavailable",
        `${runtime.provider === "groq" ? "Groq" : "Cloudflare Workers AI"} is not configured.`,
      );
    const report = await c.env.DB.prepare(
      `SELECT reports.id, reports.run_id FROM reports JOIN runs ON runs.id = reports.run_id
     WHERE (reports.id = ? OR reports.run_id = ?) AND runs.owner_id = ?`,
    )
      .bind(c.req.param("reportId"), c.req.param("reportId"), identity.ownerId)
      .first<{ id: string; run_id: string }>();
    if (!report)
      return jsonError(c, 404, "report_not_found", "Report not found.");
    const run = await loadOwnedRun(c.env, report.run_id, identity.ownerId);
    if (!run) return jsonError(c, 404, "run_not_found", "Run not found.");
    const input = c.req.valid("json");
    const timestamp = now();
    const midnight = new Date();
    midnight.setUTCHours(0, 0, 0, 0);
    const usage = await c.env.DB.prepare(
      "SELECT COALESCE(SUM(units), 0) AS units FROM usage_ledger WHERE owner_id = ? AND kind = 'ai_chat' AND created_at >= ?",
    )
      .bind(identity.ownerId, midnight.toISOString())
      .first<{ units: number }>();
    const dailyLimit = 100;
    if (Number(usage?.units ?? 0) >= dailyLimit) {
      return jsonError(
        c,
        429,
        "ai_quota_exceeded",
        `Daily report-chat quota of ${dailyLimit} messages exceeded.`,
      );
    }
    const history = await c.env.DB.prepare(
      `SELECT role, content FROM chat_messages WHERE report_id = ? AND owner_id = ?
     ORDER BY created_at DESC LIMIT 20`,
    )
      .bind(report.id, identity.ownerId)
      .all<{ role: "user" | "assistant"; content: string }>();
    await c.env.DB.prepare(
      "INSERT INTO chat_messages (id, report_id, owner_id, role, content, created_at) VALUES (?, ?, ?, 'user', ?, ?)",
    )
      .bind(
        crypto.randomUUID(),
        report.id,
        identity.ownerId,
        input.message,
        timestamp,
      )
      .run();

    const chatStartedAt = Date.now();
    const chatProviderOptions = aiProviderOptions(
      runtime.provider,
      "reportChat",
    );
    const result = streamText({
      model: createAiLanguageModel(
        c.env,
        runtime.models.reportChat.modelId,
        `report:${report.id}`,
        "reportChat",
      ),
      ...(chatProviderOptions ? { providerOptions: chatProviderOptions } : {}),
      maxRetries: 0,
      timeout: REPORT_CHAT_TIMEOUT_MS,
      system: reportChatSystemPrompt(
        run.tasks.map((task) => ({
          analysis: task.analysisKey,
          status: task.status,
        })),
      ),
      messages: [
        ...history.results
          .reverse()
          .map((message) => ({ role: message.role, content: message.content })),
        { role: "user" as const, content: input.message },
      ],
      stopWhen: stepCountIs(8),
      tools: {
        getReportOutline: tool({
          description:
            "Discover analysis sections, completion state, available stored field names, and persisted scalar metric/fact values. Answer from scalarValues when they contain the requested evidence; field names alone are never an answer. Use getAnalysisSummary when more context is needed.",
          inputSchema: z.object({}),
          execute: async () =>
            run.tasks.map((task) => ({
              analysis: task.analysisKey,
              status: task.status,
              available: task.result
                ? {
                    metrics: Object.keys(task.result.payload.metrics),
                    facts: Object.keys(task.result.payload.facts),
                    tables: Object.keys(task.result.payload.tables),
                    series: Object.keys(task.result.payload.series),
                    matrices: Object.keys(task.result.payload.matrices),
                  }
                : undefined,
              scalarValues: task.result
                ? {
                    metrics: task.result.payload.metrics,
                    facts: task.result.payload.facts,
                  }
                : undefined,
            })),
        }),
        getAnalysisSummary: tool({
          description:
            "Read all scalar metrics, grounded facts, warnings, assumptions, and available data names for one analysis.",
          inputSchema: z.object({ analysisKey: z.string() }),
          execute: async ({ analysisKey }) => {
            const task = run.tasks.find(
              (candidate) => candidate.analysisKey === analysisKey,
            );
            return task?.result
              ? {
                  metrics: task.result.payload.metrics,
                  facts: task.result.payload.facts,
                  warnings: task.result.warnings,
                  assumptions: task.result.assumptions,
                  availableTables: Object.keys(task.result.payload.tables),
                  availableSeries: Object.keys(task.result.payload.series),
                  availableMatrices: Object.keys(task.result.payload.matrices),
                }
              : { error: "Analysis result is unavailable." };
          },
        }),
        getResultTable: tool({
          description:
            "Read a named stored table with bounded row pagination. Use this for row-level numerical claims.",
          inputSchema: z.object({
            analysisKey: z.string(),
            tableName: z.string(),
            offset: z.number().int().min(0).default(0),
            limit: z.number().int().min(1).max(50).default(25),
          }),
          execute: async ({ analysisKey, tableName, offset, limit }) => {
            const table = run.tasks.find(
              (task) => task.analysisKey === analysisKey,
            )?.result?.payload.tables[tableName];
            return table
              ? {
                  columns: table.columns,
                  rows: table.rows.slice(offset, offset + limit),
                  rowCount: table.row_count,
                  offset,
                }
              : { error: "The requested stored table does not exist." };
          },
        }),
        getResultSeries: tool({
          description:
            "Read a bounded page from a named stored chart series for convergence, distribution, or trace claims.",
          inputSchema: z.object({
            analysisKey: z.string(),
            seriesName: z.string(),
            offset: z.number().int().min(0).default(0),
            limit: z.number().int().min(1).max(200).default(100),
          }),
          execute: async ({ analysisKey, seriesName, offset, limit }) => {
            const series = run.tasks.find(
              (task) => task.analysisKey === analysisKey,
            )?.result?.payload.series[seriesName];
            return series
              ? {
                  name: series.name,
                  xLabel: series.x_label,
                  yLabel: series.y_label,
                  x: series.x.slice(offset, offset + limit),
                  y: series.y.slice(offset, offset + limit),
                  pointCount: Math.max(series.x.length, series.y.length),
                  offset,
                }
              : { error: "The requested stored series does not exist." };
          },
        }),
        getResultMatrix: tool({
          description:
            "Read a bounded window from a named stored matrix for correlation or dependence claims.",
          inputSchema: z.object({
            analysisKey: z.string(),
            matrixName: z.string(),
            rowOffset: z.number().int().min(0).default(0),
            rowLimit: z.number().int().min(1).max(50).default(25),
            columnOffset: z.number().int().min(0).default(0),
            columnLimit: z.number().int().min(1).max(50).default(25),
          }),
          execute: async ({
            analysisKey,
            matrixName,
            rowOffset,
            rowLimit,
            columnOffset,
            columnLimit,
          }) => {
            const matrix = run.tasks.find(
              (task) => task.analysisKey === analysisKey,
            )?.result?.payload.matrices[matrixName];
            return matrix
              ? {
                  rowLabels: matrix.row_labels.slice(
                    rowOffset,
                    rowOffset + rowLimit,
                  ),
                  columnLabels: matrix.column_labels.slice(
                    columnOffset,
                    columnOffset + columnLimit,
                  ),
                  values: matrix.values
                    .slice(rowOffset, rowOffset + rowLimit)
                    .map((row) =>
                      row.slice(columnOffset, columnOffset + columnLimit),
                    ),
                  rowCount: matrix.row_labels.length,
                  columnCount: matrix.column_labels.length,
                  rowOffset,
                  columnOffset,
                }
              : { error: "The requested stored matrix does not exist." };
          },
        }),
      },
      onFinish: async ({ text }) => {
        if (text) {
          await c.env.DB.batch([
            c.env.DB.prepare(
              "INSERT INTO chat_messages (id, report_id, owner_id, role, content, created_at) VALUES (?, ?, ?, 'assistant', ?, ?)",
            ).bind(
              crypto.randomUUID(),
              report.id,
              identity.ownerId,
              text,
              now(),
            ),
            c.env.DB.prepare(
              "INSERT INTO usage_ledger (id, owner_id, kind, units, reference_id, created_at) VALUES (?, ?, 'ai_chat', 1, ?, ?)",
            ).bind(crypto.randomUUID(), identity.ownerId, report.id, now()),
          ]);
          console.log(
            JSON.stringify({
              event: "report_chat_generation_succeeded",
              requestId: c.get("requestId"),
              reportId: report.id,
              aiProvider: runtime.provider,
              aiModelId: runtime.models.reportChat.modelId,
              durationMs: Date.now() - chatStartedAt,
              outputCharacters: text.length,
            }),
          );
        }
      },
      onError: ({ error }) => {
        console.error(
          JSON.stringify({
            event: "report_chat_generation_failed",
            requestId: c.get("requestId"),
            reportId: report.id,
            aiProvider: runtime.provider,
            aiModelId: runtime.models.reportChat.modelId,
            durationMs: Date.now() - chatStartedAt,
            error: error instanceof Error ? error.message : String(error),
          }),
        );
      },
    });
    return result.toTextStreamResponse();
  },
);

app.notFound((c) => {
  if (c.env.ASSETS && !c.req.path.startsWith("/api/"))
    return c.env.ASSETS.fetch(c.req.raw);
  return jsonError(c, 404, "not_found", "Route not found.");
});
app.onError((error, c) => {
  console.error(
    JSON.stringify({ requestId: c.get("requestId"), error: String(error) }),
  );
  return jsonError(c, 500, "internal_error", "An unexpected error occurred.");
});

export default {
  fetch: app.fetch,
  async queue(batch: MessageBatch<RunTaskMessage>, env: Env): Promise<void> {
    for (const message of batch.messages) {
      try {
        await processRunTask(env, {
          ...message.body,
          attempt: Math.max(0, message.attempts - 1),
        });
        message.ack();
      } catch (error) {
        console.error(
          JSON.stringify({ taskId: message.body.taskId, error: String(error) }),
        );
        const computeError =
          error instanceof ComputeRequestError ? error : null;
        if (computeError && !computeError.retryable) {
          await failRunTask(env, message.body.taskId, {
            code: computeError.code,
            message: computeError.message,
          });
          message.ack();
        } else if (message.attempts >= 3) {
          await failRunTask(env, message.body.taskId, {
            code: computeError?.code ?? "compute_retries_exhausted",
            message:
              computeError?.message ??
              "The compute service remained unavailable after the retry budget was exhausted.",
          });
          message.ack();
        } else {
          await requeueRunTask(env, message.body.taskId, message.attempts);
          message.retry({ delaySeconds: Math.min(60, 2 ** message.attempts) });
        }
      }
    }
  },
};

export { app };
export { ContainerProxy } from "@cloudflare/sandbox";
export { IsolatedComputeSandbox } from "./sandbox";
