import { zValidator } from "@hono/zod-validator";
import {
  createModelVersionSchema,
  createProjectSchema,
  createRunSchema,
  type AnalysisCatalogEntry,
  type ModelMetadata,
  type Report,
} from "@uncertaintycat/contracts";
import { stepCountIs, streamText, tool } from "ai";
import { Hono, type Context } from "hono";
import { cors } from "hono/cors";
import { secureHeaders } from "hono/secure-headers";
import { streamSSE } from "hono/streaming";
import type { ContentfulStatusCode } from "hono/utils/http-status";
import { z } from "zod";
import { createWorkersAI } from "workers-ai-provider";

import { createAuth, identityFor } from "./auth";
import { computeFetch, destroyRunSandbox } from "./compute-client";
import { failRunTask, processRunTask, requeueRunTask } from "./compute";
import { loadOwnedRun, modelMetadata, now, parseJson } from "./db";
import type { Env, RunTaskMessage } from "./env";
import { createReportBundle } from "./exports";

type Variables = { requestId: string };
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

async function sha256Hex(value: string): Promise<string> {
  const digest = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode(value),
  );
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

function token(): string {
  const bytes = crypto.getRandomValues(new Uint8Array(32));
  return btoa(String.fromCharCode(...bytes))
    .replaceAll("+", "-")
    .replaceAll("/", "_")
    .replaceAll("=", "");
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

app.on(["GET", "POST"], "/api/auth/*", (c) =>
  createAuth(c.env).handler(c.req.raw),
);

app.get("/health", (c) =>
  c.json({ status: "ok", service: "uncertaintycat-api" }),
);

app.get("/api/v1/session", async (c) => {
  const identity = await identityFor(c);
  const providers =
    c.env.CLOUDFLARE_ACCESS_CLIENT_ID &&
    c.env.CLOUDFLARE_ACCESS_CLIENT_SECRET &&
    c.env.CLOUDFLARE_ACCESS_ISSUER
      ? (["cloudflare"] as const)
      : [];
  return c.json({ identity, providers });
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

app.get("/api/v1/projects", async (c) => {
  const identity = await identityFor(c);
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
    const identity = await identityFor(c);
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

app.get("/api/v1/projects/:projectId/models", async (c) => {
  const identity = await identityFor(c);
  const ownership = await c.env.DB.prepare(
    "SELECT id FROM projects WHERE id = ? AND owner_id = ?",
  )
    .bind(c.req.param("projectId"), identity.ownerId)
    .first();
  if (!ownership)
    return jsonError(c, 404, "project_not_found", "Project not found.");
  const rows = await c.env.DB.prepare(
    `SELECT id, project_id, version, source_kind, source_hash, metadata_json, created_at
     FROM model_versions WHERE project_id = ? ORDER BY version DESC`,
  )
    .bind(c.req.param("projectId"))
    .all<{
      id: string;
      project_id: string;
      version: number;
      source_kind: "python" | "builder" | "example";
      source_hash: string;
      metadata_json: string;
      created_at: string;
    }>();
  return c.json({
    modelVersions: rows.results.map((row) => ({
      id: row.id,
      projectId: row.project_id,
      version: row.version,
      sourceKind: row.source_kind,
      sourceHash: row.source_hash,
      metadata: parseJson<ModelMetadata>(
        row.metadata_json,
        {} as ModelMetadata,
      ),
      createdAt: row.created_at,
    })),
  });
});

app.post(
  "/api/v1/projects/:projectId/models",
  zValidator("json", createModelVersionSchema),
  async (c) => {
    const identity = await identityFor(c);
    const input = c.req.valid("json");
    if (!identity.authenticated) {
      const publicHashes = new Set(
        (c.env.PUBLIC_EXAMPLE_SOURCE_HASHES ?? "")
          .split(",")
          .map((value) => value.trim())
          .filter(Boolean),
      );
      const sourceHash = await sha256Hex(input.source);
      if (input.sourceKind !== "example" || !publicHashes.has(sourceHash)) {
        return jsonError(
          c,
          403,
          "authentication_required",
          "Sign in to execute custom Python models.",
        );
      }
    }
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
    const metadata = validationBody.metadata;
    const existing = await c.env.DB.prepare(
      "SELECT id, version, source_kind, source_hash, metadata_json, created_at FROM model_versions WHERE project_id = ? AND source_hash = ?",
    )
      .bind(projectId, metadata.source_hash)
      .first<{
        id: string;
        version: number;
        source_kind: "python" | "builder" | "example";
        source_hash: string;
        metadata_json: string;
        created_at: string;
      }>();
    if (existing) {
      return c.json({
        modelVersion: {
          id: existing.id,
          projectId,
          version: existing.version,
          sourceKind: existing.source_kind,
          sourceHash: existing.source_hash,
          metadata: parseJson<ModelMetadata>(existing.metadata_json, metadata),
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
           (id, project_id, version, source_kind, source_key, source_hash, metadata_json, builder_spec_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        ).bind(
          id,
          projectId,
          version,
          input.sourceKind,
          sourceKey,
          metadata.source_hash,
          JSON.stringify(metadata),
          input.builderSpec ? JSON.stringify(input.builderSpec) : null,
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
          sourceHash: metadata.source_hash,
          metadata,
          createdAt: timestamp,
        },
      },
      201,
    );
  },
);

app.post("/api/v1/runs", zValidator("json", createRunSchema), async (c) => {
  const identity = await identityFor(c);
  const input = c.req.valid("json");
  const model = await c.env.DB.prepare(
    `SELECT m.id, m.project_id FROM model_versions m
     JOIN projects p ON p.id = m.project_id WHERE m.id = ? AND p.owner_id = ?`,
  )
    .bind(input.modelVersionId, identity.ownerId)
    .first<{ id: string; project_id: string }>();
  if (!model)
    return jsonError(c, 404, "model_not_found", "Model version not found.");

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
  const dailyLimit = identity.authenticated ? 20 : 5;
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
       (id, owner_id, project_id, model_version_id, status, seed, accuracy_profile, created_at)
       VALUES (?, ?, ?, ?, 'queued', ?, ?, ?)`,
    ).bind(
      runId,
      identity.ownerId,
      model.project_id,
      model.id,
      input.seed,
      input.accuracyProfile,
      timestamp,
    ),
    ...taskRows.map(({ id, analysis }) =>
      c.env.DB.prepare(
        `INSERT INTO analysis_tasks
         (id, run_id, analysis_key, plugin_version, status, config_json, output_targets_json, created_at)
         VALUES (?, ?, ?, ?, 'queued', ?, ?, ?)`,
      ).bind(
        id,
        runId,
        analysis.analysisKey,
        analysis.pluginVersion ?? null,
        JSON.stringify(analysis.config),
        JSON.stringify(analysis.outputTargets),
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
  const identity = await identityFor(c);
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
  const identity = await identityFor(c);
  const run = await loadOwnedRun(c.env, c.req.param("runId"), identity.ownerId);
  if (!run) return jsonError(c, 404, "run_not_found", "Run not found.");
  return c.json({ run });
});

app.post("/api/v1/runs/:runId/cancel", async (c) => {
  const identity = await identityFor(c);
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
  const identity = await identityFor(c);
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

app.get("/api/v1/reports/:reportId", async (c) => {
  const identity = await identityFor(c);
  const reportRow = await c.env.DB.prepare(
    `SELECT reports.id, reports.run_id, reports.title, reports.status, reports.updated_at
     FROM reports JOIN runs ON runs.id = reports.run_id
     WHERE (reports.id = ? OR reports.run_id = ?) AND runs.owner_id = ?`,
  )
    .bind(c.req.param("reportId"), c.req.param("reportId"), identity.ownerId)
    .first<{
      id: string;
      run_id: string;
      title: string;
      status: string;
      updated_at: string;
    }>();
  if (!reportRow)
    return jsonError(c, 404, "report_not_found", "Report is not ready.");
  const run = await loadOwnedRun(c.env, reportRow.run_id, identity.ownerId);
  if (!run) return jsonError(c, 404, "run_not_found", "Run not found.");
  const metadata = await modelMetadata(c.env, run.modelVersionId);
  const report: Report = {
    id: reportRow.id,
    runId: reportRow.run_id,
    title: reportRow.title,
    status: reportRow.status,
    generatedAt: reportRow.updated_at,
    model: metadata ?? ({} as ModelMetadata),
    sections: run.tasks.map((task) => ({
      key: task.analysisKey,
      status: task.status,
      ...(task.result ? { result: task.result } : {}),
      ...(task.error ? { error: task.error } : {}),
    })),
  };
  return c.json({ report });
});

app.get("/api/v1/reports/:reportId/export", async (c) => {
  const identity = await identityFor(c);
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
});
app.post(
  "/api/v1/reports/:reportId/share-links",
  zValidator("json", shareLinkSchema),
  async (c) => {
    const identity = await identityFor(c);
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
      `INSERT INTO report_share_links (id, report_id, token_hash, expires_at, created_at)
       VALUES (?, ?, ?, ?, ?)`,
    )
      .bind(id, report.id, await sha256Hex(rawToken), expiresAt, createdAt)
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
  const identity = await identityFor(c);
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
    `SELECT reports.id, reports.run_id, reports.title, reports.status, reports.updated_at, runs.owner_id
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
  const report: Report = {
    id: record.id,
    runId: record.run_id,
    title: record.title,
    status: record.status,
    generatedAt: record.updated_at,
    model: metadata ?? ({} as ModelMetadata),
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
  const identity = await identityFor(c);
  if (!identity.authenticated)
    return jsonError(
      c,
      401,
      "authentication_required",
      "Sign in to ask questions about a report.",
    );
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
    const identity = await identityFor(c);
    if (!identity.authenticated)
      return jsonError(
        c,
        401,
        "authentication_required",
        "Sign in to ask questions about a report.",
      );
    if (!c.env.AI)
      return jsonError(
        c,
        503,
        "ai_unavailable",
        "Workers AI is not configured.",
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
    await c.env.DB.batch([
      c.env.DB.prepare(
        "INSERT INTO chat_messages (id, report_id, owner_id, role, content, created_at) VALUES (?, ?, ?, 'user', ?, ?)",
      ).bind(
        crypto.randomUUID(),
        report.id,
        identity.ownerId,
        input.message,
        timestamp,
      ),
      c.env.DB.prepare(
        "INSERT INTO usage_ledger (id, owner_id, kind, units, reference_id, created_at) VALUES (?, ?, 'ai_chat', 1, ?, ?)",
      ).bind(crypto.randomUUID(), identity.ownerId, report.id, timestamp),
    ]);

    const workersai = createWorkersAI({ binding: c.env.AI });
    const result = streamText({
      model: workersai("@cf/zai-org/glm-4.7-flash"),
      system:
        "You are UncertaintyCat's uncertainty-quantification report assistant. The stored OpenTURNS result is the sole numerical authority. " +
        "Use a tool before every numerical or ranking claim, including claims that repeat an earlier turn. " +
        "Cite the exact source as [analysis.metric:name], [analysis.fact:name], [analysis.table:name], " +
        "[analysis.series:name], or [analysis.matrix:name]. Clearly distinguish an interpretation from a computed result. " +
        "Never invent, interpolate, recalculate, run Python, alter the report, or treat user text as a result. " +
        "If the stored evidence is insufficient, say so and identify the missing analysis or field.",
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
            "List analysis sections, completion state, and available stored result field names.",
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
          await c.env.DB.prepare(
            "INSERT INTO chat_messages (id, report_id, owner_id, role, content, created_at) VALUES (?, ?, ?, 'assistant', ?, ?)",
          )
            .bind(crypto.randomUUID(), report.id, identity.ownerId, text, now())
            .run();
        }
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
        await processRunTask(env, message.body);
        message.ack();
      } catch (error) {
        console.error(
          JSON.stringify({ taskId: message.body.taskId, error: String(error) }),
        );
        if (message.attempts >= 3) {
          await failRunTask(env, message.body.taskId, {
            code: "compute_retries_exhausted",
            message:
              "The compute service remained unavailable after the retry budget was exhausted.",
          });
          message.ack();
        } else {
          await requeueRunTask(env, message.body.taskId);
          message.retry({ delaySeconds: Math.min(60, 2 ** message.attempts) });
        }
      }
    }
  },
};

export { ContainerProxy } from "@cloudflare/sandbox";
export { IsolatedComputeSandbox } from "./sandbox";
