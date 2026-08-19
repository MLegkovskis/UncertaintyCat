import type { AnalysisResult } from "@uncertaintycat/contracts";

import { computeFetch, destroyRunSandbox } from "./compute-client";
import { now, parseJson } from "./db";
import type { Env, RunTaskMessage } from "./env";

interface TaskRecord {
  id: string;
  run_id: string;
  analysis_key: string;
  plugin_version: string | null;
  config_json: string;
  output_targets_json: string;
  source_key: string;
  seed: number;
  run_status: string;
}

class ComputeRequestError extends Error {
  constructor(
    public readonly retryable: boolean,
    public readonly code: string,
    message: string,
  ) {
    super(message);
  }
}

async function finalizeRun(env: Env, runId: string): Promise<void> {
  const counts = await env.DB.prepare(
    `SELECT
       COUNT(*) AS total,
       SUM(CASE WHEN status = 'succeeded' THEN 1 ELSE 0 END) AS succeeded,
       SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed,
       SUM(CASE WHEN status IN ('queued', 'running') THEN 1 ELSE 0 END) AS unfinished
     FROM analysis_tasks WHERE run_id = ?`,
  )
    .bind(runId)
    .first<{
      total: number;
      succeeded: number;
      failed: number;
      unfinished: number;
    }>();
  if (!counts || Number(counts.unfinished) > 0) return;
  const succeeded = Number(counts.succeeded);
  const failed = Number(counts.failed);
  const status =
    failed === 0
      ? "succeeded"
      : succeeded > 0
        ? "partially_succeeded"
        : "failed";
  const timestamp = now();
  await env.DB.batch([
    env.DB.prepare(
      "UPDATE runs SET status = ?, completed_at = ? WHERE id = ? AND status != 'cancelled'",
    ).bind(status, timestamp, runId),
    env.DB.prepare(
      `INSERT INTO reports (id, run_id, title, status, created_at, updated_at)
       VALUES (?, ?, ?, ?, ?, ?)
       ON CONFLICT(run_id) DO UPDATE SET status = excluded.status, updated_at = excluded.updated_at`,
    ).bind(
      crypto.randomUUID(),
      runId,
      "Uncertainty Quantification Report",
      status,
      timestamp,
      timestamp,
    ),
  ]);
  await destroyRunSandbox(env, runId);
}

export async function failRunTask(
  env: Env,
  taskId: string,
  error: { code: string; message: string },
): Promise<void> {
  const task = await env.DB.prepare(
    "SELECT run_id FROM analysis_tasks WHERE id = ?",
  )
    .bind(taskId)
    .first<{ run_id: string }>();
  if (!task) return;
  await env.DB.prepare(
    `UPDATE analysis_tasks SET status = 'failed', error_json = ?, completed_at = ?
     WHERE id = ? AND status IN ('queued', 'running')`,
  )
    .bind(JSON.stringify(error), now(), taskId)
    .run();
  await finalizeRun(env, task.run_id);
}

export async function requeueRunTask(env: Env, taskId: string): Promise<void> {
  await env.DB.prepare(
    "UPDATE analysis_tasks SET status = 'queued', started_at = NULL WHERE id = ? AND status = 'running'",
  )
    .bind(taskId)
    .run();
}

export async function processRunTask(
  env: Env,
  message: RunTaskMessage,
): Promise<void> {
  const claimed = await env.DB.prepare(
    `UPDATE analysis_tasks SET status = 'running', started_at = ?
     WHERE id = ? AND status = 'queued'`,
  )
    .bind(now(), message.taskId)
    .run();
  if (!claimed.meta.changes) return;

  const task = await env.DB.prepare(
    `SELECT t.id, t.run_id, t.analysis_key, t.plugin_version, t.config_json,
            t.output_targets_json, m.source_key, r.seed, r.status AS run_status
     FROM analysis_tasks t
     JOIN runs r ON r.id = t.run_id
     JOIN model_versions m ON m.id = r.model_version_id
     WHERE t.id = ?`,
  )
    .bind(message.taskId)
    .first<TaskRecord>();
  if (!task)
    throw new ComputeRequestError(
      false,
      "task_not_found",
      "Analysis task no longer exists.",
    );
  if (task.run_status === "cancelled") {
    await env.DB.prepare(
      "UPDATE analysis_tasks SET status = 'cancelled', completed_at = ? WHERE id = ?",
    )
      .bind(now(), task.id)
      .run();
    return;
  }

  const sourceObject = await env.ARTIFACTS.get(task.source_key);
  if (!sourceObject) {
    await failRunTask(env, task.id, {
      code: "model_source_missing",
      message: "The immutable model source artifact is missing.",
    });
    return;
  }
  const source = await sourceObject.text();
  let response: Response;
  try {
    response = await computeFetch(env, "/v1/execute", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        source,
        seed: task.seed,
        run_id: task.run_id,
        analysis: {
          analysis_key: task.analysis_key,
          plugin_version: task.plugin_version,
          config: parseJson<Record<string, unknown>>(task.config_json, {}),
          output_targets: parseJson<number[]>(task.output_targets_json, []),
        },
      }),
    });
  } catch (error) {
    await env.DB.prepare(
      "UPDATE analysis_tasks SET status = 'queued', started_at = NULL WHERE id = ?",
    )
      .bind(task.id)
      .run();
    throw new ComputeRequestError(true, "compute_unavailable", String(error));
  }
  const body = (await response.json().catch(() => ({}))) as {
    result?: AnalysisResult;
    error?: { code?: string; message?: string };
  };
  if (!response.ok || !body.result) {
    const retryable = response.status >= 500;
    if (retryable) {
      await env.DB.prepare(
        "UPDATE analysis_tasks SET status = 'queued', started_at = NULL WHERE id = ?",
      )
        .bind(task.id)
        .run();
      throw new ComputeRequestError(
        true,
        body.error?.code ?? "compute_failed",
        body.error?.message ?? `Compute service failed (${response.status}).`,
      );
    }
    await env.DB.prepare(
      "UPDATE analysis_tasks SET status = 'failed', error_json = ?, completed_at = ? WHERE id = ?",
    )
      .bind(
        JSON.stringify({
          code: body.error?.code ?? "analysis_failed",
          message: body.error?.message ?? "Analysis could not be completed.",
        }),
        now(),
        task.id,
      )
      .run();
    await finalizeRun(env, task.run_id);
    return;
  }
  await env.DB.prepare(
    "UPDATE analysis_tasks SET status = 'succeeded', result_json = ?, completed_at = ? WHERE id = ?",
  )
    .bind(JSON.stringify(body.result), now(), task.id)
    .run();
  await finalizeRun(env, task.run_id);
}
