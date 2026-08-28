import type {
  AnalysisResult,
  AnalysisTask,
  ModelAssessment,
  ModelMetadata,
} from "@uncertaintycat/contracts";

import {
  computeFetch,
  destroyRunSandbox,
  type ComputeProgress,
} from "./compute-client";
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
  metadata_json: string;
  assessment_json: string | null;
  surrogate_model_id: string | null;
  surrogate_method: "pce" | "gpr" | null;
  surrogate_object_key: string | null;
  surrogate_status: string | null;
  surrogate_validation_json: string | null;
  seed: number;
  run_status: string;
}

type TaskProgress = NonNullable<AnalysisTask["progress"]>;

function taskProgress(
  phase: string,
  percent: number,
  message: string,
  options: { indeterminate?: boolean; attempt?: number } = {},
): TaskProgress {
  return {
    phase,
    percent: Math.max(0, Math.min(100, Math.round(percent))),
    message,
    indeterminate: options.indeterminate ?? false,
    attempt: options.attempt ?? 0,
    updatedAt: now(),
  };
}

async function writeTaskProgress(
  env: Env,
  taskId: string,
  progress: TaskProgress,
): Promise<void> {
  await env.DB.prepare(
    "UPDATE analysis_tasks SET progress_json = ? WHERE id = ? AND status IN ('queued', 'running')",
  )
    .bind(JSON.stringify(progress), taskId)
    .run();
}

function encodeBase64(value: ArrayBuffer): string {
  const bytes = new Uint8Array(value);
  const chunks: string[] = [];
  for (let offset = 0; offset < bytes.length; offset += 0x8000)
    chunks.push(
      String.fromCharCode(...bytes.subarray(offset, offset + 0x8000)),
    );
  return btoa(chunks.join(""));
}

export class ComputeRequestError extends Error {
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
    `UPDATE analysis_tasks SET status = 'failed', error_json = ?, progress_json = ?, completed_at = ?
     WHERE id = ? AND status IN ('queued', 'running')`,
  )
    .bind(
      JSON.stringify(error),
      JSON.stringify(taskProgress("failed", 100, error.message)),
      now(),
      taskId,
    )
    .run();
  await finalizeRun(env, task.run_id);
}

export async function requeueRunTask(
  env: Env,
  taskId: string,
  nextAttempt = 1,
): Promise<void> {
  await env.DB.prepare(
    "UPDATE analysis_tasks SET status = 'queued', started_at = NULL, progress_json = ? WHERE id = ? AND status = 'running'",
  )
    .bind(
      JSON.stringify(
        taskProgress(
          "retrying",
          0,
          `Compute capacity was interrupted. Retry ${nextAttempt} is queued automatically.`,
          { indeterminate: true, attempt: nextAttempt },
        ),
      ),
      taskId,
    )
    .run();
}

export async function processRunTask(
  env: Env,
  message: RunTaskMessage,
): Promise<void> {
  const attempt = Math.max(0, message.attempt);
  const claimedAt = now();
  const claimed = await env.DB.prepare(
    `UPDATE analysis_tasks SET status = 'running', started_at = ?, progress_json = ?
     WHERE id = ? AND status = 'queued'`,
  )
    .bind(
      claimedAt,
      JSON.stringify(
        taskProgress("capacity_acquired", 3, "Compute capacity acquired.", {
          indeterminate: true,
          attempt,
        }),
      ),
      message.taskId,
    )
    .run();
  if (!claimed.meta.changes) return;
  await env.DB.prepare(
    "UPDATE runs SET status = 'running', started_at = COALESCE(started_at, ?) WHERE id = ? AND status = 'queued'",
  )
    .bind(claimedAt, message.runId)
    .run();

  let latestProgress = taskProgress(
    "model_artifact",
    5,
    "Loading immutable numerical inputs.",
    { indeterminate: true, attempt },
  );
  let progressWrites = writeTaskProgress(env, message.taskId, latestProgress);
  const publishProgress = (progress: ComputeProgress) => {
    latestProgress = taskProgress(
      progress.phase,
      progress.percent,
      progress.message,
      {
        indeterminate: progress.indeterminate,
        attempt,
      },
    );
    const persistedProgress = latestProgress;
    progressWrites = progressWrites
      .catch(() => undefined)
      .then(() => writeTaskProgress(env, message.taskId, persistedProgress));
  };

  try {
    const task = await env.DB.prepare(
      `SELECT t.id, t.run_id, t.analysis_key, t.plugin_version, t.config_json,
            t.output_targets_json, m.source_key, m.metadata_json, m.assessment_json,
            r.surrogate_model_id, s.method AS surrogate_method,
            s.object_key AS surrogate_object_key, s.status AS surrogate_status,
            s.validation_json AS surrogate_validation_json,
            r.seed, r.status AS run_status
     FROM analysis_tasks t
     JOIN runs r ON r.id = t.run_id
     JOIN model_versions m ON m.id = r.model_version_id
     LEFT JOIN surrogate_models s ON s.id = r.surrogate_model_id
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
        "UPDATE analysis_tasks SET status = 'cancelled', progress_json = ?, completed_at = ? WHERE id = ?",
      )
        .bind(
          JSON.stringify(taskProgress("cancelled", 100, "Analysis cancelled.")),
          now(),
          task.id,
        )
        .run();
      return;
    }

    const analysis = {
      analysis_key: task.analysis_key,
      plugin_version: task.plugin_version,
      config: parseJson<Record<string, unknown>>(task.config_json, {}),
      output_targets: parseJson<number[]>(task.output_targets_json, []),
    };
    let response: Response;
    if (task.surrogate_model_id) {
      if (
        task.surrogate_status !== "promoted" ||
        !task.surrogate_object_key ||
        !task.surrogate_method ||
        !task.surrogate_validation_json ||
        !task.assessment_json
      ) {
        await failRunTask(env, task.id, {
          code: "surrogate_unavailable",
          message: "The explicitly selected promoted surrogate is unavailable.",
        });
        return;
      }
      const surrogateObject = await env.ARTIFACTS.get(
        task.surrogate_object_key,
      );
      if (!surrogateObject) {
        await failRunTask(env, task.id, {
          code: "surrogate_artifact_missing",
          message: "The promoted surrogate XML artifact is missing.",
        });
        return;
      }
      const validation = parseJson<{
        outputTargets?: number[];
      } | null>(task.surrogate_validation_json, null);
      const surrogateOutputTarget = validation?.outputTargets?.[0];
      if (surrogateOutputTarget === undefined) {
        await failRunTask(env, task.id, {
          code: "surrogate_provenance_incomplete",
          message:
            "The promoted surrogate does not retain its source output target.",
        });
        return;
      }
      publishProgress({
        phase: "surrogate_artifact",
        percent: 10,
        message: "Loading the promoted surrogate artifact.",
        indeterminate: true,
      });
      response = await computeFetch(
        env,
        "/v1/surrogates/execute",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            xml_base64: encodeBase64(await surrogateObject.arrayBuffer()),
            method: task.surrogate_method,
            analysis: {
              ...analysis,
              output_targets: analysis.output_targets.length ? [0] : [],
            },
            metadata: parseJson<ModelMetadata>(
              task.metadata_json,
              {} as ModelMetadata,
            ),
            assessment: parseJson<ModelAssessment>(
              task.assessment_json,
              {} as ModelAssessment,
            ),
            surrogate_id: task.surrogate_model_id,
            surrogate_output_target: surrogateOutputTarget,
            seed: task.seed,
            run_id: task.run_id,
          }),
        },
        { onProgress: publishProgress },
      );
    } else {
      const sourceObject = await env.ARTIFACTS.get(task.source_key);
      if (!sourceObject) {
        await failRunTask(env, task.id, {
          code: "model_source_missing",
          message: "The immutable model source artifact is missing.",
        });
        return;
      }
      publishProgress({
        phase: "model_artifact",
        percent: 10,
        message:
          "Model artifact loaded; starting isolated OpenTURNS execution.",
        indeterminate: true,
      });
      response = await computeFetch(
        env,
        "/v1/execute",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            source: await sourceObject.text(),
            seed: task.seed,
            run_id: task.run_id,
            analysis,
          }),
        },
        { onProgress: publishProgress },
      );
    }
    publishProgress({
      phase: "persistence",
      percent: 97,
      message: "Persisting the completed numerical evidence.",
      indeterminate: false,
    });
    await progressWrites;
    const body = (await response.json().catch(() => ({}))) as {
      result?: AnalysisResult;
      error?: { code?: string; message?: string };
    };
    if (!response.ok || !body.result) {
      const retryable = response.status >= 500;
      if (retryable) {
        throw new ComputeRequestError(
          true,
          body.error?.code ?? "compute_failed",
          body.error?.message ?? `Compute service failed (${response.status}).`,
        );
      }
      await failRunTask(env, task.id, {
        code: body.error?.code ?? "analysis_failed",
        message: body.error?.message ?? "Analysis could not be completed.",
      });
      return;
    }
    await env.DB.prepare(
      "UPDATE analysis_tasks SET status = 'succeeded', result_json = ?, progress_json = ?, completed_at = ? WHERE id = ?",
    )
      .bind(
        JSON.stringify(body.result),
        JSON.stringify(
          taskProgress("complete", 100, "Numerical evidence persisted."),
        ),
        now(),
        task.id,
      )
      .run();
    await finalizeRun(env, task.run_id);
  } catch (error) {
    if (error instanceof ComputeRequestError) throw error;
    throw new ComputeRequestError(true, "compute_unavailable", String(error));
  } finally {
    await progressWrites.catch(() => undefined);
  }
}
