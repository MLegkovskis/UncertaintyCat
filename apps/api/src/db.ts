import type {
  AnalysisResult,
  ModelMetadata,
  Run,
} from "@uncertaintycat/contracts";

import type { Env } from "./env";

export function now(): string {
  return new Date().toISOString();
}

export function parseJson<T>(value: string | null, fallback: T): T {
  if (!value) return fallback;
  try {
    return JSON.parse(value) as T;
  } catch {
    return fallback;
  }
}

interface RunRow {
  id: string;
  project_id: string;
  model_version_id: string;
  status: Run["status"];
  seed: number;
  accuracy_profile: Run["accuracyProfile"];
  created_at: string;
  completed_at: string | null;
}

interface TaskRow {
  id: string;
  analysis_key: string;
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  result_json: string | null;
  error_json: string | null;
}

export async function loadOwnedRun(
  env: Env,
  runId: string,
  ownerId: string,
): Promise<Run | null> {
  const run = await env.DB.prepare(
    `SELECT id, project_id, model_version_id, status, seed, accuracy_profile, created_at, completed_at
     FROM runs WHERE id = ? AND owner_id = ?`,
  )
    .bind(runId, ownerId)
    .first<RunRow>();
  if (!run) return null;
  const tasks = await env.DB.prepare(
    `SELECT id, analysis_key, status, result_json, error_json
     FROM analysis_tasks WHERE run_id = ? ORDER BY created_at`,
  )
    .bind(runId)
    .all<TaskRow>();
  return {
    id: run.id,
    projectId: run.project_id,
    modelVersionId: run.model_version_id,
    status: run.status,
    seed: run.seed,
    accuracyProfile: run.accuracy_profile,
    createdAt: run.created_at,
    completedAt: run.completed_at,
    tasks: tasks.results.map((task) => ({
      id: task.id,
      analysisKey: task.analysis_key,
      status: task.status,
      ...(task.result_json
        ? {
            result: parseJson<AnalysisResult>(
              task.result_json,
              {} as AnalysisResult,
            ),
          }
        : {}),
      ...(task.error_json
        ? {
            error: parseJson<{ code: string; message: string }>(
              task.error_json,
              { code: "unknown", message: "Unknown failure" },
            ),
          }
        : {}),
    })),
  };
}

export async function modelMetadata(
  env: Env,
  modelVersionId: string,
): Promise<ModelMetadata | null> {
  const row = await env.DB.prepare(
    "SELECT metadata_json FROM model_versions WHERE id = ?",
  )
    .bind(modelVersionId)
    .first<{ metadata_json: string }>();
  return row
    ? parseJson<ModelMetadata>(row.metadata_json, {} as ModelMetadata)
    : null;
}
