import type {
  AnalysisResult,
  ModelDefinition,
  ModelAssessment,
  ModelMetadata,
  ModelVersion,
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
  surrogate_model_id: string | null;
  status: Run["status"];
  seed: number;
  accuracy_profile: Run["accuracyProfile"];
  created_at: string;
  completed_at: string | null;
  project_name: string;
  model_display_name: string;
  model_version: number;
  source_kind: ModelVersion["sourceKind"];
}

interface TaskRow {
  id: string;
  analysis_key: string;
  plugin_version: string | null;
  config_json: string;
  output_targets_json: string;
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
    `SELECT r.id, r.project_id, r.model_version_id, r.surrogate_model_id, r.status, r.seed,
            r.accuracy_profile, r.created_at, r.completed_at,
            p.name AS project_name, m.display_name AS model_display_name,
            m.version AS model_version, m.source_kind
     FROM runs r
     JOIN projects p ON p.id = r.project_id
     JOIN model_versions m ON m.id = r.model_version_id
     WHERE r.id = ? AND r.owner_id = ?`,
  )
    .bind(runId, ownerId)
    .first<RunRow>();
  if (!run) return null;
  const tasks = await env.DB.prepare(
    `SELECT id, analysis_key, plugin_version, config_json, output_targets_json,
            status, result_json, error_json
     FROM analysis_tasks WHERE run_id = ? ORDER BY created_at`,
  )
    .bind(runId)
    .all<TaskRow>();
  return {
    id: run.id,
    projectId: run.project_id,
    modelVersionId: run.model_version_id,
    surrogateModelId: run.surrogate_model_id,
    evidenceSource: run.surrogate_model_id ? "surrogate" : "direct",
    projectName: run.project_name,
    modelDisplayName: run.model_display_name,
    modelVersion: run.model_version,
    sourceKind: run.source_kind,
    status: run.status,
    seed: run.seed,
    accuracyProfile: run.accuracy_profile,
    createdAt: run.created_at,
    completedAt: run.completed_at,
    tasks: tasks.results.map((task) => ({
      id: task.id,
      analysisKey: task.analysis_key,
      pluginVersion: task.plugin_version,
      config: parseJson<Record<string, unknown>>(task.config_json, {}),
      outputTargets: parseJson<number[]>(task.output_targets_json, []),
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

export async function loadModelDefinition(
  env: Env,
  modelVersionId: string,
  ownerId: string,
  visibility: ModelDefinition["visibility"] = "owner",
): Promise<ModelDefinition | null> {
  const row = await env.DB.prepare(
    `SELECT m.id, m.project_id, m.version, m.source_kind, m.display_name,
            m.source_key, m.source_hash, m.metadata_json, m.assessment_json,
            m.builder_spec_json,
            m.parent_version_id, m.derivation_json, m.created_at,
            p.name AS project_name, p.description AS project_description,
            p.created_at AS project_created_at, p.updated_at AS project_updated_at
     FROM model_versions m
     JOIN projects p ON p.id = m.project_id
     WHERE m.id = ? AND p.owner_id = ?`,
  )
    .bind(modelVersionId, ownerId)
    .first<{
      id: string;
      project_id: string;
      version: number;
      source_kind: ModelVersion["sourceKind"];
      display_name: string;
      source_key: string;
      source_hash: string;
      metadata_json: string;
      assessment_json: string | null;
      builder_spec_json: string | null;
      parent_version_id: string | null;
      derivation_json: string | null;
      created_at: string;
      project_name: string;
      project_description: string;
      project_created_at: string;
      project_updated_at: string;
    }>();
  if (!row) return null;
  const sourceObject = await env.ARTIFACTS.get(row.source_key);
  if (!sourceObject) return null;
  return {
    modelVersion: {
      id: row.id,
      projectId: row.project_id,
      version: row.version,
      sourceKind: row.source_kind,
      displayName: row.display_name,
      sourceHash: row.source_hash,
      metadata: parseJson<ModelMetadata>(row.metadata_json, {} as ModelMetadata),
      assessment: parseJson<ModelAssessment | null>(row.assessment_json, null),
      parentVersionId: row.parent_version_id,
      derivation: parseJson<Record<string, unknown> | null>(row.derivation_json, null),
      createdAt: row.created_at,
    },
    project: {
      id: row.project_id,
      name: row.project_name,
      description: row.project_description,
      createdAt: row.project_created_at,
      updatedAt: row.project_updated_at,
    },
    source: await sourceObject.text(),
    builderSpec: parseJson<Record<string, unknown> | null>(row.builder_spec_json, null),
    visibility,
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
