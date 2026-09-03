import type { OperatorOverview } from "@uncertaintycat/contracts";

import type { Env } from "./env";

const WINDOWS = new Set([24, 168, 720]);
const ROW_LIMIT = 100;
const ERROR_LIMIT = 240;

function count(value: number | string | null | undefined): number {
  return Number(value ?? 0);
}

function isoFromAuthTimestamp(value: number): string {
  const milliseconds = value < 10_000_000_000 ? value * 1000 : value;
  return new Date(milliseconds).toISOString();
}

function safeError(value: string | null): { code: string; message: string } {
  try {
    const parsed = JSON.parse(value ?? "{}") as {
      code?: unknown;
      message?: unknown;
    };
    return {
      code:
        typeof parsed.code === "string"
          ? parsed.code.slice(0, 80)
          : "operation_failed",
      message:
        typeof parsed.message === "string"
          ? parsed.message.replaceAll(/\s+/g, " ").slice(0, ERROR_LIMIT)
          : "The operation failed without a stored public-safe message.",
    };
  } catch {
    return {
      code: "operation_failed",
      message: "The operation failed with an unreadable stored error.",
    };
  }
}

export function operatorWindow(value: string | undefined): 24 | 168 | 720 {
  const parsed = Number(value ?? 168);
  return WINDOWS.has(parsed) ? (parsed as 24 | 168 | 720) : 168;
}

export async function loadOperatorOverview(
  env: Env,
  windowHours: 24 | 168 | 720,
): Promise<OperatorOverview> {
  const generatedAt = new Date();
  const cutoff = new Date(
    generatedAt.getTime() - windowHours * 3_600_000,
  ).toISOString();
  const staleCutoff = new Date(
    generatedAt.getTime() - 15 * 60_000,
  ).toISOString();

  const [
    totals,
    runStatuses,
    taskTotals,
    analyses,
    recentRuns,
    users,
    projects,
    analysisIssues,
    fitIssues,
    understandingIssues,
    staleTasks,
  ] = await Promise.all([
    env.DB.prepare(
      `SELECT
           (SELECT COUNT(*) FROM user) AS users,
           (SELECT COUNT(*) FROM projects) AS projects,
           (SELECT COUNT(*) FROM model_versions) AS models`,
    ).first<{ users: number; projects: number; models: number }>(),
    env.DB.prepare(
      `SELECT status, COUNT(*) AS count FROM runs
         WHERE created_at >= ? GROUP BY status ORDER BY count DESC`,
    )
      .bind(cutoff)
      .all<{ status: string; count: number }>(),
    env.DB.prepare(
      `SELECT COUNT(*) AS total,
                SUM(CASE WHEN t.status = 'succeeded' THEN 1 ELSE 0 END) AS succeeded,
                SUM(CASE WHEN t.status = 'failed' THEN 1 ELSE 0 END) AS failed,
                SUM(CASE WHEN t.status IN ('queued', 'running') THEN 1 ELSE 0 END) AS active
         FROM analysis_tasks t JOIN runs r ON r.id = t.run_id
         WHERE r.created_at >= ?`,
    )
      .bind(cutoff)
      .first<{
        total: number;
        succeeded: number;
        failed: number;
        active: number;
      }>(),
    env.DB.prepare(
      `SELECT t.analysis_key AS key, COUNT(*) AS total,
                SUM(CASE WHEN t.status = 'succeeded' THEN 1 ELSE 0 END) AS succeeded,
                SUM(CASE WHEN t.status = 'failed' THEN 1 ELSE 0 END) AS failed,
                SUM(CASE WHEN t.status IN ('queued', 'running') THEN 1 ELSE 0 END) AS active,
                AVG(CASE WHEN t.started_at IS NOT NULL AND t.completed_at IS NOT NULL
                    THEN (julianday(t.completed_at) - julianday(t.started_at)) * 86400000 END) AS average_duration_ms
         FROM analysis_tasks t JOIN runs r ON r.id = t.run_id
         WHERE r.created_at >= ? GROUP BY t.analysis_key ORDER BY total DESC, key ASC`,
    )
      .bind(cutoff)
      .all<{
        key: string;
        total: number;
        succeeded: number;
        failed: number;
        active: number;
        average_duration_ms: number | null;
      }>(),
    env.DB.prepare(
      `SELECT r.id, r.project_id, p.name AS project_name, m.display_name AS model_name,
                COALESCE(u.email, r.owner_id) AS owner_email, r.status, r.created_at, r.completed_at,
                CASE WHEN r.started_at IS NOT NULL AND r.completed_at IS NOT NULL
                     THEN (julianday(r.completed_at) - julianday(r.started_at)) * 86400000 END AS duration_ms,
                COUNT(t.id) AS task_count,
                SUM(CASE WHEN t.status = 'failed' THEN 1 ELSE 0 END) AS failed_task_count
         FROM runs r
         JOIN projects p ON p.id = r.project_id
         JOIN model_versions m ON m.id = r.model_version_id
         LEFT JOIN user u ON u.id = r.owner_id
         LEFT JOIN analysis_tasks t ON t.run_id = r.id
         WHERE r.created_at >= ?
         GROUP BY r.id ORDER BY r.created_at DESC LIMIT ?`,
    )
      .bind(cutoff, ROW_LIMIT)
      .all<{
        id: string;
        project_id: string;
        project_name: string;
        model_name: string;
        owner_email: string;
        status: string;
        created_at: string;
        completed_at: string | null;
        duration_ms: number | null;
        task_count: number;
        failed_task_count: number;
      }>(),
    env.DB.prepare(
      `SELECT u.id, u.name, u.email, u.createdAt AS registered_at,
                COUNT(DISTINCT p.id) AS project_count,
                COUNT(DISTINCT CASE WHEN r.created_at >= ? THEN r.id END) AS period_run_count,
                COUNT(DISTINCT CASE WHEN r.created_at >= ? AND r.status IN ('failed', 'partially_succeeded') THEN r.id END) AS period_failed_run_count,
                COALESCE(MAX(r.created_at), MAX(p.updated_at)) AS last_activity_at
         FROM user u LEFT JOIN projects p ON p.owner_id = u.id
         LEFT JOIN runs r ON r.project_id = p.id
         GROUP BY u.id ORDER BY last_activity_at DESC, u.createdAt DESC LIMIT ?`,
    )
      .bind(cutoff, cutoff, ROW_LIMIT)
      .all<{
        id: string;
        name: string;
        email: string;
        registered_at: number;
        project_count: number;
        period_run_count: number;
        period_failed_run_count: number;
        last_activity_at: string | null;
      }>(),
    env.DB.prepare(
      `SELECT p.id, p.name, p.updated_at, COALESCE(u.name, p.owner_id) AS owner_name,
                COALESCE(u.email, p.owner_id) AS owner_email,
                COUNT(DISTINCT m.id) AS model_count,
                COUNT(DISTINCT CASE WHEN r.created_at >= ? THEN r.id END) AS period_run_count,
                COUNT(DISTINCT CASE WHEN r.created_at >= ? AND r.status IN ('failed', 'partially_succeeded') THEN r.id END) AS period_failed_run_count
         FROM projects p LEFT JOIN user u ON u.id = p.owner_id
         LEFT JOIN model_versions m ON m.project_id = p.id
         LEFT JOIN runs r ON r.project_id = p.id
         GROUP BY p.id ORDER BY p.updated_at DESC LIMIT ?`,
    )
      .bind(cutoff, cutoff, ROW_LIMIT)
      .all<{
        id: string;
        name: string;
        updated_at: string;
        owner_name: string;
        owner_email: string;
        model_count: number;
        period_run_count: number;
        period_failed_run_count: number;
      }>(),
    env.DB.prepare(
      `SELECT t.id, t.analysis_key, t.run_id, t.status, t.error_json, COALESCE(t.completed_at, t.created_at) AS occurred_at,
                p.id AS project_id, p.name AS project_name, COALESCE(u.email, r.owner_id) AS owner_email
         FROM analysis_tasks t JOIN runs r ON r.id = t.run_id JOIN projects p ON p.id = r.project_id
         LEFT JOIN user u ON u.id = r.owner_id
         WHERE t.status = 'failed' AND t.created_at >= ? ORDER BY occurred_at DESC LIMIT ?`,
    )
      .bind(cutoff, ROW_LIMIT)
      .all<{
        id: string;
        analysis_key: string;
        run_id: string;
        status: string;
        error_json: string | null;
        occurred_at: string;
        project_id: string;
        project_name: string;
        owner_email: string;
      }>(),
    env.DB.prepare(
      `SELECT d.id, d.status, d.error_json, COALESCE(d.completed_at, d.created_at) AS occurred_at,
                p.id AS project_id, p.name AS project_name, COALESCE(u.email, d.owner_id) AS owner_email
         FROM data_analysis_runs d JOIN datasets ds ON ds.id = d.dataset_id
         JOIN projects p ON p.id = ds.project_id LEFT JOIN user u ON u.id = d.owner_id
         WHERE d.status = 'failed' AND d.created_at >= ? ORDER BY occurred_at DESC LIMIT ?`,
    )
      .bind(cutoff, ROW_LIMIT)
      .all<{
        id: string;
        status: string;
        error_json: string | null;
        occurred_at: string;
        project_id: string;
        project_name: string;
        owner_email: string;
      }>(),
    env.DB.prepare(
      `SELECT mu.id, mu.status, mu.error, mu.updated_at AS occurred_at,
                p.id AS project_id, p.name AS project_name, COALESCE(u.email, p.owner_id) AS owner_email
         FROM model_understandings mu JOIN model_versions m ON m.id = mu.model_version_id
         JOIN projects p ON p.id = m.project_id LEFT JOIN user u ON u.id = p.owner_id
         WHERE mu.status = 'failed' AND mu.updated_at >= ? ORDER BY occurred_at DESC LIMIT ?`,
    )
      .bind(cutoff, ROW_LIMIT)
      .all<{
        id: string;
        status: string;
        error: string | null;
        occurred_at: string;
        project_id: string;
        project_name: string;
        owner_email: string;
      }>(),
    env.DB.prepare(
      `SELECT t.id, t.analysis_key, t.run_id, t.status, t.created_at AS occurred_at,
                p.id AS project_id, p.name AS project_name, COALESCE(u.email, r.owner_id) AS owner_email
         FROM analysis_tasks t JOIN runs r ON r.id = t.run_id JOIN projects p ON p.id = r.project_id
         LEFT JOIN user u ON u.id = r.owner_id
         WHERE t.status IN ('queued', 'running') AND COALESCE(t.started_at, t.created_at) < ?
         ORDER BY occurred_at ASC LIMIT ?`,
    )
      .bind(staleCutoff, ROW_LIMIT)
      .all<{
        id: string;
        analysis_key: string;
        run_id: string;
        status: string;
        occurred_at: string;
        project_id: string;
        project_name: string;
        owner_email: string;
      }>(),
  ]);

  const status = new Map(
    runStatuses.results.map((row) => [row.status, count(row.count)]),
  );
  const runCount = [...status.values()].reduce((sum, value) => sum + value, 0);
  const successfulRuns = count(status.get("succeeded"));
  const failedRuns =
    count(status.get("failed")) + count(status.get("partially_succeeded"));
  const activeRuns = count(status.get("queued")) + count(status.get("running"));
  const finishedRuns =
    successfulRuns + failedRuns + count(status.get("cancelled"));

  const issues: OperatorOverview["issues"] = [
    ...analysisIssues.results.map((row) => ({
      id: row.id,
      kind: "analysis" as const,
      ...safeError(row.error_json),
      status: row.status,
      analysisKey: row.analysis_key,
      runId: row.run_id,
      projectId: row.project_id,
      projectName: row.project_name,
      ownerEmail: row.owner_email,
      occurredAt: row.occurred_at,
    })),
    ...fitIssues.results.map((row) => ({
      id: row.id,
      kind: "distribution_fit" as const,
      ...safeError(row.error_json),
      status: row.status,
      projectId: row.project_id,
      projectName: row.project_name,
      ownerEmail: row.owner_email,
      occurredAt: row.occurred_at,
    })),
    ...understandingIssues.results.map((row) => ({
      id: row.id,
      kind: "model_understanding" as const,
      code: "model_understanding_failed",
      message: (
        row.error ??
        "Model Understanding failed without a stored public-safe message."
      )
        .replaceAll(/\s+/g, " ")
        .slice(0, ERROR_LIMIT),
      status: row.status,
      projectId: row.project_id,
      projectName: row.project_name,
      ownerEmail: row.owner_email,
      occurredAt: row.occurred_at,
    })),
    ...staleTasks.results.map((row) => ({
      id: `stale-${row.id}`,
      kind: "stale_task" as const,
      code: "task_progress_stale",
      message:
        "No terminal state was recorded within 15 minutes. Inspect the queue and Worker logs.",
      status: row.status,
      analysisKey: row.analysis_key,
      runId: row.run_id,
      projectId: row.project_id,
      projectName: row.project_name,
      ownerEmail: row.owner_email,
      occurredAt: row.occurred_at,
    })),
  ]
    .sort((left, right) => right.occurredAt.localeCompare(left.occurredAt))
    .slice(0, ROW_LIMIT);

  return {
    generatedAt: generatedAt.toISOString(),
    windowHours,
    refreshAfterSeconds: 30,
    summary: {
      users: count(totals?.users),
      projects: count(totals?.projects),
      models: count(totals?.models),
      runs: runCount,
      successfulRuns,
      failedRuns,
      activeRuns,
      tasks: count(taskTotals?.total),
      successfulTasks: count(taskTotals?.succeeded),
      failedTasks: count(taskTotals?.failed),
      activeTasks: count(taskTotals?.active),
      runSuccessRate: finishedRuns ? successfulRuns / finishedRuns : null,
    },
    runStatus: runStatuses.results.map((row) => ({
      status: row.status,
      count: count(row.count),
    })),
    analyses: analyses.results.map((row) => {
      const completed = count(row.succeeded) + count(row.failed);
      return {
        key: row.key,
        total: count(row.total),
        succeeded: count(row.succeeded),
        failed: count(row.failed),
        active: count(row.active),
        successRate: completed ? count(row.succeeded) / completed : null,
        averageDurationMs:
          row.average_duration_ms === null
            ? null
            : Math.max(0, Math.round(row.average_duration_ms)),
      };
    }),
    issues,
    recentRuns: recentRuns.results.map((row) => ({
      id: row.id,
      projectId: row.project_id,
      projectName: row.project_name,
      modelName: row.model_name,
      ownerEmail: row.owner_email,
      status: row.status,
      createdAt: row.created_at,
      completedAt: row.completed_at,
      durationMs:
        row.duration_ms === null
          ? null
          : Math.max(0, Math.round(row.duration_ms)),
      tasks: count(row.task_count),
      failedTasks: count(row.failed_task_count),
    })),
    users: users.results.map((row) => ({
      id: row.id,
      name: row.name,
      email: row.email,
      registeredAt: isoFromAuthTimestamp(row.registered_at),
      projectCount: count(row.project_count),
      periodRunCount: count(row.period_run_count),
      periodFailedRunCount: count(row.period_failed_run_count),
      lastActivityAt: row.last_activity_at,
    })),
    projects: projects.results.map((row) => ({
      id: row.id,
      name: row.name,
      ownerName: row.owner_name,
      ownerEmail: row.owner_email,
      modelCount: count(row.model_count),
      periodRunCount: count(row.period_run_count),
      periodFailedRunCount: count(row.period_failed_run_count),
      lastActivityAt: row.updated_at,
    })),
  };
}
