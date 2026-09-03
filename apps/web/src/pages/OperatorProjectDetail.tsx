import { useQuery } from "@tanstack/react-query";
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  Braces,
  Clock3,
  FileBarChart,
  RefreshCw,
} from "lucide-react";
import { useEffect } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";

import { api } from "../api";
import { EmptyState, StatusBadge } from "../components/Status";

function time(value: string | null): string {
  return value ? new Date(value).toLocaleString() : "Not recorded";
}

function duration(value: number | null): string {
  if (value === null) return "—";
  if (value < 1000) return `${value} ms`;
  if (value < 60_000) return `${(value / 1000).toFixed(1)} s`;
  return `${(value / 60_000).toFixed(1)} min`;
}

const REPORT_STATUSES = new Set(["succeeded", "partially_succeeded", "failed"]);

export function OperatorProjectDetail() {
  const { projectId = "" } = useParams();
  const [searchParams, setSearchParams] = useSearchParams();
  const selectedRunId = searchParams.get("run");
  const pageQuery = searchParams.get("page");
  const parsedPage = pageQuery === null ? undefined : Number(pageQuery);
  const pageNumber =
    parsedPage !== undefined && Number.isFinite(parsedPage)
      ? Math.max(1, Math.floor(parsedPage))
      : undefined;
  const project = useQuery({
    queryKey: ["operator-project", projectId, pageNumber, selectedRunId],
    queryFn: () =>
      api.operatorProject(projectId, pageNumber, selectedRunId ?? undefined),
    enabled: Boolean(projectId),
    refetchInterval: 30_000,
    refetchIntervalInBackground: false,
    retry: false,
  });
  const data = project.data;

  useEffect(() => {
    if (!data || !selectedRunId) return;
    document
      .getElementById(`operator-run-${selectedRunId}`)
      ?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [data, selectedRunId]);

  if (project.isPending) {
    return (
      <div className="route-loading">
        Reading the project’s retained operational record…
      </div>
    );
  }
  if (project.isError || !data) {
    return (
      <div className="page auth-required-page">
        <section className="auth-required-card">
          <AlertTriangle />
          <span className="section-kicker">Project unavailable</span>
          <h1>This project could not be inspected.</h1>
          <p>
            It may have been deleted, or the operational snapshot could not be
            loaded. Return to Operations and refresh the current D1 state.
          </p>
          <Link className="button secondary" to="/operator">
            <ArrowLeft /> Back to Operations
          </Link>
        </section>
      </div>
    );
  }

  return (
    <div className="page operations-page operator-project-page">
      <nav className="breadcrumbs" aria-label="Breadcrumb">
        <Link to="/operator">Operations</Link>
        <span>/</span>
        <span>{data.project.name}</span>
      </nav>
      <div className="page-heading split">
        <div>
          <span className="section-kicker">Operator project inspection</span>
          <h1>{data.project.name}</h1>
          <p>
            {data.project.ownerName} · {data.project.ownerEmail} · updated{" "}
            {time(data.project.updatedAt)}
          </p>
        </div>
        <button
          className="button secondary"
          type="button"
          onClick={() => void project.refetch()}
          disabled={project.isFetching}
        >
          <RefreshCw className={project.isFetching ? "spin" : ""} /> Refresh
        </button>
      </div>

      <div className="operator-readonly-note">
        <FileBarChart />
        <div>
          <strong>Read-only operational view</strong>
          <span>
            Inspect retained model versions, analyses, numerical reports, and
            failures without changing the user’s project.
          </span>
        </div>
      </div>

      <section className="operator-kpis" aria-label="Project summary">
        <article>
          <Braces />
          <span>Models</span>
          <strong>{data.project.modelCount}</strong>
          <small>{data.models.length} versions shown</small>
        </article>
        <article>
          <Activity />
          <span>Runs</span>
          <strong>{data.project.runCount}</strong>
          <small>
            {data.runPage.start}–{data.runPage.end} shown
          </small>
        </article>
        <article>
          <Clock3 />
          <span>Analysis tasks</span>
          <strong>{data.project.taskCount}</strong>
          <small>{data.project.activeTaskCount} currently active</small>
        </article>
        <article>
          <AlertTriangle />
          <span>Failed tasks</span>
          <strong>{data.project.failedTaskCount}</strong>
          <small>all retained executions</small>
        </article>
      </section>

      <section className="operator-panel">
        <header>
          <div>
            <span className="section-kicker">Retained definitions</span>
            <h2>Model versions</h2>
          </div>
          <span>{data.models.length} shown</span>
        </header>
        {data.models.length ? (
          <div
            className="operator-table-wrap"
            role="region"
            aria-label="Project model versions"
            tabIndex={0}
          >
            <table className="operator-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Version</th>
                  <th>Definition</th>
                  <th>Shape</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {data.models.map((model) => (
                  <tr key={model.id}>
                    <td>
                      <strong>{model.displayName}</strong>
                    </td>
                    <td>v{model.version}</td>
                    <td>{model.sourceKind}</td>
                    <td>
                      {model.inputDimension ?? "?"} inputs →{" "}
                      {model.outputDimension ?? "?"} outputs
                    </td>
                    <td>{time(model.createdAt)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState
            title="No retained models"
            body="This user has not yet validated a model in the project."
          />
        )}
      </section>

      <section className="operator-panel operator-runs-panel">
        <header>
          <div>
            <span className="section-kicker">Execution evidence</span>
            <h2>Runs and analyses</h2>
          </div>
          <span>
            {data.runPage.start}–{data.runPage.end} of {data.runPage.totalRuns}
          </span>
        </header>
        {data.runs.length ? (
          <div className="operator-run-list">
            {data.runs.map((run) => (
              <article
                id={`operator-run-${run.id}`}
                className={
                  run.id === selectedRunId
                    ? "operator-run-card selected"
                    : "operator-run-card"
                }
                key={run.id}
              >
                <header>
                  <div>
                    <strong>
                      {run.modelName} <span>v{run.modelVersion}</span>
                    </strong>
                    <small>
                      Run {run.id} · started{" "}
                      {time(run.startedAt ?? run.createdAt)}
                      {run.durationMs !== null
                        ? ` · ${duration(run.durationMs)}`
                        : ""}
                    </small>
                  </div>
                  <div className="operator-run-actions">
                    <StatusBadge status={run.status} />
                    {REPORT_STATUSES.has(run.status) && (
                      <Link
                        className="button secondary small"
                        to={`/operator/reports/${run.id}`}
                      >
                        <FileBarChart /> Open numerical report
                      </Link>
                    )}
                  </div>
                </header>
                <div
                  className="operator-table-wrap"
                  role="region"
                  aria-label={`${run.modelName} analysis tasks`}
                  tabIndex={0}
                >
                  <table className="operator-table">
                    <thead>
                      <tr>
                        <th>Analysis</th>
                        <th>Plugin</th>
                        <th>Status</th>
                        <th>Duration</th>
                        <th>Diagnostic</th>
                      </tr>
                    </thead>
                    <tbody>
                      {run.tasks.map((task) => (
                        <tr key={task.id}>
                          <td>
                            <strong>
                              {task.analysisKey.replaceAll("_", " ")}
                            </strong>
                          </td>
                          <td>{task.pluginVersion ?? "—"}</td>
                          <td>
                            <StatusBadge status={task.status} />
                          </td>
                          <td>{duration(task.durationMs)}</td>
                          <td className={task.error ? "operator-error" : ""}>
                            {task.error
                              ? `${task.error.code}: ${task.error.message}`
                              : "No stored error"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </article>
            ))}
          </div>
        ) : (
          <EmptyState
            title="No previous runs"
            body="The project has no retained analysis executions yet."
          />
        )}
        {data.runPage.totalPages > 1 && (
          <nav className="operator-pagination" aria-label="Analysis run pages">
            <button
              className="button secondary small"
              type="button"
              disabled={data.runPage.page <= 1}
              onClick={() =>
                setSearchParams({ page: String(data.runPage.page - 1) })
              }
            >
              Previous 50
            </button>
            <span>
              Page {data.runPage.page} of {data.runPage.totalPages}
            </span>
            <button
              className="button secondary small"
              type="button"
              disabled={data.runPage.page >= data.runPage.totalPages}
              onClick={() =>
                setSearchParams({ page: String(data.runPage.page + 1) })
              }
            >
              Next 50
            </button>
          </nav>
        )}
      </section>
    </div>
  );
}
