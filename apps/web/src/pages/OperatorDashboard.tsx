import { useQuery } from "@tanstack/react-query";
import type { OperatorOverview } from "@uncertaintycat/contracts";
import type { EChartsOption } from "echarts";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Clock3,
  Database,
  FolderKanban,
  RefreshCw,
  Users,
} from "lucide-react";
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";

import { api } from "../api";
import { EChart } from "../components/EChart";
import { EmptyState, StatusBadge } from "../components/Status";

type WindowHours = 24 | 168 | 720;

const integer = new Intl.NumberFormat();
const percent = new Intl.NumberFormat(undefined, {
  style: "percent",
  maximumFractionDigits: 1,
});

function time(value: string | null): string {
  return value ? new Date(value).toLocaleString() : "No activity";
}

function duration(value: number | null): string {
  if (value === null) return "—";
  if (value < 1000) return `${value} ms`;
  if (value < 60_000) return `${(value / 1000).toFixed(1)} s`;
  return `${(value / 60_000).toFixed(1)} min`;
}

function metricRate(value: number | null): string {
  return value === null ? "No completed runs" : percent.format(value);
}

function issueLabel(kind: OperatorOverview["issues"][number]["kind"]): string {
  return kind.replaceAll("_", " ");
}

export function OperatorDashboard() {
  const [windowHours, setWindowHours] = useState<WindowHours>(168);
  const overview = useQuery({
    queryKey: ["operator-overview", windowHours],
    queryFn: () => api.operatorOverview(windowHours),
    refetchInterval: 30_000,
    refetchIntervalInBackground: false,
  });
  const data = overview.data;
  const statusOption = useMemo<EChartsOption>(
    () => ({
      tooltip: { trigger: "item" },
      legend: { bottom: 0, textStyle: { fontSize: 10 } },
      series: [
        {
          name: "Runs",
          type: "pie",
          radius: ["48%", "72%"],
          center: ["50%", "44%"],
          label: { formatter: "{b}\n{c}" },
          data: (data?.runStatus ?? []).map((entry) => ({
            name: entry.status.replaceAll("_", " "),
            value: entry.count,
          })),
        },
      ],
    }),
    [data?.runStatus],
  );
  const analysisOption = useMemo<EChartsOption>(
    () => ({
      tooltip: { trigger: "axis" },
      grid: { left: 42, right: 18, top: 18, bottom: 62 },
      xAxis: {
        type: "category",
        axisLabel: { rotate: 32, fontSize: 9 },
        data: (data?.analyses ?? []).map((entry) =>
          entry.key.replaceAll("_", " "),
        ),
      },
      yAxis: { type: "value", minInterval: 1 },
      series: [
        {
          name: "Succeeded",
          type: "bar",
          stack: "tasks",
          color: "#16744b",
          data: (data?.analyses ?? []).map((entry) => entry.succeeded),
        },
        {
          name: "Failed",
          type: "bar",
          stack: "tasks",
          color: "#b42334",
          data: (data?.analyses ?? []).map((entry) => entry.failed),
        },
        {
          name: "Active",
          type: "bar",
          stack: "tasks",
          color: "#075f6a",
          data: (data?.analyses ?? []).map((entry) => entry.active),
        },
      ],
    }),
    [data?.analyses],
  );

  return (
    <div className="page operations-page">
      <div className="page-heading split">
        <div>
          <span className="section-kicker">Operations</span>
          <h1>Application health.</h1>
          <p>
            Near-real-time D1 state for accounts, projects, runs, analyses, and
            actionable failures. Private model source and numerical payloads are
            never included.
          </p>
        </div>
        <div className="operator-controls">
          <label>
            <span>Reporting window</span>
            <select
              value={windowHours}
              onChange={(event) =>
                setWindowHours(Number(event.target.value) as WindowHours)
              }
            >
              <option value={24}>Last 24 hours</option>
              <option value={168}>Last 7 days</option>
              <option value={720}>Last 30 days</option>
            </select>
          </label>
          <button
            className="button secondary"
            type="button"
            onClick={() => void overview.refetch()}
            disabled={overview.isFetching}
          >
            <RefreshCw className={overview.isFetching ? "spin" : ""} /> Refresh
          </button>
        </div>
      </div>

      {overview.isPending && (
        <div className="route-loading">
          Reading the latest operational snapshot…
        </div>
      )}
      {overview.isError && (
        <div className="error-banner" role="alert">
          The operational snapshot could not be loaded. Retry or inspect
          Cloudflare Workers Logs.
        </div>
      )}
      {data && (
        <>
          <div
            className={`operator-health ${data.issues.length || data.summary.activeTasks ? "attention" : "healthy"}`}
          >
            {data.issues.length || data.summary.activeTasks ? (
              <AlertTriangle />
            ) : (
              <CheckCircle2 />
            )}
            <div>
              <strong>
                {data.issues.length
                  ? `${data.issues.length} operational issue${data.issues.length === 1 ? "" : "s"} need attention`
                  : data.summary.activeTasks
                    ? `${data.summary.activeTasks} analysis task${data.summary.activeTasks === 1 ? " is" : "s are"} active`
                    : "No analysis failures in this window"}
              </strong>
              <span>
                Snapshot {time(data.generatedAt)} · refreshes every{" "}
                {data.refreshAfterSeconds} seconds while this page is open
              </span>
            </div>
          </div>

          <section className="operator-kpis" aria-label="Operational summary">
            <article>
              <Users />
              <span>Registered users</span>
              <strong>{integer.format(data.summary.users)}</strong>
              <small>all time</small>
            </article>
            <article>
              <FolderKanban />
              <span>Projects</span>
              <strong>{integer.format(data.summary.projects)}</strong>
              <small>
                {integer.format(data.summary.models)} retained models
              </small>
            </article>
            <article>
              <Activity />
              <span>Runs</span>
              <strong>{integer.format(data.summary.runs)}</strong>
              <small>
                {metricRate(data.summary.runSuccessRate)} successful
              </small>
            </article>
            <article>
              <Database />
              <span>Analysis tasks</span>
              <strong>{integer.format(data.summary.tasks)}</strong>
              <small>
                {data.summary.activeTasks} active · {data.summary.failedTasks}{" "}
                failed
              </small>
            </article>
          </section>

          <section className="operator-chart-grid">
            <article className="operator-panel">
              <header>
                <div>
                  <span className="section-kicker">Execution state</span>
                  <h2>Run outcomes</h2>
                </div>
              </header>
              {data.runStatus.length ? (
                <EChart
                  option={statusOption}
                  ariaLabel="Run outcomes by status"
                  height={310}
                />
              ) : (
                <EmptyState
                  title="No runs in this window"
                  body="Completed and active runs will appear here."
                />
              )}
            </article>
            <article className="operator-panel">
              <header>
                <div>
                  <span className="section-kicker">Analysis health</span>
                  <h2>Tasks by method</h2>
                </div>
              </header>
              {data.analyses.length ? (
                <EChart
                  option={analysisOption}
                  ariaLabel="Analysis task outcomes by method"
                  height={310}
                />
              ) : (
                <EmptyState
                  title="No analysis tasks"
                  body="Analysis activity will appear here."
                />
              )}
            </article>
          </section>

          <section className="operator-panel">
            <header>
              <div>
                <span className="section-kicker">Needs attention</span>
                <h2>Errors and stale work</h2>
              </div>
              <span>{data.issues.length} items</span>
            </header>
            {data.issues.length ? (
              <div className="operator-issues" role="list">
                {data.issues.map((issue) => (
                  <article key={`${issue.kind}-${issue.id}`} role="listitem">
                    <AlertTriangle />
                    <div className="operator-issue-main">
                      <strong>
                        {issue.analysisKey?.replaceAll("_", " ") ??
                          issueLabel(issue.kind)}
                      </strong>
                      <p>{issue.message}</p>
                      <small>
                        {issue.code} · {issue.projectName ?? "No project"} ·{" "}
                        {issue.ownerEmail ?? "Unknown owner"}
                      </small>
                    </div>
                    <time dateTime={issue.occurredAt}>
                      {time(issue.occurredAt)}
                    </time>
                    {issue.runId && (
                      <Link
                        className="button secondary small"
                        to={`/runs/${issue.runId}`}
                      >
                        Open run
                      </Link>
                    )}
                  </article>
                ))}
              </div>
            ) : (
              <EmptyState
                title="Nothing needs attention"
                body="No stored failures or tasks stale for more than 15 minutes were found."
              />
            )}
          </section>

          <section className="operator-panel">
            <header>
              <div>
                <span className="section-kicker">Executions</span>
                <h2>Recent runs</h2>
              </div>
              <span>{data.recentRuns.length} shown</span>
            </header>
            <div
              className="operator-table-wrap"
              role="region"
              aria-label="Recent runs table"
              tabIndex={0}
            >
              <table className="operator-table">
                <thead>
                  <tr>
                    <th>Run</th>
                    <th>Owner</th>
                    <th>Status</th>
                    <th>Tasks</th>
                    <th>Duration</th>
                    <th>Started</th>
                  </tr>
                </thead>
                <tbody>
                  {data.recentRuns.map((run) => (
                    <tr key={run.id}>
                      <td>
                        <Link to={`/runs/${run.id}`}>
                          <strong>{run.modelName}</strong>
                          <small>{run.projectName}</small>
                        </Link>
                      </td>
                      <td>{run.ownerEmail}</td>
                      <td>
                        <StatusBadge status={run.status} />
                      </td>
                      <td>
                        {run.tasks}
                        {run.failedTasks ? ` · ${run.failedTasks} failed` : ""}
                      </td>
                      <td>{duration(run.durationMs)}</td>
                      <td>{time(run.createdAt)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="operator-panel">
            <header>
              <div>
                <span className="section-kicker">Methods</span>
                <h2>Analysis performance</h2>
              </div>
            </header>
            <div
              className="operator-table-wrap"
              role="region"
              aria-label="Analysis performance table"
              tabIndex={0}
            >
              <table className="operator-table">
                <thead>
                  <tr>
                    <th>Analysis</th>
                    <th>Total</th>
                    <th>Succeeded</th>
                    <th>Failed</th>
                    <th>Active</th>
                    <th>Success rate</th>
                    <th>Average duration</th>
                  </tr>
                </thead>
                <tbody>
                  {data.analyses.map((analysis) => (
                    <tr key={analysis.key}>
                      <td>
                        <strong>{analysis.key.replaceAll("_", " ")}</strong>
                      </td>
                      <td>{analysis.total}</td>
                      <td>{analysis.succeeded}</td>
                      <td>{analysis.failed}</td>
                      <td>{analysis.active}</td>
                      <td>{metricRate(analysis.successRate)}</td>
                      <td>{duration(analysis.averageDurationMs)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <section className="operator-split">
            <article className="operator-panel">
              <header>
                <div>
                  <span className="section-kicker">Accounts</span>
                  <h2>Users</h2>
                </div>
                <span>{data.users.length} shown</span>
              </header>
              <div
                className="operator-table-wrap"
                role="region"
                aria-label="Users table"
                tabIndex={0}
              >
                <table className="operator-table">
                  <thead>
                    <tr>
                      <th>User</th>
                      <th>Projects</th>
                      <th>Runs</th>
                      <th>Last activity</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.users.map((user) => (
                      <tr key={user.id}>
                        <td>
                          <strong>{user.name}</strong>
                          <small>{user.email}</small>
                        </td>
                        <td>{user.projectCount}</td>
                        <td>
                          {user.periodRunCount}
                          {user.periodFailedRunCount
                            ? ` · ${user.periodFailedRunCount} issues`
                            : ""}
                        </td>
                        <td>{time(user.lastActivityAt)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </article>
            <article className="operator-panel">
              <header>
                <div>
                  <span className="section-kicker">Workspaces</span>
                  <h2>Projects</h2>
                </div>
                <span>{data.projects.length} shown</span>
              </header>
              <div
                className="operator-table-wrap"
                role="region"
                aria-label="Projects table"
                tabIndex={0}
              >
                <table className="operator-table">
                  <thead>
                    <tr>
                      <th>Project</th>
                      <th>Models</th>
                      <th>Runs</th>
                      <th>Last activity</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.projects.map((project) => (
                      <tr key={project.id}>
                        <td>
                          <Link to={`/studies/${project.id}`}>
                            <strong>{project.name}</strong>
                            <small>{project.ownerEmail}</small>
                          </Link>
                        </td>
                        <td>{project.modelCount}</td>
                        <td>
                          {project.periodRunCount}
                          {project.periodFailedRunCount
                            ? ` · ${project.periodFailedRunCount} issues`
                            : ""}
                        </td>
                        <td>{time(project.lastActivityAt)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </article>
          </section>

          <p className="operator-footnote">
            <Clock3 /> D1 provides exact retained application state. Runtime
            exceptions, request traces, CPU time, and upstream failures remain
            available in Cloudflare Workers Logs and Metrics.
          </p>
        </>
      )}
    </div>
  );
}
