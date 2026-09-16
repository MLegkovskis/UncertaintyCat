import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowRight,
  CircleCheck,
  Cpu,
  TimerReset,
  RotateCcw,
  XCircle,
} from "lucide-react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { api } from "../api";
import { analysisDisplayName } from "../analysisNames";
import { ResultView } from "../components/ResultView";
import { StatusBadge } from "../components/Status";

const TERMINAL = new Set([
  "succeeded",
  "partially_succeeded",
  "failed",
  "cancelled",
]);

function fallbackTaskMessage(
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled",
) {
  if (status === "queued") return "Waiting for compute capacity.";
  if (status === "running") return "OpenTURNS computation is active.";
  if (status === "cancelled") return "Analysis cancelled before a result was retained.";
  if (status === "failed") return "Analysis failed without a numerical result.";
  return "Analysis is complete.";
}

export function RunPage() {
  const { runId = "" } = useParams();
  const navigate = useNavigate();
  const client = useQueryClient();
  const query = useQuery({
    queryKey: ["run", runId],
    queryFn: () => api.getRun(runId),
    refetchInterval: (state) =>
      state.state.status === "error" || (state.state.data && TERMINAL.has(state.state.data.run.status))
        ? false
        : 1000,
  });
  const cancel = useMutation({
    mutationFn: () => api.cancelRun(runId),
    onSuccess: () => client.invalidateQueries({ queryKey: ["run", runId] }),
  });
  const rerun = useMutation({
    mutationFn: () => api.rerun(runId),
    onSuccess: ({ run: rerunResult }) => navigate(`/runs/${rerunResult.id}`),
  });
  const run = query.data?.run;
  const completed =
    run?.tasks.filter((task) => TERMINAL.has(task.status)).length ?? 0;
  const progress = run?.tasks.length ? completed / run.tasks.length : 0;
  if (query.isPending) return <div className="page narrow-page"><div className="route-loading" role="status">Loading run and retained task progress…</div></div>;
  if (!run) return <div className="page narrow-page"><div className="error-banner" role="alert"><h1>Run unavailable</h1><p>{query.error?.message ?? "This run could not be loaded."} It may have been deleted or may belong to another account.</p><button className="button secondary" type="button" disabled={query.isFetching} onClick={() => void query.refetch()}>Retry run</button><Link className="button secondary" to="/studies">Back to Projects</Link></div></div>;
  return (
    <div className="page narrow-page">
      {run && (
        <nav className="breadcrumbs" aria-label="Breadcrumb">
          <Link to="/studies">Projects</Link>
          <span>/</span>
          <Link to={`/studies/${run.projectId}`}>
            {run.projectName ?? "Project"}
          </Link>
          <span>/</span>
          <span>
            {run.modelDisplayName ?? "Model"} v{run.modelVersion}
          </span>
        </nav>
      )}
      <div className="page-heading split">
        <div>
          <span className="section-kicker">{TERMINAL.has(run.status) ? "Retained run" : "Live run"}</span>
          <h1>
            {run && TERMINAL.has(run.status)
              ? "Analysis record"
              : "Analysis in progress"}
          </h1>
          <p>
            Each task is independently persisted. Successful evidence survives a
            partial failure.
          </p>
          {run && (
            <p className="evidence-source">
              Evidence source:{" "}
              <strong>
                {run.evidenceSource === "surrogate"
                  ? `promoted surrogate ${run.surrogateModelId?.slice(0, 8)}`
                  : "direct model"}
              </strong>
            </p>
          )}
        </div>
        <div className="run-actions">
          {run && ["queued", "running"].includes(run.status) && (
            <button
              className="button secondary small"
              onClick={() => cancel.mutate()}
              disabled={cancel.isPending}
            >
              <XCircle /> {cancel.isPending ? "Cancelling…" : "Cancel"}
            </button>
          )}
          {run && <StatusBadge status={run.status} />}
          {run && TERMINAL.has(run.status) && (
            <button
              className="button secondary small"
              onClick={() => rerun.mutate()}
              disabled={rerun.isPending}
            >
              <RotateCcw /> {rerun.isPending ? "Starting…" : "Rerun exact"}
            </button>
          )}
        </div>
      </div>
      {query.isError && <div className="error-banner" role="alert"><p>Progress updates stopped. The task states below are the last received snapshot. {query.error.message}</p><button className="button secondary" type="button" disabled={query.isFetching} onClick={() => void query.refetch()}>Retry progress</button></div>}
      {(cancel.isError || rerun.isError) && <div className="error-banner" role="alert">{cancel.isError ? `Cancellation failed. ${cancel.error.message}` : `The exact rerun could not start. ${rerun.error?.message}`} Saved numerical evidence is retained. Review the current status before retrying.</div>}
      {run && (
        <div
          className="study-meta-strip run-meta-strip"
          aria-label="Run provenance"
        >
          <span>
            Created <strong>{new Date(run.createdAt).toLocaleString()}</strong>
          </span>
          <span>
            Source <strong>{run.sourceKind ?? "unknown"}</strong>
          </span>
          <span>
            Seed <strong>{run.seed}</strong>
          </span>
          <span>
            {run.tasks.length} analyses · <strong>{run.accuracyProfile}</strong>
          </span>
        </div>
      )}
      <section className="run-card">
        <div className="run-summary">
          <div
            className="run-progress-ring"
            style={
              { "--progress": `${progress * 360}deg` } as React.CSSProperties
            }
          >
            <span>{Math.round(progress * 100)}%</span>
          </div>
          <div>
            <strong>
              {completed} of {run?.tasks.length ?? 0} tasks complete
            </strong>
            <p>
              Run ID <code>{runId}</code>
            </p>
          </div>
        </div>
        <div className="progress-track">
          <span style={{ width: `${progress * 100}%` }} />
        </div>
        <div className="task-list">
          {run?.tasks.map((task) => {
            const active =
              !TERMINAL.has(run.status) && (task.status === "queued" || task.status === "running");
            const progress = task.progress;
            return (
              <div
                className="task-row"
                data-analysis-key={task.analysisKey}
                key={task.id}
              >
                <span className="task-icon">
                  {task.status === "succeeded" ? (
                    <CircleCheck />
                  ) : task.status === "running" ? (
                    <Cpu className="pulse" />
                  ) : (
                    <TimerReset />
                  )}
                </span>
                <div>
                  <strong>{analysisDisplayName(task.analysisKey)}</strong>
                  <small>
                    {task.result
                      ? `${Math.round(task.result.runtime.duration_ms).toLocaleString()} ms · ${task.result.runtime.model_evaluations.toLocaleString()} evaluations`
                      : (task.error?.message ??
                        progress?.message ??
                        fallbackTaskMessage(task.status))}
                  </small>
                  {active && (
                    <div className="task-progress-detail">
                      <div className="task-progress-meta">
                        <span>
                          {progress?.phase.replaceAll("_", " ") ?? task.status}
                        </span>
                        <span>
                          {progress?.indeterminate
                            ? "Active"
                            : `${progress?.percent ?? 0}%`}
                          {progress && progress.attempt > 0
                            ? ` · retry ${progress.attempt}`
                            : ""}
                        </span>
                      </div>
                      <div
                        className={`task-progress-track ${progress?.indeterminate ? "indeterminate" : ""}`}
                        role="progressbar"
                        aria-label={`${analysisDisplayName(task.analysisKey)} progress`}
                        aria-valuemin={0}
                        aria-valuemax={100}
                        {...(!progress?.indeterminate
                          ? { "aria-valuenow": progress?.percent ?? 0 }
                          : {})}
                      >
                        <span
                          style={{
                            width: `${Math.max(4, progress?.percent ?? 0)}%`,
                          }}
                        />
                      </div>
                    </div>
                  )}
                </div>
                <StatusBadge status={task.status} />
              </div>
            );
          })}
        </div>
        {run?.status === "cancelled" ? (
          <div className="run-cancelled">
            <strong>Run cancelled.</strong>
            <span>Queued analyses were stopped. Any completed numerical evidence is retained below; unfinished tasks have no completed result.</span>
            {run.reportId && <Link className="button secondary" to={`/reports/${run.reportId}`}>Open retained report <ArrowRight /></Link>}
          </div>
        ) : (
          run &&
          TERMINAL.has(run.status) && (
            <div className="run-complete">
              <div>
                <strong>{run.status === "failed" ? "The run failed. Its failure record is ready." : run.status === "partially_succeeded" ? "The report is ready with partial results." : "The report is ready."}</strong>
                <span>
                  {run.status === "succeeded" ? "All numerical results and provenance have been persisted." : "Completed results and task errors are retained separately for inspection."}
                </span>
                <span>{run.tasks.filter((task) => task.status === "succeeded").length} successful · {run.tasks.filter((task) => task.status === "failed").length} failed · {run.tasks.length} total analyses</span>
              </div>
              <Link className="button primary" to={`/reports/${run.id}`}>
                Open report <ArrowRight />
              </Link>
            </div>
          )
        )}
      </section>
      {run.status === "cancelled" && run.tasks.filter((task) => task.result).map((task) => <section className="report-section" key={task.id}><h2>Retained {analysisDisplayName(task.analysisKey)} evidence</h2><ResultView result={task.result!} /></section>)}
    </div>
  );
}
