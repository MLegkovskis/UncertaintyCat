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
import { StatusBadge } from "../components/Status";

const TERMINAL = new Set([
  "succeeded",
  "partially_succeeded",
  "failed",
  "cancelled",
]);

export function RunPage() {
  const { runId = "" } = useParams();
  const navigate = useNavigate();
  const client = useQueryClient();
  const query = useQuery({
    queryKey: ["run", runId],
    queryFn: () => api.getRun(runId),
    refetchInterval: (state) =>
      state.state.data && TERMINAL.has(state.state.data.run.status)
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
  return (
    <div className="page narrow-page">
      {run && (
        <nav className="breadcrumbs" aria-label="Breadcrumb">
          <Link to="/studies">Studies</Link><span>/</span>
          <Link to={`/studies/${run.projectId}`}>{run.projectName ?? "Study"}</Link><span>/</span>
          <span>{run.modelDisplayName ?? "Model"} v{run.modelVersion}</span>
        </nav>
      )}
      <div className="page-heading split">
        <div>
          <span className="section-kicker">Live run</span>
          <h1>{run && TERMINAL.has(run.status) ? "Analysis record" : "Analysis in progress"}</h1>
          <p>
            Each task is independently persisted. Successful evidence survives a
            partial failure.
          </p>
          {run && <p className="evidence-source">Evidence source: <strong>{run.evidenceSource === "surrogate" ? `promoted surrogate ${run.surrogateModelId?.slice(0, 8)}` : "direct model"}</strong></p>}
        </div>
        <div className="run-actions">
          {run && ["queued", "running"].includes(run.status) && (
            <button
              className="button secondary small"
              onClick={() => cancel.mutate()}
              disabled={cancel.isPending}
            >
              <XCircle /> Cancel
            </button>
          )}
          {run && <StatusBadge status={run.status} />}
          {run && TERMINAL.has(run.status) && (
            <button
              className="button secondary small"
              onClick={() => rerun.mutate()}
              disabled={rerun.isPending}
            >
              <RotateCcw /> Rerun exact
            </button>
          )}
        </div>
      </div>
      {run && (
        <div className="study-meta-strip run-meta-strip" aria-label="Run provenance">
          <span>Created <strong>{new Date(run.createdAt).toLocaleString()}</strong></span>
          <span>Source <strong>{run.sourceKind ?? "unknown"}</strong></span>
          <span>Seed <strong>{run.seed}</strong></span>
          <span>{run.tasks.length} analyses · <strong>{run.accuracyProfile}</strong></span>
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
          {run?.tasks.map((task) => (
            <div className="task-row" key={task.id}>
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
                <strong>{task.analysisKey.replaceAll("_", " ")}</strong>
                <small>
                  {task.result
                    ? `${Math.round(task.result.runtime.duration_ms).toLocaleString()} ms · ${task.result.runtime.model_evaluations.toLocaleString()} evaluations`
                    : (task.error?.message ?? "Waiting for compute capacity")}
                </small>
              </div>
              <StatusBadge status={task.status} />
            </div>
          ))}
        </div>
        {run?.status === "cancelled" ? (
          <div className="run-cancelled">
            <strong>Run cancelled.</strong>
            <span>Queued analyses were stopped; no report was generated.</span>
          </div>
        ) : (
          run &&
          TERMINAL.has(run.status) && (
            <div className="run-complete">
              <div>
                <strong>The report is ready.</strong>
                <span>
                  All numerical results and provenance have been persisted.
                </span>
              </div>
              <Link className="button primary" to={`/reports/${run.id}`}>
                Open report <ArrowRight />
              </Link>
            </div>
          )
        )}
      </section>
    </div>
  );
}
