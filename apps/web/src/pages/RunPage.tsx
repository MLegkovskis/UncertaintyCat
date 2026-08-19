import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowRight,
  CircleCheck,
  Cpu,
  TimerReset,
  XCircle,
} from "lucide-react";
import { Link, useParams } from "react-router-dom";

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
  const run = query.data?.run;
  const completed =
    run?.tasks.filter((task) => TERMINAL.has(task.status)).length ?? 0;
  const progress = run?.tasks.length ? completed / run.tasks.length : 0;
  return (
    <div className="page narrow-page">
      <div className="page-heading split">
        <div>
          <span className="section-kicker">Live run</span>
          <h1>Analysis in progress</h1>
          <p>
            Each task is independently persisted. Successful evidence survives a
            partial failure.
          </p>
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
        </div>
      </div>
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
