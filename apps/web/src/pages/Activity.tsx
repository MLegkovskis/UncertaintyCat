import { useQuery } from "@tanstack/react-query";
import { ArrowRight, FlaskConical, FolderKanban } from "lucide-react";
import { Link } from "react-router-dom";

import { api } from "../api";
import { EmptyState, StatusBadge } from "../components/Status";

export function Activity() {
  const projectsQuery = useQuery({
    queryKey: ["projects"],
    queryFn: api.listProjects,
  });
  const runsQuery = useQuery({
    queryKey: ["runs"],
    queryFn: api.listRuns,
    refetchInterval: 5_000,
  });
  const projects = projectsQuery.data?.projects ?? [];
  const runs = runsQuery.data?.runs ?? [];
  return (
    <div className="page">
      <div className="page-heading">
        <span className="section-kicker">Activity</span>
        <h1>Your uncertainty studies</h1>
        <p>
          Projects retain immutable model and run history under your private
          identity.
        </p>
      </div>
      <section className="activity-section">
        <div className="section-copy">
          <span className="section-kicker">Recent runs</span>
          <h2>Numerical evidence</h2>
        </div>
        {runs.length ? (
          <div className="activity-runs">
            {runs.map((run) => (
              <Link
                to={
                  ["succeeded", "partially_succeeded", "failed"].includes(
                    run.status,
                  )
                    ? `/reports/${run.id}`
                    : `/runs/${run.id}`
                }
                className="activity-run"
                key={run.id}
              >
                <FlaskConical />
                <div>
                  <strong>
                    {run.tasks
                      .map((task) => task.analysisKey.replaceAll("_", " "))
                      .join(", ")}
                  </strong>
                  <small>
                    {new Date(run.createdAt).toLocaleString()} · seed {run.seed}{" "}
                    · {run.tasks.length} tasks
                  </small>
                </div>
                <StatusBadge status={run.status} />
                <ArrowRight />
              </Link>
            ))}
          </div>
        ) : (
          <EmptyState
            title="No runs yet"
            body="Validate a model and start an analysis suite in the workspace."
          />
        )}
      </section>
      <section className="activity-section">
        <div className="section-copy">
          <span className="section-kicker">Projects</span>
          <h2>Model workspaces</h2>
        </div>
        {projects.length ? (
          <div className="project-grid">
            {projects.map((project) => (
              <Link to="/workspace" className="project-card" key={project.id}>
                <FolderKanban />
                <div>
                  <strong>{project.name}</strong>
                  <p>{project.description || "No description"}</p>
                  <small>
                    Updated {new Date(project.updatedAt).toLocaleString()}
                  </small>
                </div>
              </Link>
            ))}
          </div>
        ) : (
          <EmptyState
            title="No studies yet"
            body="Create a project in the workspace to begin."
          />
        )}
      </section>
    </div>
  );
}
