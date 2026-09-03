import { useQuery } from "@tanstack/react-query";
import { ArrowRight, Braces, FlaskConical, Plus } from "lucide-react";
import { Link, useParams } from "react-router-dom";

import { api } from "../api";
import { ProjectNav } from "../components/ProjectNav";
import { EmptyState, StatusBadge } from "../components/Status";

export function StudyDetail() {
  const { projectId = "" } = useParams();
  const projectsQuery = useQuery({ queryKey: ["projects"], queryFn: api.listProjects });
  const project = projectsQuery.data?.projects.find((item) => item.id === projectId);
  const projectIsAvailable = Boolean(projectId && project);
  const modelsQuery = useQuery({ queryKey: ["models", projectId], queryFn: () => api.listModels(projectId), enabled: projectIsAvailable });
  const runsQuery = useQuery({ queryKey: ["runs"], queryFn: api.listRuns, enabled: projectIsAvailable, refetchInterval: 5_000 });
  const models = modelsQuery.data?.modelVersions ?? [];
  const runs = (runsQuery.data?.runs ?? []).filter((run) => run.projectId === projectId);

  if (projectsQuery.isPending) {
    return <div className="route-loading">Loading project…</div>;
  }
  if (projectsQuery.isError || !project) {
    return (
      <div className="page auth-required-page">
        <section className="auth-required-card">
          <span className="section-kicker">Project unavailable</span>
          <h1>This project is not in your workspace.</h1>
          <p>
            It may have been deleted or it may belong to another account. If
            you are investigating application activity, open it from the
            operator dashboard instead.
          </p>
          <Link className="button secondary" to="/studies">Back to Projects</Link>
        </section>
      </div>
    );
  }
  if (modelsQuery.isPending || runsQuery.isPending) {
    return <div className="route-loading">Loading retained project data…</div>;
  }
  if (modelsQuery.isError || runsQuery.isError) {
    return (
      <div className="page">
        <div className="error-banner" role="alert">
          This project’s retained models or runs could not be loaded. Return to
          Projects and retry.
        </div>
      </div>
    );
  }

  return (
    <div className="page">
      <nav className="breadcrumbs" aria-label="Breadcrumb">
        <Link to="/studies">Projects</Link><span>/</span><span>{project?.name ?? "Project"}</span>
      </nav>
      <ProjectNav projectId={projectId} projectName={project?.name} />
      <div className="page-heading split">
        <div>
          <span className="section-kicker">Project</span>
          <h1>{project?.name ?? "Loading project…"}</h1>
          <p>{project?.description || "Saved models and previous numerical runs."}</p>
        </div>
        <Link className="button primary" to={`/studies/${projectId}/workspace`}><Plus /> New analysis in this project</Link>
      </div>
      <section className="activity-section compact-activity">
        <div className="section-copy"><span className="section-kicker">Saved models</span><h2>Models ready to reuse</h2><p>Open a saved model to edit it or begin another analysis.</p></div>
        {models.length ? (
          <div className="saved-model-list">
            {models.map((model) => (
              <Link className="saved-model-row" to={`/studies/${projectId}/workspace?sourceModel=${model.id}`} key={model.id}>
                <Braces />
                <div><strong>{model.displayName}</strong><small>{model.metadata.input_dimension} inputs · saved {new Date(model.createdAt).toLocaleString()}</small></div>
                <span>Edit and analyse</span><ArrowRight />
              </Link>
            ))}
          </div>
        ) : <EmptyState title="No saved models" body="Define and validate the first model in this project." />}
      </section>
      <section className="activity-section compact-activity">
        <div className="section-copy"><span className="section-kicker">Previous runs</span><h2>Numerical results</h2><p>Open any previous execution to inspect its retained report.</p></div>
        {runs.length ? (
          <div className="activity-runs">
            {runs.map((run) => (
              <Link className="activity-run" to={["queued", "running"].includes(run.status) ? `/runs/${run.id}` : `/reports/${run.id}`} key={run.id}>
                <FlaskConical />
                <div><strong>{run.modelDisplayName}</strong><small>{run.tasks.map((task) => task.analysisKey.replaceAll("_", " ")).join(", ")} · {new Date(run.createdAt).toLocaleString()}</small></div>
                <StatusBadge status={run.status} /><ArrowRight />
              </Link>
            ))}
          </div>
        ) : <EmptyState title="No previous runs" body="Run analyses from a validated model to see results here." />}
      </section>
    </div>
  );
}
