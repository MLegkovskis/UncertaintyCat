import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { ArrowRight, Braces, FlaskConical, Plus } from "lucide-react";
import { Link, useParams } from "react-router-dom";

import { api } from "../api";
import { analysisDisplayName } from "../analysisNames";
import { ProjectNav } from "../components/ProjectNav";
import { EmptyState, StatusBadge } from "../components/Status";

export function StudyDetail() {
  const { projectId = "" } = useParams();
  const projectsQuery = useQuery({ queryKey: ["projects"], queryFn: api.listProjects });
  const project = projectsQuery.data?.projects.find((item) => item.id === projectId);
  const projectIsAvailable = Boolean(projectId && project);
  const modelsQuery = useQuery({ queryKey: ["models", projectId], queryFn: () => api.listModels(projectId), enabled: projectIsAvailable });
  const runsQuery = useInfiniteQuery({
    queryKey: ["runs", projectId],
    queryFn: ({ pageParam }) => api.listRuns(projectId, pageParam),
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (lastPage) => lastPage.nextCursor ?? undefined,
    enabled: projectIsAvailable,
    refetchInterval: (state) => state.state.status !== "error" && state.state.data?.pages.some((page) => page.runs.some((run) => run.status === "queued" || run.status === "running")) ? 5_000 : false,
    refetchOnWindowFocus: true,
  });
  const models = modelsQuery.data?.modelVersions ?? [];
  const runs = runsQuery.data?.pages.flatMap((page) => page.runs) ?? [];

  if (projectsQuery.isPending) {
    return <div className="route-loading" role="status">Loading project…</div>;
  }
  if (projectsQuery.isError) {
    return <div className="page"><div className="error-banner" role="alert"><h1>Project could not be loaded.</h1><p>{projectsQuery.error.message}</p><button className="button secondary" type="button" disabled={projectsQuery.isFetching} onClick={() => void projectsQuery.refetch()}>Retry project</button><Link className="button secondary" to="/studies">Back to Projects</Link></div></div>;
  }
  if (!project) {
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
      <div className="study-meta-strip" aria-label="Retained project objects">
        <span>Saved models <strong>{modelsQuery.isSuccess ? models.length : "…"}</strong></span>
        <span>Runs loaded <strong>{runsQuery.data ? runs.length : "…"}</strong>{runsQuery.hasNextPage ? " · older runs available below" : ""}</span>
      </div>
      <section className="activity-section compact-activity">
        <div className="section-copy"><span className="section-kicker">Saved models</span><h2>Models ready to reuse</h2><p>Open a saved model to edit it or begin another analysis.</p></div>
        {modelsQuery.isPending ? <p role="status">Loading saved models…</p> : modelsQuery.isError ? <div className="error-banner" role="alert"><p>Saved models could not be loaded. {modelsQuery.error.message}</p><button className="button secondary" disabled={modelsQuery.isFetching} onClick={() => void modelsQuery.refetch()}>Retry models</button></div> : models.length ? (
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
        <button className="button secondary small" type="button" disabled={runsQuery.isFetching} onClick={() => void runsQuery.refetch()}>{runsQuery.isFetching && !runsQuery.isFetchingNextPage ? "Refreshing history…" : "Refresh history"}</button>
        {runsQuery.isPending ? <p role="status">Loading previous runs…</p> : runs.length ? (
          <div className="activity-runs">
            {runs.map((run) => (
              <Link className="activity-run" to={["queued", "running", "cancelled"].includes(run.status) ? `/runs/${run.id}` : `/reports/${run.id}`} key={run.id}>
                <FlaskConical />
                <div><strong>{run.modelDisplayName}</strong><small>{run.tasks.map((task) => analysisDisplayName(task.analysisKey)).join(", ")} · {new Date(run.createdAt).toLocaleString()}</small></div>
                <StatusBadge status={run.status} /><ArrowRight />
              </Link>
            ))}
          </div>
        ) : !runsQuery.isError ? <EmptyState title="No previous runs" body="Run analyses from a validated model to see results here." /> : null}
        {runsQuery.isError && <div className="error-banner" role="alert"><p>Previous runs could not be loaded. {runsQuery.error.message}</p><button className="button secondary" disabled={runsQuery.isFetching} onClick={() => void (runsQuery.isFetchNextPageError ? runsQuery.fetchNextPage() : runsQuery.refetch())}>Retry runs</button></div>}
        {runsQuery.hasNextPage && !runsQuery.isError && <button className="button secondary" type="button" disabled={runsQuery.isFetching} onClick={() => void runsQuery.fetchNextPage()}>{runsQuery.isFetchingNextPage ? "Loading older runs…" : "Load older runs"}</button>}
      </section>
    </div>
  );
}
