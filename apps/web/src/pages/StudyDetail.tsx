import { useQuery } from "@tanstack/react-query";
import { ArrowRight, Braces, Database, FlaskConical, Plus, Waves } from "lucide-react";
import { Link, useParams } from "react-router-dom";

import { api } from "../api";
import { EmptyState, StatusBadge } from "../components/Status";

export function StudyDetail() {
  const { projectId = "" } = useParams();
  const projectsQuery = useQuery({ queryKey: ["projects"], queryFn: api.listProjects });
  const modelsQuery = useQuery({
    queryKey: ["models", projectId],
    queryFn: () => api.listModels(projectId),
    enabled: Boolean(projectId),
  });
  const runsQuery = useQuery({ queryKey: ["runs"], queryFn: api.listRuns, refetchInterval: 5_000 });
  const sessionQuery = useQuery({ queryKey: ["session-policy"], queryFn: api.session });
  const datasetsQuery = useQuery({ queryKey: ["datasets", projectId], queryFn: () => api.listDatasets(projectId), enabled: Boolean(projectId) && sessionQuery.data?.identity.authenticated === true });
  const surrogatesQuery = useQuery({ queryKey: ["surrogates", projectId], queryFn: () => api.listSurrogates(projectId), enabled: Boolean(projectId) && sessionQuery.data?.identity.authenticated === true });
  const project = projectsQuery.data?.projects.find((item) => item.id === projectId);
  const models = modelsQuery.data?.modelVersions ?? [];
  const runs = (runsQuery.data?.runs ?? []).filter((run) => run.projectId === projectId);
  const datasets = datasetsQuery.data?.datasets ?? [];
  const surrogates = surrogatesQuery.data?.surrogates ?? [];

  return (
    <div className="page">
      <nav className="breadcrumbs" aria-label="Breadcrumb">
        <Link to="/studies">Studies</Link><span>/</span><span>{project?.name ?? "Study"}</span>
      </nav>
      <div className="page-heading split">
        <div>
          <span className="section-kicker">Study detail</span>
          <h1>{project?.name ?? "Loading study…"}</h1>
          <p>{project?.description || "Model versions and chronological numerical evidence."}</p>
        </div>
        <Link className="button primary" to={`/studies/${projectId}/workspace`}>
          <Plus /> Prepare new version
        </Link>
      </div>
      <div className="study-meta-strip">
        <span><Braces /> {models.length} model versions</span>
        <span><FlaskConical /> {runs.length} runs</span>
        <span><Database /> {datasets.length} datasets</span>
        <span><Waves /> {surrogates.length} surrogates</span>
      </div>
      <section className="activity-section">
        <div className="section-copy"><span className="section-kicker">Models</span><h2>Immutable versions</h2></div>
        {models.length ? (
          <div className="model-version-grid">
            {models.map((model) => (
              <article className="model-version-card" key={model.id}>
                <div><strong>{model.displayName}</strong><span>Version {model.version}</span></div>
                <p>{model.metadata.input_dimension} inputs · {model.metadata.output_dimension} outputs · {model.sourceKind}</p>
                <small>{new Date(model.createdAt).toLocaleString()} · {model.sourceHash.slice(0, 12)}</small>
                <Link to={`/studies/${projectId}/workspace?sourceModel=${model.id}`}>Open as new version <ArrowRight /></Link>
              </article>
            ))}
          </div>
        ) : <EmptyState title="No model versions" body="Prepare and validate the first model version." />}
      </section>
      {sessionQuery.data?.identity.authenticated === true && (
        <section className="activity-section">
          <div className="section-copy"><span className="section-kicker">Private assets</span><h2>Datasets and promoted surrogates</h2></div>
          <div className="model-version-grid">
            {datasets.map((dataset) => <article className="model-version-card" key={dataset.id}><div><strong>{dataset.name}</strong><span>Dataset</span></div><p>{dataset.rowCount} rows · {dataset.columns.length} columns</p><Link to={`/data-lab?projectId=${projectId}`}>Open Data Lab <ArrowRight /></Link></article>)}
            {surrogates.map((surrogate) => <article className="model-version-card" key={surrogate.id}><div><strong>{surrogate.method.toUpperCase()} surrogate</strong><StatusBadge status={surrogate.status} /></div><p>Score {surrogate.validation.guidance.score.toPrecision(4)} · normalized RMSE {surrogate.validation.guidance.normalizedRmse.toPrecision(4)}</p>{surrogate.status === "promoted" && <a href={`/api/v1/surrogates/${surrogate.id}/artifact`} download>Download OpenTURNS XML <ArrowRight /></a>}</article>)}
          </div>
        </section>
      )}
      <section className="activity-section">
        <div className="section-copy"><span className="section-kicker">Chronology</span><h2>Runs and reports</h2></div>
        {runs.length ? (
          <div className="activity-runs">
            {runs.map((run) => (
              <Link
                className="activity-run"
                to={["queued", "running"].includes(run.status) ? `/runs/${run.id}` : `/reports/${run.id}`}
                key={run.id}
              >
                <FlaskConical />
                <div>
                  <strong>{run.modelDisplayName} · v{run.modelVersion}</strong>
                  <small>{run.tasks.map((task) => task.analysisKey).join(", ")} · seed {run.seed} · {new Date(run.createdAt).toLocaleString()}</small>
                </div>
                <StatusBadge status={run.status} /><ArrowRight />
              </Link>
            ))}
          </div>
        ) : <EmptyState title="No runs yet" body="Select analyses from a validated model version." />}
      </section>
    </div>
  );
}
