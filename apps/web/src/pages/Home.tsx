import { useQuery } from "@tanstack/react-query";
import {
  ArrowRight,
  Blocks,
  Braces,
  ChartNoAxesCombined,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import { Link } from "react-router-dom";

import { api } from "../api";

export function Home() {
  const catalog = useQuery({ queryKey: ["catalog"], queryFn: api.catalog });
  const session = useQuery({ queryKey: ["session-policy"], queryFn: api.session });
  const projects = useQuery({
    queryKey: ["projects"],
    queryFn: api.listProjects,
    enabled: session.data?.identity.authenticated === true,
  });
  const runs = useQuery({
    queryKey: ["runs"],
    queryFn: api.listRuns,
    enabled: session.data?.identity.authenticated === true,
  });
  if (session.data?.identity.authenticated) {
    const recentProjects = projects.data?.projects.slice(0, 3) ?? [];
    const recentRuns = runs.data?.runs.slice(0, 4) ?? [];
    const continueRun = recentRuns[0];
    return (
      <div className="page dashboard-page">
        <div className="page-heading split">
          <div>
            <span className="section-kicker">Dashboard</span>
            <h1>Continue your uncertainty work.</h1>
            <p>Recent studies and exact retained configurations are ready to reopen.</p>
          </div>
          <Link className="button primary" to="/new-analysis">New analysis <ArrowRight /></Link>
        </div>
        {continueRun && (
          <Link
            className="continue-card"
            to={["queued", "running"].includes(continueRun.status) ? `/runs/${continueRun.id}` : `/reports/${continueRun.id}`}
          >
            <div><span className="section-kicker">Continue working</span><h2>{continueRun.projectName}</h2></div>
            <p>{continueRun.modelDisplayName} · version {continueRun.modelVersion} · {continueRun.tasks.length} analyses</p>
            <ArrowRight />
          </Link>
        )}
        <section className="dashboard-grid">
          <div>
            <div className="section-copy"><span className="section-kicker">Recent studies</span><h2>Studies</h2></div>
            {recentProjects.map((project) => (
              <Link className="dashboard-row" to={`/studies/${project.id}`} key={project.id}>
                <strong>{project.name}</strong><small>{new Date(project.updatedAt).toLocaleString()}</small><ArrowRight />
              </Link>
            ))}
            {!recentProjects.length && <p className="muted-copy">No retained studies yet.</p>}
          </div>
          <div>
            <div className="section-copy"><span className="section-kicker">Recent executions</span><h2>Runs</h2></div>
            {recentRuns.map((run) => (
              <Link className="dashboard-row" to={["queued", "running"].includes(run.status) ? `/runs/${run.id}` : `/reports/${run.id}`} key={run.id}>
                <strong>{run.modelDisplayName}</strong><small>{run.status} · {new Date(run.createdAt).toLocaleString()}</small><ArrowRight />
              </Link>
            ))}
            {!recentRuns.length && <p className="muted-copy">No numerical runs yet.</p>}
          </div>
        </section>
      </div>
    );
  }
  return (
    <div className="page home-page">
      <section className="hero">
        <div className="eyebrow">
          <Sparkles size={15} /> Open-source uncertainty engineering
        </div>
        <h1>
          Turn uncertain inputs into <em>defensible decisions.</em>
        </h1>
        <p>
          Define an OpenTURNS model, run a traceable analysis suite, and explore
          every assumption, sensitivity index, and convergence result in one
          durable report.
        </p>
        <div className="hero-actions">
          <Link className="button primary" to="/new-analysis">
            Start a new analysis <ArrowRight size={17} />
          </Link>
          <a
            className="button secondary"
            href="https://openturns.github.io/openturns/latest/"
            target="_blank"
            rel="noreferrer"
          >
            OpenTURNS methods
          </a>
        </div>
        <div className="trust-row">
          <span>
            <ShieldCheck /> Isolated Python
          </span>
          <span>
            <Braces /> Versioned results
          </span>
          <span>
            <ChartNoAxesCombined /> Interactive reports
          </span>
        </div>
      </section>
      <section className="architecture-strip">
        <div>
          <strong>{catalog.data?.analyses.length ?? "—"}</strong>
          <span>versioned analysis plugins</span>
        </div>
        <div>
          <strong>23</strong>
          <span>validated reference models</span>
        </div>
        <div>
          <strong>100%</strong>
          <span>numerical provenance retained</span>
        </div>
      </section>
      <section className="feature-section">
        <div className="section-copy">
          <span className="section-kicker">Analysis catalog</span>
          <h2>A focused toolkit, built to expand.</h2>
          <p>
            Each method declares its assumptions, supported model shape,
            configuration schema, and result version. New OpenTURNS capability
            can arrive as a plugin instead of a UI rewrite.
          </p>
        </div>
        <div className="catalog-grid">
          {catalog.isLoading &&
            [1, 2, 3].map((item) => (
              <div className="catalog-card skeleton" key={item} />
            ))}
          {catalog.data?.analyses.map((analysis) => (
            <article className="catalog-card" key={analysis.key}>
              <div className="catalog-icon">
                <Blocks />
              </div>
              <span>{analysis.category}</span>
              <h3>{analysis.name}</h3>
              <p>{analysis.description}</p>
              <footer>
                <code>
                  {analysis.key}@{analysis.version}
                </code>
                <span className={`resource ${analysis.resource_class}`}>
                  {analysis.resource_class}
                </span>
              </footer>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}
