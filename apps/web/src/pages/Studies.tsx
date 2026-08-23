import { useQueries, useQuery } from "@tanstack/react-query";
import { ArrowRight, FolderKanban, Search } from "lucide-react";
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";

import { api } from "../api";
import { EmptyState, StatusBadge } from "../components/Status";

export function Studies() {
  const [search, setSearch] = useState("");
  const projectsQuery = useQuery({ queryKey: ["projects"], queryFn: api.listProjects });
  const runsQuery = useQuery({ queryKey: ["runs"], queryFn: api.listRuns, refetchInterval: 5_000 });
  const projects = projectsQuery.data?.projects ?? [];
  const modelQueries = useQueries({
    queries: projects.map((project) => ({
      queryKey: ["models", project.id],
      queryFn: () => api.listModels(project.id),
      staleTime: 30_000,
    })),
  });
  const runs = runsQuery.data?.runs ?? [];
  const visible = useMemo(() => {
    const needle = search.trim().toLocaleLowerCase();
    return projects.filter((project) =>
      !needle || `${project.name} ${project.description}`.toLocaleLowerCase().includes(needle),
    );
  }, [projects, search]);

  return (
    <div className="page">
      <div className="page-heading split">
        <div>
          <span className="section-kicker">Studies</span>
          <h1>One home for durable evidence.</h1>
          <p>Search model versions, executions, datasets, surrogates, and reports by study.</p>
        </div>
        <label className="study-search">
          <Search aria-hidden="true" />
          <span className="sr-only">Search studies</span>
          <input
            aria-label="Search studies"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search studies"
          />
        </label>
      </div>
      {visible.length ? (
        <div className="studies-table" role="list">
          {visible.map((project) => {
            const projectIndex = projects.findIndex((item) => item.id === project.id);
            const lastModel = modelQueries[projectIndex]?.data?.modelVersions[0];
            const lastRun = runs.find((run) => run.projectId === project.id);
            return (
              <Link to={`/studies/${project.id}`} className="study-row" key={project.id} role="listitem">
                <FolderKanban aria-hidden="true" />
                <div className="study-primary">
                  <strong>{project.name}</strong>
                  <span>{project.description || "No description"}</span>
                </div>
                <div><small>Last model</small><strong>{lastModel ? `${lastModel.displayName} · v${lastModel.version}` : "None"}</strong></div>
                <div><small>Last run</small><strong>{lastRun ? new Date(lastRun.createdAt).toLocaleString() : "None"}</strong></div>
                {lastRun ? <StatusBadge status={lastRun.status} /> : <span className="status-badge">Draft</span>}
                <time dateTime={project.updatedAt}>{new Date(project.updatedAt).toLocaleDateString()}</time>
                <ArrowRight aria-hidden="true" />
              </Link>
            );
          })}
        </div>
      ) : (
        <EmptyState
          title={projects.length ? "No matching studies" : "No studies yet"}
          body={projects.length ? "Try a different search term." : "Create a study in New analysis to begin."}
        />
      )}
    </div>
  );
}
