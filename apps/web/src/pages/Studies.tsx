import { useQuery } from "@tanstack/react-query";
import { ArrowRight, FolderKanban, Plus, Search } from "lucide-react";
import { useMemo, useState } from "react";
import { Link } from "react-router-dom";

import { api } from "../api";
import { EmptyState } from "../components/Status";

export function Studies() {
  const [search, setSearch] = useState("");
  const projectsQuery = useQuery({ queryKey: ["projects"], queryFn: api.listProjects });
  const projects = projectsQuery.data?.projects ?? [];
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
          <span className="section-kicker">Projects</span>
          <h1>Your uncertainty projects.</h1>
          <p>Each project keeps its saved models and previous numerical runs together.</p>
        </div>
        <div className="heading-actions">
          <label className="study-search">
            <Search aria-hidden="true" />
            <span className="sr-only">Search projects</span>
            <input aria-label="Search projects" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search projects" />
          </label>
          <Link className="button primary" to="/new-analysis"><Plus /> New analysis</Link>
        </div>
      </div>
      {visible.length ? (
        <div className="project-list" role="list">
          {visible.map((project) => (
            <Link to={`/studies/${project.id}`} className="project-row" key={project.id} role="listitem">
              <FolderKanban aria-hidden="true" />
              <div>
                <strong>{project.name}</strong>
                <span>{project.description || "Uncertainty analysis project"}</span>
              </div>
              <time dateTime={project.updatedAt}>Updated {new Date(project.updatedAt).toLocaleString()}</time>
              <ArrowRight aria-hidden="true" />
            </Link>
          ))}
        </div>
      ) : (
        <EmptyState title={projects.length ? "No matching projects" : "No projects yet"} body={projects.length ? "Try a different search term." : "Create a project from New analysis to begin."} />
      )}
    </div>
  );
}
