import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowRight, FolderKanban, Plus, Search } from "lucide-react";
import { useMemo, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";

import { api } from "../api";
import { EmptyState } from "../components/Status";

export function Studies() {
  const [search, setSearch] = useState("");
  const [searchParams] = useSearchParams();
  const [creating, setCreating] = useState(searchParams.get("new") === "1");
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const client = useQueryClient();
  const navigate = useNavigate();
  const projectsQuery = useQuery({ queryKey: ["projects"], queryFn: api.listProjects });
  const projects = projectsQuery.data?.projects ?? [];
  const visible = useMemo(() => {
    const needle = search.trim().toLocaleLowerCase();
    return projects.filter((project) =>
      !needle || `${project.name} ${project.description}`.toLocaleLowerCase().includes(needle),
    );
  }, [projects, search]);
  const createProject = useMutation({
    mutationFn: () => api.createProject({ name: name.trim(), description: description.trim() }),
    onSuccess: async ({ project }) => {
      await client.invalidateQueries({ queryKey: ["projects"] });
      navigate(`/studies/${project.id}`);
    },
  });

  return (
    <div className="page">
      <div className="page-heading split">
        <div>
          <span className="section-kicker">Projects</span>
          <h1>Your projects.</h1>
          <p>Each project keeps its models, data, analyses, surrogates, and previous results together.</p>
        </div>
        <div className="heading-actions">
          <label className="study-search">
            <Search aria-hidden="true" />
            <span className="sr-only">Search projects</span>
            <input aria-label="Search projects" value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search projects" />
          </label>
          <button className="button primary" type="button" onClick={() => setCreating(true)}><Plus /> New project</button>
        </div>
      </div>
      {creating && (
        <section className="project-creator" aria-label="Create project">
          <div><span className="section-kicker">New project</span><h2>What are you investigating?</h2><p>You can add models, datasets, and any number of analysis runs after creation.</p></div>
          <label><span>Project name</span><input autoFocus value={name} onChange={(event) => setName(event.target.value)} placeholder="e.g. Turbine blade reliability" /></label>
          <label><span>Description <small>optional</small></span><input value={description} onChange={(event) => setDescription(event.target.value)} placeholder="Purpose, system, or decision being studied" /></label>
          <div className="project-creator-actions">
            <button className="button secondary" type="button" onClick={() => { setCreating(false); setName(""); setDescription(""); }}>Cancel</button>
            <button className="button primary" type="button" disabled={!name.trim() || createProject.isPending} onClick={() => createProject.mutate()}>{createProject.isPending ? "Creating…" : "Create project"} <ArrowRight /></button>
          </div>
          {createProject.isError && <div className="inline-error" role="alert">{createProject.error instanceof Error ? createProject.error.message : "Project creation failed."}</div>}
        </section>
      )}
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
        <div className="projects-empty">
          <EmptyState title={projects.length ? "No matching projects" : "No projects yet"} body={projects.length ? "Try a different search term." : "Create a project, then add your first Python model."} />
          {!projects.length && <button className="button primary" type="button" onClick={() => setCreating(true)}><Plus /> Create first project</button>}
        </div>
      )}
    </div>
  );
}
