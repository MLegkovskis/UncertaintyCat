import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { Project } from "@uncertaintycat/contracts";
import { ArrowRight, FolderKanban, Plus, Search, Trash2, X } from "lucide-react";
import { useMemo, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";

import { api } from "../api";
import { EmptyState } from "../components/Status";

export function Studies() {
  const [search, setSearch] = useState("");
  const [searchParams] = useSearchParams();
  const sourceModelId = searchParams.get("sourceModel") ?? "";
  const surrogateId = searchParams.get("surrogate") ?? "";
  const isModelHandoff = Boolean(sourceModelId);
  const [creating, setCreating] = useState(searchParams.get("new") === "1");
  const [name, setName] = useState(searchParams.get("suggestedName") ?? "");
  const [description, setDescription] = useState("");
  const [projectToDelete, setProjectToDelete] = useState<Project>();
  const [deleteConfirmation, setDeleteConfirmation] = useState("");
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
    mutationFn: async () => {
      const { project: createdProject } = await api.createProject({
        name: name.trim(),
        description: description.trim(),
      });
      if (!sourceModelId) return { project: createdProject };
      try {
        const { definition } = await api.getModelDefinition(sourceModelId);
        const { modelVersion } = await api.createModel(createdProject.id, {
          source: definition.source,
          sourceKind: definition.modelVersion.sourceKind,
          displayName: definition.modelVersion.displayName,
          ...(definition.builderSpec ? { builderSpec: definition.builderSpec } : {}),
          derivation: {
            type: "project_model_handoff",
            sourceModelVersionId: definition.modelVersion.id,
            sourceProjectId: definition.project.id,
          },
        });
        const copiedSurrogate = surrogateId
          ? await api.copySurrogate(surrogateId, {
              targetProjectId: createdProject.id,
              targetModelVersionId: modelVersion.id,
            })
          : undefined;
        return {
          project: createdProject,
          modelVersion,
          surrogate: copiedSurrogate?.surrogate,
        };
      } catch (error) {
        await api.deleteProject(createdProject.id).catch(() => undefined);
        throw error;
      }
    },
    onSuccess: async ({ project, modelVersion, surrogate }) => {
      await client.invalidateQueries({ queryKey: ["projects"] });
      if (modelVersion) {
        const params = new URLSearchParams({ sourceModel: modelVersion.id });
        if (surrogate) params.set("surrogate", surrogate.id);
        navigate(`/studies/${project.id}/workspace?${params.toString()}`);
      } else {
        navigate(`/studies/${project.id}`);
      }
    },
  });
  const deleteProject = useMutation({
    mutationFn: (projectId: string) => api.deleteProject(projectId),
    onSuccess: async () => {
      setProjectToDelete(undefined);
      setDeleteConfirmation("");
      await client.invalidateQueries({ queryKey: ["projects"] });
    },
  });
  const cancelCreator = () => {
    setCreating(false);
    setName("");
    setDescription("");
    if (isModelHandoff) navigate("/studies", { replace: true });
  };

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
          <div><span className="section-kicker">{isModelHandoff ? "Model handoff" : "New project"}</span><h2>{isModelHandoff ? `Start a new project with this ${surrogateId ? "surrogate" : "model"}.` : "What are you investigating?"}</h2><p>{isModelHandoff ? `The complete validated model${surrogateId ? " and promoted surrogate" : ""} will be copied into the new project with explicit provenance.` : "You can add models, datasets, and any number of analysis runs after creation."}</p></div>
          <label><span>Project name</span><input autoFocus value={name} onChange={(event) => setName(event.target.value)} placeholder="e.g. Turbine blade reliability" /></label>
          <label><span>Description <small>optional</small></span><input value={description} onChange={(event) => setDescription(event.target.value)} placeholder="Purpose, system, or decision being studied" /></label>
          <div className="project-creator-actions">
            <button className="button secondary" type="button" onClick={cancelCreator}>Cancel</button>
            <button className="button primary" type="button" disabled={!name.trim() || createProject.isPending} onClick={() => createProject.mutate()}>{createProject.isPending ? (isModelHandoff ? "Copying model…" : "Creating…") : (isModelHandoff ? `Create project with ${surrogateId ? "surrogate" : "model"}` : "Create project")} <ArrowRight /></button>
          </div>
          {createProject.isError && <div className="inline-error" role="alert">{createProject.error instanceof Error ? createProject.error.message : "Project creation failed."}</div>}
        </section>
      )}
      {visible.length ? (
        <div className="project-list" role="list">
          {visible.map((project) => (
            <div className="project-row" key={project.id} role="listitem">
              <FolderKanban aria-hidden="true" />
              <Link className="project-row-main" to={`/studies/${project.id}`}>
                <strong>{project.name}</strong>
                <span>{project.description || "Uncertainty analysis project"}</span>
              </Link>
              <time dateTime={project.updatedAt}>Updated {new Date(project.updatedAt).toLocaleString()}</time>
              <button className="icon-button danger-icon" type="button" aria-label={`Delete ${project.name}`} onClick={() => { setProjectToDelete(project); setDeleteConfirmation(""); }}><Trash2 /></button>
              <Link className="project-row-open" to={`/studies/${project.id}`} aria-label={`Open ${project.name}`}><ArrowRight aria-hidden="true" /></Link>
            </div>
          ))}
        </div>
      ) : (
        <div className="projects-empty">
          <EmptyState title={projects.length ? "No matching projects" : "No projects yet"} body={projects.length ? "Try a different search term." : "Create a project, then add your first Python model."} />
          {!projects.length && <button className="button primary" type="button" onClick={() => setCreating(true)}><Plus /> Create first project</button>}
        </div>
      )}
      {projectToDelete && (
        <div className="dialog-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) setProjectToDelete(undefined); }}>
          <section className="confirmation-dialog" role="dialog" aria-modal="true" aria-labelledby="delete-project-title">
            <button className="dialog-close" type="button" aria-label="Close delete confirmation" onClick={() => setProjectToDelete(undefined)}><X /></button>
            <span className="section-kicker danger-copy">Permanent deletion</span>
            <h2 id="delete-project-title">Delete “{projectToDelete.name}”?</h2>
            <p>This removes every model, dataset, run, report, chat, surrogate, and stored artifact in this project. This action cannot be undone.</p>
            <label><span>Type the project name to confirm</span><input autoFocus value={deleteConfirmation} onChange={(event) => setDeleteConfirmation(event.target.value)} aria-label="Project name confirmation" /></label>
            {deleteProject.isError && <div className="inline-error" role="alert">{deleteProject.error instanceof Error ? deleteProject.error.message : "Project deletion failed."}</div>}
            <div className="dialog-actions">
              <button className="button secondary" type="button" onClick={() => setProjectToDelete(undefined)}>Cancel</button>
              <button className="button danger-button" type="button" disabled={deleteConfirmation !== projectToDelete.name || deleteProject.isPending} onClick={() => deleteProject.mutate(projectToDelete.id)}>{deleteProject.isPending ? "Deleting…" : "Delete project permanently"}</button>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
