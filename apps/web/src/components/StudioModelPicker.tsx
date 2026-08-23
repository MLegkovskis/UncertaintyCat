import { useQuery } from "@tanstack/react-query";
import type { ExampleCatalogEntry } from "@uncertaintycat/contracts";
import { ArrowRight, Braces } from "lucide-react";
import { useEffect } from "react";
import { Link } from "react-router-dom";

import { api } from "../api";
import { EmptyState } from "./Status";

export function StudioModelPicker({
  projectId,
  modelId,
  onProjectChange,
  onModelChange,
  returnTo,
}: {
  projectId: string;
  modelId: string;
  onProjectChange: (id: string) => void;
  onModelChange: (id: string) => void;
  returnTo: "dimension-reduction" | "surrogates";
}) {
  const projectsQuery = useQuery({ queryKey: ["projects"], queryFn: api.listProjects });
  const examplesQuery = useQuery({ queryKey: ["examples"], queryFn: api.examples });
  const projects = projectsQuery.data?.projects ?? [];
  const activeProjectId = projectId || projects[0]?.id || "";
  const modelsQuery = useQuery({
    queryKey: ["models", activeProjectId],
    queryFn: () => api.listModels(activeProjectId),
    enabled: Boolean(activeProjectId),
  });
  const models = modelsQuery.data?.modelVersions ?? [];

  useEffect(() => {
    if (!projectId && projects[0]) onProjectChange(projects[0].id);
  }, [onProjectChange, projectId, projects]);
  useEffect(() => {
    if (models.length && !models.some((model) => model.id === modelId)) {
      onModelChange(models[0]!.id);
    }
  }, [modelId, models, onModelChange]);

  return (
    <section className="studio-model-picker">
      <div className="picker-fields">
        <label><span>Project</span><select value={activeProjectId} onChange={(event) => { onProjectChange(event.target.value); onModelChange(""); }}>{projects.map((project) => <option key={project.id} value={project.id}>{project.name}</option>)}</select></label>
        <label><span>Saved model</span><select value={modelId} onChange={(event) => onModelChange(event.target.value)} disabled={!models.length}>{models.map((model) => <option key={model.id} value={model.id}>{model.displayName}</option>)}</select></label>
      </div>
      {!modelsQuery.isLoading && !models.length && (
        <EmptyState title="No validated model in this project" body="Choose a reference example or define a model in New analysis first." />
      )}
      <details className="studio-example-shortcuts">
        <summary>Start from a reference model</summary>
        <div>
          {(examplesQuery.data?.examples ?? []).slice(0, 8).map((example: ExampleCatalogEntry) => (
            <Link key={example.id} to={`/new-analysis?example=${example.id}&next=${returnTo}`}>
              <Braces /><span><strong>{example.title}</strong><small>{example.domain} · {example.inputDimension} inputs</small></span><ArrowRight />
            </Link>
          ))}
        </div>
      </details>
    </section>
  );
}
