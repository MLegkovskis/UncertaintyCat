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
  onModelChange,
  returnTo,
}: {
  projectId: string;
  modelId: string;
  onModelChange: (id: string) => void;
  returnTo: "dimension-reduction" | "surrogates";
}) {
  const projectsQuery = useQuery({ queryKey: ["projects"], queryFn: api.listProjects });
  const examplesQuery = useQuery({ queryKey: ["examples"], queryFn: api.examples });
  const project = projectsQuery.data?.projects.find((item) => item.id === projectId);
  const modelsQuery = useQuery({
    queryKey: ["models", projectId],
    queryFn: () => api.listModels(projectId),
    enabled: Boolean(projectId),
  });
  const models = modelsQuery.data?.modelVersions ?? [];

  useEffect(() => {
    if (models.length && !models.some((model) => model.id === modelId)) {
      onModelChange(models[0]!.id);
    }
  }, [modelId, models, onModelChange]);

  return (
    <section className="studio-model-picker">
      <div className="picker-fields">
        <div className="fixed-project-field"><span>Project</span><strong>{project?.name ?? "Loading project…"}</strong></div>
        <label><span>Saved model</span><select value={modelId} onChange={(event) => onModelChange(event.target.value)} disabled={!models.length}>{models.map((model) => <option key={model.id} value={model.id}>{model.displayName}</option>)}</select></label>
      </div>
      {!modelsQuery.isLoading && !models.length && (
        <EmptyState title="No validated model in this project" body="Add a Python model or choose a reference model from Model & analyses first." />
      )}
      <details className="studio-example-shortcuts">
        <summary>Start from a reference model</summary>
        <div>
          {(examplesQuery.data?.examples ?? []).slice(0, 8).map((example: ExampleCatalogEntry) => (
            <Link key={example.id} to={`/studies/${projectId}/workspace?example=${example.id}&next=${returnTo}`}>
              <Braces /><span><strong>{example.title}</strong><small>{example.domain} · {example.inputDimension} inputs</small></span><ArrowRight />
            </Link>
          ))}
        </div>
      </details>
    </section>
  );
}
