import { useMutation, useQuery } from "@tanstack/react-query";
import { ArrowRight, Play, ScanSearch } from "lucide-react";
import { useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";

import { api } from "../api";
import { StudioModelPicker } from "../components/StudioModelPicker";

export function DimensionalityReduction() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [projectId, setProjectId] = useState(searchParams.get("projectId") ?? "");
  const [modelId, setModelId] = useState(searchParams.get("modelId") ?? "");
  const [trajectories, setTrajectories] = useState(10);
  const [levels, setLevels] = useState(6);
  const [error, setError] = useState<string>();
  const modelsQuery = useQuery({ queryKey: ["models", projectId], queryFn: () => api.listModels(projectId), enabled: Boolean(projectId) });
  const model = modelsQuery.data?.modelVersions.find((item) => item.id === modelId);
  const run = useMutation({
    mutationFn: () => api.createRun({
      modelVersionId: modelId,
      analyses: [{ analysisKey: "morris", config: { trajectories, levels }, outputTargets: [0] }],
      seed: 42,
      accuracyProfile: "standard",
      idempotencyKey: crypto.randomUUID(),
    }),
    onSuccess: ({ run: created }) => navigate(`/runs/${created.id}`),
    onError: (caught) => setError(caught instanceof Error ? caught.message : "Morris screening could not be started."),
  });

  return (
    <div className="page scientific-studio-page">
      <div className="page-heading split">
        <div>
          <span className="section-kicker">Dimensionality Reduction Studio</span>
          <h1>Screen inputs before expensive analysis.</h1>
          <p>Use Morris elementary effects to identify potentially negligible, linear, and nonlinear or interacting factors with a comparatively small design.</p>
        </div>
        <a className="button secondary" href="https://openturns.github.io/otmorris/master/user_manual/_generated/otmorris.Morris.html" target="_blank" rel="noreferrer">OTMorris method <ArrowRight /></a>
      </div>
      <div className="scientific-method-note"><ScanSearch /><div><strong>When this route is recommended</strong><p>UncertaintyCat recommends screening first at 15 or more inputs. For 8–14 inputs it remains available as an optional exploration; the original model is never modified automatically.</p></div></div>
      <StudioModelPicker projectId={projectId} modelId={modelId} onProjectChange={setProjectId} onModelChange={setModelId} returnTo="dimension-reduction" />
      {model && (
        <section className="method-workbench">
          <div className="section-copy"><span className="section-kicker">Selected model</span><h2>{model.displayName}</h2><p>{model.metadata.input_dimension} inputs · projected {Math.round(model.assessment?.profile.projected_1000_evaluation_runtime_ms ?? 0).toLocaleString()} ms per 1,000 direct evaluations.</p></div>
          <div className="method-controls">
            <label><span>Trajectories</span><input type="number" min="4" max="100" value={trajectories} onChange={(event) => setTrajectories(Number(event.target.value))} /></label>
            <label><span>Grid levels</span><input type="number" min="4" max="20" step="2" value={levels} onChange={(event) => setLevels(Number(event.target.value))} /></label>
            <div><span>Projected evaluations</span><strong>{(trajectories * (model.metadata.input_dimension + 1)).toLocaleString()}</strong></div>
          </div>
          <p className="method-caveat">Screening is qualitative evidence, not proof that an input is irrelevant. The report asks you to confirm every fixed value before creating a reduced model.</p>
          <button className="button primary" disabled={run.isPending} onClick={() => run.mutate()}><Play /> {run.isPending ? "Queuing screening…" : "Run Morris screening"}</button>
        </section>
      )}
      {error && <div className="error-banner" role="alert">{error}</div>}
    </div>
  );
}
