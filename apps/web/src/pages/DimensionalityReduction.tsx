import { useMutation, useQuery } from "@tanstack/react-query";
import { ArrowRight, Play, ScanSearch } from "lucide-react";
import { useState } from "react";
import { useNavigate, useParams, useSearchParams } from "react-router-dom";

import { api } from "../api";
import { ProjectNav } from "../components/ProjectNav";
import { StudioModelPicker } from "../components/StudioModelPicker";

export function DimensionalityReduction() {
  const navigate = useNavigate();
  const { projectId = "" } = useParams();
  const [searchParams, setSearchParams] = useSearchParams();
  const modelId = searchParams.get("modelId") ?? "";
  const setModelId = (id: string) => setSearchParams((current) => {
    const next = new URLSearchParams(current);
    next.set("modelId", id);
    return next;
  });
  const [trajectories, setTrajectories] = useState(10);
  const [levels, setLevels] = useState(6);
  const [outputTarget, setOutputTarget] = useState(0);
  const [error, setError] = useState<string>();
  const modelsQuery = useQuery({ queryKey: ["models", projectId], queryFn: () => api.listModels(projectId), enabled: Boolean(projectId) });
  const projectsQuery = useQuery({ queryKey: ["projects"], queryFn: api.listProjects });
  const model = modelsQuery.data?.modelVersions.find((item) => item.id === modelId);
  const project = projectsQuery.data?.projects.find((item) => item.id === projectId);
  const recommendation = model?.assessment?.recommendations.find((item) => item.capability === "morris");
  const incompatibility = recommendation?.status === "incompatible"
    ? recommendation.compatibility_warnings.join(" ") || "Morris screening is incompatible with this model."
    : model?.assessment?.profile.dependent_inputs || model?.metadata.dependent_inputs
      ? "Morris probability-space trajectories require independent inputs."
      : undefined;
  const configurationError = !Number.isInteger(trajectories) || trajectories < 4 || trajectories > 100
    ? "Choose 4–100 whole trajectories for this bounded screening workflow."
    : !Number.isInteger(levels) || levels < 4 || levels > 20
      ? "Choose 4–20 whole grid levels."
      : undefined;
  const selectedOutput = model?.metadata.outputs.some((output) => output.index === outputTarget) ? outputTarget : 0;
  const run = useMutation({
    mutationFn: () => {
      if (!model || incompatibility || configurationError) throw new Error(incompatibility ?? configurationError ?? "Choose a saved model.");
      return api.createRun({
      modelVersionId: modelId,
      analyses: [{ analysisKey: "morris", config: { trajectories, levels }, outputTargets: [selectedOutput] }],
      seed: 42,
      accuracyProfile: "standard",
      idempotencyKey: crypto.randomUUID(),
    });
    },
    onSuccess: ({ run: created }) => navigate(`/runs/${created.id}`),
    onError: (caught) => setError(caught instanceof Error ? caught.message : "Morris screening could not be started."),
  });

  return (
    <div className="page scientific-studio-page">
      <ProjectNav projectId={projectId} projectName={project?.name} />
      <div className="page-heading split">
        <div>
          <span className="section-kicker">Dimensionality Reduction Studio</span>
          <h1>Screen inputs before expensive analysis.</h1>
          <p>Start from a saved Python/OpenTURNS model and produce screening evidence. Use Morris elementary effects to identify potentially negligible, linear, and nonlinear or interacting factors with a comparatively small design.</p>
        </div>
        <a className="button secondary" href="https://openturns.github.io/otmorris/master/user_manual/_generated/otmorris.Morris.html" target="_blank" rel="noreferrer">OTMorris method <ArrowRight /></a>
      </div>
      <div className="scientific-method-note"><ScanSearch /><div><strong>When this route is recommended</strong><p>UncertaintyCat recommends screening first at 15 or more inputs. For 8–14 inputs it remains available as an optional exploration; the original model is never modified automatically.</p></div></div>
      <StudioModelPicker projectId={projectId} modelId={modelId} onModelChange={setModelId} returnTo="dimension-reduction" />
      {model && (
        <section className="method-workbench">
          <div className="section-copy"><span className="section-kicker">Selected model</span><h2>{model.displayName}</h2><p>{model.metadata.input_dimension} inputs · projected {Math.round(model.assessment?.profile.projected_1000_evaluation_runtime_ms ?? 0).toLocaleString()} ms per 1,000 direct evaluations.</p></div>
          <div className="method-controls">
            <label><span>Trajectories</span><input type="number" min="4" max="100" aria-describedby="morris-budget-help" value={trajectories} onChange={(event) => setTrajectories(Number(event.target.value))} /></label>
            <label><span>Grid levels</span><input type="number" min="4" max="20" step="1" aria-describedby="morris-grid-help" value={levels} onChange={(event) => setLevels(Number(event.target.value))} /></label>
            <div><span>Projected evaluations</span><strong>{(trajectories * (model.metadata.input_dimension + 1)).toLocaleString()}</strong></div>
          </div>
          <p id="morris-budget-help" className="muted-copy">Each trajectory evaluates the selected response once at its starting point and once per input. More trajectories improve the screening evidence at greater model-evaluation cost; this studio allows 4–100 (default 10).</p>
          <p id="morris-grid-help" className="muted-copy">Grid levels divide each input's probability range for the elementary-effect steps: 4–20 whole levels (default 6). They are probability-space settings, not physical units.</p>
          {model.metadata.output_dimension > 1 && <label><span>Screened model output</span><select value={selectedOutput} onChange={(event) => setOutputTarget(Number(event.target.value))}>{model.metadata.outputs.map((output) => <option key={output.index} value={output.index}>{output.name}</option>)}</select></label>}
          {(incompatibility || configurationError) && <p className="inline-error" role="alert">{incompatibility ?? configurationError}</p>}
          <p className="method-caveat">Screening is qualitative evidence, not proof that an input is irrelevant. The report asks you to confirm every fixed value before creating a reduced model.</p>
          <button className="button primary" disabled={run.isPending || Boolean(incompatibility || configurationError)} onClick={() => run.mutate()}><Play /> {run.isPending ? "Queuing screening…" : "Run Morris screening"}</button>
        </section>
      )}
      {error && <div className="error-banner" role="alert">{error}</div>}
    </div>
  );
}
