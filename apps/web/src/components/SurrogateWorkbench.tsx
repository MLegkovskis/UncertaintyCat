import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { ModelVersion, SurrogateModel } from "@uncertaintycat/contracts";
import { ArrowRight, CheckCircle2 } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router-dom";

import { api } from "../api";
import { ResultView } from "./ResultView";

export type GprKernel = "MATERN_1_5" | "MATERN_2_5" | "SQUARED_EXPONENTIAL";
export type GprTrend = "CONSTANT" | "LINEAR";

export function SurrogateWorkbench({ model, projectId }: { model: ModelVersion; projectId: string }) {
  const client = useQueryClient();
  const [method, setMethod] = useState<"pce" | "gpr">("gpr");
  const [sampleSize, setSampleSize] = useState(256);
  const [validationSize, setValidationSize] = useState(256);
  const [sparse, setSparse] = useState(true);
  const [outputTarget, setOutputTarget] = useState(0);
  const [pceDegree, setPceDegree] = useState(3);
  const [gprKernel, setGprKernel] = useState<GprKernel>("MATERN_2_5");
  const [gprTrend, setGprTrend] = useState<GprTrend>("CONSTANT");
  const [current, setCurrent] = useState<SurrogateModel>();
  const [acknowledge, setAcknowledge] = useState(false);
  const [reason, setReason] = useState("");
  const [error, setError] = useState<string>();
  const query = useQuery({ queryKey: ["surrogates", projectId], queryFn: () => api.listSurrogates(projectId) });
  const previous = (query.data?.surrogates ?? []).filter((item) => item.sourceModelVersionId === model.id);
  const recommendation = model.assessment?.recommendations.find((item) => item.capability === method);
  const incompatibility = recommendation?.status === "incompatible"
    ? recommendation.compatibility_warnings.join(" ") || "This method is incompatible with the selected model."
    : undefined;
  const minimumTraining = method === "gpr" ? 16 : 30;
  const maximumTraining = method === "gpr" ? 512 : 10_000;
  const configurationError = !Number.isInteger(sampleSize) || sampleSize < minimumTraining || sampleSize > maximumTraining
    ? `Choose ${minimumTraining}–${maximumTraining.toLocaleString()} whole training evaluations for ${method.toUpperCase()}.`
    : !Number.isInteger(validationSize) || validationSize < 20 || validationSize > 2_000
      ? "Choose 20–2,000 whole independent validation evaluations."
      : method === "pce" && (!Number.isInteger(pceDegree) || pceDegree < 1 || pceDegree > 12)
        ? "Choose a whole PCE degree from 1 to 12."
        : method === "gpr" && gprTrend === "LINEAR" && sampleSize <= model.metadata.input_dimension + 1
          ? `A linear GPR trend requires more than ${model.metadata.input_dimension + 1} training points for this model.`
          : undefined;
  const selectCandidate = (candidate?: SurrogateModel) => {
    setCurrent(candidate);
    setAcknowledge(candidate?.acknowledgement?.acknowledgeOverride ?? false);
    setReason(candidate?.acknowledgement?.reason ?? "");
    setError(undefined);
  };
  const build = useMutation({
    mutationFn: () => {
      if (incompatibility || configurationError) throw new Error(incompatibility ?? configurationError);
      return api.createSurrogate(model.id, {
      method,
      config: method === "pce"
        ? { degree: pceDegree, training_size: sampleSize, validation_size: validationSize, sparse }
        : { training_size: sampleSize, validation_size: validationSize, kernel: gprKernel, trend: gprTrend },
      outputTarget,
      seed: 42,
    });
    },
    onSuccess: async ({ surrogate }) => {
      setCurrent(surrogate); setAcknowledge(false); setReason(""); setError(undefined);
      await client.invalidateQueries({ queryKey: ["surrogates", projectId] });
    },
    onError: (caught) => setError(caught instanceof Error ? caught.message : "Surrogate build failed."),
  });
  const promote = useMutation({
    mutationFn: () => api.promoteSurrogate(current?.id ?? "", { acknowledgeOverride: acknowledge, reason }),
    onSuccess: async ({ surrogate }) => {
      setCurrent(surrogate); setError(undefined);
      await client.invalidateQueries({ queryKey: ["surrogates", projectId] });
    },
    onError: (caught) => setError(caught instanceof Error ? caught.message : "Promotion failed."),
  });
  const guidance = current?.validation.guidance;

  return (
    <section className="surrogate-workbench">
      <div className="section-copy"><span className="section-kicker">Selected source model</span><h2>{model.displayName}</h2><p>A candidate keeps its own validation evidence. Promotion creates a retained approximation without replacing this source model.</p></div>
      <div className="surrogate-guidance">
        <span>Measured direct projection <strong>{Math.round(model.assessment?.profile.projected_1000_evaluation_runtime_ms ?? 0).toLocaleString()} ms / 1,000 evaluations</strong></span>
        <span>Promotion guidance <strong>Q²/R² ≥ 0.95</strong> and <strong>normalized RMSE ≤ 0.10</strong></span>
      </div>
      <div className="surrogate-controls">
        <label><span>Method</span><select aria-label="Method" value={method} onChange={(event) => setMethod(event.target.value as "pce" | "gpr")}><option value="gpr">Gaussian process regression</option><option value="pce">Polynomial chaos expansion</option></select></label>
        <label><span>Training budget</span><input type="number" min={minimumTraining} max={maximumTraining} aria-describedby="surrogate-budget-help" value={sampleSize} onChange={(event) => setSampleSize(Number(event.target.value))} /></label>
        <label><span>Validation budget</span><input type="number" min="20" max="2000" aria-describedby="surrogate-budget-help" value={validationSize} onChange={(event) => setValidationSize(Number(event.target.value))} /></label>
        {model.metadata.output_dimension > 1 && <label><span>Output</span><select value={outputTarget} onChange={(event) => setOutputTarget(Number(event.target.value))}>{model.metadata.outputs.map((output) => <option key={output.index} value={output.index}>{output.name}</option>)}</select></label>}
        {method === "pce" ? <label><span>PCE total degree</span><input type="number" min="1" max="12" aria-describedby="surrogate-method-help" value={pceDegree} onChange={(event) => setPceDegree(Number(event.target.value))} /></label> : <><label><span>GPR kernel</span><select aria-describedby="surrogate-method-help" value={gprKernel} onChange={(event) => setGprKernel(event.target.value as GprKernel)}><option value="MATERN_1_5">Matérn 3/2</option><option value="MATERN_2_5">Matérn 5/2</option><option value="SQUARED_EXPONENTIAL">Squared exponential</option></select></label><label><span>GPR trend</span><select aria-describedby="surrogate-method-help" value={gprTrend} onChange={(event) => setGprTrend(event.target.value as GprTrend)}><option value="CONSTANT">Constant</option><option value="LINEAR">Linear</option></select></label></>}
      </div>
      {method === "pce" && <label className="confirmation-check"><input type="checkbox" checked={sparse} onChange={(event) => setSparse(event.target.checked)} /><span>Use sparse polynomial selection</span></label>}
      <p id="surrogate-budget-help" className="muted-copy">Training evaluates the source model to fit the candidate; validation uses a separate hold-out sample. {method === "gpr" ? "Exact GPR allows 16–512 training points because fitting cost grows cubically." : "This studio allows 30–10,000 training points for PCE."} Validation allows 20–2,000 points. The requested total is {(sampleSize + validationSize).toLocaleString()} source-model evaluations.</p>
      <p id="surrogate-method-help" className="muted-copy">{method === "gpr" ? "The kernel describes response smoothness: Matérn 3/2 is less smooth than Matérn 5/2; squared exponential assumes a very smooth response. The constant or linear trend describes the broad response before the Gaussian-process correction." : "Total degree (1–12) limits the polynomial basis; higher degree can increase fitting cost and overfitting. Sparse selection retains a subset of that basis. Independent continuous inputs are required."}</p>
      {(incompatibility || configurationError) && <p className="inline-error" role="alert">{incompatibility ?? configurationError}</p>}
      <button className="button primary" onClick={() => build.mutate()} disabled={build.isPending || Boolean(incompatibility || configurationError)}>{build.isPending ? "Building and validating…" : `Build ${method.toUpperCase()} candidate`}</button>
      {current && guidance && (
        <div className={`surrogate-validation ${guidance.meetsDefault ? "accepted" : "review"}`}>
          <div><span>{current.method === "pce" ? "Hold-out Q²" : "Hold-out R²"}</span><strong>{guidance.score.toPrecision(5)}</strong></div>
          <div><span>Normalized RMSE</span><strong>{guidance.normalizedRmse.toPrecision(5)}</strong></div>
          <div><span>Guidance</span><strong>{guidance.meetsDefault ? "Meets default" : "Override required"}</strong></div>
          <p>{current.method.toUpperCase()} {current.status === "promoted" ? "promoted surrogate" : "candidate"} · {model.metadata.outputs.find((output) => output.index === current.validation.outputTargets[0])?.name ?? "Selected output"} · created {new Date(current.createdAt).toLocaleString()}</p>
          {current.acknowledgement?.acknowledgeOverride && <p><strong>Recorded promotion override:</strong> {current.acknowledgement.reason}</p>}
          <details className="surrogate-evidence" open><summary>Independent hold-out evidence</summary><ResultView result={current.validation.result} /></details>
          {!guidance.meetsDefault && current.status !== "promoted" && <><label className="confirmation-check"><input type="checkbox" checked={acknowledge} onChange={(event) => setAcknowledge(event.target.checked)} /><span>I acknowledge the validation is below the default promotion guidance.</span></label><label><span>Recorded reason</span><input minLength={10} maxLength={1000} value={reason} onChange={(event) => setReason(event.target.value)} placeholder="Why this approximation is acceptable…" /><small>Enter at least 10 characters; this reason is retained with the promoted surrogate.</small></label></>}
          {current.status === "promoted" ? (
            <div className="promoted-next-step"><CheckCircle2 /><div><strong>Surrogate promoted</strong><small>Choose whether this validated approximation remains with its source model or starts a separate investigation.</small></div><div className="model-handoff-actions"><Link className="model-handoff-option primary-option" to={`/studies/${projectId}/workspace?sourceModel=${model.id}&surrogate=${current.id}`}><span>Continue in this project</span><strong>Start a new analysis with this surrogate</strong><small>Use the promoted surrogate with its source model and existing project evidence.</small><ArrowRight /></Link><Link className="model-handoff-option" to={`/studies?new=1&sourceModel=${encodeURIComponent(model.id)}&surrogate=${encodeURIComponent(current.id)}&suggestedName=${encodeURIComponent(`${model.displayName} surrogate study`)}`}><span>Separate investigation</span><strong>Start a new project with this surrogate</strong><small>Copy the exact source model, validated OpenTURNS artifact, and provenance.</small><ArrowRight /></Link></div></div>
          ) : <button className="button primary" onClick={() => promote.mutate()} disabled={promote.isPending || (!guidance.meetsDefault && (!acknowledge || reason.trim().length < 10))}>{promote.isPending ? "Serializing OpenTURNS XML…" : "Promote validated surrogate"}</button>}
        </div>
      )}
      {previous.length > 0 && <label><span>Retained surrogate evidence</span><select aria-label="Retained surrogate evidence" disabled={build.isPending || promote.isPending} value={current?.id ?? ""} onChange={(event) => selectCandidate(previous.find((item) => item.id === event.target.value))}><option value="">Choose a candidate or promoted surrogate…</option>{previous.map((item) => <option key={item.id} value={item.id}>{item.method.toUpperCase()} · {item.status === "promoted" ? "promoted surrogate" : "candidate"} · {new Date(item.createdAt).toLocaleString()}</option>)}</select></label>}
      {query.isError && <p className="inline-error" role="alert">Retained surrogate evidence could not be loaded. <button className="button secondary" onClick={() => void query.refetch()}>Retry surrogate history</button></p>}
      {error && <div className="inline-error" role="alert">{error}</div>}
    </section>
  );
}
