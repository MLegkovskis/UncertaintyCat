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
  const build = useMutation({
    mutationFn: () => api.createSurrogate(model.id, {
      method,
      config: method === "pce"
        ? { degree: pceDegree, training_size: Math.max(30, Math.min(sampleSize, 10_000)), validation_size: Math.max(20, Math.min(sampleSize, 2_000)), sparse: true }
        : { training_size: Math.max(16, Math.min(sampleSize, 512)), validation_size: Math.max(20, Math.min(sampleSize, 2_000)), kernel: gprKernel, trend: gprTrend },
      outputTarget,
      seed: 42,
    }),
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
      <div className="surrogate-guidance">
        <span>Measured direct projection <strong>{Math.round(model.assessment?.profile.projected_1000_evaluation_runtime_ms ?? 0).toLocaleString()} ms / 1,000 evaluations</strong></span>
        <span>Promotion guidance <strong>Q²/R² ≥ 0.95</strong> and <strong>normalized RMSE ≤ 0.10</strong></span>
      </div>
      <div className="surrogate-controls">
        <label><span>Method</span><select value={method} onChange={(event) => setMethod(event.target.value as "pce" | "gpr")}><option value="gpr">Gaussian process regression</option><option value="pce">Polynomial chaos expansion</option></select></label>
        <label><span>Training budget</span><input type="number" min="20" max="10000" value={sampleSize} onChange={(event) => setSampleSize(Number(event.target.value))} /></label>
        {model.metadata.output_dimension > 1 && <label><span>Output</span><select value={outputTarget} onChange={(event) => setOutputTarget(Number(event.target.value))}>{model.metadata.outputs.map((output) => <option key={output.index} value={output.index}>{output.name}</option>)}</select></label>}
        {method === "pce" ? <label><span>PCE total degree</span><input type="number" min="1" max="12" value={pceDegree} onChange={(event) => setPceDegree(Number(event.target.value))} /></label> : <><label><span>GPR kernel</span><select value={gprKernel} onChange={(event) => setGprKernel(event.target.value as GprKernel)}><option value="MATERN_1_5">Matérn 3/2</option><option value="MATERN_2_5">Matérn 5/2</option><option value="SQUARED_EXPONENTIAL">Squared exponential</option></select></label><label><span>GPR trend</span><select value={gprTrend} onChange={(event) => setGprTrend(event.target.value as GprTrend)}><option value="CONSTANT">Constant</option><option value="LINEAR">Linear</option></select></label></>}
      </div>
      <button className="button primary" onClick={() => build.mutate()} disabled={build.isPending}>{build.isPending ? "Building and validating…" : `Build ${method.toUpperCase()} candidate`}</button>
      {current && guidance && (
        <div className={`surrogate-validation ${guidance.meetsDefault ? "accepted" : "review"}`}>
          <div><span>{method === "pce" ? "Hold-out Q²" : "Hold-out R²"}</span><strong>{guidance.score.toPrecision(5)}</strong></div>
          <div><span>Normalized RMSE</span><strong>{guidance.normalizedRmse.toPrecision(5)}</strong></div>
          <div><span>Guidance</span><strong>{guidance.meetsDefault ? "Meets default" : "Override required"}</strong></div>
          <details className="surrogate-evidence" open><summary>Independent hold-out evidence</summary><ResultView result={current.validation.result} /></details>
          {!guidance.meetsDefault && <><label className="confirmation-check"><input type="checkbox" checked={acknowledge} onChange={(event) => setAcknowledge(event.target.checked)} /><span>I acknowledge the validation is below the default promotion guidance.</span></label><label><span>Recorded reason</span><input value={reason} onChange={(event) => setReason(event.target.value)} placeholder="Why this approximation is acceptable…" /></label></>}
          {current.status === "promoted" ? (
            <div className="promoted-next-step"><CheckCircle2 /><div><strong>Surrogate promoted</strong><small>Choose whether this validated approximation remains with its source model or starts a separate investigation.</small></div><div className="model-handoff-actions"><Link className="model-handoff-option primary-option" to={`/studies/${projectId}/workspace?sourceModel=${model.id}&surrogate=${current.id}`}><span>Continue in this project</span><strong>Start a new analysis with this surrogate</strong><small>Use the promoted surrogate with its source model and existing project evidence.</small><ArrowRight /></Link><Link className="model-handoff-option" to={`/studies?new=1&sourceModel=${encodeURIComponent(model.id)}&surrogate=${encodeURIComponent(current.id)}&suggestedName=${encodeURIComponent(`${model.displayName} surrogate study`)}`}><span>Separate investigation</span><strong>Start a new project with this surrogate</strong><small>Copy the exact source model, validated OpenTURNS artifact, and provenance.</small><ArrowRight /></Link></div></div>
          ) : <button className="button primary" onClick={() => promote.mutate()} disabled={promote.isPending || (!guidance.meetsDefault && (!acknowledge || reason.trim().length < 10))}>{promote.isPending ? "Serializing OpenTURNS XML…" : "Promote validated surrogate"}</button>}
        </div>
      )}
      {previous.length > 0 && <p className="muted-copy">{previous.length} previous surrogate candidate{previous.length === 1 ? "" : "s"} retained for this model.</p>}
      {error && <div className="inline-error" role="alert">{error}</div>}
    </section>
  );
}
