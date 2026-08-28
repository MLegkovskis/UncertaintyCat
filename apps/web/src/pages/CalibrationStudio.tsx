import { useMutation, useQuery } from "@tanstack/react-query";
import { ArrowRight, FileUp, Play, Target } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";

import { api } from "../api";
import {
  MAX_CALIBRATION_PARAMETERS,
  MAX_CALIBRATION_ROWS,
  OFFICIAL_CALIBRATION_CSV,
  isOfficialCalibrationModel,
  parseCalibrationCsv,
} from "../calibration";
import { ProjectNav } from "../components/ProjectNav";
import { StudioModelPicker } from "../components/StudioModelPicker";

const MAX_UPLOAD_BYTES = 250_000;

export function CalibrationStudio() {
  const navigate = useNavigate();
  const { projectId = "" } = useParams();
  const [searchParams] = useSearchParams();
  const [modelId, setModelId] = useState(searchParams.get("modelId") ?? "");
  const [selectedParameters, setSelectedParameters] = useState<number[]>([]);
  const [startingValues, setStartingValues] = useState<Record<number, string>>({});
  const [outputTarget, setOutputTarget] = useState(0);
  const [csv, setCsv] = useState("");
  const [maximumCalls, setMaximumCalls] = useState(250);
  const [error, setError] = useState<string>();
  const projectsQuery = useQuery({ queryKey: ["projects"], queryFn: api.listProjects });
  const examplesQuery = useQuery({ queryKey: ["examples"], queryFn: api.examples });
  const modelsQuery = useQuery({ queryKey: ["models", projectId], queryFn: () => api.listModels(projectId), enabled: Boolean(projectId) });
  const project = projectsQuery.data?.projects.find((item) => item.id === projectId);
  const model = modelsQuery.data?.modelVersions.find((item) => item.id === modelId);
  const officialExampleHash = examplesQuery.data?.examples.find(
    (example) => example.id === "calibration_exponential",
  )?.sha256;
  const officialModel = isOfficialCalibrationModel(model?.sourceHash, officialExampleHash);

  useEffect(() => {
    if (!model) return;
    const official = isOfficialCalibrationModel(model.sourceHash, officialExampleHash);
    const indices = official ? [0, 1, 2] : [];
    setSelectedParameters(indices);
    setStartingValues(Object.fromEntries(model.metadata.inputs.map((item) => [
      item.index,
      String(official && item.index < 3 ? 1 : item.mean ?? 0),
    ])));
    setOutputTarget(0);
    setCsv(official ? OFFICIAL_CALIBRATION_CSV : "");
    setError(undefined);
  }, [model?.id, officialExampleHash]);

  const observedInputNames = useMemo(() => model?.metadata.inputs
    .filter((item) => !selectedParameters.includes(item.index))
    .map((item) => item.name) ?? [], [model, selectedParameters]);
  const outputName = model?.metadata.outputs[outputTarget]?.name ?? "output";
  const parsed = useMemo(() => {
    if (!model || !csv.trim()) return { data: undefined, error: "Paste or upload named observation data." };
    try {
      const data = parseCalibrationCsv(csv, observedInputNames, outputName);
      if (data.outputs.length < selectedParameters.length + 2) {
        return { data: undefined, error: `At least ${selectedParameters.length + 2} observations are required for ${selectedParameters.length} parameters.` };
      }
      if (Math.max(...data.outputs) === Math.min(...data.outputs)) {
        return { data: undefined, error: "Observed output values must vary." };
      }
      return { data, error: undefined };
    } catch (caught) {
      return { data: undefined, error: caught instanceof Error ? caught.message : "Observation data are invalid." };
    }
  }, [csv, model, observedInputNames, outputName, selectedParameters.length]);
  const starts = selectedParameters.map((index) => Number(startingValues[index]));
  const startValuesValid = starts.every(Number.isFinite);
  const modelWithinBound = (model?.metadata.input_dimension ?? 0) <= 32;
  const canRun = Boolean(
    model
    && selectedParameters.length
    && selectedParameters.length <= MAX_CALIBRATION_PARAMETERS
    && parsed.data
    && startValuesValid
    && modelWithinBound
    && maximumCalls >= 10
    && maximumCalls <= 500,
  );

  const run = useMutation({
    mutationFn: () => {
      if (!model || !parsed.data || !canRun) throw new Error(parsed.error ?? "Complete the calibration setup first.");
      return api.createRun({
        modelVersionId: model.id,
        analyses: [{
          analysisKey: "calibration_nlls",
          config: {
            parameter_indices: selectedParameters,
            starting_values: starts,
            observed_input_names: parsed.data.inputNames,
            observed_output_name: outputName,
            observed_inputs: parsed.data.inputs,
            observed_outputs: parsed.data.outputs,
            maximum_calls: maximumCalls,
          },
          outputTargets: [outputTarget],
        }],
        seed: 42,
        accuracyProfile: "standard",
        idempotencyKey: crypto.randomUUID(),
      });
    },
    onSuccess: ({ run: created }) => navigate(`/runs/${created.id}`),
    onError: (caught) => setError(caught instanceof Error ? caught.message : "Calibration could not be started."),
  });

  const toggleParameter = (index: number) => {
    setSelectedParameters((current) => current.includes(index)
      ? current.filter((item) => item !== index)
      : current.length < MAX_CALIBRATION_PARAMETERS ? [...current, index].sort((a, b) => a - b) : current);
  };
  const readFile = async (file?: File) => {
    if (!file) return;
    if (file.size > MAX_UPLOAD_BYTES) {
      setError("Calibration CSV files are limited to 250 KB and 250 observation rows.");
      return;
    }
    setCsv(await file.text());
    setError(undefined);
  };

  return (
    <div className="page scientific-studio-page">
      <ProjectNav projectId={projectId} projectName={project?.name} />
      <div className="page-heading split">
        <div>
          <span className="section-kicker">Calibration Studio</span>
          <h1>Estimate model parameters from observations.</h1>
          <p>Condition the current project model on named explanatory inputs and observed outputs using bounded OpenTURNS nonlinear least squares.</p>
        </div>
        <div className="documentation-links">
          <a className="button secondary" href="https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.NonLinearLeastSquaresCalibration.html" target="_blank" rel="noreferrer">OpenTURNS method <ArrowRight /></a>
          <a className="button secondary" href="https://openturns.github.io/openturns/latest/auto_calibration/least_squares_and_gaussian_calibration/plot_calibration_quickstart.html" target="_blank" rel="noreferrer">Official example <ArrowRight /></a>
        </div>
      </div>
      <div className="scientific-method-note"><Target /><div><strong>Deterministic fit, local uncertainty approximation</strong><p>OpenTURNS supplies every computed value. Approximate parameter intervals use its local linear Gaussian approximation with bootstrap disabled; they are not exact confidence guarantees.</p></div></div>
      <StudioModelPicker projectId={projectId} modelId={modelId} onModelChange={setModelId} returnTo="calibration" />
      {model && (
        <section className="calibration-workbench">
          <div className="section-copy"><span className="section-kicker">Current project model</span><h2>{model.displayName}</h2><p>Select constant unknown parameters. Every remaining model input must appear as a named observation column.</p></div>
          {!modelWithinBound && <div className="error-banner" role="alert">Calibration supports models with at most 32 inputs.</div>}
          <fieldset className="calibration-parameters">
            <legend>Unknown calibration parameters <small>1–{MAX_CALIBRATION_PARAMETERS} continuous inputs</small></legend>
            {model.metadata.inputs.map((input) => {
              const checked = selectedParameters.includes(input.index);
              const continuous = input.kind === "continuous";
              return <div key={input.index} className={checked ? "selected" : ""}>
                <label><input type="checkbox" checked={checked} disabled={!continuous || (!checked && selectedParameters.length >= MAX_CALIBRATION_PARAMETERS)} onChange={() => toggleParameter(input.index)} /><span><strong>{input.name}</strong><small>{continuous ? input.distribution ?? "Continuous" : `${input.kind ?? "Unknown"} · unavailable`}</small></span></label>
                {checked && <label className="starting-value"><span>Starting value</span><input aria-label={`Starting value for ${input.name}`} type="number" value={startingValues[input.index] ?? ""} onChange={(event) => setStartingValues((current) => ({ ...current, [input.index]: event.target.value }))} /></label>}
              </div>;
            })}
          </fieldset>
          <div className="calibration-observation-heading">
            <div><h3>Named observations</h3><p>Expected CSV columns: <code>{[...observedInputNames, outputName].join(", ")}</code>. Column order may vary.</p></div>
            <label><span>Observed model output</span><select value={outputTarget} onChange={(event) => setOutputTarget(Number(event.target.value))}>{model.metadata.outputs.map((output) => <option key={output.index} value={output.index}>{output.name}</option>)}</select></label>
          </div>
          {officialModel ? <div className="calibration-example-note"><strong>Official OpenTURNS exponential example loaded</strong><span>Fixed seed 0 generated these 10 noisy observations for truth a=2.8, b=1.2, c=0.5. Edit or replace them to calibrate your own data.</span><button className="button secondary" onClick={() => { setSelectedParameters([0, 1, 2]); setStartingValues((current) => ({ ...current, 0: "1", 1: "1", 2: "1" })); setOutputTarget(0); setCsv(OFFICIAL_CALIBRATION_CSV); }}>Reload example</button></div> : <div className="calibration-example-note"><strong>Want the verified reference case?</strong><span>Add the four-input exponential calibration model to this project, then return here with its official observations pre-filled.</span><Link className="button secondary" to={`/studies/${projectId}/workspace?example=calibration_exponential&next=calibration`}>Add reference model</Link></div>}
          <div className="calibration-data-entry">
            <label><span>Observed-input and observed-output CSV</span><textarea aria-label="Calibration observation CSV" rows={12} value={csv} onChange={(event) => setCsv(event.target.value)} placeholder={`${[...observedInputNames, outputName].join(",")}\n…`} /></label>
            <div className="calibration-upload"><label className="button secondary"><FileUp /> Upload CSV<input aria-label="Upload calibration CSV" type="file" accept=".csv,text/csv" onChange={(event) => void readFile(event.target.files?.[0])} /></label><small>Maximum {MAX_CALIBRATION_ROWS} rows and 250 KB. All values must be finite.</small></div>
          </div>
          <div className="calibration-run-controls">
            <label><span>Maximum optimizer calls</span><input type="number" min="10" max="500" value={maximumCalls} onChange={(event) => setMaximumCalls(Number(event.target.value))} /></label>
            <div><span>Observation rows</span><strong>{parsed.data?.outputs.length ?? 0} / {MAX_CALIBRATION_ROWS}</strong></div>
            <div><span>Calibrated parameters</span><strong>{selectedParameters.length} / {MAX_CALIBRATION_PARAMETERS}</strong></div>
            <div><span>Observed explanatory inputs</span><strong>{observedInputNames.length}</strong></div>
          </div>
          {(parsed.error || !startValuesValid || !selectedParameters.length) && <div className="inline-error" role="alert">{!selectedParameters.length ? "Select at least one continuous calibration parameter." : !startValuesValid ? "Every selected parameter needs a finite starting value." : parsed.error}</div>}
          <p className="method-caveat">The optimizer is capped at {maximumCalls} calls. The report records the exact OpenTURNS atomic model-evaluation delta, including derivative work. Fit alone does not establish identifiability, causality, or predictive validity outside these observations.</p>
          <button className="button primary" disabled={!canRun || run.isPending} onClick={() => run.mutate()}><Play /> {run.isPending ? "Queuing calibration…" : "Run nonlinear least-squares calibration"}</button>
        </section>
      )}
      {error && <div className="error-banner" role="alert">{error}</div>}
    </div>
  );
}
