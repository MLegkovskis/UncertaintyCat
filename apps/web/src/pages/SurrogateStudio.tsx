import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { DataSurrogateModel, Dataset } from "@uncertaintycat/contracts";
import type { EChartsOption } from "echarts";
import { ArrowRight, CheckCircle2, Database, FileSpreadsheet, Waves } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useParams, useSearchParams } from "react-router-dom";

import { api } from "../api";
import { EChart } from "../components/EChart";
import { ProjectNav } from "../components/ProjectNav";
import { StudioModelPicker } from "../components/StudioModelPicker";
import { SurrogateWorkbench } from "../components/SurrogateWorkbench";

const EXAMPLE_SURROGATE_CSV = ["x1,x2,response", ...Array.from({ length: 36 }, (_, index) => {
  const x1 = -2 + index * 4 / 35;
  const x2 = ((index * 7) % 17) / 4 - 2;
  const response = Math.sin(x1) + 0.35 * x2 * x2 + 0.08 * x1 * x2;
  return `${x1.toFixed(6)},${x2.toFixed(6)},${response.toFixed(6)}`;
})].join("\n");

async function textBase64(value: string) {
  const bytes = new TextEncoder().encode(value);
  let binary = "";
  bytes.forEach((byte) => { binary += String.fromCharCode(byte); });
  return btoa(binary);
}

function DataSurrogateEvidence({ surrogate }: { surrogate: DataSurrogateModel }) {
  const option: EChartsOption = {
    tooltip: { trigger: "item" },
    xAxis: { type: "value", name: "Observed", scale: true },
    yAxis: { type: "value", name: "Predicted", scale: true },
    series: [
      {
        name: "Hold-out predictions",
        type: "scatter",
        symbolSize: 8,
        data: surrogate.validation.observed.map((value, index) => [value, surrogate.validation.predicted[index]]),
      },
    ],
  };
  return (
    <section className={`data-surrogate-evidence ${surrogate.validation.meetsDefault ? "accepted" : "review"}`}>
      <header><CheckCircle2 /><div><strong>Data-driven GPR retained</strong><small>OpenTURNS {surrogate.openturnsVersion} · independent hold-out set</small></div></header>
      <div className="surrogate-metrics">
        <span><small>Hold-out R²</small><strong>{surrogate.validation.r2.toPrecision(5)}</strong></span>
        <span><small>Normalized RMSE</small><strong>{surrogate.validation.normalizedRmse.toPrecision(5)}</strong></span>
        <span><small>Training / validation</small><strong>{surrogate.validation.trainingSize} / {surrogate.validation.validationSize}</strong></span>
        <span><small>Guidance</small><strong>{surrogate.validation.meetsDefault ? "Meets default" : "Review required"}</strong></span>
      </div>
      <EChart option={option} ariaLabel="Observed versus predicted hold-out values" height={340} />
      <p>This surrogate is fitted from empirical pairs and retained with its XML artifact. Define an input uncertainty distribution before using it for uncertainty propagation or sensitivity analysis.</p>
    </section>
  );
}

function DataSurrogateWorkbench({ projectId }: { projectId: string }) {
  const client = useQueryClient();
  const datasetsQuery = useQuery({ queryKey: ["datasets", projectId], queryFn: () => api.listDatasets(projectId) });
  const previousQuery = useQuery({ queryKey: ["data-surrogates", projectId], queryFn: () => api.listDataSurrogates(projectId) });
  const datasets = datasetsQuery.data?.datasets ?? [];
  const [datasetId, setDatasetId] = useState("");
  const dataset = datasets.find((item) => item.id === datasetId) ?? datasets[0];
  const numericColumns = useMemo(() => dataset?.columns.filter((column) => column.type === "numeric").map((column) => column.name) ?? [], [dataset]);
  const [outputColumn, setOutputColumn] = useState("");
  const [inputColumns, setInputColumns] = useState<string[]>([]);
  const [kernel, setKernel] = useState<"MATERN_1_5" | "MATERN_2_5" | "SQUARED_EXPONENTIAL">("MATERN_2_5");
  const [trend, setTrend] = useState<"CONSTANT" | "LINEAR">("CONSTANT");
  const [pasted, setPasted] = useState(EXAMPLE_SURROGATE_CSV);
  const [current, setCurrent] = useState<DataSurrogateModel>();
  const [error, setError] = useState<string>();

  useEffect(() => {
    if (!dataset || numericColumns.length < 2) return;
    const output = numericColumns.at(-1) ?? "";
    setDatasetId(dataset.id);
    setOutputColumn(output);
    setInputColumns(numericColumns.filter((column) => column !== output));
    setCurrent(undefined);
  }, [dataset?.id, numericColumns.join("|")]);

  const upload = useMutation({
    mutationFn: async () => api.uploadDataset({
      projectId,
      name: "GPR example data",
      sourceKind: "paste",
      contentBase64: await textBase64(pasted),
    }),
    onSuccess: async ({ dataset: created }) => {
      await client.invalidateQueries({ queryKey: ["datasets", projectId] });
      setDatasetId(created.id);
      setError(undefined);
    },
    onError: (caught) => setError(caught instanceof Error ? caught.message : "Dataset validation failed."),
  });
  const build = useMutation({
    mutationFn: () => {
      if (!dataset) throw new Error("Choose a validated dataset first.");
      return api.createDataSurrogate(dataset.id, {
        inputColumns,
        outputColumn,
        validationFraction: 0.2,
        kernel,
        trend,
        seed: 42,
      });
    },
    onSuccess: async ({ surrogate }) => {
      setCurrent(surrogate);
      setError(undefined);
      await client.invalidateQueries({ queryKey: ["data-surrogates", projectId] });
    },
    onError: (caught) => setError(caught instanceof Error ? caught.message : "Data-driven surrogate fitting failed."),
  });

  return (
    <section className="data-surrogate-workbench">
      <div className="data-surrogate-source">
        <div><span className="section-kicker">Paired observations</span><h2>Choose a dataset with input and output columns.</h2><p>Rows with missing or non-finite selected values are excluded before OpenTURNS fitting.</p></div>
        <label><span>Dataset</span><select value={dataset?.id ?? ""} onChange={(event) => setDatasetId(event.target.value)}><option value="">Choose a dataset…</option>{datasets.map((item: Dataset) => <option key={item.id} value={item.id}>{item.name} · {item.rowCount} rows</option>)}</select></label>
      </div>
      {!datasets.length && (
        <div className="surrogate-example-data">
          <FileSpreadsheet />
          <div><strong>No dataset in this project yet</strong><p>The pre-filled nonlinear example has two inputs and one response. Validate it as a private project dataset, or add your own in Distribution fitting.</p></div>
          <textarea aria-label="Example surrogate CSV" rows={8} value={pasted} onChange={(event) => setPasted(event.target.value)} />
          <button className="button primary" disabled={upload.isPending || !pasted.trim()} onClick={() => upload.mutate()}>{upload.isPending ? "Validating data…" : "Add example dataset"}</button>
        </div>
      )}
      {dataset && (
        <>
          <div className="data-surrogate-columns">
            <label><span>Output column</span><select value={outputColumn} onChange={(event) => { const value = event.target.value; setOutputColumn(value); setInputColumns((current) => current.filter((column) => column !== value)); }}>{numericColumns.map((column) => <option key={column} value={column}>{column}</option>)}</select></label>
            <fieldset><legend>Input columns</legend>{numericColumns.filter((column) => column !== outputColumn).map((column) => <label key={column}><input type="checkbox" checked={inputColumns.includes(column)} onChange={() => setInputColumns((current) => current.includes(column) ? current.filter((item) => item !== column) : [...current, column])} /><span>{column}</span></label>)}</fieldset>
            <label><span>Kernel</span><select value={kernel} onChange={(event) => setKernel(event.target.value as typeof kernel)}><option value="MATERN_1_5">Matérn 3/2</option><option value="MATERN_2_5">Matérn 5/2</option><option value="SQUARED_EXPONENTIAL">Squared exponential</option></select></label>
            <label><span>Trend</span><select value={trend} onChange={(event) => setTrend(event.target.value as typeof trend)}><option value="CONSTANT">Constant</option><option value="LINEAR">Linear</option></select></label>
          </div>
          <button className="button primary" disabled={build.isPending || !outputColumn || !inputColumns.length} onClick={() => build.mutate()}>{build.isPending ? "Fitting and validating…" : "Build data-driven GPR"}</button>
        </>
      )}
      {current && <DataSurrogateEvidence surrogate={current} />}
      {!current && (previousQuery.data?.surrogates.length ?? 0) > 0 && <p className="muted-copy">{previousQuery.data?.surrogates.length} previous data-driven surrogate{previousQuery.data?.surrogates.length === 1 ? "" : "s"} retained in this project.</p>}
      {error && <div className="inline-error" role="alert">{error}</div>}
    </section>
  );
}

export function SurrogateStudio() {
  const [searchParams] = useSearchParams();
  const { projectId = "" } = useParams();
  const [modelId, setModelId] = useState(searchParams.get("modelId") ?? "");
  const [sourceMode, setSourceMode] = useState<"model" | "data">("model");
  const modelsQuery = useQuery({ queryKey: ["models", projectId], queryFn: () => api.listModels(projectId), enabled: Boolean(projectId) });
  const projectsQuery = useQuery({ queryKey: ["projects"], queryFn: api.listProjects });
  const model = modelsQuery.data?.modelVersions.find((item) => item.id === modelId);
  const project = projectsQuery.data?.projects.find((item) => item.id === projectId);

  return (
    <div className="page scientific-studio-page">
      <ProjectNav projectId={projectId} projectName={project?.name} />
      <div className="page-heading split">
        <div>
          <span className="section-kicker">Surrogate Studio</span>
          <h1>Approximate a response deliberately.</h1>
          <p>Build from a saved Python model or from paired empirical observations, then inspect independent validation evidence.</p>
        </div>
        <div className="documentation-links">
          <a className="button secondary" href="https://openturns.github.io/openturns/latest/theory/meta_modeling/gaussian_process_regression.html" target="_blank" rel="noreferrer">GPR method <ArrowRight /></a>
          <a className="button secondary" href="https://openturns.github.io/openturns/latest/theory/meta_modeling/functional_chaos.html" target="_blank" rel="noreferrer">Functional chaos <ArrowRight /></a>
        </div>
      </div>
      <div className="scientific-method-note"><Waves /><div><strong>Two valid starting points</strong><p>A model-based surrogate samples a declared input distribution. A data-driven GPR fits directly to paired input/output observations. Both receive an independent hold-out assessment.</p></div></div>
      <div className="source-choice" role="tablist" aria-label="Surrogate source">
        <button role="tab" aria-selected={sourceMode === "model"} className={sourceMode === "model" ? "active" : ""} onClick={() => setSourceMode("model")}><Waves /> From saved model</button>
        <button role="tab" aria-selected={sourceMode === "data"} className={sourceMode === "data" ? "active" : ""} onClick={() => setSourceMode("data")}><Database /> From empirical data</button>
      </div>
      {sourceMode === "model" ? (
        <>
          <StudioModelPicker projectId={projectId} modelId={modelId} onModelChange={setModelId} returnTo="surrogates" />
          {model && <SurrogateWorkbench model={model} projectId={projectId} />}
        </>
      ) : <DataSurrogateWorkbench projectId={projectId} />}
    </div>
  );
}
