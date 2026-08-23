import type {
  Dataset,
  DistributionFitInput,
  DistributionFitRun,
} from "@uncertaintycat/contracts";
import type { EChartsOption } from "echarts";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  Check,
  Database,
  FileSpreadsheet,
  FlaskConical,
  Upload,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";

import { api } from "../api";
import { EmptyState } from "../components/Status";
import { EChart } from "../components/EChart";

const CANDIDATES = [
  "Normal",
  "Uniform",
  "LogNormal",
  "Exponential",
  "Gamma",
  "Beta",
  "Triangular",
  "KernelSmoothing",
] as const;

const BEAM_SAMPLE_CSV = `E,F,L,I
34359951.21260305,21373.816129377272,253.30596872570482,397.88130837467713
28057355.742907558,30617.791560420934,253.61292155138702,403.4412280138856
35431982.11609094,34967.84428462613,253.18989774837783,371.9317609626851
28041948.68744233,21315.022018381096,250.94796999240342,366.9290196269449
36444562.18607961,26905.249605116507,252.8766960385893,377.62193952816585
32950211.408753864,31469.358723163783,251.49617198482918,345.7680011092174
33677615.77853969,26997.73301158137,256.37362522786367,408.4063413130727
31834518.543026865,23339.72271578812,252.35290689487118,359.5817317946926
28161856.421815425,31068.600024391148,259.1431665172396,382.2666962937509
35060848.40812222,27555.065901060505,257.05175477014944,367.28420402594907
33354364.845005997,24828.70269255394,253.50862841887505,345.0763173318328
32650036.582528695,53321.48491239969,252.51017732171073,414.9274064196715
31760377.08707273,34937.858468744904,257.6311315822481,340.28593650735166
32321024.64565009,20504.113025865678,257.4377496313702,373.25560204725014
40480798.857369,22558.58388028789,256.47255779337155,355.2800073040295
40143213.0470579,22229.65890564795,253.61837159264246,397.64449171332876
29241084.37305043,21022.119848015464,255.41206776536532,339.7936733322751
35948475.09559152,30486.29962848546,253.14320123388524,383.7108728726487
40332295.4311597,38288.9911272209,253.06103816644242,327.9031490269908
42878117.60289985,34599.749892857115,253.0475822307583,348.51918605201456
28917407.48150608,32391.62514913473,257.92338089448367,340.0480896185756
31124461.501452688,37326.20297954747,254.36194254454924,339.94152981659533
28102348.784948435,23932.159217643522,256.3260137381918,388.43591698148145
34510293.02632851,27236.86580021063,251.96030298968432,401.81672134221003
30678059.64601908,41836.36403213085,259.8298329366686,331.60670950206725
28304220.129013382,21257.526247229827,255.9109881215413,313.13466368006584
31603114.40831752,21404.793980640985,258.47968745329126,338.4672971089906
32467849.909416374,37964.710300525665,256.2989255933696,347.86857537574633
29982074.795751978,30463.040249978054,254.3339949054922,433.7465660056944
36148382.16508266,33058.36309138205,257.29636883851657,326.3142753120113`;

async function blobBase64(blob: Blob): Promise<string> {
  return await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error ?? new Error("Could not read the dataset."));
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1] ?? "");
    reader.readAsDataURL(blob);
  });
}

function DatasetPreview({ dataset }: { dataset: Dataset }) {
  const columns = dataset.columns.map((column) => column.name);
  return (
    <div className="table-scroll" tabIndex={0} aria-label={`${dataset.name} data preview`}>
      <table className="engineering-table">
        <thead><tr>{columns.map((column) => <th key={column}>{column}</th>)}</tr></thead>
        <tbody>
          {dataset.preview.map((row, index) => (
            <tr key={index}>
              {columns.map((column) => <td key={column}>{String(row[column] ?? "—")}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function FitCharts({ column }: { column: NonNullable<DistributionFitRun["result"]>["columns"][number] }) {
  const sample = column.plot.sample;
  const minimum = Math.min(...sample);
  const maximum = Math.max(...sample);
  const binCount = Math.max(5, Math.min(30, Math.ceil(Math.sqrt(sample.length))));
  const width = maximum > minimum ? (maximum - minimum) / binCount : 1;
  const counts = Array<number>(binCount).fill(0);
  sample.forEach((value) => {
    const index = Math.min(binCount - 1, Math.max(0, Math.floor((value - minimum) / width)));
    counts[index] = (counts[index] ?? 0) + 1;
  });
  const histogram = counts.map((count, index) => [minimum + (index + 0.5) * width, count / sample.length / width]);
  const toolbox = { feature: { dataZoom: {}, restore: {}, saveAsImage: { title: "Export image" } } };
  const density: EChartsOption = {
    tooltip: { trigger: "axis" }, legend: { data: ["Empirical density", "Fitted PDF"] }, toolbox,
    xAxis: { type: "value", scale: true }, yAxis: { type: "value", name: "Density" },
    dataZoom: [{ type: "inside" }],
    series: [
      { name: "Empirical density", type: "bar", data: histogram, barWidth: "95%" },
      { name: "Fitted PDF", type: "line", data: column.plot.pdf.x.map((x, index) => [x, column.plot.pdf.y[index]]), showSymbol: false },
    ],
  };
  const cdf: EChartsOption = {
    tooltip: { trigger: "axis" }, legend: { data: ["Empirical CDF", "Fitted CDF"] }, toolbox,
    xAxis: { type: "value", scale: true }, yAxis: { type: "value", min: 0, max: 1 }, dataZoom: [{ type: "inside" }],
    series: [
      { name: "Empirical CDF", type: "line", step: "end", showSymbol: false, data: column.plot.cdf.empiricalX.map((x, index) => [x, column.plot.cdf.empiricalY[index]]) },
      { name: "Fitted CDF", type: "line", showSymbol: false, data: column.plot.cdf.fittedX.map((x, index) => [x, column.plot.cdf.fittedY[index]]) },
    ],
  };
  const qqValues = column.plot.qq.theoretical.map((x, index) => [x, column.plot.qq.observed[index]] as [number, number]);
  const qqMin = Math.min(...qqValues.flat());
  const qqMax = Math.max(...qqValues.flat());
  const qq: EChartsOption = {
    tooltip: { trigger: "item" }, toolbox,
    xAxis: { type: "value", name: "Theoretical quantile", scale: true }, yAxis: { type: "value", name: "Observed quantile", scale: true },
    dataZoom: [{ type: "inside" }],
    series: [
      { name: "QQ", type: "scatter", data: qqValues, symbolSize: 6 },
      { name: "Reference", type: "line", data: [[qqMin, qqMin], [qqMax, qqMax]], showSymbol: false, lineStyle: { type: "dashed" } },
    ],
  };
  return <div className="distribution-chart-grid">
    <div><h4>Histogram and PDF</h4><EChart option={density} ariaLabel={`Histogram and fitted density for ${column.column}`} height={280} /></div>
    <div><h4>Empirical and fitted CDF</h4><EChart option={cdf} ariaLabel={`Empirical and fitted CDF for ${column.column}`} height={280} /></div>
    <div><h4>QQ plot</h4><EChart option={qq} ariaLabel={`Quantile-quantile plot for ${column.column}`} height={280} /></div>
  </div>;
}

function FitEvidence({
  run,
  selections,
  setSelections,
}: {
  run: DistributionFitRun;
  selections: Record<string, string>;
  setSelections: (next: Record<string, string>) => void;
}) {
  if (!run.result) return null;
  return (
    <div className="fit-evidence">
      {run.result.columns.map((column) => (
        <section className="fit-column" key={column.column}>
          <div className="section-copy split">
            <div>
              <span className="section-kicker">Marginal fit · n={column.sampleSize}</span>
              <h3>{column.column}</h3>
            </div>
            <label>
              <span>Selected marginal</span>
              <select
                aria-label={`Selected marginal for ${column.column}`}
                value={selections[column.column] ?? ""}
                onChange={(event) =>
                  setSelections({ ...selections, [column.column]: event.target.value })
                }
              >
                <option value="">Choose explicitly…</option>
                {column.rankings.map((ranking) => (
                  <option key={ranking.candidate} value={ranking.candidate}>
                    {ranking.candidate}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="fit-summary-grid">
            <div className="fit-plot-summary"><FlaskConical /><strong>{column.rankings[0]?.candidate}</strong><span>lowest parametric BIC</span><small>{column.plot.sample.length} observations</small></div>
            <div className="table-scroll" tabIndex={0}>
              <table className="engineering-table fit-ranking-table">
                <thead>
                  <tr><th>Candidate</th><th>BIC</th><th>AIC</th><th>AICc</th><th>Fit test</th><th>p-value</th></tr>
                </thead>
                <tbody>
                  {column.rankings.map((ranking) => (
                    <tr key={ranking.candidate}>
                      <td><strong>{ranking.candidate}</strong></td>
                      <td>{ranking.bic?.toPrecision(5) ?? "non-parametric"}</td>
                      <td>{ranking.aic?.toPrecision(5) ?? "—"}</td>
                      <td>{ranking.aicc?.toPrecision(5) ?? "—"}</td>
                      <td>{ranking.test.name}{ranking.test.rejected ? " · rejected" : " · not rejected"}</td>
                      <td>{ranking.test.pValue.toPrecision(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          <FitCharts column={column} />
          <details className="chart-data-fallback"><summary>Exact plotted sample</summary><pre>{column.plot.sample.join("\n")}</pre></details>
          {column.rejectedCandidates.length > 0 && (
            <details>
              <summary>{column.rejectedCandidates.length} rejected candidate(s)</summary>
              <ul>{column.rejectedCandidates.map((item) => <li key={item.candidate}><strong>{item.candidate}:</strong> {item.reason}</li>)}</ul>
            </details>
          )}
        </section>
      ))}
    </div>
  );
}

export function DataLab() {
  const client = useQueryClient();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const fileRef = useRef<HTMLInputElement>(null);
  const projectsQuery = useQuery({ queryKey: ["projects"], queryFn: api.listProjects });
  const projects = projectsQuery.data?.projects ?? [];
  const [projectId, setProjectId] = useState(searchParams.get("projectId") ?? "");
  const activeProjectId = projectId || projects[0]?.id || "";
  const datasetsQuery = useQuery({
    queryKey: ["datasets", activeProjectId],
    queryFn: () => api.listDatasets(activeProjectId),
    enabled: Boolean(activeProjectId),
  });
  const datasets = datasetsQuery.data?.datasets ?? [];
  const [datasetId, setDatasetId] = useState("");
  const dataset = datasets.find((item) => item.id === datasetId) ?? datasets[0];
  const [pasted, setPasted] = useState(BEAM_SAMPLE_CSV);
  const [selectedColumns, setSelectedColumns] = useState<string[]>([]);
  const [candidates, setCandidates] = useState<DistributionFitInput["candidates"]>([
    ...CANDIDATES,
  ]);
  const [selections, setSelections] = useState<Record<string, string>>({});
  const [copula, setCopula] = useState<DistributionFitInput["copula"]>("independent");
  const [fitRun, setFitRun] = useState<DistributionFitRun>();
  const [error, setError] = useState<string>();

  useEffect(() => {
    if (projectId || !projects[0]) return;
    setProjectId(projects[0].id);
  }, [projectId, projects]);
  useEffect(() => {
    if (!dataset) return;
    setDatasetId(dataset.id);
    setSelectedColumns(dataset.columns.filter((column) => column.type === "numeric").map((column) => column.name));
    setSelections({});
    setFitRun(undefined);
  }, [dataset?.id]);

  const upload = useMutation({
    mutationFn: async ({ blob, name, sourceKind }: { blob: Blob; name: string; sourceKind: "csv" | "xlsx" | "paste" }) => {
      if (!activeProjectId) throw new Error("Choose a study first.");
      return api.uploadDataset({
        projectId: activeProjectId,
        name,
        sourceKind,
        contentBase64: await blobBase64(blob),
      });
    },
    onSuccess: async ({ dataset: created }) => {
      await client.invalidateQueries({ queryKey: ["datasets", activeProjectId] });
      setDatasetId(created.id);
      setError(undefined);
    },
    onError: (caught) => setError(caught instanceof Error ? caught.message : "Upload failed."),
  });
  const fit = useMutation({
    mutationFn: (input: DistributionFitInput) => {
      if (!dataset) throw new Error("Choose a dataset first.");
      return api.fitDataset(dataset.id, input);
    },
    onSuccess: ({ fitRun: completed }) => {
      setFitRun(completed);
      setError(undefined);
    },
    onError: (caught) => setError(caught instanceof Error ? caught.message : "Fit failed."),
  });
  const fitInput = useMemo<DistributionFitInput>(() => ({
    selectedColumns,
    candidates,
    selectedMarginals: {},
    copula,
    significanceLevel: 0.05,
  }), [candidates, copula, selectedColumns]);
  const selectionComplete = selectedColumns.length > 0 && selectedColumns.every((column) => selections[column]);

  return (
    <div className="page data-lab-page">
      <div className="page-heading split">
        <div>
          <span className="section-kicker">Distribution Fitting</span>
          <h1>Fit uncertainty from empirical data.</h1>
          <p>Validate private observations, compare OpenTURNS fits, then explicitly compose a problem definition.</p>
        </div>
        <label className="project-select">
          <span>Project</span>
          <select value={activeProjectId} onChange={(event) => {
            const next = event.target.value;
            setProjectId(next);
            setSearchParams({ projectId: next });
            setDatasetId("");
          }}>
            {projects.map((project) => <option key={project.id} value={project.id}>{project.name}</option>)}
          </select>
        </label>
      </div>
      {!projects.length && !projectsQuery.isLoading ? (
        <div className="empty-state">
          <EmptyState title="Create a project first" body="Data provenance is always scoped to an owned project." />
          <Link to="/new-analysis" className="button primary">Create project</Link>
        </div>
      ) : (
        <div className="data-lab-layout">
          <aside className="data-source-panel">
            <div className="panel-heading"><Database /><div><strong>Private datasets</strong><small>Local/R2 originals · owner only</small></div></div>
            <input
              ref={fileRef}
              type="file"
              hidden
              accept=".csv,.xlsx,text/csv,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (!file) return;
                upload.mutate({ blob: file, name: file.name, sourceKind: file.name.toLowerCase().endsWith(".xlsx") ? "xlsx" : "csv" });
                event.target.value = "";
              }}
            />
            <button className="button primary" disabled={upload.isPending} onClick={() => fileRef.current?.click()}><Upload /> Upload CSV/XLSX</button>
            <label>
              <span>Or paste comma-separated data</span>
              <textarea value={pasted} onChange={(event) => setPasted(event.target.value)} rows={12} />
            </label>
            <button className="button secondary" disabled={upload.isPending || !pasted.trim()} onClick={() => upload.mutate({ blob: new Blob([pasted], { type: "text/csv" }), name: `Beam uncertainty sample ${new Date().toLocaleDateString()}`, sourceKind: "paste" })}><FileSpreadsheet /> Validate pasted data</button>
            <div className="dataset-list">
              {datasets.map((item) => (
                <button key={item.id} className={dataset?.id === item.id ? "active" : ""} onClick={() => setDatasetId(item.id)}>
                  <strong>{item.name}</strong><span>{item.rowCount} rows · {item.columns.length} columns</span><small>{new Date(item.createdAt).toLocaleString()}</small>
                </button>
              ))}
            </div>
          </aside>
          <main className="data-work-panel">
            {!dataset ? <EmptyState title="Upload or select a dataset" body="The original is validated before it is retained." /> : (
              <>
                <div className="data-metadata-strip">
                  <span><strong>{dataset.name}</strong></span><span>{dataset.rowCount} rows</span><span>SHA-256 {dataset.sha256.slice(0, 12)}</span><span>{dataset.sourceKind.toUpperCase()}</span>
                </div>
                {dataset.warnings.length > 0 && <div className="warning-panel"><AlertTriangle /> <ul>{dataset.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul></div>}
                <DatasetPreview dataset={dataset} />
                <section className="fit-controls">
                  <div><span className="section-kicker">1 · Variables</span><h2>Choose numeric columns</h2></div>
                  <div className="chip-options">
                    {dataset.columns.map((column) => (
                      <label className={column.type !== "numeric" ? "disabled" : ""} key={column.name}>
                        <input type="checkbox" disabled={column.type !== "numeric"} checked={selectedColumns.includes(column.name)} onChange={() => setSelectedColumns((current) => current.includes(column.name) ? current.filter((name) => name !== column.name) : [...current, column.name])} />
                        <span>{column.name}<small>{column.type} · {column.finiteCount} finite</small></span>
                      </label>
                    ))}
                  </div>
                  <div><span className="section-kicker">2 · Candidates</span><h2>Compare distribution families</h2></div>
                  <div className="chip-options compact">
                    {CANDIDATES.map((candidate) => <label key={candidate}><input type="checkbox" checked={candidates.includes(candidate)} onChange={() => setCandidates((current) => current.includes(candidate) ? current.filter((name) => name !== candidate) : [...current, candidate])} /><span>{candidate}</span></label>)}
                  </div>
                  <button className="button primary" disabled={fit.isPending || !selectedColumns.length || !candidates.length} onClick={() => fit.mutate(fitInput)}><FlaskConical /> {fit.isPending ? "Fitting with OpenTURNS…" : "Rank candidate fits"}</button>
                </section>
                {fitRun && <FitEvidence run={fitRun} selections={selections} setSelections={setSelections} />}
                {fitRun?.result && (
                  <section className="problem-composer">
                    <div><span className="section-kicker">3 · Dependence and provenance</span><h2>Compose a new uncertainty problem</h2><p>Selections are never applied to an existing model silently.</p></div>
                    <label><span>Copula</span><select value={copula} onChange={(event) => setCopula(event.target.value as DistributionFitInput["copula"])}><option value="independent">Independent composition</option><option value="normal">Fitted Normal copula</option><option value="bernstein">Fitted Bernstein copula</option></select></label>
                    <button className="button primary" disabled={!selectionComplete || fit.isPending} onClick={() => fit.mutate({ ...fitInput, selectedMarginals: selections })}><Check /> Generate problem definition</button>
                    {fitRun.result.generatedSource && (
                      <div className="generated-problem">
                        <div className="editor-title"><span>generated_problem.py</span><small>OpenTURNS {fitRun.openturnsVersion}</small></div>
                        <pre><code>{fitRun.result.generatedSource}</code></pre>
                        <button className="button secondary" onClick={() => {
                          window.sessionStorage.setItem("uncertaintycat-data-lab-draft", JSON.stringify({ fitRunId: fitRun.id, datasetId: dataset.id, source: `${fitRun.result?.generatedSource ?? ""}\n# Define an OpenTURNS Function named model before validation.\n`, builderSpec: fitRun.result?.builderSpec }));
                          navigate(`/studies/${activeProjectId}/workspace?dataFit=${fitRun.id}`);
                        }}>Prepare model draft</button>
                      </div>
                    )}
                  </section>
                )}
              </>
            )}
          </main>
        </div>
      )}
      {error && <div className="error-banner" role="alert">{error}</div>}
    </div>
  );
}
