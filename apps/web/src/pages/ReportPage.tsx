import { useMutation, useQuery } from "@tanstack/react-query";
import {
  Check,
  Code2,
  Download,
  FileText,
  RotateCcw,
  Share2,
  ShieldCheck,
} from "lucide-react";
import { useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { api } from "../api";
import { ChatPanel } from "../components/ChatPanel";
import { Markdown } from "../components/Markdown";
import { ResultView } from "../components/ResultView";
import { StatusBadge } from "../components/Status";
import type { AnalysisResult, ModelMetadata } from "@uncertaintycat/contracts";

function SymbolicDefinitionSummary({ spec }: { spec: Record<string, unknown> | null | undefined }) {
  if (!spec || !Array.isArray(spec.variables) || !Array.isArray(spec.outputs)) return null;
  const variables = spec.variables as Array<{
    name?: string;
    distribution?: string;
    parameters?: unknown[];
  }>;
  const outputs = spec.outputs as Array<{ name?: string; formula?: string }>;
  const copula = spec.copula as
    | { kind?: string; correlation?: unknown[][] }
    | undefined;
  return (
    <div className="symbolic-definition">
      <div>
        <strong>Rendered equations</strong>
        <Markdown>
          {outputs
            .filter((output) => output.name && output.formula)
            .map((output) => `$$\\mathrm{${output.name}} = ${output.formula}$$`)
            .join("\n\n")}
        </Markdown>
      </div>
      <div>
        <strong>Inputs and marginals</strong>
        <div className="table-scroll" tabIndex={0}>
          <table className="engineering-table">
            <thead><tr><th>Input</th><th>Distribution</th><th>Parameters</th></tr></thead>
            <tbody>
              {variables.map((variable, index) => (
                <tr key={`${variable.name ?? "input"}-${index}`}>
                  <td>{variable.name}</td>
                  <td>{variable.distribution}</td>
                  <td>{variable.parameters?.map(String).join(", ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <p>
        Function: <strong>OpenTURNS SymbolicFunction</strong> · exact analytical
        gradient and Hessian · dependence: <strong>{copula?.kind === "normal" ? "Normal copula" : "independent"}</strong>
      </p>
      {copula?.kind === "normal" && Array.isArray(copula.correlation) && (
        <details><summary>Validated correlation matrix</summary><pre><code>{JSON.stringify(copula.correlation, null, 2)}</code></pre></details>
      )}
    </div>
  );
}

function MorrisReduction({
  result,
  model,
  modelVersionId,
  runId,
  projectId,
}: {
  result: AnalysisResult;
  model: ModelMetadata;
  modelVersionId: string;
  runId: string;
  projectId: string;
}) {
  const table = result.payload.tables.effects;
  const rows = table?.rows ?? [];
  const defaults = Object.fromEntries(
    rows.map((row, index) => [String(row[0]), Boolean(row[5]) || index === 0]),
  );
  const [retained, setRetained] = useState<Record<string, boolean>>(defaults);
  const [fixedValues, setFixedValues] = useState<Record<string, number>>(
    Object.fromEntries(model.inputs.map((input) => [input.name, input.mean ?? 0])),
  );
  const [displayName, setDisplayName] = useState("Morris-screened model");
  const [confirmed, setConfirmed] = useState(false);
  const [createdModelId, setCreatedModelId] = useState<string>();
  const [error, setError] = useState<string>();
  const fixedVariables = model.inputs
    .filter((input) => !retained[input.name])
    .map((input) => ({ index: input.index, value: fixedValues[input.name] ?? 0 }));
  const mutation = useMutation({
    mutationFn: () =>
      api.createReducedModel(modelVersionId, {
        morrisRunId: runId,
        displayName,
        fixedVariables,
        confirmed: true,
      }),
    onSuccess: ({ modelVersion }) => setCreatedModelId(modelVersion.id),
    onError: (caught) =>
      setError(caught instanceof Error ? caught.message : "Could not create the derived model."),
  });
  const createdDefinition = useQuery({
    queryKey: ["model-definition", createdModelId],
    queryFn: () => api.getModelDefinition(createdModelId ?? ""),
    enabled: Boolean(createdModelId),
  });
  if (!table) return null;
  return (
    <section className="morris-reduction">
      <div className="section-copy">
        <span className="section-kicker">Optional derived version</span>
        <h3>Confirm active and fixed variables</h3>
        <p>
          The default candidate rule is {Number(result.payload.metrics.candidate_threshold_fraction ?? 0.05) * 100}% of the largest mean absolute effect. It is a screening rule, not proof of irrelevance.
        </p>
      </div>
      <div className="table-scroll" tabIndex={0}>
        <table className="engineering-table">
          <thead><tr><th>Retain</th><th>Input</th><th>Mean |effect|</th><th>Rank</th><th>Fixed value if removed</th></tr></thead>
          <tbody>
            {rows.map((row) => {
              const name = String(row[0]);
              return (
                <tr key={name}>
                  <td><input aria-label={`Retain ${name}`} type="checkbox" checked={retained[name] ?? false} onChange={(event) => setRetained({ ...retained, [name]: event.target.checked })} /></td>
                  <td><strong>{name}</strong></td>
                  <td>{Number(row[2]).toPrecision(5)}</td>
                  <td>{String(row[4])}</td>
                  <td><input aria-label={`Fixed value for ${name}`} type="number" value={fixedValues[name] ?? 0} disabled={retained[name] ?? false} onChange={(event) => setFixedValues({ ...fixedValues, [name]: Number(event.target.value) })} /></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="reduction-confirmation">
        <label><span>Derived model name</span><input value={displayName} onChange={(event) => setDisplayName(event.target.value)} /></label>
        <label className="confirmation-check"><input type="checkbox" checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} /><span>I confirm these explicit fixed values and understand the original model remains unchanged.</span></label>
        <button className="button primary" disabled={!confirmed || !displayName.trim() || fixedVariables.length === 0 || fixedVariables.length >= model.input_dimension || mutation.isPending} onClick={() => mutation.mutate()}><Check /> {mutation.isPending ? "Validating derived model…" : "Create derived version"}</button>
      </div>
      {createdModelId && (
        <div className="reduced-model-result">
          <Check />
          <div>
            <strong>Reduced model created</strong>
            <p>The original model is unchanged. Continue with this explicit reduced model in the current project, or copy its complete Python definition into a new project.</p>
          </div>
          <div className="reduced-model-actions">
            <Link className="button primary" to={`/studies/${projectId}/workspace?sourceModel=${createdModelId}`}>Analyse reduced model</Link>
            <button className="button secondary" type="button" disabled={!createdDefinition.data?.definition.source} onClick={() => void navigator.clipboard.writeText(createdDefinition.data?.definition.source ?? "")}>Copy Python model</button>
            <Link className="button secondary" to="/studies?new=1">Create another project</Link>
          </div>
          {createdDefinition.data?.definition.source && <pre><code>{createdDefinition.data.definition.source}</code></pre>}
        </div>
      )}
      {error && <div className="inline-error" role="alert">{error}</div>}
    </section>
  );
}

export function ReportPage({ shared = false }: { shared?: boolean }) {
  const { reportId = "" } = useParams();
  const { token = "" } = useParams();
  const navigate = useNavigate();
  const reportDocument = useRef<HTMLElement>(null);
  const [shareUrl, setShareUrl] = useState<string>();
  const [shareOpen, setShareOpen] = useState(false);
  const [includeModelDefinition, setIncludeModelDefinition] = useState(false);
  const [downloadingPdf, setDownloadingPdf] = useState(false);
  const query = useQuery({
    queryKey: [shared ? "shared-report" : "report", shared ? token : reportId],
    queryFn: () =>
      shared ? api.getSharedReport(token) : api.getReport(reportId),
  });
  const report = query.data?.report;
  const definitionQuery = useQuery({
    queryKey: ["model-definition", report?.modelVersion.id],
    queryFn: () => api.getModelDefinition(report?.modelVersion.id ?? ""),
    enabled: !shared && Boolean(report?.modelVersion.id),
  });
  const share = useMutation({
    mutationFn: () =>
      api.createShareLink(
        report?.id ?? reportId,
        30,
        includeModelDefinition,
      ),
    onSuccess: async ({ shareLink }) => {
      setShareUrl(shareLink.url);
      await navigator.clipboard
        ?.writeText(shareLink.url)
        .catch(() => undefined);
      setShareOpen(false);
    },
  });
  const rerun = useMutation({
    mutationFn: () => api.rerun(report?.runId ?? reportId),
    onSuccess: ({ run }) => navigate(`/runs/${run.id}`),
  });
  const downloadPdf = async () => {
    if (!reportDocument.current || !report) return;
    setDownloadingPdf(true);
    try {
      const [{ default: html2canvas }, { jsPDF }] = await Promise.all([
        import("html2canvas"),
        import("jspdf"),
      ]);
      const canvas = await html2canvas(reportDocument.current, {
        backgroundColor: "#ffffff",
        scale: Math.min(window.devicePixelRatio || 1, 2),
        useCORS: true,
        onclone: (document) => document.documentElement.removeAttribute("data-theme"),
      });
      const pdf = new jsPDF({ orientation: "portrait", unit: "pt", format: "a4", compress: true });
      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      const imageHeight = canvas.height * pageWidth / canvas.width;
      const image = canvas.toDataURL("image/jpeg", 0.92);
      let offset = 0;
      pdf.addImage(image, "JPEG", 0, offset, pageWidth, imageHeight, undefined, "FAST");
      while (offset + imageHeight > pageHeight) {
        offset -= pageHeight;
        pdf.addPage();
        pdf.addImage(image, "JPEG", 0, offset, pageWidth, imageHeight, undefined, "FAST");
      }
      const safeName = report.modelVersion.displayName.replace(/[^a-z0-9_-]+/gi, "-").replace(/^-|-$/g, "") || "uncertainty-report";
      pdf.save(`${safeName}-report.pdf`);
    } finally {
      setDownloadingPdf(false);
    }
  };
  if (query.isLoading)
    return (
      <div className="page">
        <div className="report-loading">Assembling persisted results…</div>
      </div>
    );
  if (!report)
    return (
      <div className="page">
        <div className="error-banner">The report is not available yet.</div>
      </div>
    );
  return (
    <div className="report-layout">
      <article className="report-document" ref={reportDocument}>
        <nav className="breadcrumbs" aria-label="Breadcrumb">
          <Link to="/studies">Studies</Link><span>/</span>
          <Link to={`/studies/${report.project.id}`}>{report.project.name}</Link><span>/</span>
          <span>{report.modelVersion.displayName} v{report.modelVersion.version}</span>
        </nav>
        <header className="report-header">
          <div>
            <span className="section-kicker">Comprehensive UQ report</span>
            <h1>{report.title}</h1>
            <p>
              Generated {new Date(report.generatedAt).toLocaleString()} · Run{" "}
              <code>{report.runId}</code>
            </p>
            {shareUrl && (
              <p className="share-confirmation">
                Share link copied: <a href={shareUrl}>{shareUrl}</a>
              </p>
            )}
          </div>
          <div className="report-actions">
            {!shared && (
              <>
                <a
                  className="button secondary small"
                  href={`/api/v1/reports/${report.id}/export`}
                  download
                >
                  <Download /> Data bundle
                </a>
                <button
                  className="button secondary small"
                  onClick={() => setShareOpen((value) => !value)}
                >
                  <Share2 /> Share
                </button>
                <button
                  className="button secondary small"
                  onClick={() => rerun.mutate()}
                  disabled={rerun.isPending}
                >
                  <RotateCcw /> {rerun.isPending ? "Starting…" : "Rerun exact"}
                </button>
              </>
            )}
            <button
              className="button secondary small"
              onClick={() => void downloadPdf()}
              disabled={downloadingPdf}
            >
              <Download /> {downloadingPdf ? "Preparing PDF…" : "Download PDF"}
            </button>
          </div>
        </header>
        {shareOpen && !shared && (
          <section className="share-dialog" role="dialog" aria-label="Share report">
            <div>
              <strong>Create a read-only report link</strong>
              <small>Numerical metadata is included. Exact model source stays private by default.</small>
            </div>
            <label>
              <input
                type="checkbox"
                checked={includeModelDefinition}
                onChange={(event) => setIncludeModelDefinition(event.target.checked)}
              />
              Include model definition
            </label>
            <button className="button primary small" onClick={() => share.mutate()} disabled={share.isPending}>
              {share.isPending ? "Creating…" : "Create share link"}
            </button>
          </section>
        )}
        <section className="provenance-banner">
          <ShieldCheck />
          <div>
            <strong>Reproducible numerical record</strong>
            <p>
              {report.project.name} · {report.modelVersion.displayName} v{report.modelVersion.version} · {report.modelVersion.sourceKind} · OpenTURNS {report.model.openturns_version} · model{" "}
              {report.model.source_hash.slice(0, 12)} ·{" "}
              {report.model.input_dimension} inputs ·{" "}
              {report.model.output_dimension} outputs
            </p>
            <p className="evidence-source">
              Evidence source: <strong>{report.evidenceSource === "surrogate" ? `explicit promoted ${report.surrogate?.method.toUpperCase()} surrogate` : "direct model"}</strong>
              {report.surrogate ? ` · surrogate ${report.surrogate.id.slice(0, 8)} · plugin ${report.surrogate.pluginVersion}` : ""}
            </p>
          </div>
          <StatusBadge status={report.status} />
        </section>
        {(definitionQuery.data?.definition ?? report.modelDefinition) && (
          <section className="model-definition-section">
            <div className="section-copy">
              <span className="section-kicker">Model definition and provenance</span>
              <h2>Exact immutable source</h2>
              <p>Created {new Date(report.modelVersion.createdAt).toLocaleString()} · seed {report.seed} · {report.accuracyProfile} accuracy</p>
            </div>
            <SymbolicDefinitionSummary
              spec={(definitionQuery.data?.definition ?? report.modelDefinition)?.builderSpec}
            />
            <pre><code>{(definitionQuery.data?.definition ?? report.modelDefinition)?.source}</code></pre>
            {!shared && (
              <div className="model-source-actions">
                <a className="button secondary small" href={`/api/v1/model-versions/${report.modelVersion.id}/source`} download><Download /> Source</a>
                <Link className="button secondary small" to={`/studies/${report.project.id}/workspace?sourceModel=${report.modelVersion.id}`}><Code2 /> Open as new version</Link>
              </div>
            )}
          </section>
        )}
        <nav className="report-toc" aria-label="Report sections">
          <strong>Contents</strong>
          {report.sections.map((section, index) => (
            <a key={section.key} href={`#section-${section.key}`}>
              <span>{String(index + 1).padStart(2, "0")}</span>
              {section.key.replaceAll("_", " ")}
            </a>
          ))}
        </nav>
        {report.sections.map((section, index) => (
          <section
            className="report-section"
            id={`section-${section.key}`}
            key={section.key}
          >
            <header>
              <span>{String(index + 1).padStart(2, "0")}</span>
              <div>
                <h2>{section.key.replaceAll("_", " ")}</h2>
                <p>
                  Versioned numerical result and method-specific provenance.
                </p>
              </div>
              <StatusBadge status={section.status} />
            </header>
            {section.result ? (
              <>
                <ResultView result={section.result} />
                {!shared && section.key === "morris" && (
                  <MorrisReduction
                    result={section.result}
                    model={report.model}
                    modelVersionId={report.modelVersion.id}
                    runId={report.runId}
                    projectId={report.project.id}
                  />
                )}
              </>
            ) : (
              <div className="section-error">
                <FileText />
                <p>
                  {section.error?.message ??
                    "This section did not produce a result."}
                </p>
              </div>
            )}
          </section>
        ))}
      </article>
      {!shared && <ChatPanel reportId={report.id} />}
    </div>
  );
}
