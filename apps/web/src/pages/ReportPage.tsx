import { useMutation, useQuery } from "@tanstack/react-query";
import {
  ArrowRight,
  Check,
  Code2,
  Download,
  FileText,
  RotateCcw,
  Share2,
  ShieldCheck,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { api } from "../api";
import { analysisDisplayName } from "../analysisNames";
import { ChatPanel } from "../components/ChatPanel";
import { Markdown } from "../components/Markdown";
import { PythonSource } from "../components/PythonSource";
import { ResultView } from "../components/ResultView";
import { StatusBadge } from "../components/Status";
import type { AnalysisResult, ModelMetadata } from "@uncertaintycat/contracts";

function equationMarkdownText(value: string) {
  return value.replace(/([\\`*_[\]{}()#+.!|>~-])/g, "\\$1");
}

function equationMathBody(value: string) {
  return value.replaceAll("$", "\\$");
}

function ModelEquationSummary({
  model,
  spec,
}: {
  model: ModelMetadata;
  spec: Record<string, unknown> | null | undefined;
}) {
  const symbolicOutputs = Array.isArray(spec?.outputs)
    ? (spec.outputs as Array<{ name?: string; formula?: string }>).flatMap(
        (output) =>
          output.name && output.formula
            ? [{ outputName: output.name, latex: `${output.name}=${output.formula}` }]
            : [],
      )
    : [];
  const equations = model.equations?.length
    ? model.equations.map((equation) => ({
        outputName: equation.output_name,
        latex: equation.latex,
      }))
    : symbolicOutputs;
  if (!equations.length) return null;
  return (
    <div className="symbolic-definition">
      <div>
        <strong>Rendered equations</strong>
        <Markdown>
          {equations
            .map(
              (equation) =>
                `**${equationMarkdownText(equation.outputName)}**\n\n$$${equationMathBody(equation.latex)}$$`,
            )
            .join("\n\n")}
        </Markdown>
      </div>
    </div>
  );
}

function SymbolicDefinitionSummary({ spec }: { spec: Record<string, unknown> | null | undefined }) {
  if (!spec || !Array.isArray(spec.variables) || !Array.isArray(spec.outputs)) return null;
  const variables = spec.variables as Array<{
    name?: string;
    distribution?: string;
    parameters?: unknown[];
  }>;
  const copula = spec.copula as
    | { kind?: string; correlation?: unknown[][] }
    | undefined;
  return (
    <div className="symbolic-definition">
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
  const [fixedValues, setFixedValues] = useState<Record<string, string>>(
    Object.fromEntries(model.inputs.map((input) => [input.name, String(input.mean ?? 0)])),
  );
  const [displayName, setDisplayName] = useState("Morris-screened model");
  const [confirmed, setConfirmed] = useState(false);
  const [createdModelId, setCreatedModelId] = useState<string>();
  const [error, setError] = useState<string>();
  const [copyStatus, setCopyStatus] = useState<string>();
  const fixedVariables = model.inputs
    .filter((input) => !retained[input.name])
    .map((input) => ({ index: input.index, value: Number(fixedValues[input.name]) }));
  const validFixedValues = model.inputs.filter((input) => !retained[input.name]).every((input) => Boolean(fixedValues[input.name]?.trim()) && Number.isFinite(Number(fixedValues[input.name])));
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
                  <td><input aria-label={`Retain ${name}`} type="checkbox" checked={retained[name] ?? false} onChange={(event) => { setRetained({ ...retained, [name]: event.target.checked }); setConfirmed(false); }} /></td>
                  <td><strong>{name}</strong></td>
                  <td>{Number(row[2]).toPrecision(5)}</td>
                  <td>{String(row[4])}</td>
                  <td><input aria-label={`Fixed value for ${name}`} type="number" value={fixedValues[name] ?? ""} disabled={retained[name] ?? false} onChange={(event) => { setFixedValues({ ...fixedValues, [name]: event.target.value }); setConfirmed(false); }} /></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="reduction-confirmation">
        <label><span>Derived model name</span><input value={displayName} onChange={(event) => setDisplayName(event.target.value)} /></label>
        <label className="confirmation-check"><input type="checkbox" checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} /><span>I confirm these explicit fixed values and understand the original model remains unchanged.</span></label>
        <button className="button primary" disabled={!confirmed || !displayName.trim() || !validFixedValues || fixedVariables.length === 0 || fixedVariables.length >= model.input_dimension || mutation.isPending} onClick={() => { setError(undefined); mutation.mutate(); }}><Check /> {mutation.isPending ? "Validating derived model…" : "Create derived version"}</button>
        {!validFixedValues && <p role="alert">Enter a finite fixed value for every removed input before confirming.</p>}
      </div>
      {createdModelId && (
        <div className="reduced-model-result">
          <Check />
          <div>
            <strong>Reduced model created</strong>
            <p>The original model is unchanged. Choose where the validated reduced model should be analysed next.</p>
          </div>
          <div className="model-handoff-actions">
            <Link className="model-handoff-option primary-option" to={`/studies/${projectId}/workspace?sourceModel=${createdModelId}`}><span>Continue in this project</span><strong>Start a new analysis with the reduced model</strong><small>Keep the original and reduced versions together in this project.</small><ArrowRight /></Link>
            <Link className="model-handoff-option" to={`/studies?new=1&sourceModel=${encodeURIComponent(createdModelId)}&suggestedName=${encodeURIComponent(`${displayName} analysis`)}`}><span>Separate investigation</span><strong>Start a new project with the reduced model</strong><small>Copy the complete validated Python model and its provenance into a new project.</small><ArrowRight /></Link>
          </div>
          <div className="reduced-model-actions">
            <button className="button secondary" type="button" disabled={!createdDefinition.data?.definition.source} onClick={async () => {
              try { await navigator.clipboard.writeText(createdDefinition.data?.definition.source ?? ""); setCopyStatus("Python model copied."); }
              catch { setCopyStatus("Clipboard access failed. Select and copy the Python source below."); }
            }}>Copy Python model</button>
          </div>
          {copyStatus && <p role="status">{copyStatus}</p>}
          {createdDefinition.isPending && <p role="status">Loading the retained reduced model source…</p>}
          {createdDefinition.isError && <div className="inline-error" role="alert"><p>The reduced model was saved, but its source could not be loaded. {createdDefinition.error.message}</p><button className="button secondary" disabled={createdDefinition.isFetching} onClick={() => void createdDefinition.refetch()}>Retry reduced model source</button></div>}
          {createdDefinition.data?.definition.source && (
            <PythonSource
              source={createdDefinition.data.definition.source}
              label="Reduced Python model source"
            />
          )}
        </div>
      )}
      {error && <div className="inline-error" role="alert">{error}</div>}
    </section>
  );
}

export function ReportPage({
  shared = false,
  operator = false,
}: {
  shared?: boolean;
  operator?: boolean;
}) {
  const { reportId = "" } = useParams();
  const { token = "" } = useParams();
  const navigate = useNavigate();
  const reportDocument = useRef<HTMLElement>(null);
  const [shareUrl, setShareUrl] = useState<string>();
  const [shareOpen, setShareOpen] = useState(false);
  const [includeModelDefinition, setIncludeModelDefinition] = useState(false);
  const [downloadingPdf, setDownloadingPdf] = useState(false);
  const [pdfError, setPdfError] = useState<string>();
  const [shareCopied, setShareCopied] = useState(false);
  const query = useQuery({
    queryKey: [
      shared ? "shared-report" : operator ? "operator-report" : "report",
      shared ? token : reportId,
    ],
    queryFn: () =>
      shared
        ? api.getSharedReport(token)
        : operator
          ? api.operatorReport(reportId)
          : api.getReport(reportId),
  });
  const report = query.data?.report;
  const definitionQuery = useQuery({
    queryKey: ["model-definition", report?.modelVersion.id],
    queryFn: () => api.getModelDefinition(report?.modelVersion.id ?? ""),
    enabled: !shared && !operator && Boolean(report?.modelVersion.id),
  });
  // A disabled owner query can still expose its cached data. Shared reports must
  // render only the definition explicitly included by their share contract.
  const visibleDefinition = shared ? report?.modelDefinition : operator ? undefined : definitionQuery.data?.definition;
  const share = useMutation({
    mutationFn: () =>
      api.createShareLink(
        report?.id ?? reportId,
        30,
        includeModelDefinition,
      ),
    onSuccess: async ({ shareLink }) => {
      setShareUrl(shareLink.url);
      setShareCopied(false);
      try {
        if (navigator.clipboard) {
          await navigator.clipboard.writeText(shareLink.url);
          setShareCopied(true);
        }
      } catch { /* The retained link remains selectable if clipboard access fails. */ }
      setShareOpen(false);
    },
  });
  const rerun = useMutation({
    mutationFn: () => api.rerun(report?.runId ?? reportId),
    onSuccess: ({ run }) => navigate(`/runs/${run.id}`),
  });
  useEffect(() => {
    setShareUrl(undefined);
    setShareOpen(false);
    setIncludeModelDefinition(false);
    setPdfError(undefined);
    setShareCopied(false);
    share.reset();
    rerun.reset();
  }, [reportId, token, shared, operator]);
  const downloadPdf = async () => {
    if (!reportDocument.current || !report) return;
    setPdfError(undefined);
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
    } catch (error) {
      setPdfError(error instanceof Error ? error.message : "The report could not be rendered as a PDF.");
    } finally {
      setDownloadingPdf(false);
    }
  };
  if (query.isLoading)
    return (
      <div className="page">
        <div className="report-loading" role="status">Assembling persisted results…</div>
      </div>
    );
  if (!report)
    return (
      <div className="page">
        <div className="error-banner" role="alert">
          <h1>Report unavailable</h1>
          <p>{query.error?.message ?? "The persisted report could not be loaded."} {shared ? "This link may have expired or been revoked." : "A report may still be pending, or this record may have been deleted."}</p>
          <button className="button secondary" type="button" disabled={query.isFetching} onClick={() => void query.refetch()}>Retry report</button>
          <Link className="button secondary" to={operator ? "/operator" : "/studies"}>{operator ? "Back to Operations" : "Back to Projects"}</Link>
        </div>
      </div>
    );
  return (
    <div className="report-layout">
      <article className="report-document" ref={reportDocument}>
        <nav className="breadcrumbs" aria-label="Breadcrumb">
          <Link to={operator ? "/operator" : "/studies"}>{operator ? "Operations" : "Projects"}</Link><span>/</span>
          {shared ? <span>{report.project.name}</span> : <Link to={operator ? `/operator/projects/${report.project.id}` : `/studies/${report.project.id}`}>{report.project.name}</Link>}<span>/</span>
          <span>{report.modelVersion.displayName} v{report.modelVersion.version}</span>
        </nav>
        <header className="report-header">
          <div>
            <span className="section-kicker">Persisted numerical report</span>
            <h1>{report.title}</h1>
            <p>
              Generated {new Date(report.generatedAt).toLocaleString()} · Run{" "}
              <code>{report.runId}</code>
            </p>
            {shareUrl && (
              <p className="share-confirmation" role="status">
                {shareCopied ? "Share link copied:" : "Share link created; copy this link:"} <Link to={shareUrl}>{shareUrl}</Link>
              </p>
            )}
          </div>
          <div className="report-actions">
            {!shared && !operator && (
              <>
                <Link className="button secondary small" to={`/runs/${report.runId}`}>Run details</Link>
                <a
                  className="button secondary small"
                  href={`/api/v1/reports/${report.id}/export`}
                  download
                >
                  <Download /> Data bundle
                </a>
                <button
                  className="button secondary small"
                  onClick={() => { share.reset(); setShareOpen((value) => !value); }}
                  aria-expanded={shareOpen}
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
        {rerun.isError && <div className="error-banner" role="alert">The exact rerun could not start. {rerun.error.message} Your retained report is unchanged; retry when ready.</div>}
        {pdfError && <div className="error-banner" role="alert">PDF download failed. {pdfError} Retry Download PDF, or use the data bundle for exact numerical evidence.</div>}
        {shareOpen && !shared && !operator && (
          <section className="share-dialog" role="dialog" aria-label="Share report">
            <div>
              <strong>Create a read-only report link</strong>
              <small>Recipients must sign in to UncertaintyCat. This read-only link expires after 30 days. Numerical evidence is included; exact model source stays private by default.</small>
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
            {share.isError && <div className="inline-error" role="alert">Share link creation failed. {share.error.message} Retry when ready.</div>}
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
        {shared && <section className="operator-readonly-note"><ShieldCheck /><div><strong>Shared report · read only</strong><span>This authenticated link grants access to this report. The source project remains with its owner.</span></div></section>}
        {operator && (
          <section className="operator-readonly-note report-operator-note">
            <ShieldCheck />
            <div>
              <strong>Operator inspection · read only</strong>
              <span>
                This is the user’s retained numerical evidence. Rerun, sharing,
                source download, derived-model actions, and chat are disabled.
              </span>
            </div>
          </section>
        )}
        {operator && report.model.equations?.length ? (
          <section className="model-definition-section">
            <div className="section-copy">
              <span className="section-kicker">Validated model</span>
              <h2>Model equations</h2>
            </div>
            <ModelEquationSummary model={report.model} spec={null} />
          </section>
        ) : null}
        {!shared && !operator && definitionQuery.isPending && <p role="status">Loading the exact model definition…</p>}
        {!shared && !operator && definitionQuery.isError && <div className="error-banner" role="alert"><p>The exact model definition could not be loaded. Numerical evidence below is still available. {definitionQuery.error.message}</p><button className="button secondary" disabled={definitionQuery.isFetching} onClick={() => void definitionQuery.refetch()}>Retry model definition</button></div>}
        {visibleDefinition && (
          <section className="model-definition-section">
            <div className="section-copy">
              <span className="section-kicker">Model definition and provenance</span>
              <h2>Exact immutable source</h2>
              <p>Created {new Date(report.modelVersion.createdAt).toLocaleString()} · seed {report.seed} · {report.accuracyProfile} accuracy</p>
            </div>
            <ModelEquationSummary
              model={report.model}
              spec={visibleDefinition.builderSpec}
            />
            <SymbolicDefinitionSummary
              spec={visibleDefinition.builderSpec}
            />
            <PythonSource
              source={visibleDefinition.source}
              label="Exact immutable Python model source"
            />
            {!shared && !operator && (
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
              {analysisDisplayName(section.key)}
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
                <h2>{analysisDisplayName(section.key)}</h2>
                <p>
                  Versioned numerical result and method-specific provenance.
                </p>
              </div>
              <StatusBadge status={section.status} />
            </header>
            {section.result ? (
              <>
                <ResultView result={section.result} />
                {!shared && !operator && section.key === "morris" && (
                  <MorrisReduction
                    key={report.runId}
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
      {!shared && !operator && <ChatPanel reportId={report.id} />}
    </div>
  );
}
