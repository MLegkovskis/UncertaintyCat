import { useMutation, useQuery } from "@tanstack/react-query";
import { Download, FileText, Printer, Share2, ShieldCheck } from "lucide-react";
import { useState } from "react";
import { useParams } from "react-router-dom";

import { api } from "../api";
import { ChatPanel } from "../components/ChatPanel";
import { ResultView } from "../components/ResultView";
import { StatusBadge } from "../components/Status";

export function ReportPage({ shared = false }: { shared?: boolean }) {
  const { reportId = "" } = useParams();
  const { token = "" } = useParams();
  const [shareUrl, setShareUrl] = useState<string>();
  const query = useQuery({
    queryKey: [shared ? "shared-report" : "report", shared ? token : reportId],
    queryFn: () =>
      shared ? api.getSharedReport(token) : api.getReport(reportId),
  });
  const report = query.data?.report;
  const share = useMutation({
    mutationFn: () => api.createShareLink(report?.id ?? reportId),
    onSuccess: async ({ shareLink }) => {
      setShareUrl(shareLink.url);
      await navigator.clipboard
        ?.writeText(shareLink.url)
        .catch(() => undefined);
    },
  });
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
      <article className="report-document">
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
                  onClick={() => share.mutate()}
                  disabled={share.isPending}
                >
                  <Share2 /> {share.isPending ? "Sharing…" : "Share"}
                </button>
              </>
            )}
            <button
              className="button secondary small"
              onClick={() => window.print()}
            >
              <Printer /> PDF
            </button>
          </div>
        </header>
        <section className="provenance-banner">
          <ShieldCheck />
          <div>
            <strong>Reproducible numerical record</strong>
            <p>
              OpenTURNS {report.model.openturns_version} · model{" "}
              {report.model.source_hash.slice(0, 12)} ·{" "}
              {report.model.input_dimension} inputs ·{" "}
              {report.model.output_dimension} outputs
            </p>
          </div>
          <StatusBadge status={report.status} />
        </section>
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
              <ResultView result={section.result} />
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
