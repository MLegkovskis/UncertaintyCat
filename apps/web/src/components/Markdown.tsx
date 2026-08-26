import type { ComponentProps } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

const EVIDENCE_CITATION =
  /(?:\[|【)analysis\.(metric|fact|table|series|matrix):([A-Za-z0-9_-]+)\.([A-Za-z0-9_.-]+)(?:\]|】)/g;
const EVIDENCE_LINK_PREFIX = "#evidence-";

const ANALYSIS_LABELS: Record<string, string> = {
  eda: "EDA",
  fast: "FAST",
  hsic: "HSIC",
  morris: "Morris",
  sobol: "Sobol",
  taylor: "Taylor",
};

function humanize(value: string) {
  return value
    .replaceAll("_", " ")
    .replaceAll(".", " · ")
    .replace(/\b[a-z]/g, (letter) => letter.toUpperCase());
}

export function formatEvidenceCitations(markdown: string) {
  return markdown.replace(
    EVIDENCE_CITATION,
    (token, kind: string, analysis: string, field: string) => {
      const analysisLabel = ANALYSIS_LABELS[analysis.toLowerCase()] ?? humanize(analysis);
      const label = `Source: ${analysisLabel} · ${humanize(field)}`;
      return `[${label}](${EVIDENCE_LINK_PREFIX}${encodeURIComponent(token)} "Stored ${kind}")`;
    },
  );
}

export function Markdown({
  children,
  className,
  evidenceCitations = false,
}: {
  children: string;
  className?: string;
  evidenceCitations?: boolean;
}) {
  const content = evidenceCitations ? formatEvidenceCitations(children) : children;
  return (
    <div className={`markdown ${className ?? ""}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        components={{
          a: ({ href, children: linkChildren, ...props }: ComponentProps<"a">) => {
            if (href?.startsWith(EVIDENCE_LINK_PREFIX)) {
              const rawToken = decodeURIComponent(href.slice(EVIDENCE_LINK_PREFIX.length));
              return (
                <span
                  className="evidence-citation"
                  title={`${props.title ?? "Stored evidence"}: ${rawToken}`}
                  data-evidence={rawToken}
                >
                  {linkChildren}
                </span>
              );
            }
            return <a href={href} target="_blank" rel="noreferrer" {...props}>{linkChildren}</a>;
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
