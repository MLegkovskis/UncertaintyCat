import { useQuery } from "@tanstack/react-query";
import {
  ArrowRight,
  Blocks,
  Braces,
  ChartNoAxesCombined,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import { Link } from "react-router-dom";

import { api } from "../api";

export function Home() {
  const catalog = useQuery({ queryKey: ["catalog"], queryFn: api.catalog });
  return (
    <div className="page home-page">
      <section className="hero">
        <div className="eyebrow">
          <Sparkles size={15} /> Open-source uncertainty engineering
        </div>
        <h1>
          Turn uncertain inputs into <em>defensible decisions.</em>
        </h1>
        <p>
          Define an OpenTURNS model, run a traceable analysis suite, and explore
          every assumption, sensitivity index, and convergence result in one
          durable report.
        </p>
        <div className="hero-actions">
          <Link className="button primary" to="/workspace">
            Open the workspace <ArrowRight size={17} />
          </Link>
          <a
            className="button secondary"
            href="https://openturns.github.io/openturns/latest/"
            target="_blank"
            rel="noreferrer"
          >
            OpenTURNS methods
          </a>
        </div>
        <div className="trust-row">
          <span>
            <ShieldCheck /> Isolated Python
          </span>
          <span>
            <Braces /> Versioned results
          </span>
          <span>
            <ChartNoAxesCombined /> Interactive reports
          </span>
        </div>
      </section>
      <section className="architecture-strip">
        <div>
          <strong>{catalog.data?.analyses.length ?? "—"}</strong>
          <span>versioned analysis plugins</span>
        </div>
        <div>
          <strong>23</strong>
          <span>validated reference models</span>
        </div>
        <div>
          <strong>100%</strong>
          <span>numerical provenance retained</span>
        </div>
      </section>
      <section className="feature-section">
        <div className="section-copy">
          <span className="section-kicker">Analysis catalog</span>
          <h2>A focused toolkit, built to expand.</h2>
          <p>
            Each method declares its assumptions, supported model shape,
            configuration schema, and result version. New OpenTURNS capability
            can arrive as a plugin instead of a UI rewrite.
          </p>
        </div>
        <div className="catalog-grid">
          {catalog.isLoading &&
            [1, 2, 3].map((item) => (
              <div className="catalog-card skeleton" key={item} />
            ))}
          {catalog.data?.analyses.map((analysis) => (
            <article className="catalog-card" key={analysis.key}>
              <div className="catalog-icon">
                <Blocks />
              </div>
              <span>{analysis.category}</span>
              <h3>{analysis.name}</h3>
              <p>{analysis.description}</p>
              <footer>
                <code>
                  {analysis.key}@{analysis.version}
                </code>
                <span className={`resource ${analysis.resource_class}`}>
                  {analysis.resource_class}
                </span>
              </footer>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}
