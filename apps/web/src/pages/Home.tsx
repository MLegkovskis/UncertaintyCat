import { useQuery } from "@tanstack/react-query";
import {
  ArrowRight,
  BarChart3,
  Braces,
  Cloud,
  Database,
  Gauge,
  ScanSearch,
  Waves,
} from "lucide-react";
import { Navigate } from "react-router-dom";

import { api } from "../api";
import { authClient } from "../auth-client";

const methodGroups = [
  {
    icon: <BarChart3 />,
    title: "Uncertainty quantification",
    methods: "Output distributions · summary statistics · expectation convergence · exploratory analysis",
  },
  {
    icon: <Gauge />,
    title: "Sensitivity analysis",
    methods: "Sobol · FAST · HSIC · Taylor expansion · correlation analysis",
  },
  {
    icon: <ScanSearch />,
    title: "Screening & reduction",
    methods: "Morris elementary effects · explicit reduced-model generation",
  },
  {
    icon: <Waves />,
    title: "Surrogate modelling",
    methods: "Gaussian process regression · polynomial chaos · independent validation",
  },
  {
    icon: <Database />,
    title: "Distribution fitting",
    methods: "Marginal ranking · goodness-of-fit evidence · copula composition",
  },
  {
    icon: <Gauge />,
    title: "Reliability",
    methods: "FORM · SORM · Monte Carlo · directional and subset simulation",
  },
] as const;

function DistributionPreview() {
  return (
    <svg viewBox="0 0 520 240" role="img" aria-label="Example output distribution and uncertainty interval">
      <defs>
        <linearGradient id="area" x1="0" y1="0" x2="0" y2="1">
          <stop stopColor="currentColor" stopOpacity="0.38" />
          <stop offset="1" stopColor="currentColor" stopOpacity="0.02" />
        </linearGradient>
      </defs>
      <g className="landing-grid">
        <path d="M34 30v174h458M34 66h458M34 102h458M34 138h458M34 174h458M126 30v174M218 30v174M310 30v174M402 30v174" />
      </g>
      <path className="landing-area" d="M34 204C82 203 99 200 128 190c34-12 55-45 78-82 25-41 44-64 72-64 31 0 48 30 73 73 23 39 43 67 76 77 24 7 39 9 65 10v0H34Z" />
      <path className="landing-curve" d="M34 204C82 203 99 200 128 190c34-12 55-45 78-82 25-41 44-64 72-64 31 0 48 30 73 73 23 39 43 67 76 77 24 7 39 9 65 10" />
      <path className="landing-bound" d="M166 204V139M389 204v-36" />
      <text x="150" y="225">2.5%</text><text x="375" y="225">97.5%</text>
    </svg>
  );
}

function SensitivityPreview() {
  const bars = [160, 112, 76, 43];
  return (
    <svg viewBox="0 0 420 230" role="img" aria-label="Example sensitivity index ranking">
      <g className="landing-grid"><path d="M100 24v174h286M171 24v174M242 24v174M313 24v174M384 24v174" /></g>
      {bars.map((width, index) => (
        <g key={width} transform={`translate(0 ${index * 42})`}>
          <text x="20" y="58">x{index + 1}</text>
          <rect className="landing-bar-track" x="100" y="38" width="260" height="22" rx="3" />
          <rect className="landing-bar" x="100" y="38" width={width} height="22" rx="3" />
        </g>
      ))}
      <text x="100" y="218">Influence on output variance</text>
    </svg>
  );
}

export function Home() {
  const session = useQuery({ queryKey: ["session-policy"], queryFn: api.session });
  if (session.data?.identity.authenticated) return <Navigate to="/studies" replace />;

  const signIn = () =>
    authClient.signIn.social({
      provider: "cloudflare",
      callbackURL: `${window.location.origin}/studies`,
    });

  return (
    <div className="page home-page">
      <section className="landing-hero">
        <div className="landing-hero-copy">
          <span className="eyebrow"><Braces /> OpenTURNS in the browser</span>
          <h1>Understand what uncertainty does to your model.</h1>
          <p>
            UncertaintyCat is an interactive web application for uncertainty
            quantification and sensitivity analysis. Define a Python model as
            <code> y = f(x)</code>, describe uncertain inputs with OpenTURNS,
            and turn numerical runs into clear, reproducible reports.
          </p>
          <div className="hero-actions">
            <button className="button primary" type="button" onClick={signIn}>
              <Cloud /> Sign in with Cloudflare
            </button>
            <a className="button secondary" href="#methods">
              Explore methods <ArrowRight />
            </a>
          </div>
        </div>
        <div className="hero-result-card" aria-label="Example uncertainty result">
          <header><span>Output distribution</span><small>10,000 evaluations</small></header>
          <DistributionPreview />
          <footer>
            <span><small>Mean</small><strong>14.07</strong></span>
            <span><small>Std. deviation</small><strong>3.21</strong></span>
            <span><small>95% interval</small><strong>8.12–20.64</strong></span>
          </footer>
        </div>
      </section>

      <section className="workflow-story" aria-labelledby="workflow-title">
        <div className="section-copy">
          <span className="section-kicker">One scientific workflow</span>
          <h2 id="workflow-title">From Python function to numerical evidence.</h2>
          <p>The application keeps the model, uncertainty definition, selected method, run configuration, and result together inside one project.</p>
        </div>
        <div className="workflow-layers">
          <article><i>01</i><Braces /><div><strong>Define the model</strong><span>Python <code>f(x)</code> or an OpenTURNS SymbolicFunction</span></div></article>
          <article><i>02</i><Database /><div><strong>Describe uncertainty</strong><span>Marginal distributions and input dependence</span></div></article>
          <article><i>03</i><Gauge /><div><strong>Validate &amp; assess</strong><span>Shape, finite evaluations, cost, and method compatibility</span></div></article>
          <article><i>04</i><BarChart3 /><div><strong>Run OpenTURNS</strong><span>UQ, sensitivity, screening, surrogates, or reliability</span></div></article>
          <article><i>05</i><ArrowRight /><div><strong>Inspect the report</strong><span>Interactive plots, exact source, provenance, and PDF export</span></div></article>
        </div>
      </section>

      <section className="feature-section" id="methods">
        <div className="section-copy">
          <span className="section-kicker">Methods available</span>
          <h2>Choose the analysis that answers your engineering question.</h2>
          <p>Every numerical method is implemented by OpenTURNS or its official OTMorris extension. UncertaintyCat supplies the workflow, validation, visualisation, and retained evidence.</p>
        </div>
        <div className="method-grid">
          {methodGroups.map((group) => (
            <article className="method-card" key={group.title}>
              <span>{group.icon}</span><h3>{group.title}</h3><p>{group.methods}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="landing-showcase">
        <div className="showcase-chart"><SensitivityPreview /></div>
        <div className="section-copy">
          <span className="section-kicker">Readable by design</span>
          <h2>See which inputs matter—and why.</h2>
          <p>Interactive reports expose indices, diagnostics, assumptions, convergence evidence, and the exact model source. AI explanations are clearly separated from deterministic numerical results.</p>
          <button className="button primary" type="button" onClick={signIn}>Create your first project <ArrowRight /></button>
        </div>
      </section>
    </div>
  );
}
