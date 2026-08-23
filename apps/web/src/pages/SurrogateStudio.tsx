import { useQuery } from "@tanstack/react-query";
import { ArrowRight, Waves } from "lucide-react";
import { useState } from "react";
import { useSearchParams } from "react-router-dom";

import { api } from "../api";
import { StudioModelPicker } from "../components/StudioModelPicker";
import { SurrogateWorkbench } from "../components/SurrogateWorkbench";

export function SurrogateStudio() {
  const [searchParams] = useSearchParams();
  const [projectId, setProjectId] = useState(searchParams.get("projectId") ?? "");
  const [modelId, setModelId] = useState(searchParams.get("modelId") ?? "");
  const modelsQuery = useQuery({ queryKey: ["models", projectId], queryFn: () => api.listModels(projectId), enabled: Boolean(projectId) });
  const model = modelsQuery.data?.modelVersions.find((item) => item.id === modelId);

  return (
    <div className="page scientific-studio-page">
      <div className="page-heading split">
        <div>
          <span className="section-kicker">Surrogate Studio</span>
          <h1>Approximate an expensive model deliberately.</h1>
          <p>Build and validate a Gaussian process or polynomial chaos metamodel before explicitly using it as an analysis evidence source.</p>
        </div>
        <div className="documentation-links">
          <a className="button secondary" href="https://openturns.github.io/openturns/latest/theory/meta_modeling/gaussian_process_regression.html" target="_blank" rel="noreferrer">GPR method <ArrowRight /></a>
          <a className="button secondary" href="https://openturns.github.io/openturns/latest/theory/meta_modeling/functional_chaos.html" target="_blank" rel="noreferrer">Functional chaos <ArrowRight /></a>
        </div>
      </div>
      <div className="scientific-method-note"><Waves /><div><strong>When this route is recommended</strong><p>UncertaintyCat recommends a surrogate when the measured direct projection exceeds five seconds per 1,000 evaluations and baseline eligibility is established. Every candidate receives independent hold-out validation before promotion.</p></div></div>
      <StudioModelPicker projectId={projectId} modelId={modelId} onProjectChange={setProjectId} onModelChange={setModelId} returnTo="surrogates" />
      {model && <SurrogateWorkbench model={model} projectId={projectId} />}
    </div>
  );
}
