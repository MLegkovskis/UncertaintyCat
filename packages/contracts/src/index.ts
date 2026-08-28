import { z } from "zod";

export interface ExampleCatalogEntry {
  id: string;
  title: string;
  filename: string;
  domain: string;
  inputDimension: number;
  outputDimension: number;
  summary: string;
  difficulty: "introductory" | "intermediate" | "advanced";
  suggestedAnalyses: readonly string[];
  equations?: readonly {
    outputName: string;
    latex: string;
  }[];
  source: string;
  sha256: string;
}

export { EXAMPLE_CATALOG } from "./example-catalog.generated";

export const analysisRequestSchema = z.object({
  analysisKey: z.string().min(1),
  pluginVersion: z.string().optional(),
  config: z.record(z.string(), z.unknown()).default({}),
  outputTargets: z.array(z.number().int().nonnegative()).default([]),
});

export const createRunSchema = z.object({
  modelVersionId: z.string().min(1),
  surrogateModelId: z.string().min(1).optional(),
  analyses: z.array(analysisRequestSchema).min(1),
  seed: z.number().int().nonnegative().max(2_147_483_647).default(42),
  accuracyProfile: z.enum(["preview", "standard", "high"]).default("standard"),
  idempotencyKey: z.string().max(128).optional(),
});

export const createProjectSchema = z.object({
  name: z.string().trim().min(1).max(120),
  description: z.string().trim().max(2_000).default(""),
});

export const createModelVersionSchema = z.object({
  source: z.string().min(1).max(262_144),
  sourceKind: z.enum(["python", "builder", "example"]).default("python"),
  displayName: z.string().trim().min(1).max(120).optional(),
  builderSpec: z.record(z.string(), z.unknown()).optional(),
  parentVersionId: z.string().min(1).optional(),
  derivation: z.record(z.string(), z.unknown()).optional(),
});

export const uploadDatasetSchema = z.object({
  projectId: z.string().min(1),
  name: z.string().trim().min(1).max(160),
  sourceKind: z.enum(["csv", "xlsx", "paste"]),
  contentBase64: z.string().min(1).max(20_000_000),
});

export const distributionFitSchema = z.object({
  selectedColumns: z.array(z.string().min(1)).min(1).max(10),
  candidates: z
    .array(
      z.enum([
        "Normal",
        "Uniform",
        "LogNormal",
        "Exponential",
        "Gamma",
        "Beta",
        "Triangular",
        "KernelSmoothing",
      ]),
    )
    .min(1)
    .max(8),
  selectedMarginals: z.record(z.string(), z.string()).default({}),
  copula: z.enum(["independent", "normal", "bernstein"]).default("independent"),
  significanceLevel: z.number().gt(0).lt(1).default(0.05),
});

export const createReducedModelSchema = z.object({
  morrisRunId: z.string().min(1),
  displayName: z.string().trim().min(1).max(120),
  fixedVariables: z
    .array(
      z.object({
        index: z.number().int().nonnegative(),
        value: z.number().finite(),
      }),
    )
    .min(1),
  confirmed: z.literal(true),
});

export const createSurrogateSchema = z.object({
  method: z.enum(["pce", "gpr"]),
  config: z.record(z.string(), z.unknown()).default({}),
  outputTarget: z.number().int().nonnegative().default(0),
  seed: z.number().int().nonnegative().max(2_147_483_647).default(42),
});

export const createDataSurrogateSchema = z.object({
  inputColumns: z.array(z.string().min(1)).min(1).max(40),
  outputColumn: z.string().min(1).max(200),
  validationFraction: z.number().min(0.1).max(0.5).default(0.2),
  kernel: z.enum(["MATERN_1_5", "MATERN_2_5", "SQUARED_EXPONENTIAL"]).default("MATERN_2_5"),
  trend: z.enum(["CONSTANT", "LINEAR"]).default("CONSTANT"),
  seed: z.number().int().nonnegative().max(2_147_483_647).default(42),
});

export const promoteSurrogateSchema = z.object({
  acknowledgeOverride: z.boolean().default(false),
  reason: z.string().trim().max(1_000).default(""),
});

export const copySurrogateSchema = z.object({
  targetProjectId: z.string().min(1),
  targetModelVersionId: z.string().min(1),
});

export type AnalysisRequest = z.infer<typeof analysisRequestSchema>;
export type CreateRun = z.infer<typeof createRunSchema>;
export type CreateProject = z.infer<typeof createProjectSchema>;
export type CreateModelVersion = z.infer<typeof createModelVersionSchema>;
export type UploadDataset = z.infer<typeof uploadDatasetSchema>;
export type DistributionFitInput = z.infer<typeof distributionFitSchema>;
export type CreateReducedModel = z.infer<typeof createReducedModelSchema>;
export type CreateSurrogate = z.infer<typeof createSurrogateSchema>;
export type CreateDataSurrogate = z.infer<typeof createDataSurrogateSchema>;
export type PromoteSurrogate = z.infer<typeof promoteSurrogateSchema>;
export type CopySurrogate = z.infer<typeof copySurrogateSchema>;

export interface VariableMetadata {
  index: number;
  name: string;
  distribution?: string | null;
  parameters: number[];
  kind?: "continuous" | "discrete" | "mixed" | "unknown";
  mean?: number | null;
  standard_deviation?: number | null;
}

export interface ModelMetadata {
  source_hash: string;
  input_dimension: number;
  output_dimension: number;
  inputs: VariableMetadata[];
  outputs: Array<{ index: number; name: string }>;
  openturns_version: string;
  batch_evaluation_supported: boolean;
  validation_runtime_ms: number;
  warnings: string[];
  validation_sample_size?: number;
  function_type?: string;
  exact_gradient_available?: boolean;
  exact_hessian_available?: boolean;
  copula?: string;
  dependent_inputs?: boolean;
}

export interface PilotOutputSummary {
  output_index: number;
  output_name: string;
  minimum: number;
  maximum: number;
  mean: number;
  standard_deviation: number;
  quantile_05: number;
  quantile_95: number;
  variable: boolean;
}

export interface AnalysisRecommendation {
  capability: string;
  status: "recommended" | "available" | "incompatible";
  priority: number;
  rationale_codes: string[];
  projected_evaluations?: number | null;
  projected_runtime_ms?: number | null;
  compatibility_warnings: string[];
}

export interface ModelAssessment {
  version: string;
  profile: {
    input_dimension: number;
    output_dimension: number;
    continuous_marginals: number;
    discrete_marginals: number;
    copula: string;
    dependent_inputs: boolean;
    function_type: string;
    batch_support: boolean;
    validation_evaluation_runtime_ms: number;
    projected_1000_evaluation_runtime_ms: number;
    pilot_sample_size: number;
    pilot_outputs: PilotOutputSummary[];
  };
  workflow?: {
    path: "direct" | "dimensionality_reduction" | "surrogate";
    rationale_codes: string[];
  };
  recommendations: AnalysisRecommendation[];
}

export interface TableData {
  columns: string[];
  rows: Array<Array<string | number | boolean | null>>;
  row_count: number;
  truncated: boolean;
}

export interface MatrixData {
  row_labels: string[];
  column_labels: string[];
  values: Array<Array<number | null>>;
}

export interface SeriesData {
  name: string;
  x: unknown[];
  y: unknown[];
  x_label?: string;
  y_label?: string;
}

export interface AnalysisResult {
  analysis_key: string;
  plugin_version: string;
  result_schema_version: string;
  model_hash: string;
  seed: number;
  uq_core_version: string;
  openturns_version: string;
  status: "succeeded";
  started_at: string;
  completed_at: string;
  runtime: { duration_ms: number; model_evaluations: number; sample_size?: number | null };
  warnings: string[];
  assumptions: string[];
  payload: {
    metrics: Record<string, string | number | boolean | null>;
    tables: Record<string, TableData>;
    series: Record<string, SeriesData>;
    matrices: Record<string, MatrixData>;
    facts: Record<string, string | number | boolean | null>;
  };
}

export interface AnalysisCatalogEntry {
  key: string;
  version: string;
  name: string;
  category: string;
  description: string;
  assumptions: string[];
  supports_dependent_inputs: boolean;
  requires_dependent_inputs: boolean;
  supports_multi_output: boolean;
  resource_class: "lite" | "standard" | "heavy";
  config_schema: Record<string, unknown>;
}

export interface Project {
  id: string;
  name: string;
  description: string;
  createdAt: string;
  updatedAt: string;
}

export interface SessionPolicy {
  identity: {
    ownerId: string;
    authenticated: boolean;
    name?: string;
    email?: string;
  };
  providers: Array<"cloudflare">;
  ai: {
    provider: "groq" | "cloudflare";
    configured: boolean;
    modelUnderstanding: { modelId: string; label: string };
    reportChat: { modelId: string; label: string };
  };
}

export interface ModelVersion {
  id: string;
  projectId: string;
  version: number;
  sourceKind: "python" | "builder" | "example";
  displayName: string;
  sourceHash: string;
  metadata: ModelMetadata;
  assessment?: ModelAssessment | null;
  parentVersionId?: string | null;
  derivation?: Record<string, unknown> | null;
  createdAt: string;
}

export interface ModelDefinition {
  modelVersion: ModelVersion;
  project: Project;
  source: string;
  builderSpec?: Record<string, unknown> | null;
  visibility: "owner" | "shared";
}

export interface ModelUnderstanding {
  id: string;
  modelVersionId: string;
  modelHash: string;
  promptVersion: string;
  aiModelId: string;
  status: "pending" | "generating" | "succeeded" | "failed";
  content?: string | null;
  error?: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface DatasetColumn {
  name: string;
  type: "numeric" | "text";
  missingCount: number;
  invalidNumericCount: number;
  nonFiniteCount: number;
  finiteCount: number;
  uniqueCount: number;
  minimum?: number;
  maximum?: number;
  mean?: number;
}

export interface Dataset {
  id: string;
  projectId: string;
  name: string;
  sourceKind: "csv" | "xlsx" | "paste";
  sha256: string;
  rowCount: number;
  columns: DatasetColumn[];
  preview: Array<Record<string, string | number | boolean | null>>;
  warnings: string[];
  createdAt: string;
}

export interface DistributionFitRanking {
  candidate: string;
  distribution: string;
  parameters: number[];
  parameterDescription: string[];
  bic: number | null;
  aic: number | null;
  aicc: number | null;
  test: {
    name: string;
    statistic: number;
    pValue: number;
    significanceLevel: number;
    rejected: boolean;
  };
}

export interface FittedColumnResult {
  column: string;
  sampleSize: number;
  warnings: string[];
  rankings: DistributionFitRanking[];
  rejectedCandidates: Array<{ candidate: string; reason: string }>;
  selectedMarginal?: string | null;
  plot: {
    sample: number[];
    pdf: { x: number[]; y: number[] };
    cdf: { empiricalX: number[]; empiricalY: number[]; fittedX: number[]; fittedY: number[] };
    qq: { theoretical: number[]; observed: number[] };
  };
}

export interface DistributionFitResult {
  openturnsVersion: string;
  columns: FittedColumnResult[];
  copula?: {
    kind: "independent" | "normal" | "bernstein";
    className: string;
    correlation?: number[][];
  } | null;
  generatedSource?: string | null;
  builderSpec?: Record<string, unknown> | null;
  assumptions: string[];
}

export interface DistributionFitRun {
  id: string;
  datasetId: string;
  status: "queued" | "running" | "succeeded" | "failed";
  config: DistributionFitInput;
  result?: DistributionFitResult | null;
  generatedSource?: string | null;
  error?: { code: string; message: string } | null;
  openturnsVersion?: string | null;
  createdAt: string;
  completedAt?: string | null;
}

export interface SurrogateModel {
  id: string;
  projectId: string;
  sourceModelVersionId: string;
  sourceModelHash: string;
  method: "pce" | "gpr";
  pluginVersion: string;
  openturnsVersion: string;
  status: "draft" | "validated" | "promoted" | "rejected";
  validation: {
    config: Record<string, unknown>;
    outputTargets: number[];
    seed: number;
    result: AnalysisResult;
    guidance: {
      score: number;
      normalizedRmse: number;
      scoreThreshold: number;
      normalizedRmseThreshold: number;
      meetsDefault: boolean;
    };
  };
  acknowledgement?: { acknowledgeOverride: boolean; reason: string } | null;
  artifact?: { sha256: string; sizeBytes: number; resultType: string } | null;
  createdAt: string;
  promotedAt?: string | null;
}

export interface DataSurrogateModel {
  id: string;
  projectId: string;
  datasetId: string;
  method: "gpr";
  pluginVersion: string;
  openturnsVersion: string;
  inputColumns: string[];
  outputColumn: string;
  config: {
    kernel: "MATERN_1_5" | "MATERN_2_5" | "SQUARED_EXPONENTIAL";
    trend: "CONSTANT" | "LINEAR";
    seed: number;
    validationFraction: number;
  };
  validation: {
    trainingSize: number;
    validationSize: number;
    r2: number;
    rmse: number;
    normalizedRmse: number;
    meetsDefault: boolean;
    observed: number[];
    predicted: number[];
  };
  artifact: { sha256: string; sizeBytes: number; resultType: string };
  createdAt: string;
}

export interface AnalysisTask {
  id: string;
  analysisKey: string;
  pluginVersion?: string | null;
  config: Record<string, unknown>;
  outputTargets: number[];
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  result?: AnalysisResult;
  error?: { code: string; message: string };
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: string;
}

export interface Run {
  id: string;
  projectId: string;
  modelVersionId: string;
  surrogateModelId?: string | null;
  evidenceSource?: "direct" | "surrogate";
  projectName?: string;
  modelDisplayName?: string;
  modelVersion?: number;
  sourceKind?: ModelVersion["sourceKind"];
  status: "queued" | "running" | "succeeded" | "partially_succeeded" | "failed" | "cancelled";
  seed: number;
  accuracyProfile: "preview" | "standard" | "high";
  createdAt: string;
  completedAt?: string | null;
  tasks: AnalysisTask[];
}

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly code: string,
    message: string,
  ) {
    super(message);
  }
}

export class ApiClient {
  constructor(private readonly baseUrl = "/api/v1") {}

  private async request<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
      credentials: "include",
    });
    if (!response.ok) {
      const body = (await response.json().catch(() => ({}))) as {
        error?: { code?: string; message?: string };
      };
      throw new ApiError(
        response.status,
        body.error?.code ?? "request_failed",
        body.error?.message ?? `Request failed (${response.status})`,
      );
    }
    return (await response.json()) as T;
  }

  catalog = () => this.request<{ analyses: AnalysisCatalogEntry[] }>("/analyses/catalog");
  examples = () => this.request<{ examples: readonly ExampleCatalogEntry[] }>("/examples");
  example = (id: string) => this.request<{ example: ExampleCatalogEntry }>(`/examples/${id}`);
  session = () => this.request<SessionPolicy>("/session");
  listProjects = () => this.request<{ projects: Project[] }>("/projects");
  createProject = (input: CreateProject) =>
    this.request<{ project: Project }>("/projects", { method: "POST", body: JSON.stringify(input) });
  deleteProject = (projectId: string) =>
    this.request<{ deletedProjectId: string; deletedArtifactCount: number }>(
      `/projects/${projectId}`,
      { method: "DELETE" },
    );
  createModel = (projectId: string, input: CreateModelVersion) =>
    this.request<{ modelVersion: ModelVersion }>(`/projects/${projectId}/models`, {
      method: "POST",
      body: JSON.stringify(input),
    });
  listModels = (projectId: string) =>
    this.request<{ modelVersions: ModelVersion[] }>(`/projects/${projectId}/models`);
  createRun = (input: CreateRun) =>
    this.request<{ run: Run }>("/runs", { method: "POST", body: JSON.stringify(input) });
  listRuns = () => this.request<{ runs: Run[] }>("/runs");
  getRun = (id: string) => this.request<{ run: Run }>(`/runs/${id}`);
  rerun = (id: string) =>
    this.request<{ run: Run }>(`/runs/${id}/rerun`, { method: "POST" });
  cancelRun = (id: string) =>
    this.request<{ status: "cancelled" }>(`/runs/${id}/cancel`, { method: "POST" });
  getReport = (id: string) => this.request<{ report: Report }>(`/reports/${id}`);
  getSharedReport = (token: string) => this.request<{ report: Report }>(`/shared-reports/${token}`);
  createShareLink = (
    id: string,
    expiresInDays: number | null = 30,
    includeModelDefinition = false,
  ) =>
    this.request<{ shareLink: { id: string; url: string; expiresAt: string | null; createdAt: string } }>(
      `/reports/${id}/share-links`,
      {
        method: "POST",
        body: JSON.stringify({ expiresInDays, includeModelDefinition }),
      },
    );
  getModelDefinition = (id: string) =>
    this.request<{ definition: ModelDefinition }>(`/model-versions/${id}/definition`);
  getModelUnderstanding = (id: string) =>
    this.request<{ understanding: ModelUnderstanding | null }>(
      `/model-versions/${id}/understanding`,
    );
  listDatasets = (projectId: string) =>
    this.request<{ datasets: Dataset[] }>(`/projects/${projectId}/datasets`);
  uploadDataset = (input: UploadDataset) =>
    this.request<{ dataset: Dataset }>("/datasets", {
      method: "POST",
      body: JSON.stringify(input),
    });
  fitDataset = (datasetId: string, input: DistributionFitInput) =>
    this.request<{ fitRun: DistributionFitRun }>(`/datasets/${datasetId}/fits`, {
      method: "POST",
      body: JSON.stringify(input),
    });
  listDistributionFits = (datasetId: string) =>
    this.request<{ fitRuns: DistributionFitRun[] }>(`/datasets/${datasetId}/fits`);
  createReducedModel = (modelVersionId: string, input: CreateReducedModel) =>
    this.request<{ modelVersion: ModelVersion }>(
      `/model-versions/${modelVersionId}/derived-reduction`,
      { method: "POST", body: JSON.stringify(input) },
    );
  listSurrogates = (projectId: string) =>
    this.request<{ surrogates: SurrogateModel[] }>(`/projects/${projectId}/surrogates`);
  createSurrogate = (modelVersionId: string, input: CreateSurrogate) =>
    this.request<{ surrogate: SurrogateModel }>(`/model-versions/${modelVersionId}/surrogates`, {
      method: "POST",
      body: JSON.stringify(input),
    });
  listDataSurrogates = (projectId: string) =>
    this.request<{ surrogates: DataSurrogateModel[] }>(`/projects/${projectId}/data-surrogates`);
  createDataSurrogate = (datasetId: string, input: CreateDataSurrogate) =>
    this.request<{ surrogate: DataSurrogateModel }>(`/datasets/${datasetId}/surrogates`, {
      method: "POST",
      body: JSON.stringify(input),
    });
  promoteSurrogate = (surrogateId: string, input: PromoteSurrogate) =>
    this.request<{ surrogate: SurrogateModel }>(`/surrogates/${surrogateId}/promote`, {
      method: "POST",
      body: JSON.stringify(input),
    });
  copySurrogate = (surrogateId: string, input: CopySurrogate) =>
    this.request<{ surrogate: SurrogateModel }>(`/surrogates/${surrogateId}/copy`, {
      method: "POST",
      body: JSON.stringify(input),
    });
  getChatMessages = (id: string) =>
    this.request<{ messages: ChatMessage[] }>(`/reports/${id}/chat`);
}

export interface Report {
  id: string;
  runId: string;
  title: string;
  status: string;
  generatedAt: string;
  project: { id: string; name: string };
  modelVersion: {
    id: string;
    version: number;
    displayName: string;
    sourceKind: ModelVersion["sourceKind"];
    createdAt: string;
    parentVersionId?: string | null;
  };
  seed: number;
  accuracyProfile: Run["accuracyProfile"];
  evidenceSource: "direct" | "surrogate";
  surrogate?: {
    id: string;
    method: "pce" | "gpr";
    pluginVersion: string;
    openturnsVersion: string;
  } | null;
  model: ModelMetadata;
  modelDefinition?: ModelDefinition;
  sections: Array<{
    key: string;
    status: string;
    result?: AnalysisResult;
    error?: { code: string; message: string };
  }>;
}
