import { z } from "zod";

export const analysisRequestSchema = z.object({
  analysisKey: z.string().min(1),
  pluginVersion: z.string().optional(),
  config: z.record(z.string(), z.unknown()).default({}),
  outputTargets: z.array(z.number().int().nonnegative()).default([]),
});

export const createRunSchema = z.object({
  modelVersionId: z.string().min(1),
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
  builderSpec: z.record(z.string(), z.unknown()).optional(),
});

export type AnalysisRequest = z.infer<typeof analysisRequestSchema>;
export type CreateRun = z.infer<typeof createRunSchema>;
export type CreateProject = z.infer<typeof createProjectSchema>;
export type CreateModelVersion = z.infer<typeof createModelVersionSchema>;

export interface VariableMetadata {
  index: number;
  name: string;
  distribution?: string | null;
  parameters: number[];
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

export interface ModelVersion {
  id: string;
  projectId: string;
  version: number;
  sourceKind: "python" | "builder" | "example";
  sourceHash: string;
  metadata: ModelMetadata;
  createdAt: string;
}

export interface AnalysisTask {
  id: string;
  analysisKey: string;
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
  listProjects = () => this.request<{ projects: Project[] }>("/projects");
  createProject = (input: CreateProject) =>
    this.request<{ project: Project }>("/projects", { method: "POST", body: JSON.stringify(input) });
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
  cancelRun = (id: string) =>
    this.request<{ status: "cancelled" }>(`/runs/${id}/cancel`, { method: "POST" });
  getReport = (id: string) => this.request<{ report: Report }>(`/reports/${id}`);
  getSharedReport = (token: string) => this.request<{ report: Report }>(`/shared-reports/${token}`);
  createShareLink = (id: string, expiresInDays: number | null = 30) =>
    this.request<{ shareLink: { id: string; url: string; expiresAt: string | null; createdAt: string } }>(
      `/reports/${id}/share-links`,
      { method: "POST", body: JSON.stringify({ expiresInDays }) },
    );
  getChatMessages = (id: string) =>
    this.request<{ messages: ChatMessage[] }>(`/reports/${id}/chat`);
}

export interface Report {
  id: string;
  runId: string;
  title: string;
  status: string;
  generatedAt: string;
  model: ModelMetadata;
  sections: Array<{
    key: string;
    status: string;
    result?: AnalysisResult;
    error?: { code: string; message: string };
  }>;
}
