import { describe, expect, it, vi } from "vitest";
import { boundedSubsetConfigSchema, subsetSamplingIncompatibility, type ModelAssessment } from "@uncertaintycat/contracts";
import type { Env } from "./env";

vi.mock("@cloudflare/sandbox", () => ({ getSandbox: vi.fn(), Sandbox: class {} }));
vi.mock("./sandbox", () => ({ IsolatedComputeSandbox: class {} }));
const { app } = await import("./index");

const assessment: ModelAssessment = {
  version: "1.4.0",
  profile: { input_dimension: 20, output_dimension: 1, continuous_marginals: 20,
    discrete_marginals: 0, copula: "IndependentCopula", dependent_inputs: false,
    function_type: "SymbolicFunction", batch_support: true,
    validation_evaluation_runtime_ms: 1, projected_1000_evaluation_runtime_ms: 125, pilot_sample_size: 8,
    pilot_outputs: [{ output_index: 0, output_name: "y", minimum: -1, maximum: 1,
      mean: 0, standard_deviation: 1, quantile_05: -0.9, quantile_95: 0.9, variable: true }] },
  recommendations: [{ capability: "reliability", status: "available", priority: 4,
    rationale_codes: [], compatibility_warnings: [], safe_config: {
    subset_sampling_available: true, subset_maximum_evaluations: 50_000,
  } }],
};

describe("bounded subset contract", () => {
  it("admits the maximum-dimensional UI default and rejects the first excessive budget", () => {
    expect(subsetSamplingIncompatibility(assessment, 0)).toBeUndefined();
    const config = boundedSubsetConfigSchema.parse({ method: "SUBSET_SAMPLING", threshold: 0 });
    expect(config).toMatchObject({ subset_sample_size: 2000, maximum_evaluations: 20000, block_size: 1 });
    for (const extra of [
      { maximum_evaluations: 50001 }, { maximum_evaluations: 2000000 },
      { maximum_evaluations: 1000 }, { subset_sample_size: 101 },
      { threshold: Infinity }, { sample_size: 1000 }, { block_size: 10 }, { unknown: true },
    ]) expect(boundedSubsetConfigSchema.safeParse({ ...config, ...extra }).success).toBe(false);
  });

  it.each([
    [{ ...assessment, profile: { ...assessment.profile, input_dimension: 21 } }, "20 inputs"],
    [{ ...assessment, profile: { ...assessment.profile, continuous_marginals: 19 } }, "continuous inputs"],
    [{ ...assessment, profile: { ...assessment.profile, pilot_outputs: [{ ...assessment.profile.pilot_outputs[0]!, variable: false }] } }, "varies"],
    [null, "Revalidate"],
  ])("preserves model-specific admission", (value, reason) => {
    expect(subsetSamplingIncompatibility(value as ModelAssessment | null, 0)).toContain(reason);
  });

  it.each([
    [{ method: "SUBSET_SAMPLING", threshold: 0, maximum_evaluations: 50001 }, assessment, "invalid_subset_config"],
    [{ method: "SUBSET_SAMPLING", threshold: 0 }, { ...assessment, profile: { ...assessment.profile, continuous_marginals: 0 } }, "analysis_incompatible"],
    [{ method: "SUBSET_SAMPLING", threshold: 0, output_targets: [1] }, assessment, "invalid_subset_config"],
  ])("rejects unsafe authenticated requests before queue or persistence", async (config, modelAssessment, code) => {
    const first = vi.fn().mockResolvedValue({ id: "model", project_id: "project", assessment_json: JSON.stringify(modelAssessment) });
    const bind = vi.fn().mockReturnValue({ first });
    const prepare = vi.fn().mockReturnValue({ bind });
    const send = vi.fn();
    const env = { DB: { prepare }, RUN_QUEUE: { send }, BETTER_AUTH_URL: "http://127.0.0.1:8787", DEV_AUTH_BYPASS: "true" } as unknown as Env;
    const response = await app.request("/api/v1/runs", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ modelVersionId: "model", analyses: [{ analysisKey: "reliability", config, outputTargets: [0] }] }),
    }, env);
    expect(response.status).toBe(422);
    expect(await response.json()).toMatchObject({ error: { code } });
    expect(prepare).toHaveBeenCalledTimes(1);
    expect(prepare.mock.calls[0]![0]).toContain("p.owner_id = ?");
    expect(send).not.toHaveBeenCalled();
  });
});
