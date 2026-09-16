import { describe, expect, it, vi } from "vitest";
import { distributionFitSchema } from "@uncertaintycat/contracts";
import type { Env } from "./env";

const compute = vi.hoisted(() => vi.fn());
vi.mock("@cloudflare/sandbox", () => ({ getSandbox: vi.fn(), Sandbox: class {} }));
vi.mock("./sandbox", () => ({ IsolatedComputeSandbox: class {} }));
vi.mock("./compute-client", () => ({ computeFetch: compute, destroyRunSandbox: vi.fn() }));
const { app } = await import("./index");

describe("distribution fitting seed provenance", () => {
  it("admits whole bounded seeds and supplies the documented default", () => {
    const input = { selectedColumns: ["x"], candidates: ["Normal"] };
    expect(distributionFitSchema.parse(input).seed).toBe(42);
    expect(distributionFitSchema.parse({ ...input, seed: 2_147_483_647 }).seed).toBe(2_147_483_647);
    for (const seed of [-1, 2_147_483_648, 1.5, Infinity]) {
      expect(distributionFitSchema.safeParse({ ...input, seed }).success).toBe(false);
    }
  });

  it.each([undefined, 73])("forwards and persists effective fit seed %s", async (seed) => {
    const effectiveSeed = seed ?? 42;
    const fit = { openturnsVersion: "1.27.post1", fittingVersion: "1.1.0", seed: effectiveSeed, columns: [], assumptions: [] };
    compute.mockReset().mockResolvedValue(new Response(JSON.stringify({ fit }), { headers: { "Content-Type": "application/json" } }));
    const writes: Array<{ sql: string; values: unknown[] }> = [];
    const prepare = vi.fn((sql: string) => ({
      bind: (...values: unknown[]) => ({
        first: async () => ({ id: "dataset-1", source_kind: "paste", object_key: "synthetic-dataset" }),
        run: async () => { writes.push({ sql, values }); return { success: true }; },
      }),
    }));
    const env = {
      DB: { prepare },
      ARTIFACTS: { get: async () => ({ arrayBuffer: async () => new TextEncoder().encode("x\n1\n2\n3\n4\n5\n").buffer }) },
      BETTER_AUTH_URL: "http://127.0.0.1:8787", DEV_AUTH_BYPASS: "true",
    } as unknown as Env;
    const response = await app.request("/api/v1/datasets/dataset-1/fits", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ selectedColumns: ["x"], candidates: ["Normal"], ...(seed === undefined ? {} : { seed }) }),
    }, env);
    expect(response.status).toBe(201);
    expect(JSON.parse(compute.mock.calls[0]![2].body)).toMatchObject({ seed: effectiveSeed });
    expect(prepare.mock.calls[0]![0]).toContain("owner_id = ?");
    const storedConfig = writes.find((entry) => entry.sql.includes("INSERT INTO data_analysis_runs"))!.values[3];
    expect(JSON.parse(String(storedConfig))).toMatchObject({ seed: effectiveSeed });
    expect(await response.json()).toMatchObject({ fitRun: { config: { seed: effectiveSeed }, result: fit } });
  });
});
