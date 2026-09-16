import { readFileSync, readdirSync } from "node:fs";
import { convertV4MiniflareOptions, Miniflare } from "miniflare";
import { expect, it, vi } from "vitest";
import type { Env } from "./env";

vi.mock("@cloudflare/sandbox", () => ({ getSandbox: vi.fn(), Sandbox: class {} }));
vi.mock("./sandbox", () => ({ IsolatedComputeSandbox: class {} }));
vi.mock("./compute-client", () => ({ computeFetch: vi.fn(), destroyRunSandbox: vi.fn().mockResolvedValue(undefined) }));
const { app } = await import("./index");
const { computeFetch, destroyRunSandbox } = await import("./compute-client");
const { failRunTask, finalizeRun, processRunTask, requeueRunTask } = await import("./compute");

it("finalizes cancelled reports and preserves successful evidence through late compute success or failure", async () => {
  const runtime = new Miniflare(convertV4MiniflareOptions({
    modules: true,
    script: "export default { fetch() { return new Response('ok') } }",
    compatibilityDate: "2026-09-15",
    d1Databases: { DB: "cancellation-test" },
  }));
  try {
    const db = await runtime.getD1Database("DB");
    const directory = new URL("../migrations/", import.meta.url);
    for (const name of readdirSync(directory).filter((file) => file.endsWith(".sql")).sort()) {
      for (const statement of readFileSync(new URL(name, directory), "utf8").split(/;\s*(?:\r?\n|$)/)) {
        if (statement.trim()) await db.prepare(statement.trim()).run();
      }
    }
    await db.prepare("INSERT INTO projects (id, owner_id, name, created_at, updated_at) VALUES ('project', 'dev-user', 'Cancellation fixture', '2026-09-16', '2026-09-16')").run();
    await db.prepare("INSERT INTO model_versions (id, project_id, version, source_kind, source_key, source_hash, metadata_json, display_name, created_at) VALUES ('model', 'project', 1, 'example', 'synthetic-key', 'synthetic-hash', '{}', 'Synthetic model', '2026-09-16')").run();
    const result = { analysis_key: "monte_carlo", payload: { metrics: { mean: 1.25 } } };
    const resultJson = JSON.stringify(result);
    const createRun = async (id: string, status = "running", owner = "dev-user") => {
      await db.prepare("INSERT INTO runs (id, owner_id, project_id, model_version_id, status, seed, accuracy_profile, created_at) VALUES (?, ?, 'project', 'model', ?, 42, 'standard', '2026-09-16')").bind(id, owner, status).run();
    };
    const createTask = async (id: string, runId: string, status: string, retained: string | null = null) => {
      await db.prepare("INSERT INTO analysis_tasks (id, run_id, analysis_key, status, config_json, output_targets_json, result_json, created_at) VALUES (?, ?, ?, ?, '{}', '[0]', ?, '2026-09-16')").bind(id, runId, id === "active" ? "hsic" : id === "queued" ? "sobol" : "monte_carlo", status, retained).run();
    };
    await createRun("mixed");
    await createTask("retained", "mixed", "succeeded", resultJson);
    await createTask("active", "mixed", "queued");
    await createTask("queued", "mixed", "queued");
    const env = {
      DB: db,
      DEV_AUTH_BYPASS: "true",
      BETTER_AUTH_URL: "http://127.0.0.1:8787",
      ARTIFACTS: { get: vi.fn().mockResolvedValue({ text: async () => "synthetic source fixture" }) },
    } as unknown as Env;
    let finishCompute!: (response: Response) => void;
    vi.mocked(computeFetch).mockImplementation(() => new Promise<Response>((resolve) => { finishCompute = resolve; }));
    const inFlight = processRunTask(env, { runId: "mixed", taskId: "active", attempt: 0 });
    await vi.waitFor(() => expect(computeFetch).toHaveBeenCalledOnce());
    const cancel = (id: string) => app.request(`/api/v1/runs/${id}/cancel`, { method: "POST" }, env);
    expect((await cancel("mixed")).status).toBe(200);
    const reportResponse = await app.request("/api/v1/reports/mixed", undefined, env);
    expect(reportResponse.status).toBe(200);
    const reportBody = await reportResponse.json() as { report: { status: string; sections: { status: string; result?: unknown }[] } };
    expect(reportBody.report.status).toBe("cancelled");
    const retainedRun = await (await app.request("/api/v1/runs/mixed", undefined, env)).json() as { run: { reportId: string } };
    expect(retainedRun.run.reportId).toEqual(expect.any(String));
    expect(reportBody.report.sections).toEqual(expect.arrayContaining([
      expect.objectContaining({ status: "succeeded", result }),
      expect.objectContaining({ status: "cancelled" }),
    ]));
    expect(destroyRunSandbox).toHaveBeenCalledWith(env, "mixed");
    finishCompute(new Response(JSON.stringify({ result }), { headers: { "Content-Type": "application/json" } }));
    await inFlight;
    await failRunTask(env, "active", { code: "late_failure", message: "Late transport failure." });
    await requeueRunTask(env, "active", 1);
    await finalizeRun(env, "mixed");
    expect(await db.prepare("SELECT status FROM runs WHERE id = 'mixed'").first("status")).toBe("cancelled");
    expect(await db.prepare("SELECT status FROM reports WHERE run_id = 'mixed'").first("status")).toBe("cancelled");
    expect(await db.prepare("SELECT COUNT(*) FROM reports WHERE run_id = 'mixed'").first("COUNT(*)")).toBe(1);
    expect(await db.prepare("SELECT result_json FROM analysis_tasks WHERE id = 'retained'").first("result_json")).toBe(resultJson);
    const cancelledTask = await db.prepare("SELECT status, result_json, error_json, progress_json FROM analysis_tasks WHERE id = 'active'").first<{ status: string; result_json: string | null; error_json: string | null; progress_json: string }>();
    expect(cancelledTask).toMatchObject({ status: "cancelled", result_json: null, error_json: null });
    expect(JSON.parse(cancelledTask!.progress_json)).toMatchObject({ phase: "cancelled", percent: 100, indeterminate: false });

    await createRun("queued-only", "queued");
    await createTask("queued-only-task", "queued-only", "queued");
    expect((await cancel("queued-only")).status).toBe(200);
    expect(await db.prepare("SELECT status FROM reports WHERE run_id = 'queued-only'").first("status")).toBe("cancelled");
    expect((await cancel("queued-only")).status).toBe(409);

    await createRun("foreign", "running", "another-owner");
    await createTask("foreign-task", "foreign", "queued");
    expect((await cancel("foreign")).status).toBe(409);
    expect(await db.prepare("SELECT status FROM runs WHERE id = 'foreign'").first("status")).toBe("running");
    expect(await db.prepare("SELECT COUNT(*) FROM reports WHERE run_id = 'foreign'").first("COUNT(*)")).toBe(0);
  } finally {
    await runtime.dispose();
  }
}, 30_000);
