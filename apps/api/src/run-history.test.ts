import { readFileSync, readdirSync } from "node:fs";
import { convertV4MiniflareOptions, Miniflare } from "miniflare";
import { expect, it, vi } from "vitest";
import type { Env } from "./env";

vi.mock("@cloudflare/sandbox", () => ({ getSandbox: vi.fn(), Sandbox: class {} }));
vi.mock("./sandbox", () => ({ IsolatedComputeSandbox: class {} }));
const { app } = await import("./index");

it("retains owner-scoped project history beyond 50 newer runs and paginates timestamp ties", async () => {
  const runtime = new Miniflare(convertV4MiniflareOptions({
    modules: true, script: "export default { fetch() { return new Response('ok') } }",
    compatibilityDate: "2026-09-15", d1Databases: { DB: "run-history-test" },
  }));
  try {
    const db = await runtime.getD1Database("DB");
    const directory = new URL("../migrations/", import.meta.url);
    for (const name of readdirSync(directory).filter((file) => file.endsWith(".sql")).sort()) {
      for (const statement of readFileSync(new URL(name, directory), "utf8").split(/;\s*(?:\r?\n|$)/)) {
        if (statement.trim()) await db.prepare(statement.trim()).run();
      }
    }
    for (const [id, owner] of [["audit", "dev-user"], ["other", "dev-user"], ["private", "someone-else"]]) {
      await db.prepare("INSERT INTO projects (id, owner_id, name, created_at, updated_at) VALUES (?, ?, ?, ?, ?)")
        .bind(id, owner, id, "2026-09-16", "2026-09-16").run();
      await db.prepare("INSERT INTO model_versions (id, project_id, version, source_kind, source_key, source_hash, metadata_json, display_name, created_at) VALUES (?, ?, 1, 'example', 'synthetic-key', 'synthetic-hash', '{}', 'Synthetic model', ?)")
        .bind(`model-${id}`, id, "2026-09-16").run();
    }
    const insert = async (project: string, count: number, timestamp: string) => {
      await db.batch(Array.from({ length: count }, (_, index) => db.prepare(
        "INSERT INTO runs (id, owner_id, project_id, model_version_id, status, seed, accuracy_profile, created_at) VALUES (?, ?, ?, ?, 'succeeded', 42, 'standard', ?)"
      ).bind(`${project}-${String(index).padStart(3, "0")}`, project === "private" ? "someone-else" : "dev-user", project, `model-${project}`, timestamp)));
    };
    await insert("audit", 52, "2026-09-15T00:00:00Z");
    await insert("other", 55, "2026-09-16T00:00:00Z");
    await insert("private", 1, "2026-09-17T00:00:00Z");
    const plan = await db.prepare("EXPLAIN QUERY PLAN SELECT id FROM runs WHERE owner_id = ? AND project_id = ? ORDER BY created_at DESC, id DESC LIMIT 51")
      .bind("dev-user", "audit").all<{ detail: string }>();
    expect(plan.results.some((row) => row.detail.includes("runs_owner_project_history_idx"))).toBe(true);
    const env = { DB: db, DEV_AUTH_BYPASS: "true", BETTER_AUTH_URL: "http://127.0.0.1:8787" } as unknown as Env;
    const get = (query: string) => app.request(`/api/v1/runs${query}`, undefined, env);
    const first = await get("?projectId=audit");
    expect(first.status).toBe(200);
    const page = await first.json() as { runs: { id: string; projectId: string }[]; nextCursor: string };
    expect(page.runs).toHaveLength(50);
    expect(page.runs.every((run) => run.projectId === "audit")).toBe(true);
    expect(page.nextCursor).toBe("audit-002");
    const next = await (await get(`?projectId=audit&cursor=${page.nextCursor}`)).json() as { runs: { id: string }[]; nextCursor: null };
    expect(next.runs.map((run) => run.id)).toEqual(["audit-001", "audit-000"]);
    expect(next.nextCursor).toBeNull();
    expect(new Set([...page.runs, ...next.runs].map((run) => run.id)).size).toBe(52);
    expect((await get("?projectId=private")).status).toBe(404);
    expect((await get("?projectId=missing")).status).toBe(404);
    expect((await get("?projectId=audit&cursor=other-000")).status).toBe(400);
    expect((await get("?cursor=private-000")).status).toBe(400);
    expect((await get("?cursor=missing")).status).toBe(400);
    const all = await (await get("")).json() as { runs: { projectId: string }[] };
    expect(all.runs.every((run) => run.projectId === "other")).toBe(true);
  } finally { await runtime.dispose(); }
}, 30_000);
