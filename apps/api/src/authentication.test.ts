import { describe, expect, it, vi } from "vitest";

import type { Env } from "./env";

vi.mock("@cloudflare/sandbox", () => ({
  getSandbox: vi.fn(),
  Sandbox: class {},
}));
vi.mock("./sandbox", () => ({ IsolatedComputeSandbox: class {} }));

const { app } = await import("./index");

const unauthenticatedEnv = {
  DB: {},
  BETTER_AUTH_URL: "https://uncertaintycat.test",
} as unknown as Env;

describe("authenticated application boundary", () => {
  it("keeps health and session discovery public without creating a guest identity", async () => {
    const health = await app.request("/health", undefined, unauthenticatedEnv);
    expect(health.status).toBe(200);

    const session = await app.request(
      "/api/v1/session",
      undefined,
      unauthenticatedEnv,
    );
    expect(session.status).toBe(200);
    await expect(session.json()).resolves.toMatchObject({
      identity: { ownerId: "", authenticated: false },
      providers: [],
    });
    expect(session.headers.get("set-cookie") ?? "").not.toContain(
      "uncertaintycat_guest",
    );
  });

  it.each([
    "/api/v1/analyses/catalog",
    "/api/v1/examples",
    "/api/v1/projects",
    "/api/v1/runs",
    "/api/v1/reports/report-id",
    "/api/v1/shared-reports/share-token",
  ])("rejects unauthenticated access to %s", async (path) => {
    const response = await app.request(path, undefined, unauthenticatedEnv);
    expect(response.status).toBe(401);
    await expect(response.json()).resolves.toMatchObject({
      error: { code: "authentication_required" },
    });
  });
});
