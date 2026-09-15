import { readFileSync } from "node:fs";

import { convertV4MiniflareOptions, Miniflare } from "miniflare";
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
  it("initializes Better Auth before an OAuth button can be offered", async () => {
    const response = await app.request("/api/auth/ok", undefined, {
      ...unauthenticatedEnv,
      CLOUDFLARE_ACCESS_CLIENT_ID: "test-client",
      CLOUDFLARE_ACCESS_CLIENT_SECRET: "test-secret",
      CLOUDFLARE_ACCESS_ISSUER: "https://example.com/oidc",
    });
    expect(response.status).toBe(200);
  });

  it("initiates Cloudflare OAuth without contacting a real identity provider", async () => {
    const miniflare = new Miniflare(
      convertV4MiniflareOptions({
        modules: true,
        script: "export default { fetch() { return new Response('ok') } }",
        compatibilityDate: "2026-09-15",
        d1Databases: { DB: "auth-integration-test" },
      }),
    );
    try {
      const db = await miniflare.getD1Database("DB");
      const migration = (name: string) =>
        readFileSync(new URL(`../migrations/${name}`, import.meta.url), "utf8");
      for (const name of [
        "0001_initial.sql",
        "0002_auth_account_issuer.sql",
        "0008_auth_account_issuer_compat.sql",
      ]) {
        // D1's exec() expects each statement on one line; migrations deliberately
        // use readable multi-line SQL, so apply statements through prepared D1.
        for (const statement of migration(name).split(/;\s*(?:\r?\n|$)/)) {
          if (statement.trim()) await db.prepare(statement.trim()).run();
        }
      }
      const issuer = "https://identity.example.test/oidc";
      const discoveryUrl = `${issuer}/.well-known/openid-configuration`;
      const discovery = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
        const url = input instanceof Request ? input.url : String(input);
        if (url !== discoveryUrl) {
          throw new Error(`Unexpected identity-provider request: ${url}`);
        }
        return Response.json({
          issuer,
          authorization_endpoint: `${issuer}/authorize`,
          token_endpoint: `${issuer}/token`,
          userinfo_endpoint: `${issuer}/userinfo`,
          jwks_uri: `${issuer}/jwks`,
          response_types_supported: ["code"],
          subject_types_supported: ["public"],
          id_token_signing_alg_values_supported: ["RS256"],
          code_challenge_methods_supported: ["S256"],
        });
      });
      try {
        const response = await app.request(
          "https://uncertaintycat.test/api/auth/sign-in/social",
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Origin: "https://uncertaintycat.test",
            },
            body: JSON.stringify({
              provider: "cloudflare",
              callbackURL: "/projects",
            }),
          },
          {
            ...unauthenticatedEnv,
            DB: db,
            CLOUDFLARE_ACCESS_CLIENT_ID: "test-client",
            CLOUDFLARE_ACCESS_CLIENT_SECRET: "test-secret",
            CLOUDFLARE_ACCESS_ISSUER: issuer,
          },
        );
        expect(response.status).toBe(200);
        const result = (await response.json()) as { url?: string };
        expect(result.url).toBeDefined();
        const authorizeUrl = new URL(result.url!);
        expect(authorizeUrl.origin).toBe("https://identity.example.test");
        expect(authorizeUrl.pathname).toBe("/oidc/authorize");
        expect(authorizeUrl.searchParams.get("client_id")).toBe("test-client");
        expect(authorizeUrl.searchParams.get("redirect_uri")).toBe(
          "https://uncertaintycat.test/api/auth/callback/cloudflare",
        );
        expect(authorizeUrl.searchParams.get("code_challenge_method")).toBe("S256");
        expect(authorizeUrl.searchParams.get("state")).toBeTruthy();
        expect(discovery).toHaveBeenCalled();
      } finally {
        discovery.mockRestore();
      }
    } finally {
      await miniflare.dispose();
    }
  });

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
      ai: {
        provider: "groq",
        configured: false,
        modelUnderstanding: { modelId: "openai/gpt-oss-20b" },
        reportChat: { modelId: "openai/gpt-oss-120b" },
      },
    });
    expect(session.headers.get("set-cookie") ?? "").not.toContain(
      "uncertaintycat_guest",
    );
  });

  it.each([
    "/api/v1/analyses/catalog",
    "/api/v1/examples",
    "/api/v1/projects",
    "/api/v1/operator/overview",
    "/api/v1/operator/projects/project-id",
    "/api/v1/operator/reports/run-id",
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

  it("keeps operational telemetry behind a separate operator allowlist", async () => {
    const ordinaryUserEnv = {
      DB: {},
      BETTER_AUTH_URL: "http://127.0.0.1:8787",
      DEV_AUTH_BYPASS: "true",
      OPERATOR_EMAILS: "someone-else@example.com",
    } as unknown as Env;
    for (const path of [
      "/api/v1/operator/overview",
      "/api/v1/operator/projects/project-id",
      "/api/v1/operator/reports/run-id",
    ]) {
      const denied = await app.request(path, undefined, ordinaryUserEnv);
      expect(denied.status).toBe(403);
      await expect(denied.json()).resolves.toMatchObject({
        error: { code: "operator_access_required" },
      });
    }

    const operatorSession = await app.request("/api/v1/session", undefined, {
      ...ordinaryUserEnv,
      OPERATOR_EMAILS: " DEVELOPER@LOCALHOST ",
    });
    await expect(operatorSession.json()).resolves.toMatchObject({
      identity: { authenticated: true, operator: true },
    });
  });

  it.each([
    ["DELETE", "/api/v1/projects/project-id"],
    ["POST", "/api/v1/surrogates/surrogate-id/copy"],
  ])("rejects unauthenticated %s access to %s", async (method, path) => {
    const response = await app.request(
      path,
      { method, headers: { "Content-Type": "application/json" }, body: "{}" },
      unauthenticatedEnv,
    );
    expect(response.status).toBe(401);
  });
});
