import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

test("production serves the static overview and rejects every private API surface", async ({
  page,
  request,
}) => {
  const health = await request.get("/health");
  expect(health.ok()).toBe(true);
  expect(await health.json()).toMatchObject({ status: "ok", service: "uncertaintycat-api" });
  expect(health.headers()["strict-transport-security"]).toBeTruthy();
  expect(health.headers()["x-content-type-options"]).toBe("nosniff");

  const session = await request.get("/api/v1/session");
  expect(session.ok()).toBe(true);
  expect(await session.json()).toMatchObject({
    identity: { authenticated: false },
    providers: ["cloudflare"],
  });
  expect(session.headers()["set-cookie"] ?? "").not.toContain("uncertaintycat_guest");

  for (const path of [
    "/api/v1/analyses/catalog",
    "/api/v1/examples",
    "/api/v1/projects",
    "/api/v1/runs",
    "/api/v1/shared-reports/not-a-token",
  ]) {
    const response = await request.get(path);
    expect(response.status(), path).toBe(401);
    expect(await response.json(), path).toMatchObject({
      error: { code: "authentication_required" },
    });
  }

  await page.goto("/");
  await expect(page.getByRole("heading", { name: /Turn uncertain inputs/ })).toBeVisible();
  await expect(page.getByText("Ishigami", { exact: true })).toBeVisible();
  await expect(page.getByRole("link", { name: "New analysis" })).toHaveCount(0);
  let violations = (await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
    .analyze()).violations.filter(
    (item) => item.impact === "serious" || item.impact === "critical",
  );
  expect(violations.map((item) => item.id)).toEqual([]);

  await page.goto("/workspace");
  await expect(page.getByRole("heading", { name: "Sign in before starting an analysis." })).toBeVisible();
  violations = (await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
    .analyze()).violations.filter(
    (item) => item.impact === "serious" || item.impact === "critical",
  );
  expect(violations.map((item) => item.id)).toEqual([]);
});

test("production Cloudflare identity initiation uses the configured OIDC application", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Not signed in Sign in" }).click();
  await expect(page.getByRole("menuitem", { name: "Continue with Cloudflare" })).toBeVisible();
  const result = await page.evaluate(async () => {
    const response = await fetch("/api/auth/sign-in/social", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ provider: "cloudflare", callbackURL: window.location.href }),
    });
    return { ok: response.ok, body: await response.json() };
  });
  expect(result.ok).toBe(true);
  const body = result.body as { url: string; redirect: boolean };
  expect(body.redirect).toBe(true);
  const authorization = new URL(body.url);
  expect(authorization.origin).toBe("https://uncertaintycat.cloudflareaccess.com");
  expect(authorization.pathname).toContain("/cdn-cgi/access/sso/oidc/");
  expect(authorization.searchParams.get("redirect_uri")).toBe(
    "https://uncertaintycat.com/api/auth/callback/cloudflare",
  );
  expect(authorization.searchParams.get("code_challenge")).toBeTruthy();
});
