import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

test("production serves every public screen, the real catalog, and security headers", async ({
  page,
  request,
}) => {
  // A Worker revision becomes available before Cloudflare has necessarily
  // converged every newly built Sandbox container. Poll the compute-backed
  // catalog for the release-specific contract instead of treating a healthy
  // old image during that bounded rollout window as the final deployment.
  test.setTimeout(5 * 60_000);
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

  const examplesResponse = await request.get("/api/v1/examples");
  expect(examplesResponse.ok()).toBe(true);
  const examples = (await examplesResponse.json()).examples as Array<{
    id: string;
    source: string;
    sha256: string;
  }>;
  expect(examples).toHaveLength(23);
  expect(new Set(examples.map((example) => example.sha256)).size).toBe(23);

  const deniedTamper = await request.post(
    "/api/v1/projects/nonexistent/models",
    {
      data: {
        source: `${examples[0]!.source}\n# tampered`,
        sourceKind: "example",
      },
    },
  );
  expect(deniedTamper.status()).toBe(403);
  const exactCatalogSource = await request.post(
    "/api/v1/projects/nonexistent/models",
    {
      data: { source: examples[0]!.source, sourceKind: "example" },
    },
  );
  expect(exactCatalogSource.status()).toBe(404);

  let liveCatalog: Array<{
    key: string;
    config_schema: { properties?: Record<string, { maximum?: number }> };
  }> = [];
  await expect
    .poll(
      async () => {
        const response = await request.get("/api/v1/analyses/catalog");
        if (!response.ok()) return false;
        liveCatalog = (await response.json()).analyses;
        const gpr = liveCatalog.find((entry) => entry.key === "gpr");
        return (
          liveCatalog.length === 12 &&
          gpr?.config_schema.properties?.training_size?.maximum === 512
        );
      },
      {
        timeout: 4 * 60_000,
        intervals: [5_000, 10_000, 15_000],
        message: "Cloudflare Sandbox catalog did not converge to the deployed GPR contract",
      },
    )
    .toBe(true);

  for (const [path, heading] of [
    ["/", /Turn uncertain inputs/],
    ["/workspace", "Start with a durable project."],
    ["/activity", "One home for durable evidence."],
  ] as const) {
    await page.goto(path);
    await expect(page.getByRole("heading", { name: heading })).toBeVisible();
    const violations = (await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .analyze()).violations.filter(
      (item) => item.impact === "serious" || item.impact === "critical",
    );
    expect(violations.map((item) => item.id)).toEqual([]);
  }
});

test("production Cloudflare identity initiation uses the configured OIDC application", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: /Guest workspace/ }).click();
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

test("optional live mutation exercises guest D1/R2/Queue/Sandbox/report/share and denies chat", async ({
  page,
}) => {
  test.skip(process.env.E2E_LIVE_MUTATIONS !== "true", "Enable explicitly to create and execute disposable production data.");
  const studyName = `E2E live ${Date.now()}`;
  await page.goto("/workspace");
  await page.getByLabel("Project name").fill(studyName);
  await page.getByRole("button", { name: /Create project/ }).click();
  await expect(page.getByRole("heading", { name: studyName })).toBeVisible();
  await page.getByRole("button", { name: "Validate & save" }).click();
  await expect(page.getByText(/Validated as version 1/)).toBeVisible({ timeout: 120_000 });

  const options = page.locator(".analysis-option");
  await expect(options).toHaveCount(12);
  for (let index = 0; index < 12; index += 1) {
    const option = options.nth(index);
    const checkbox = option.locator("input[type=checkbox]");
    if ((await option.innerText()).includes("Gaussian Process Surrogate")) {
      await checkbox.check();
    } else {
      await checkbox.uncheck();
    }
  }
  await page.getByLabel("Standard sample budget").fill("64");
  await expect(page.getByText("1 analysis task")).toBeVisible();
  await page.getByRole("button", { name: "Run analyses" }).click();
  await expect(page.getByText("The report is ready.")).toBeVisible({ timeout: 7 * 60_000 });
  await expect(page.locator(".task-row .status-succeeded")).toHaveCount(1);
  await page.getByRole("link", { name: /Open report/ }).click();
  await expect(page.locator(".report-section")).toHaveCount(1);

  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("link", { name: "Data bundle" }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toMatch(/^uncertaintycat-.+\.zip$/);
  await page.getByRole("button", { name: "Share" }).click();
  await expect(page.getByRole("checkbox", { name: "Include model definition" })).not.toBeChecked();
  await page.getByRole("button", { name: "Create share link" }).click();
  await expect(page.locator(".share-confirmation a")).toBeVisible();
  await expect(page.getByText("Ask this report")).toHaveCount(0);
  const reportId = page.url().split("/").at(-1)!;
  const denied = await page.evaluate(async (id) => {
    const response = await fetch(`/api/v1/reports/${id}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ message: "This guest request must be denied." }),
    });
    return { status: response.status, body: await response.json() };
  }, reportId);
  expect(denied.status).toBe(401);
  expect(denied.body).toMatchObject({
    error: { code: "authentication_required" },
  });
});
