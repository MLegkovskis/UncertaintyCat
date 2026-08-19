import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

test("production serves every public screen, the real catalog, and security headers", async ({
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

  const catalog = await request.get("/api/v1/analyses/catalog");
  expect(catalog.ok()).toBe(true);
  expect((await catalog.json()).analyses).toHaveLength(11);

  for (const [path, heading] of [
    ["/", /Turn uncertain inputs/],
    ["/workspace", "Start with a durable project."],
    ["/activity", "Your uncertainty studies"],
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
  await page.route("https://uncertaintycat.cloudflareaccess.com/**", (route) => route.abort());
  await page.goto("/");
  await page.getByRole("button", { name: /Guest workspace/ }).click();
  const responsePromise = page.waitForResponse(
    (response) => response.url().includes("/api/auth/sign-in/social") && response.request().method() === "POST",
  );
  await page.getByRole("button", { name: "Continue with Cloudflare" }).click();
  const response = await responsePromise;
  expect(response.ok()).toBe(true);
  const body = (await response.json()) as { url: string; redirect: boolean };
  expect(body.redirect).toBe(true);
  const authorization = new URL(body.url);
  expect(authorization.origin).toBe("https://uncertaintycat.cloudflareaccess.com");
  expect(authorization.pathname).toContain("/cdn-cgi/access/sso/oidc/");
  expect(authorization.searchParams.get("redirect_uri")).toBe(
    "https://uncertaintycat.com/api/auth/callback/cloudflare",
  );
  expect(authorization.searchParams.get("code_challenge")).toBeTruthy();
});

test("optional live mutation exercises guest D1/R2/Queue/Sandbox/report/share/export/chat", async ({
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
  await expect(options).toHaveCount(11);
  for (const key of ["eda", "sobol"]) {
    await options.filter({ has: page.getByText(new RegExp(key === "eda" ? "Exploratory" : "Sobol", "i")) })
      .locator("input[type=checkbox]")
      .uncheck();
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
  await expect(page.locator(".share-confirmation a")).toBeVisible();

  await page.getByLabel("Question about report").fill("What is the computed mean? Cite its stored source.");
  await page.getByRole("button", { name: "Send question" }).click();
  await expect(page.locator(".chat-message.assistant p")).not.toHaveText("Thinking…", { timeout: 120_000 });
  await expect(page.locator(".chat-message.assistant p")).toContainText("[");
});
