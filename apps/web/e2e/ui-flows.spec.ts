import { expect, test } from "@playwright/test";

import { catalog, installMockApi, makeReport, makeRun, project } from "./fixtures";

test.describe("application shell and identity", () => {
  test("all primary routes, the mobile drawer, and Cloudflare sign-in are operable", async ({
    page,
  }) => {
    await installMockApi(page);
    await page.route("https://uncertaintycat.cloudflareaccess.com/**", (route) =>
      route.fulfill({ contentType: "text/html", body: "<h1>Cloudflare identity</h1>" }),
    );
    await page.goto("/");

    await expect(page.getByRole("heading", { name: /Turn uncertain inputs/ })).toBeVisible();
    await expect(page.getByRole("navigation", { name: "Primary navigation" })).toBeVisible();
    await page.getByRole("link", { name: "Workspace", exact: true }).click();
    await expect(page).toHaveURL(/\/workspace$/);
    await page.getByRole("link", { name: "Activity" }).click();
    await expect(page).toHaveURL(/\/activity$/);
    await page.getByRole("link", { name: "Overview" }).click();

    await page.setViewportSize({ width: 390, height: 844 });
    await page.getByRole("button", { name: "Open navigation" }).click();
    await expect(page.locator(".sidebar")).toHaveClass(/sidebar-open/);
    await page
      .getByRole("button", { name: "Close navigation overlay" })
      .click({ position: { x: 330, y: 100 } });
    await expect(page.locator(".sidebar")).not.toHaveClass(/sidebar-open/);

    await page.getByRole("button", { name: /Guest workspace/ }).click();
    await expect(page.getByText("Keep custom models private")).toBeVisible();
    await page.getByRole("button", { name: "Continue with Cloudflare" }).click();
    await expect(page).toHaveURL(/uncertaintycat\.cloudflareaccess\.com/);
    await expect(page.getByRole("heading", { name: "Cloudflare identity" })).toBeVisible();
  });

  test("a retained user sees account details and can sign out", async ({ page }) => {
    await installMockApi(page, { authenticated: true, projects: [project] });
    let signedOut = false;
    await page.route("**/api/auth/sign-out", async (route) => {
      signedOut = true;
      await route.fulfill({ contentType: "application/json", body: JSON.stringify({ success: true }) });
    });
    await page.goto("/");
    await page.getByRole("button", { name: /Mark Legkovskis/ }).click();
    await expect(page.getByText("mlegkovskis@gmail.com")).toBeVisible();
    await page.getByRole("button", { name: "Sign out" }).click();
    await expect.poll(() => signedOut).toBe(true);
  });
});

test.describe("model studio", () => {
  test("creates a project, authors with the guided builder, configures all plugins, and queues the suite", async ({
    page,
  }) => {
    await installMockApi(page, { authenticated: true });
    let runBody: Record<string, unknown> | undefined;
    page.on("request", (request) => {
      if (request.url().endsWith("/api/v1/runs") && request.method() === "POST") {
        runBody = request.postDataJSON() as Record<string, unknown>;
      }
    });

    await page.goto("/workspace");
    await expect(page.getByRole("heading", { name: "Start with a durable project." })).toBeVisible();
    await page.getByLabel("Project name").fill("Complete browser study");
    await page.getByRole("button", { name: /Create project/ }).click();
    await expect(page.getByRole("heading", { name: "Complete browser study" })).toBeVisible();

    await expect(page.getByRole("button", { name: /Run analyses/ })).toBeDisabled();
    await page.getByRole("button", { name: "Guided builder" }).click();
    await page.getByRole("button", { name: "Add variable" }).click();
    await page.getByLabel("Variable 3 name").fill("pressure");
    await page.getByLabel("Response formula").fill("x1 + x2^2 + pressure");
    await page.getByRole("button", { name: "Validate & save" }).click();
    await expect(page.getByText("Validated as version 1")).toBeVisible();

    const checkboxes = page.locator(".analysis-option input[type=checkbox]");
    await expect(checkboxes).toHaveCount(catalog.length);
    for (let index = 0; index < catalog.length; index += 1) {
      await checkboxes.nth(index).check();
    }
    await expect(page.getByText("11 analysis tasks")).toBeVisible();
    await page.getByLabel("Standard sample budget").fill("128");
    await page.getByLabel("Reliability method").selectOption("MONTE_CARLO");
    await page.getByLabel("Failure event").selectOption("<");
    await page.getByRole("spinbutton", { name: "Threshold" }).fill("-2.5");
    await page.getByLabel("PCE total degree").fill("4");
    await page.getByRole("button", { name: "Run analyses" }).click();

    await expect(page).toHaveURL(/\/runs\/run-1$/);
    await expect.poll(() => runBody).toBeTruthy();
    const analyses = runBody?.analyses as Array<{
      analysisKey: string;
      config: Record<string, unknown>;
    }>;
    expect(analyses.map((item) => item.analysisKey).sort()).toEqual(
      catalog.map((item) => item.key).sort(),
    );
    expect(analyses.find((item) => item.analysisKey === "reliability")?.config).toMatchObject({
      method: "MONTE_CARLO",
      operator: "<",
      threshold: -2.5,
    });
    expect(analyses.find((item) => item.analysisKey === "pce")?.config.degree).toBe(4);
  });

  test("surfaces model validation and catalog failures without enabling a run", async ({ page }) => {
    await installMockApi(page, { projects: [project] });
    await page.route("**/api/v1/analyses/catalog", (route) =>
      route.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({ error: { message: "offline" } }) }),
    );
    await page.route("**/api/v1/projects/*/models", (route) =>
      route.fulfill({ status: 422, contentType: "application/json", body: JSON.stringify({ error: { code: "invalid_model", message: "Model must define model and distribution." } }) }),
    );
    await page.goto("/workspace");
    await page.getByRole("button", { name: "Validate & save" }).click();
    await expect(page.getByText("Model must define model and distribution.")).toBeVisible();
    await expect(page.getByText("Catalog unavailable")).toBeVisible();
    await expect(page.getByRole("button", { name: "Run analyses" })).toBeDisabled();
  });
});

test.describe("run lifecycle", () => {
  test("renders live task progress and sends cancellation", async ({ page }) => {
    await installMockApi(page, { projects: [project], runs: [makeRun("running")] });
    let cancelRequested = false;
    await page.route("**/api/v1/runs/run-1/cancel", async (route) => {
      cancelRequested = true;
      await route.fulfill({ contentType: "application/json", body: JSON.stringify({ status: "cancelled" }) });
    });
    await page.goto("/runs/run-1");
    await expect(page.getByText("1 of 3 tasks complete")).toBeVisible();
    await expect(page.locator(".task-row")).toHaveCount(3);
    await page.getByRole("button", { name: "Cancel" }).click();
    await expect.poll(() => cancelRequested).toBe(true);
  });

  test("opens the report from a terminal run", async ({ page }) => {
    await installMockApi(page, { projects: [project], runs: [makeRun("succeeded")] });
    await page.goto("/runs/run-1");
    await expect(page.getByText("The report is ready.")).toBeVisible();
    await page.getByRole("link", { name: /Open report/ }).click();
    await expect(page.getByRole("heading", { name: "Verification report" })).toBeVisible();
  });
});

test.describe("reports and grounded chat", () => {
  test("renders every evidence type and operates export, share, print, and streaming chat", async ({ page }) => {
    await installMockApi(page, { projects: [project], runs: [makeRun()], report: makeReport() });
    await page.addInitScript(() => {
      window.print = () => document.documentElement.setAttribute("data-print-called", "true");
    });
    await page.goto("/reports/report-1");

    await expect(page.getByRole("heading", { name: "Verification report" })).toBeVisible();
    await expect(page.locator(".metric")).toHaveCount(2);
    await expect(page.getByRole("table")).toBeVisible();
    await expect(page.locator("svg.series-chart")).toBeVisible();
    await expect(page.locator(".matrix-cell")).toHaveCount(3);
    await expect(page.getByText("Deliberate partial-failure evidence.")).toBeVisible();
    await page.getByText("Method assumptions and provenance").click();
    await expect(page.getByText(/Core 0.2.0/)).toBeVisible();

    await expect(page.getByRole("link", { name: "Data bundle" })).toHaveAttribute(
      "href",
      "/api/v1/reports/report-1/export",
    );

    await page.getByRole("button", { name: "Share" }).click();
    await expect(page.getByText(/Share link copied:/)).toBeVisible();
    await page.getByRole("button", { name: "PDF" }).click();
    await expect(page.locator("html")).toHaveAttribute("data-print-called", "true");

    await page.getByRole("button", { name: "Which input has the greatest influence?" }).click();
    await expect(page.getByLabel("Question about report")).toHaveValue(/greatest influence/);
    await page.getByRole("button", { name: "Send question" }).click();
    await expect(page.getByText(/x1 is greatest/)).toBeVisible();
    await expect(page.getByText(/monte_carlo\.fact:strongest_input/)).toBeVisible();
  });

  test("shared reports are read-only and hide owner chat", async ({ page }) => {
    await installMockApi(page, { report: makeReport() });
    await page.goto("/shared/share-token");
    await expect(page.getByRole("heading", { name: "Verification report" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Share" })).toHaveCount(0);
    await expect(page.getByRole("link", { name: "Data bundle" })).toHaveCount(0);
    await expect(page.getByText("Ask this report")).toHaveCount(0);
    await expect(page.getByRole("button", { name: "PDF" })).toBeVisible();
  });

  test("chat failures are explicit and preserve the user's question", async ({ page }) => {
    await installMockApi(page, { report: makeReport() });
    await page.route("**/api/v1/reports/report-1/chat", async (route) => {
      if (route.request().method() === "GET") {
        await route.fulfill({ contentType: "application/json", body: JSON.stringify({ messages: [] }) });
      } else {
        await route.fulfill({ status: 429, contentType: "application/json", body: JSON.stringify({ error: { message: "Daily report-chat quota exceeded." } }) });
      }
    });
    await page.goto("/reports/report-1");
    await page.getByLabel("Question about report").fill("Summarise the result");
    await page.getByLabel("Question about report").press("Enter");
    await expect(page.getByText("Daily report-chat quota exceeded.")).toBeVisible();
    await expect(page.getByText("Summarise the result")).toBeVisible();
  });
});

test.describe("durable activity", () => {
  test("shows empty states and later links retained projects and runs", async ({ page }) => {
    await installMockApi(page);
    await page.goto("/activity");
    await expect(page.getByText("No runs yet")).toBeVisible();
    await expect(page.getByText("No studies yet")).toBeVisible();

    const secondPage = await page.context().newPage();
    await installMockApi(secondPage, { projects: [project], runs: [makeRun()] });
    await secondPage.goto("/activity");
    await expect(secondPage.getByText(project.name)).toBeVisible();
    await secondPage.locator(".activity-run").click();
    await expect(secondPage).toHaveURL(/\/reports\/run-1$/);
    await secondPage.close();
  });
});
