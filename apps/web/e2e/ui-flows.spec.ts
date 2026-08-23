import { expect, test } from "@playwright/test";

import {
  analysisResult,
  catalog,
  installMockApi,
  makeReport,
  makeRun,
  project,
} from "./fixtures";

test.describe("application shell and identity", () => {
  test("the public overview, private-route gate, mobile drawer, and Cloudflare sign-in are operable", async ({
    page,
  }) => {
    await installMockApi(page);
    await page.route("https://uncertaintycat.cloudflareaccess.com/**", (route) =>
      route.fulfill({ contentType: "text/html", body: "<h1>Cloudflare identity</h1>" }),
    );
    await page.goto("/");

    await expect(page.getByRole("heading", { name: /Turn uncertain inputs/ })).toBeVisible();
    await expect(page.getByRole("navigation", { name: "Primary navigation" })).toBeVisible();
    await expect(page.getByRole("link", { name: "New analysis", exact: true })).toHaveCount(0);
    await expect(page.getByText("Ishigami", { exact: true })).toBeVisible();
    await page.goto("/new-analysis");
    await expect(page.getByRole("heading", { name: "Sign in before starting an analysis." })).toBeVisible();
    await expect(page.getByRole("button", { name: "Continue with Cloudflare" })).toBeVisible();
    await page.goto("/");

    await page.setViewportSize({ width: 390, height: 844 });
    await page.getByRole("button", { name: "Open navigation" }).click();
    await expect(page.locator(".sidebar")).toHaveClass(/sidebar-open/);
    await page
      .getByRole("button", { name: "Close navigation overlay" })
      .click({ position: { x: 330, y: 100 } });
    await expect(page.locator(".sidebar")).not.toHaveClass(/sidebar-open/);

    await page.getByRole("button", { name: "Not signed in Sign in" }).click();
    await expect(page.getByText("Sign in to use the workspace")).toBeVisible();
    await page.getByRole("menuitem", { name: "Continue with Cloudflare" }).click();
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
    await expect(page.getByRole("heading", { name: "Your projects." })).toBeVisible();
    await expect(page.getByText("Recent executions")).toHaveCount(0);
    const accountButton = page.getByRole("button", { name: /Mark Legkovskis/ });
    await accountButton.click();
    await expect(page.getByText("mlegkovskis@gmail.com")).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(page.getByRole("menuitem", { name: "Sign out" })).toHaveCount(0);
    await expect(accountButton).toBeFocused();
    await accountButton.click();
    await page.getByRole("menuitem", { name: "Sign out" }).click();
    await expect.poll(() => signedOut).toBe(true);
  });

  test("blank profile names fall back to email initials and theme choice persists", async ({ page }) => {
    await installMockApi(page, { authenticated: true, projects: [project] });
    await page.route("**/api/auth/get-session", (route) =>
      route.fulfill({ contentType: "application/json", body: JSON.stringify({ session: { id: "session-blank", expiresAt: "2099-01-01T00:00:00Z" }, user: { id: "user-1", name: "   ", email: "m.legkovskis@gmail.com" } }) }),
    );
    await page.goto("/");
    await expect(page.locator(".avatar")).toHaveText("ML");
    await expect(page.getByRole("button", { name: /m\.legkovskis@gmail\.com/ })).toContainText("Signed in");
    await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
    await page.getByRole("button", { name: "Switch to dark theme" }).click();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
    await page.reload();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  });
});

test.describe("model studio", () => {
  test("unifies all 23 examples with an editable Python model", async ({ page }) => {
    await installMockApi(page, { authenticated: true, projects: [project] });
    await page.goto("/new-analysis");
    await expect(page.locator(".example-card")).toHaveCount(23);
    await expect(page.getByLabel("Model name")).toHaveValue("");
    await expect(page.getByRole("textbox", { name: "Python model source" })).toBeVisible();
    await page.getByLabel("Search reference models").fill("rocket");
    await expect(page.locator(".example-card")).toHaveCount(1);
    await page.locator(".example-card").click();
    await expect(page.locator(".example-card.selected")).toContainText("Rocket");
    await expect(page.getByRole("textbox", { name: "Python model source" })).toBeVisible();
    await expect(page.getByLabel("Model name")).toHaveValue("Rocket trajectory");
    await page.getByLabel("Model name").fill("My adapted rocket");
    await page.getByLabel("Search reference models").fill("beam");
    await page.locator(".example-card").first().click();
    await expect(page.getByLabel("Model name")).toHaveValue("My adapted rocket");
  });

  test("creates a new project from the workspace project selector", async ({ page }) => {
    await installMockApi(page, { authenticated: true, projects: [project] });
    await page.goto("/new-analysis");
    await page.getByLabel("Project").selectOption("__create__");
    await page.getByLabel("New project name").fill("Fresh engineering project");
    await page.getByRole("button", { name: "Create project", exact: true }).click();
    await expect(page.getByRole("heading", { name: "Fresh engineering project" })).toBeVisible();
  });

  test("keeps the validated understanding header inside its panel", async ({ page }, testInfo) => {
    await installMockApi(page, { authenticated: true, projects: [project] });
    await page.setViewportSize({ width: 1920, height: 1200 });
    await page.goto("/new-analysis");
    await page.getByLabel("Search reference models").fill("Ishigami");
    await page.locator(".example-card").click();
    await page.getByRole("button", { name: "Validate & Assess" }).click();
    await expect(page.getByRole("heading", { name: "Model Understanding" })).toBeVisible();
    await page.locator(".validated-studio").evaluate((element) => {
      window.scrollTo({ top: element.getBoundingClientRect().top + window.scrollY - 86, behavior: "auto" });
    });
    const pane = await page.locator(".understanding-pane").boundingBox();
    const header = await page.locator(".understanding-pane > header").boundingBox();
    expect(pane).not.toBeNull();
    expect(header).not.toBeNull();
    expect(header!.x).toBeGreaterThanOrEqual(pane!.x);
    expect(header!.x + header!.width).toBeLessThanOrEqual(pane!.x + pane!.width + 1);
    expect(header!.y).toBeGreaterThanOrEqual(pane!.y);
    await expect(page.locator(".understanding-pane > header")).toHaveCSS("position", "static");
    const screenshot = await page.screenshot({ fullPage: true });
    await testInfo.attach("validated-workspace", { body: screenshot, contentType: "image/png" });
    if (process.env.UI_REVIEW_PATH) {
      await page.screenshot({ path: process.env.UI_REVIEW_PATH, fullPage: true });
    }
  });

  test("creates a project, authors with the guided builder, configures all direct plugins, and queues the suite", async ({
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
    await page.getByLabel("Model name").fill("Guided browser model");
    await page.getByRole("button", { name: "Guided builder" }).click();
    await page.getByRole("button", { name: "Add variable" }).click();
    await page.getByLabel("Variable 3 name").fill("pressure");
    await page.getByLabel("Output 1 formula").fill("x1 + x2^2 + pressure");
    await page.getByRole("button", { name: "Validate & Assess" }).click();
    await expect(page.getByText("Validated as version 1")).toBeVisible();

    const checkboxes = page.locator(".analysis-option input[type=checkbox]");
    const directCatalog = catalog.filter((item) => !["morris", "pce", "gpr"].includes(item.key));
    await expect(checkboxes).toHaveCount(directCatalog.length);
    for (let index = 0; index < directCatalog.length; index += 1) {
      await checkboxes.nth(index).check();
    }
    await expect(page.getByText("9 analysis tasks")).toBeVisible();
    await page.getByLabel("Standard sample budget").fill("128");
    await page.getByLabel("Reliability method").selectOption("MONTE_CARLO");
    await page.getByLabel("Failure event").selectOption("<");
    await page.getByRole("spinbutton", { name: "Threshold" }).fill("-2.5");
    await page.getByRole("button", { name: "Run analyses" }).click();

    await expect(page).toHaveURL(/\/runs\/run-1$/);
    await expect.poll(() => runBody).toBeTruthy();
    const analyses = runBody?.analyses as Array<{
      analysisKey: string;
      config: Record<string, unknown>;
    }>;
    expect(analyses.map((item) => item.analysisKey).sort()).toEqual(
      directCatalog.map((item) => item.key).sort(),
    );
    expect(analyses.find((item) => item.analysisKey === "reliability")?.config).toMatchObject({
      method: "MONTE_CARLO",
      operator: "<",
      threshold: -2.5,
    });
    expect(analyses.some((item) => ["morris", "pce", "gpr"].includes(item.analysisKey))).toBe(false);
  });

  test("surfaces model validation and catalog failures without enabling a run", async ({ page }) => {
    await installMockApi(page, { authenticated: true, projects: [project] });
    await page.route("**/api/v1/analyses/catalog", (route) =>
      route.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({ error: { message: "offline" } }) }),
    );
    await page.route("**/api/v1/projects/*/models", (route) =>
      route.fulfill({ status: 422, contentType: "application/json", body: JSON.stringify({ error: { code: "invalid_model", message: "Model must define model and distribution." } }) }),
    );
    await page.goto("/workspace");
    await page.getByLabel("Search reference models").fill("Ishigami");
    await page.locator(".example-card").click();
    await page.getByRole("button", { name: "Validate & Assess" }).click();
    await expect(page.getByText("Model must define model and distribution.")).toBeVisible();
    await expect(page.getByText("Catalog unavailable")).toBeVisible();
    await expect(page.getByRole("button", { name: "Run analyses" })).toBeDisabled();
  });

  test("keeps surrogate construction in its studio and passes a promoted surrogate explicitly", async ({ page }) => {
    await installMockApi(page, { authenticated: true, projects: [project] });
    let runBody: Record<string, unknown> | undefined;
    page.on("request", (request) => {
      if (request.url().endsWith("/api/v1/runs") && request.method() === "POST") runBody = request.postDataJSON() as Record<string, unknown>;
    });
    await page.goto("/surrogates");
    await page.getByRole("button", { name: "Build GPR candidate" }).click();
    await expect(page.getByText("Meets default")).toBeVisible();
    await page.getByRole("button", { name: "Promote validated surrogate" }).click();
    await page.getByRole("link", { name: /Analyse this surrogate/ }).click();
    await expect(page.getByText("Promoted surrogate selected in Surrogate Studio")).toBeVisible();
    await page.getByRole("button", { name: "Run analyses" }).click();
    await expect.poll(() => runBody?.surrogateModelId).toBe("surrogate-1");
  });

  test("runs dimensionality screening from the dedicated studio", async ({ page }) => {
    await installMockApi(page, { authenticated: true, projects: [project] });
    let runBody: Record<string, unknown> | undefined;
    page.on("request", (request) => {
      if (request.url().endsWith("/api/v1/runs") && request.method() === "POST") runBody = request.postDataJSON() as Record<string, unknown>;
    });
    await page.goto("/dimension-reduction");
    await page.getByRole("button", { name: "Run Morris screening" }).click();
    await expect(page).toHaveURL(/\/runs\/run-1$/);
    expect((runBody?.analyses as Array<{ analysisKey: string }>)[0]?.analysisKey).toBe("morris");
  });
});

test.describe("run lifecycle", () => {
  test("renders live task progress and sends cancellation", async ({ page }) => {
    await installMockApi(page, { authenticated: true, projects: [project], runs: [makeRun("running")] });
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
    await installMockApi(page, { authenticated: true, projects: [project], runs: [makeRun("succeeded")] });
    await page.goto("/runs/run-1");
    await expect(page.getByText("The report is ready.")).toBeVisible();
    await page.getByRole("link", { name: /Open report/ }).click();
    await expect(page.getByRole("heading", { name: "Verification report" })).toBeVisible();
  });
});

test.describe("reports and grounded chat", () => {
  test("renders every evidence type and operates export, share, print, and streaming chat", async ({ page }) => {
    await installMockApi(page, { authenticated: true, projects: [project], runs: [makeRun()], report: makeReport() });
    await page.addInitScript(() => {
      window.print = () => document.documentElement.setAttribute("data-print-called", "true");
    });
    await page.goto("/reports/report-1");

    await expect(page.getByRole("heading", { name: "Verification report" })).toBeVisible();
    await expect(page.locator(".metric")).toHaveCount(2);
    await expect(page.getByRole("table").first()).toBeVisible();
    await expect(page.getByText("Rendered equations")).toBeVisible();
    await expect(page.locator(".echart")).toHaveCount(2);
    await expect(page.getByText("Exact heatmap values")).toBeVisible();
    await expect(page.getByText("Deliberate partial-failure evidence.")).toBeVisible();
    await page.getByText("Method assumptions and provenance").click();
    await expect(page.getByText(/Core 0.2.0/)).toBeVisible();

    await expect(page.getByRole("link", { name: "Data bundle" })).toHaveAttribute(
      "href",
      "/api/v1/reports/report-1/export",
    );

    await page.getByRole("button", { name: "Share" }).click();
    await expect(page.getByRole("checkbox", { name: "Include model definition" })).not.toBeChecked();
    await page.getByRole("button", { name: "Create share link" }).click();
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
    await installMockApi(page, { authenticated: true, report: makeReport() });
    await page.goto("/shared/share-token");
    await expect(page.getByRole("heading", { name: "Verification report" })).toBeVisible();
    await expect(page.getByRole("button", { name: "Share" })).toHaveCount(0);
    await expect(page.getByRole("link", { name: "Data bundle" })).toHaveCount(0);
    await expect(page.getByText("Ask this report")).toHaveCount(0);
    await expect(page.getByRole("button", { name: "PDF" })).toBeVisible();
  });

  test("chat failures are explicit and preserve the user's question", async ({ page }) => {
    await installMockApi(page, { authenticated: true, report: makeReport() });
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

  test("an unauthenticated visitor cannot open any report", async ({ page }) => {
    await installMockApi(page, { report: makeReport() });
    await page.goto("/reports/report-1");
    await expect(page.getByRole("heading", { name: "Sign in before starting an analysis." })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Verification report" })).toHaveCount(0);
    await expect(page.getByText("Ask this report")).toHaveCount(0);
  });
});

test.describe("distribution data lab", () => {
  test("uploads retained data, ranks marginals, and generates an explicit problem draft", async ({ page }) => {
    await installMockApi(page, { authenticated: true, projects: [project] });
    await page.goto("/data-lab?projectId=project-1");
    await expect(page.getByText("Or paste comma-separated data").locator("..").getByRole("textbox")).toHaveValue(/^E,F,L,I/);
    await expect(page.getByRole("button", { name: /Fixture observations\.csv/ })).toBeVisible();
    await expect(page.getByRole("table").first()).toContainText("temperature");
    await page.getByRole("button", { name: "Rank candidate fits" }).click();
    await expect(page.getByRole("heading", { name: "temperature" })).toBeVisible();
    await expect(page.locator(".distribution-chart-grid .echart canvas")).toHaveCount(6);
    await page.getByLabel("Selected marginal for temperature").selectOption("Normal");
    await page.getByLabel("Selected marginal for pressure").selectOption("Normal");
    await page.getByRole("button", { name: "Generate problem definition" }).click();
    await expect(page.getByText("generated_problem.py")).toBeVisible();
    await page.getByRole("button", { name: "Prepare model draft" }).click();
    await expect(page).toHaveURL(/\/studies\/project-1\/workspace\?dataFit=fit-2/);
    await expect(page.getByText(/Distribution draft from retained fit/)).toBeVisible();
  });
});

test.describe("dimensionality screening", () => {
  test("requires explicit fixed values before creating an immutable derived version", async ({ page }) => {
    const report = makeReport();
    const morris = analysisResult("morris");
    morris.plugin_version = "2.0.0";
    morris.payload.metrics = { candidate_threshold_fraction: 0.05 };
    morris.payload.tables = { effects: { columns: ["Variable", "Signed Mean Effect", "Mean Absolute Effect", "Effect Dispersion", "Rank", "Candidate Retained"], rows: [["x1", 0.8, 0.8, 0.1, 1, true], ["x2", 0.2, 0.2, 0.05, 2, true], ["x3", 0.01, 0.01, 0.02, 3, false]], row_count: 3, truncated: false } };
    report.sections = [{ key: "morris", status: "succeeded", result: morris }];
    await installMockApi(page, { authenticated: true, projects: [project], runs: [makeRun()], report });
    await page.goto("/reports/report-1");
    await expect(page.getByRole("heading", { name: "Confirm active and fixed variables" })).toBeVisible();
    await page.getByLabel("Fixed value for x3").fill("0.25");
    await page.getByText(/I confirm these explicit fixed values/).click();
    await page.getByRole("button", { name: "Create derived version" }).click();
    await expect(page).toHaveURL(/sourceModel=model-reduced/);
  });
});

test.describe("durable activity", () => {
  test("shows empty states and later links retained projects and runs", async ({ page }) => {
    await installMockApi(page, { authenticated: true });
    await page.goto("/activity");
    await expect(page.getByText("No projects yet")).toBeVisible();

    const secondPage = await page.context().newPage();
    await installMockApi(secondPage, { authenticated: true, projects: [project], runs: [makeRun()] });
    await secondPage.goto("/activity");
    await expect(secondPage.getByText(project.name)).toBeVisible();
    await secondPage.locator(".project-row").click();
    await expect(secondPage).toHaveURL(/\/studies\/project-1$/);
    await secondPage.close();
  });
});
