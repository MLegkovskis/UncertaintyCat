import { expect, test, type Route } from "@playwright/test";

import { analysisResult, installMockApi, makeReport, makeRun, makeVisualizationAuditReport, project } from "./fixtures";

const fail = (route: Route, message = "Temporarily unavailable.", status = 503) => route.fulfill({ status, contentType: "application/json", body: JSON.stringify({ error: { code: "fixture_failure", message } }) });

test("project loading and failed reads never masquerade as an empty workspace", async ({ page }) => {
  await installMockApi(page, { authenticated: true, projects: [project] });
  let release!: () => void;
  const pending = new Promise<void>((resolve) => { release = resolve; });
  let failed = true;
  await page.route("**/api/v1/projects", async (route) => {
    await pending;
    if (failed) await fail(route);
    else await route.fallback();
  });
  await page.goto("/studies");
  await expect(page.getByRole("status")).toContainText("Loading your projects");
  await expect(page.getByText("No projects yet")).toHaveCount(0);
  release();
  await expect(page.getByRole("alert")).toContainText("Your projects could not be loaded");
  await expect(page.getByText("No projects yet")).toHaveCount(0);
  failed = false;
  await page.getByRole("button", { name: "Retry projects" }).click();
  await expect(page.getByRole("link", { name: `Open ${project.name}` })).toBeVisible();
});

test("deletion traps focus, cancels safely, and removes cached project context", async ({ page }) => {
  await installMockApi(page, { authenticated: true, projects: [project] });
  await page.goto(`/studies/${project.id}`);
  await expect(page.getByRole("heading", { name: project.name })).toBeVisible();
  await page.getByRole("navigation", { name: "Breadcrumb" }).getByRole("link", { name: "Projects" }).click();
  const trigger = page.getByRole("button", { name: `Delete ${project.name}` });
  await trigger.click();
  const dialog = page.getByRole("dialog");
  await expect(dialog.getByRole("button", { name: "Delete project permanently" })).toBeDisabled();
  await dialog.getByRole("button", { name: "Cancel", exact: true }).focus();
  await page.keyboard.press("Tab");
  await expect(dialog.getByRole("button", { name: "Close delete confirmation" })).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(dialog).toHaveCount(0);
  await expect(trigger).toBeFocused();
  await trigger.click();
  await page.getByLabel("Project name confirmation").fill(project.name);
  await page.getByRole("button", { name: "Delete project permanently" }).click();
  await expect(page.getByRole("status")).toContainText("Project deleted");
  await expect(trigger).toHaveCount(0);
  await page.goBack();
  await expect(page.getByRole("heading", { name: "This project is not in your workspace." })).toBeVisible();
  await page.reload();
  await expect(page.getByRole("heading", { name: "This project is not in your workspace." })).toBeVisible();
});

test("project model-load recovery keeps navigation and successful run history visible", async ({ page }) => {
  await installMockApi(page, { authenticated: true, projects: [project], runs: [makeRun()] });
  let failed = true;
  await page.route("**/api/v1/projects/*/models", (route) => failed ? fail(route) : route.fallback());
  await page.goto(`/studies/${project.id}`);
  await expect(page.getByRole("alert")).toContainText("Saved models could not be loaded");
  await expect(page.getByLabel("Project workspace navigation")).toBeVisible();
  await expect(page.locator(".activity-run")).toHaveCount(1);
  failed = false;
  await page.getByRole("button", { name: "Retry models" }).click();
  await expect(page.locator(".saved-model-row")).toHaveCount(1);
});

test("project history requests scoped pages and preserves earlier rows on an older-page failure", async ({ page }) => {
  await installMockApi(page, { authenticated: true, projects: [project] });
  let olderFails = true;
  await page.route("**/api/v1/runs?*", async (route) => {
    const url = new URL(route.request().url());
    expect(url.searchParams.get("projectId")).toBe(project.id);
    if (url.searchParams.has("cursor") && olderFails) return fail(route, "Older runs unavailable.");
    const older = url.searchParams.has("cursor");
    await route.fulfill({ contentType: "application/json", body: JSON.stringify({ runs: [{ ...makeRun(), id: older ? "older-run" : "latest-run", modelDisplayName: older ? "Historical model" : "Recent model" }], nextCursor: older ? null : "latest-run" }) });
  });
  await page.goto(`/studies/${project.id}`);
  await expect(page.locator(".activity-run")).toHaveCount(1);
  await page.getByRole("button", { name: "Load older runs" }).click();
  await expect(page.getByRole("alert")).toContainText("Older runs unavailable");
  await expect(page.locator(".activity-run")).toHaveCount(1);
  olderFails = false;
  await page.getByRole("button", { name: "Retry runs" }).click();
  await expect(page.locator(".activity-run")).toHaveCount(2);
  await expect(page.getByRole("link", { name: /Historical model/ })).toHaveAttribute("href", "/reports/older-run");
  await expect(page.getByRole("button", { name: "Load older runs" })).toHaveCount(0);
});

test("terminal history stops polling and can refresh a run added elsewhere", async ({ page }) => {
  await page.clock.install();
  await installMockApi(page, { authenticated: true, projects: [project] });
  let reads = 0;
  let runs = [makeRun("succeeded")];
  await page.route("**/api/v1/runs?*", (route) => {
    reads += 1;
    return route.fulfill({ contentType: "application/json", body: JSON.stringify({ runs, nextCursor: null }) });
  });
  await page.goto(`/studies/${project.id}`);
  await expect(page.locator(".activity-run")).toHaveCount(1);
  const initialReads = reads;
  await page.clock.runFor(15_000);
  expect(reads).toBe(initialReads);
  runs = [...runs, { ...makeRun("succeeded"), id: "another-run", modelDisplayName: "Another saved result" }];
  await page.getByRole("button", { name: "Refresh history" }).click();
  await expect(page.locator(".activity-run")).toHaveCount(2);
  expect(reads).toBe(initialReads + 1);
});

for (const entry of [
  { route: "/runs/missing", endpoint: "**/api/v1/runs/missing", heading: "Run unavailable", retry: "Retry run" },
  { route: "/reports/missing", endpoint: "**/api/v1/reports/missing", heading: "Report unavailable", retry: "Retry report" },
  { route: "/shared/missing", endpoint: "**/api/v1/shared-reports/missing", heading: "Report unavailable", retry: "Retry report" },
]) {
  test(`${entry.route} has terminal failure, retry, and a route back`, async ({ page }) => {
    await installMockApi(page, { authenticated: true, projects: [project] });
    let failed = true;
    await page.route(entry.endpoint, (route) => failed ? fail(route, "Record not found.", 404) : route.fallback());
    await page.goto(entry.route);
    await expect(page.getByRole("heading", { name: entry.heading })).toBeVisible();
    await expect(page.getByRole("link", { name: "Back to Projects" })).toHaveAttribute("href", "/studies");
    await expect(page.getByText("0 of 0 tasks complete")).toHaveCount(0);
    failed = false;
    await page.getByRole("button", { name: entry.retry }).click();
    await expect(page.getByRole("heading", { name: entry.route.startsWith("/runs") ? "Analysis record" : "Verification report" })).toBeVisible();
  });
}

test("cancelled runs retain exact successful evidence and have a durable history route", async ({ page }) => {
  const run = makeRun("cancelled");
  run.tasks[0] = { ...run.tasks[0]!, status: "succeeded", result: analysisResult() };
  await installMockApi(page, { authenticated: true, projects: [project], runs: [run] });
  await page.goto(`/studies/${project.id}`);
  await expect(page.locator(".activity-run")).toHaveAttribute("href", `/runs/${run.id}`);
  await page.locator(".activity-run").click();
  await expect(page.getByText("Run cancelled.")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Retained Uncertainty Propagation evidence" })).toBeVisible();
  await expect(page.getByText("no report was generated", { exact: false })).toHaveCount(0);
  await expect(page.getByRole("progressbar")).toHaveCount(0);
  await expect(page.getByRole("link", { name: "Open retained report" })).toHaveCount(0);
  run.reportId = "cancelled-report";
  await page.reload();
  await expect(page.getByRole("heading", { name: "Retained Uncertainty Propagation evidence" })).toBeVisible();
  await expect(page.locator(".metrics-grid")).toBeVisible();
  await expect(page.getByRole("link", { name: "Open retained report" })).toHaveAttribute("href", "/reports/cancelled-report");
});

for (const status of ["failed", "partially_succeeded"] as const) {
  test(`${status} completion distinguishes retained failures from computed results`, async ({ page }) => {
    const run = makeRun(status);
    run.tasks = run.tasks.map((task, index) => {
      if (status === "partially_succeeded" && index === 0) return task;
      const failedTask = { ...task, status: "failed" as const, error: { code: "fixture_failure", message: "Analysis could not complete." } };
      delete failedTask.result;
      return failedTask;
    });
    await installMockApi(page, { authenticated: true, projects: [project], runs: [run] });
    await page.goto(`/runs/${run.id}`);
    await expect(page.getByText(status === "failed" ? "The run failed. Its failure record is ready." : "The report is ready with partial results.")).toBeVisible();
    await expect(page.getByText(status === "failed" ? "0 successful · 3 failed · 3 total analyses" : "1 successful · 2 failed · 3 total analyses")).toBeVisible();
    await expect(page.getByText("All numerical results and provenance have been persisted.")).toHaveCount(0);
  });
}

for (const entry of [
  { path: "/runs/run-1", action: "Cancel", endpoint: "**/api/v1/runs/run-1/cancel", error: "Cancellation failed" },
  { path: "/reports/report-1", action: "Rerun exact", endpoint: "**/api/v1/runs/run-1/rerun", error: "The exact rerun could not start" },
]) {
  test(`${entry.action} exposes failure without hiding retained context`, async ({ page }) => {
    await installMockApi(page, { authenticated: true, projects: [project], runs: [makeRun("running")] });
    await page.route(entry.endpoint, (route) => fail(route, "Resource limit reached."));
    await page.goto(entry.path);
    await page.getByRole("button", { name: entry.action, exact: true }).click();
    await expect(page.getByRole("alert")).toContainText(entry.error);
    await expect(page.getByRole("button", { name: entry.action, exact: true })).toBeEnabled();
    await expect(page.getByRole("navigation", { name: "Breadcrumb" })).toContainText(project.name);
  });
}

test("report sharing explains access and retains a usable link when clipboard is denied", async ({ page }) => {
  await installMockApi(page, { authenticated: true, projects: [project] });
  await page.addInitScript(() => Object.defineProperty(navigator, "clipboard", { value: { writeText: () => Promise.reject(new Error("Clipboard denied")) } }));
  let failed = true;
  await page.route("**/api/v1/reports/report-1/share-links", (route) => failed ? fail(route) : route.fulfill({ status: 201, contentType: "application/json", body: JSON.stringify({ shareLink: { id: "link-1", url: `${new URL(page.url()).origin}/shared/share-token`, expiresAt: "2026-10-16T12:00:00Z", createdAt: "2026-09-16T12:00:00Z" } }) }));
  await page.goto("/reports/report-1");
  await expect(page.getByRole("heading", { name: "Exact immutable source" })).toBeVisible();
  await page.getByRole("button", { name: "Share", exact: true }).click();
  await expect(page.getByRole("dialog", { name: "Share report" })).toContainText("Recipients must sign in");
  await page.getByRole("button", { name: "Create share link" }).click();
  await expect(page.getByRole("alert")).toContainText("Share link creation failed");
  failed = false;
  await page.getByRole("button", { name: "Create share link" }).click();
  await expect(page.getByRole("status")).toContainText("Share link created; copy this link");
  await page.locator(".share-confirmation a").click();
  await expect(page.getByText("Shared report · read only")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Exact immutable source" })).toHaveCount(0);
  await expect(page.getByRole("navigation", { name: "Breadcrumb" }).getByRole("link", { name: project.name })).toHaveCount(0);
});

test("PDF failure ends loading and keeps the report and retry action available", async ({ page }) => {
  await installMockApi(page, { authenticated: true, projects: [project] });
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  await page.goto("/reports/report-1");
  await expect(page.getByRole("heading", { name: "Exact immutable source" })).toBeVisible();
  await page.evaluate(() => { HTMLCanvasElement.prototype.toDataURL = () => { throw new Error("PDF canvas failed."); }; });
  await page.getByRole("button", { name: "Download PDF" }).click();
  await expect(page.getByRole("alert")).toContainText("PDF download failed");
  await expect(page.getByRole("button", { name: "Download PDF" })).toBeEnabled();
  await expect(page.getByRole("heading", { name: "Verification report" })).toBeVisible();
  expect(pageErrors).toEqual([]);
});

for (const deviceScaleFactor of [1, 2]) test.describe(`full report PDF rendering at DPR ${deviceScaleFactor}`, () => {
test.use({ deviceScaleFactor });
test("a complete fifteen-analysis report exports a nonempty PDF", async ({ page }, testInfo) => {
  test.setTimeout(120_000);
  const unexpectedApi: string[] = [];
  await page.route("**/api/**", (route) => { unexpectedApi.push(new URL(route.request().url()).pathname); return route.abort("blockedbyclient"); });
  await installMockApi(page, { authenticated: true, projects: [project], report: makeVisualizationAuditReport() });
  await page.goto("/reports/report-1");
  await expect(page.locator(".report-section")).toHaveCount(15);
  await expect(page.locator(".echart canvas").first()).toBeVisible();
  const bounds = await page.locator(".report-document").boundingBox();
  await testInfo.attach("complete-report-dimensions", { body: JSON.stringify(bounds), contentType: "application/json" });
  const downloadPromise = page.waitForEvent("download", { timeout: 90_000 });
  await page.getByRole("button", { name: "Download PDF" }).click();
  const download = await downloadPromise;
  expect(await download.failure()).toBeNull();
  expect(download.suggestedFilename()).toMatch(/-report\.pdf$/);
  await download.saveAs(testInfo.outputPath("complete-report.pdf"));
  const stream = await download.createReadStream();
  let bytes = 0;
  for await (const chunk of stream) bytes += chunk.length;
  expect(bytes).toBeGreaterThan(100_000);
  expect(unexpectedApi).toEqual([]);
  await expect(page.getByRole("button", { name: "Download PDF" })).toBeEnabled();
});
});

test("source load failure is recoverable without hiding report evidence", async ({ page }) => {
  await installMockApi(page, { authenticated: true, projects: [project] });
  let failed = true;
  await page.route("**/api/v1/model-versions/*/definition", (route) => failed ? fail(route) : route.fallback());
  await page.goto("/reports/report-1");
  await expect(page.getByRole("alert")).toContainText("The exact model definition could not be loaded");
  await expect(page.locator(".report-section")).toHaveCount(2);
  failed = false;
  await page.getByRole("button", { name: "Retry model definition" }).click();
  await expect(page.getByRole("heading", { name: "Exact immutable source" })).toBeVisible();
});

test("Morris fixed-value changes require renewed confirmation and blanks cannot silently become zero", async ({ page }) => {
  const report = makeReport();
  report.sections = makeVisualizationAuditReport().sections.filter((section) => section.key === "morris");
  report.sections[0]!.result!.payload.tables.effects!.rows = [
    ["x1", 0.8, 0.8, 0.1, 1, true],
    ["x2", 0.2, 0.2, 0.05, 2, true],
    ["x3", 0.01, 0.01, 0.02, 3, false],
  ];
  report.sections[0]!.result!.payload.tables.effects!.row_count = 3;
  await installMockApi(page, { authenticated: true, projects: [project], report });
  await page.addInitScript(() => Object.defineProperty(navigator, "clipboard", { value: { writeText: () => Promise.reject(new Error("Clipboard denied")) } }));
  await page.goto("/reports/report-1");
  await page.getByLabel("Retain x3", { exact: true }).uncheck();
  const confirm = page.getByRole("checkbox", { name: /I confirm these explicit fixed values/ });
  const create = page.getByRole("button", { name: "Create derived version" });
  await confirm.check();
  await expect(create).toBeEnabled();
  await page.getByLabel("Fixed value for x3").fill("");
  await expect(confirm).not.toBeChecked();
  await confirm.check();
  await expect(create).toBeDisabled();
  await expect(page.getByRole("alert")).toContainText("Enter a finite fixed value");
  await page.getByLabel("Fixed value for x3").fill("0.25");
  await expect(confirm).not.toBeChecked();
  await confirm.check();
  await expect(create).toBeEnabled();
  await create.click();
  await expect(page.getByText("Reduced model created", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Copy Python model" }).click();
  await expect(page.getByText("Clipboard access failed. Select and copy the Python source below.")).toBeVisible();
});
