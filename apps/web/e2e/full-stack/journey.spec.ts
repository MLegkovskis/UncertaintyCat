import { expect, test } from "@playwright/test";

test("retained-user journey persists a project, executes every plugin, and produces a durable report", async ({
  page,
  request,
}) => {
  const studyName = `E2E retained study ${Date.now()}`;
  await page.route("**/api/auth/get-session", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        session: { id: "e2e-session", expiresAt: "2099-01-01T00:00:00Z" },
        user: {
          id: "dev-user",
          name: "E2E Retained User",
          email: "e2e@uncertaintycat.local",
        },
      }),
    }),
  );

  await page.goto("/workspace");
  await page.getByLabel("Project name").fill(studyName);
  await page.getByRole("button", { name: /Create project/ }).click();
  await expect(page.getByRole("heading", { name: studyName })).toBeVisible();

  // Exercise both authoring modes; execute the curated Ishigami Python model so
  // every direct analysis receives a stable, well-understood scalar test function.
  await page.getByLabel("Search reference models").fill("Ishigami");
  await page.locator(".example-card").click();
  await page.getByRole("button", { name: "Guided builder" }).click();
  await page.getByRole("button", { name: "Add variable" }).click();
  await expect(page.getByLabel("Variable 3 name")).toBeVisible();
  await page.getByRole("button", { name: "Examples & Python model" }).click();
  await expect(page.getByRole("textbox", { name: "Python model source" })).toBeVisible();
  await page.getByRole("button", { name: "Validate & Assess" }).click();
  await expect(page.getByText(/Validated as version 1/)).toBeVisible({ timeout: 120_000 });
  await expect(page.getByText(/3 inputs · 1 outputs/)).toBeVisible();

  const analysisOptions = page.locator(".analysis-option");
  await expect(analysisOptions).toHaveCount(9);
  const checkboxes = analysisOptions.locator("input[type=checkbox]");
  for (let index = 0; index < 9; index += 1) {
    await checkboxes.nth(index).check();
  }
  await page.getByLabel("Standard sample budget").fill("64");
  await expect(page.getByText("9 analysis tasks")).toBeVisible();
  await page.getByRole("button", { name: "Run analyses" }).click();
  await expect(page).toHaveURL(/\/runs\/[0-9a-f-]+$/);
  const runId = page.url().split("/").at(-1)!;

  await expect(page.getByText("The report is ready.")).toBeVisible({ timeout: 7 * 60_000 });
  await expect(page.locator(".task-row")).toHaveCount(9);
  await expect(page.locator(".task-row .status-succeeded")).toHaveCount(9);
  await page.getByRole("link", { name: /Open report/ }).click();

  await expect(page.getByRole("heading", { name: "Uncertainty Quantification Report" })).toBeVisible();
  await expect(page.locator(".report-section")).toHaveCount(9);
  await expect(page.locator(".report-section .status-succeeded")).toHaveCount(9);
  await expect(page.getByText(/OpenTURNS/).first()).toBeVisible();
  await expect(page.locator(".metrics-grid, .result-block, .plot-panel").first()).toBeVisible();
  await expect(page.getByText("Ask this report")).toBeVisible();

  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("link", { name: "Data bundle" }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe(`uncertaintycat-${runId}.zip`);
  const stream = await download.createReadStream();
  let bytes = 0;
  for await (const chunk of stream) bytes += chunk.length;
  expect(bytes).toBeGreaterThan(100);

  await page.getByRole("button", { name: "Share" }).click();
  await expect(page.getByRole("checkbox", { name: "Include model definition" })).not.toBeChecked();
  await page.getByRole("button", { name: "Create share link" }).click();
  const sharedUrl = await page.locator(".share-confirmation a").getAttribute("href");
  expect(sharedUrl).toMatch(/\/shared\//);
  await page.goto(new URL(sharedUrl!).pathname);
  await expect(page.getByRole("heading", { name: "Uncertainty Quantification Report" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Share" })).toHaveCount(0);
  await expect(page.getByText("Ask this report")).toHaveCount(0);

  // The three methods intentionally removed from the direct composer are
  // exercised through their dedicated scientific workspaces.
  await page.goto("/dimension-reduction");
  await expect(page.getByRole("heading", { name: "Screen inputs before expensive analysis." })).toBeVisible();
  await page.getByRole("button", { name: "Run Morris screening" }).click();
  await expect(page.getByText("The report is ready.")).toBeVisible({ timeout: 7 * 60_000 });
  await expect(page.locator(".task-row")).toHaveCount(1);

  await page.goto("/surrogates");
  await expect(page.getByRole("heading", { name: "Approximate an expensive model deliberately." })).toBeVisible();
  await page.getByRole("button", { name: "Build GPR candidate" }).click();
  await expect(page.getByText("Hold-out R²")).toBeVisible({ timeout: 7 * 60_000 });
  await page.getByLabel("Method").selectOption("pce");
  await page.getByRole("button", { name: "Build PCE candidate" }).click();
  await expect(page.getByText("Hold-out Q²")).toBeVisible({ timeout: 7 * 60_000 });

  await page.goto("/activity");
  await expect(page.getByText(studyName)).toBeVisible();
  await expect(page.locator(".project-row")).toHaveCount(1);
  await page.reload();
  await expect(page.getByText(studyName)).toBeVisible();

  const projects = await request.get("http://127.0.0.1:8787/api/v1/projects");
  expect(projects.ok()).toBe(true);
  expect((await projects.json()).projects).toEqual(
    expect.arrayContaining([expect.objectContaining({ name: studyName })]),
  );
  const runs = await request.get("http://127.0.0.1:8787/api/v1/runs");
  expect(runs.ok()).toBe(true);
  expect((await runs.json()).runs).toEqual(
    expect.arrayContaining([expect.objectContaining({ id: runId, status: "succeeded" })]),
  );
});
