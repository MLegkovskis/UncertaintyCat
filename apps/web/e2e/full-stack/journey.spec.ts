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

  await page.goto("/studies");
  await page.getByRole("button", { name: "Create first project" }).click();
  await page.getByLabel("Project name").fill(studyName);
  await page.getByRole("button", { name: /Create project/ }).click();
  await expect(page.getByRole("heading", { name: studyName })).toBeVisible();
  const projectId = new URL(page.url()).pathname.split("/")[2]!;
  await page.getByRole("link", { name: "New analysis in this project" }).click();

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
  await expect(page.getByText("Model validated", { exact: true })).toBeVisible({ timeout: 120_000 });
  await expect(page.getByText(/3 inputs → 1 outputs/).first()).toBeVisible();

  const analysisOptions = page.locator(".analysis-option");
  await expect(analysisOptions).toHaveCount(10);
  const checkboxes = analysisOptions.locator("input[type=checkbox]:enabled");
  await expect(checkboxes).toHaveCount(9);
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
  await page.goto(`/studies/${projectId}/dimension-reduction`);
  await expect(page.getByRole("heading", { name: "Screen inputs before expensive analysis." })).toBeVisible();
  await page.getByRole("button", { name: "Run Morris screening" }).click();
  await expect(page.getByText("The report is ready.")).toBeVisible({ timeout: 7 * 60_000 });
  await expect(page.locator(".task-row")).toHaveCount(1);

  await page.goto(`/studies/${projectId}/surrogates`);
  await expect(page.getByRole("heading", { name: "Approximate a response deliberately." })).toBeVisible();
  await page.getByRole("button", { name: "Build GPR candidate" }).click();
  await expect(page.getByText("Hold-out R²")).toBeVisible({ timeout: 7 * 60_000 });
  await page.getByLabel("Method").selectOption("pce");
  await page.getByRole("button", { name: "Build PCE candidate" }).click();
  await expect(page.getByText("Hold-out Q²")).toBeVisible({ timeout: 7 * 60_000 });

  await page.getByRole("tab", { name: "From empirical data" }).click();
  await expect(page.getByLabel("Example surrogate CSV")).toBeVisible();
  await page.getByRole("button", { name: "Add example dataset" }).click();
  await expect(page.getByRole("button", { name: "Build data-driven GPR" })).toBeVisible({ timeout: 120_000 });
  await page.getByRole("button", { name: "Build data-driven GPR" }).click();
  await expect(page.getByText("Data-driven GPR retained")).toBeVisible({ timeout: 7 * 60_000 });

  // Exercise the promoted-surrogate handoff across the full Worker/D1/R2
  // boundary, including the copied source hash and immutable XML artifact.
  await page.getByRole("tab", { name: "From saved model" }).click();
  await page.getByRole("button", { name: "Build GPR candidate" }).click();
  await expect(page.getByText("Hold-out R²")).toBeVisible({ timeout: 7 * 60_000 });
  const overrideAcknowledgement = page.getByText(
    "I acknowledge the validation is below the default promotion guidance.",
  );
  if (await overrideAcknowledgement.isVisible()) {
    await overrideAcknowledgement.click();
    await page.getByLabel("Recorded reason").fill(
      "Full-stack handoff test explicitly records the validation override.",
    );
  }
  await page.getByRole("button", { name: "Promote validated surrogate" }).click();
  const newProjectLink = page.getByRole("link", { name: /Start a new project with this surrogate/ });
  await expect(newProjectLink).toBeVisible({ timeout: 120_000 });
  await newProjectLink.click();
  const handoffProjectName = `${studyName} surrogate handoff`;
  await page.getByLabel("Project name").fill(handoffProjectName);
  await page.getByRole("button", { name: "Create project with surrogate" }).click();
  await expect(page.getByText("Promoted surrogate selected in Surrogate Studio")).toBeVisible({ timeout: 120_000 });

  await page.goto("/studies");
  await page.getByRole("button", { name: `Delete ${handoffProjectName}` }).click();
  await page.getByLabel("Project name confirmation").fill(handoffProjectName);
  await page.getByRole("button", { name: "Delete project permanently" }).click();
  await expect(page.getByText(handoffProjectName)).toHaveCount(0);

  // Exercise dependent-input ANCOVA through the real builder, compute,
  // persistence, report, and export-compatible generic result path.
  await page.goto(`/studies/${projectId}/workspace`);
  await page.getByLabel("Model name").fill("Dependent ANCOVA model");
  await page.getByRole("button", { name: "Guided builder" }).click();
  await page.getByLabel("Input dependence").selectOption("normal");
  await page.getByLabel("Correlation x2 and x1").fill("0.4");
  await page.getByRole("button", { name: "Validate & Assess" }).click();
  await expect(page.getByText("Model validated", { exact: true })).toBeVisible({ timeout: 120_000 });
  const ancovaOption = page.locator(".analysis-option", {
    hasText: "ANCOVA Dependent-Input Sensitivity",
  });
  await expect(ancovaOption.locator("input")).toBeEnabled();
  await expect(
    page.locator(".analysis-option", { hasText: "Sobol Sensitivity Analysis" }).locator("input"),
  ).toBeDisabled();
  const enabledDependentAnalyses = page.locator(".analysis-option input[type=checkbox]:enabled");
  for (let index = 0; index < await enabledDependentAnalyses.count(); index += 1) {
    const checkbox = enabledDependentAnalyses.nth(index);
    if (await checkbox.isChecked()) await checkbox.uncheck();
  }
  await ancovaOption.locator("input").check();
  await page.getByLabel("Standard sample budget").fill("128");
  await expect(page.getByText(/^1 analysis task ·/)).toBeVisible();
  await page.getByRole("button", { name: "Run analyses" }).click();
  await expect(page.getByText("The report is ready.")).toBeVisible({ timeout: 7 * 60_000 });
  await expect(page.locator(".task-row .status-succeeded")).toHaveCount(1);
  await page.getByRole("link", { name: /Open report/ }).click();
  await expect(page.locator(".report-section", { hasText: "ANCOVA Contribution" })).toBeVisible();
  await expect(page.getByRole("columnheader", { name: "Correlation Contribution" })).toBeVisible();

  await page.goto("/studies");
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

  // Finish with the destructive lifecycle boundary: explicit typed confirmation,
  // D1 cascades, and authenticated absence of the deleted run/project.
  await page.goto("/studies");
  await page.getByRole("button", { name: `Delete ${studyName}` }).click();
  await expect(page.getByRole("button", { name: "Delete project permanently" })).toBeDisabled();
  await page.getByLabel("Project name confirmation").fill(studyName);
  await page.getByRole("button", { name: "Delete project permanently" }).click();
  await expect(page.getByText(studyName)).toHaveCount(0);

  const projectsAfterDelete = await request.get("http://127.0.0.1:8787/api/v1/projects");
  expect(projectsAfterDelete.ok()).toBe(true);
  expect((await projectsAfterDelete.json()).projects).not.toEqual(
    expect.arrayContaining([expect.objectContaining({ name: studyName })]),
  );
  const runsAfterDelete = await request.get("http://127.0.0.1:8787/api/v1/runs");
  expect(runsAfterDelete.ok()).toBe(true);
  expect((await runsAfterDelete.json()).runs).not.toEqual(
    expect.arrayContaining([expect.objectContaining({ id: runId })]),
  );
});
