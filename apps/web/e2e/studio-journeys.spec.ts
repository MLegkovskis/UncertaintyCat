import { expect, test } from "@playwright/test";
import type { DistributionFitRun } from "@uncertaintycat/contracts";

import { calibrationSavedModel, installMockApi, project, savedModel } from "./fixtures";

const alternateModel = { ...savedModel, id: "model-2", displayName: "Second saved model" };

for (const studio of ["dimension-reduction", "surrogates", "calibration"]) {
  test(`${studio} keeps the explicitly selected model across reload and browser history`, async ({ page }) => {
    await installMockApi(page, { authenticated: true, projects: [project], models: [savedModel, alternateModel] });
    await page.goto(`/studies/project-1/${studio}`);
    await expect(page.getByLabel("Saved model")).toHaveValue(savedModel.id);
    await page.getByLabel("Saved model").selectOption(alternateModel.id);
    await expect(page).toHaveURL(new RegExp(`/${studio}\\?modelId=model-2`));
    await page.reload();
    await expect(page.getByLabel("Saved model")).toHaveValue(alternateModel.id);
    await page.goBack();
    await expect(page.getByLabel("Saved model")).toHaveValue(savedModel.id);
    await page.goForward();
    await expect(page.getByLabel("Saved model")).toHaveValue(alternateModel.id);
    await page.goto(`/studies/project-1/${studio}?modelId=unavailable-model`);
    await expect(page.getByText("The requested saved model is not available in this project. Choose a saved model explicitly to continue.")).toBeVisible();
    await expect(page.getByLabel("Saved model")).toHaveValue("unavailable-model");
  });
}

test("screening explains and enforces applicability, whole budgets and explicit output", async ({ page }) => {
  const dependent = structuredClone(alternateModel);
  dependent.assessment!.profile.dependent_inputs = true;
  const multiOutput = structuredClone(savedModel);
  multiOutput.metadata.output_dimension = 2;
  multiOutput.metadata.outputs.push({ index: 1, name: "Second response" });
  await installMockApi(page, { authenticated: true, projects: [project], models: [multiOutput, dependent] });
  let body: Record<string, unknown> | undefined;
  page.on("request", (request) => {
    if (request.url().endsWith("/api/v1/runs") && request.method() === "POST") body = request.postDataJSON();
  });
  await page.goto("/studies/project-1/dimension-reduction?modelId=model-2");
  await expect(page.getByText("Morris probability-space trajectories require independent inputs.")).toBeVisible();
  await expect(page.getByRole("button", { name: "Run Morris screening" })).toBeDisabled();
  await page.getByLabel("Saved model").selectOption("model-1");
  await page.getByLabel("Trajectories", { exact: true }).fill("4.5");
  await expect(page.getByRole("button", { name: "Run Morris screening" })).toBeDisabled();
  await expect(page.getByText("Choose 4–100 whole trajectories for this bounded screening workflow.")).toBeVisible();
  await page.getByLabel("Trajectories", { exact: true }).fill("4");
  await page.getByLabel("Grid levels").fill("21");
  await expect(page.getByRole("button", { name: "Run Morris screening" })).toBeDisabled();
  await page.getByLabel("Grid levels").fill("6");
  await page.getByLabel("Screened model output").selectOption("1");
  await page.getByRole("button", { name: "Run Morris screening" }).click();
  await expect(page).toHaveURL(/\/runs\/run-1$/);
  expect(body?.analyses).toEqual([{ analysisKey: "morris", config: { trajectories: 4, levels: 6 }, outputTargets: [1] }]);
});

test("surrogate budgets match the request and candidates remain tied to their source and method", async ({ page }) => {
  await installMockApi(page, { authenticated: true, projects: [project], models: [savedModel, alternateModel] });
  let body: Record<string, unknown> | undefined;
  page.on("request", (request) => {
    if (/\/model-versions\/[^/]+\/surrogates$/.test(request.url())) body = request.postDataJSON();
  });
  await page.goto("/studies/project-1/surrogates");
  await expect(page.getByLabel("Training budget")).toHaveAttribute("max", "512");
  await page.getByLabel("Training budget").fill("513");
  await expect(page.getByRole("button", { name: "Build GPR candidate" })).toBeDisabled();
  await expect(page.getByText("Choose 16–512 whole training evaluations for GPR.")).toBeVisible();
  await page.getByLabel("Training budget").fill("32");
  await page.getByLabel("Validation budget").fill("20");
  await page.getByRole("button", { name: "Build GPR candidate" }).click();
  await expect(page.getByText("Hold-out R²", { exact: true })).toBeVisible();
  expect(body?.config).toMatchObject({ training_size: 32, validation_size: 20 });
  await page.getByLabel("Method", { exact: true }).selectOption("pce");
  await expect(page.getByText("Hold-out R²", { exact: true })).toBeVisible();
  await expect(page.getByText("Hold-out Q²", { exact: true })).toHaveCount(0);
  await page.getByLabel("Saved model").selectOption("model-2");
  await expect(page.getByText("Hold-out R²", { exact: true })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Promote validated surrogate" })).toHaveCount(0);
  await page.getByLabel("Saved model").selectOption("model-1");
  await page.getByLabel("Retained surrogate evidence", { exact: true }).selectOption("surrogate-1");
  await page.getByRole("button", { name: "Promote validated surrogate" }).click();
  await expect(page.getByText("Surrogate promoted", { exact: true })).toBeVisible();
  await page.reload();
  await page.getByLabel("Retained surrogate evidence", { exact: true }).selectOption("surrogate-1");
  await expect(page.getByRole("link", { name: /Start a new analysis with this surrogate/ })).toHaveAttribute("href", /sourceModel=model-1&surrogate=surrogate-1$/);
});

test("data-driven evidence can be reopened after refresh with exact values and honest next-step limits", async ({ page }) => {
  await installMockApi(page, { authenticated: true, projects: [project] });
  let releaseDataset: () => void = () => {};
  const datasetGate = new Promise<void>((resolve) => { releaseDataset = resolve; });
  await page.route("**/api/v1/projects/project-1/datasets", async (route) => {
    await datasetGate;
    await route.fallback();
  });
  await page.goto("/studies/project-1/surrogates");
  const datasetRequested = page.waitForRequest((request) => request.url().endsWith("/projects/project-1/datasets"));
  await page.getByRole("tab", { name: "From empirical data" }).click();
  await expect(page).toHaveURL(/source=data/);
  await datasetRequested;
  await expect(page.getByRole("group", { name: "Input columns" })).toHaveCount(0);
  releaseDataset();
  // The two-column fixture becomes one input and one distinct response after
  // hydration. Counting before the default response is assigned can see both.
  await expect(page.getByRole("combobox", { name: "Output column", exact: true })).toHaveValue("pressure");
  const inputColumns = page.getByRole("group", { name: "Input columns" }).getByRole("checkbox");
  await expect(inputColumns).toHaveCount(1);
  const temperature = page.getByRole("group", { name: "Input columns" }).getByRole("checkbox", { name: "temperature", exact: true });
  await expect(temperature).toBeChecked();
  await expect(page.getByRole("group", { name: "Input columns" }).getByRole("checkbox", { name: "pressure", exact: true })).toHaveCount(0);
  await temperature.uncheck();
  await expect(page.getByRole("button", { name: "Build data-driven GPR" })).toBeDisabled();
  await temperature.check();
  const buildRequest = page.waitForRequest((request) => request.url().endsWith("/datasets/dataset-1/surrogates") && request.method() === "POST");
  await page.getByRole("button", { name: "Build data-driven GPR" }).click();
  expect((await buildRequest).postDataJSON()).toEqual({ inputColumns: ["temperature"], outputColumn: "pressure", validationFraction: 0.2, kernel: "MATERN_2_5", trend: "CONSTANT", seed: 42 });
  await expect(page.getByText("Data-driven GPR retained", { exact: true })).toBeVisible();
  await page.reload();
  await expect(page.getByRole("tab", { name: "From empirical data" })).toHaveAttribute("aria-selected", "true");
  await page.getByLabel("Retained data-driven surrogate evidence").selectOption("data-surrogate-1");
  await expect(page.locator(".data-surrogate-evidence > p").filter({ hasText: "Retained paired-data fit:" })).toContainText("temperature → pressure");
  await expect(page.getByText(/running this data-driven surrogate downstream are not yet supported/)).toBeVisible();
  await page.getByText("Exact hold-out observations and predictions").click();
  await expect(page.getByRole("columnheader", { name: "Observed", exact: true })).toBeVisible();
  await expect(page.getByRole("cell", { name: "18.1", exact: true })).toBeVisible();
});

test("distribution history reopens persisted fits and changed composition cannot hand off an old draft", async ({ page }) => {
  await installMockApi(page, { authenticated: true, projects: [project] });
  const retained: DistributionFitRun[] = [];
  page.on("response", async (response) => {
    if (/\/datasets\/[^/]+\/fits$/.test(response.url()) && response.request().method() === "POST") {
      retained.push((await response.json()).fitRun);
    }
  });
  await page.route(/\/api\/v1\/datasets\/[^/]+\/fits$/, async (route) => {
    if (route.request().method() === "GET") await route.fulfill({ json: { fitRuns: retained } });
    else await route.fallback();
  });
  await page.goto("/studies/project-1/data-lab");
  await page.getByRole("button", { name: "Rank candidate fits" }).click();
  await page.getByLabel("Selected marginal for temperature").selectOption("Normal");
  await page.getByLabel("Selected marginal for pressure").selectOption("Uniform");
  await page.getByRole("button", { name: "Generate problem definition" }).click();
  await expect(page.getByRole("button", { name: "Prepare model draft" })).toBeVisible();
  await page.getByLabel("Copula", { exact: true }).selectOption("normal");
  await expect(page.getByRole("button", { name: "Prepare model draft" })).toHaveCount(0);
  await expect(page.getByText(/Marginal or dependence choices have changed/)).toBeVisible();
  await page.reload();
  await page.getByLabel("Retained distribution-fit evidence").selectOption("fit-2");
  await expect(page.getByLabel("Copula", { exact: true })).toHaveValue("independent");
  await expect(page.getByLabel("Selected marginal for pressure")).toHaveValue("Uniform");
  await expect(page.getByRole("button", { name: "Prepare model draft" })).toBeVisible();
  await page.getByLabel("Selected marginal for pressure").selectOption("Normal");
  await expect(page.getByRole("button", { name: "Prepare model draft" })).toHaveCount(0);
});

test("calibration rejects empty starting values and fractional optimizer limits before queuing", async ({ page }) => {
  await installMockApi(page, { authenticated: true, projects: [project], models: [calibrationSavedModel] });
  await page.goto("/studies/project-1/calibration");
  const run = page.getByRole("button", { name: "Run nonlinear least-squares calibration" });
  await expect(run).toBeEnabled();
  await page.getByLabel("Starting value for a").fill("");
  await expect(run).toBeDisabled();
  await expect(page.getByText("Every selected parameter needs a finite starting value.")).toBeVisible();
  await page.getByLabel("Starting value for a").fill("0");
  await expect(run).toBeEnabled();
  await page.getByLabel("Maximum optimizer calls").fill("10.5");
  await expect(run).toBeDisabled();
  await expect(page.getByText("Choose a whole optimizer-call limit from 10 to 500.")).toBeVisible();
  await page.getByLabel("Maximum optimizer calls").fill("250");
  await expect(run).toBeEnabled();
});

test("dataset validation announces pending state in its preview region without nesting main landmarks", async ({ page }) => {
  await installMockApi(page, { authenticated: true, projects: [project] });
  let finishUpload!: () => void;
  const uploadGate = new Promise<void>((resolve) => { finishUpload = resolve; });
  await page.route("**/api/v1/datasets", async (route) => {
    await uploadGate;
    await route.fallback();
  });
  await page.goto("/studies/project-1/data-lab");
  await page.getByRole("button", { name: "Validate pasted data" }).click();
  const fitting = page.getByRole("region", { name: "Dataset fitting" });
  await expect(page.getByRole("button", { name: "Validating data…" })).toBeDisabled();
  await expect(fitting.getByRole("status")).toHaveText("Validating and retaining dataset…");
  await expect(fitting).toHaveAttribute("aria-busy", "true");
  await expect(page.getByRole("main")).toHaveCount(1);
  finishUpload();
  await expect(page.getByRole("button", { name: "Validate pasted data" })).toBeEnabled();
  await expect(fitting).toHaveAttribute("aria-busy", "false");
});

for (const width of [1280, 1440, 1920, 390]) {
  test(`studio controls and distribution plots remain contained at ${width}px`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width, height: width === 390 ? 844 : 900 });
    await installMockApi(page, { authenticated: true, projects: [project] });
    await page.goto("/studies/project-1/surrogates");
    await expect(page.getByRole("button", { name: "Build GPR candidate" })).toBeVisible();
    const controls = await page.locator(".surrogate-controls input, .surrogate-controls select").evaluateAll((elements) => elements.map((element) => {
      const { left, top, right, bottom } = element.getBoundingClientRect();
      return { left, top, right, bottom };
    }));
    expect(controls).toHaveLength(5);
    for (let first = 0; first < controls.length; first += 1) {
      expect(controls[first]!.left).toBeGreaterThanOrEqual(0);
      expect(controls[first]!.right).toBeLessThanOrEqual(width + 1);
      for (let second = first + 1; second < controls.length; second += 1) {
        const a = controls[first]!;
        const b = controls[second]!;
        expect(Math.min(a.right, b.right) - Math.max(a.left, b.left) > 1 && Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top) > 1).toBe(false);
      }
    }
    await page.screenshot({ path: testInfo.outputPath(`surrogate-controls-${width}.png`), fullPage: true, mask: [page.locator(".account-menu")] });
    await page.goto("/studies/project-1/data-lab");
    await page.getByRole("button", { name: "Rank candidate fits" }).click();
    await expect(page.locator(".distribution-chart-grid")).toHaveCount(2);
    await expect(page.locator(".distribution-chart-grid canvas")).toHaveCount(6);
    await expect(page.locator(".distribution-chart-grid canvas").first()).toBeVisible();
    const plotWidths = await page.locator(".distribution-chart-grid").evaluateAll((grids) => grids.flatMap((grid) => [...grid.children].map((panel) => ({ available: grid.getBoundingClientRect().width, actual: panel.getBoundingClientRect().width }))));
    for (const plot of plotWidths) expect(plot.actual).toBeGreaterThanOrEqual(Math.min(300, plot.available) - 1);
    await page.locator(".distribution-chart-grid").first().screenshot({ path: testInfo.outputPath(`distribution-first-fit-${width}.png`), style: ".topbar { visibility: hidden; }" });
    await page.screenshot({ path: testInfo.outputPath(`distribution-plots-${width}.png`), fullPage: true, mask: [page.locator(".account-menu")] });
  });
}

test("calibration reference handoff waits for its newly saved model instead of selecting stale project history", async ({ page }) => {
  await installMockApi(page, { authenticated: true, projects: [project], models: [savedModel] });
  let modelCreated = false;
  let refreshModels!: () => void;
  const refreshedModels = new Promise<void>((resolve) => { refreshModels = resolve; });
  await page.route("**/api/v1/projects/project-1/models", async (route) => {
    if (route.request().method() === "POST") {
      modelCreated = true;
      await route.fulfill({ status: 201, json: { modelVersion: calibrationSavedModel } });
    } else {
      if (modelCreated) await refreshedModels;
      await route.fulfill({ json: { modelVersions: modelCreated ? [calibrationSavedModel, savedModel] : [savedModel] } });
    }
  });
  await page.goto("/studies/project-1/calibration?modelId=model-1");
  await page.getByRole("link", { name: "Add reference model" }).click();
  await page.getByRole("button", { name: "Validate & Assess" }).click();
  await expect(page).toHaveURL(/\/calibration\?modelId=model-calibration$/);
  await expect(page.getByLabel("Saved model")).toHaveValue("model-calibration");
  await expect(page.getByRole("heading", { name: "Ishigami reference model" })).toHaveCount(0);
  refreshModels();
  await expect(page.getByText("Official OpenTURNS exponential example loaded")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Exponential calibration benchmark" })).toBeVisible();
});

const scopedDatasets = ["dataset-1", "dataset-2"].map((id, index) => ({
  id, projectId: project.id, name: `Dataset ${index ? "B" : "A"}`, sourceKind: "paste", sha256: "synthetic-fixture", rowCount: 6,
  columns: ["temperature", "pressure"].map((name) => ({ name, type: "numeric", missingCount: 0, invalidNumericCount: 0, nonFiniteCount: 0, finiteCount: 6, uniqueCount: 6 })),
  preview: [{ temperature: 18, pressure: 1 }], warnings: [], createdAt: "2026-09-16T12:00:00Z",
}));

for (const mode of ["marginal", "paired"]) {
  test(`${mode} fitting keeps its dataset fixed while the request is pending`, async ({ page }) => {
    await installMockApi(page, { authenticated: true, projects: [project] });
    await page.route("**/api/v1/projects/project-1/datasets", (route) => route.fulfill({ json: { datasets: scopedDatasets } }));
    let complete!: () => void;
    const pending = new Promise<void>((resolve) => { complete = resolve; });
    await page.route(mode === "marginal" ? /\/api\/v1\/datasets\/[^/]+\/fits$/ : /\/api\/v1\/datasets\/[^/]+\/surrogates$/, async (route) => {
      if (route.request().method() === "POST") await pending;
      await route.fallback();
    });
    await page.goto(mode === "marginal" ? "/studies/project-1/data-lab" : "/studies/project-1/surrogates?source=data");
    await page.getByRole("button", { name: mode === "marginal" ? "Rank candidate fits" : "Build data-driven GPR" }).click();
    if (mode === "marginal") {
      await expect(page.getByRole("button", { name: /Dataset B/ })).toBeDisabled();
      await expect(page.getByRole("button", { name: "Validate pasted data" })).toBeDisabled();
    } else {
      await expect(page.getByLabel("Dataset", { exact: false })).toBeDisabled();
    }
    complete();
    if (mode === "marginal") {
      await expect(page.getByLabel("Selected marginal for temperature")).toBeVisible();
      await page.getByRole("button", { name: /Dataset B/ }).click();
      await expect(page.getByLabel("Selected marginal for temperature")).toHaveCount(0);
    } else {
      await expect(page.getByText("Data-driven GPR retained", { exact: true })).toBeVisible();
      await page.getByLabel("Dataset", { exact: false }).selectOption("dataset-2");
      await expect(page.getByText("Data-driven GPR retained", { exact: true })).toHaveCount(0);
    }
  });
}

test("promotion keeps retained-candidate selection fixed until its response is known", async ({ page }) => {
  await installMockApi(page, { authenticated: true, projects: [project] });
  await page.goto("/studies/project-1/surrogates");
  await page.getByRole("button", { name: "Build GPR candidate" }).click();
  await expect(page.getByLabel("Retained surrogate evidence", { exact: true })).toBeEnabled();
  let complete!: () => void;
  const pending = new Promise<void>((resolve) => { complete = resolve; });
  await page.route(/\/api\/v1\/surrogates\/[^/]+\/promote$/, async (route) => { await pending; await route.fallback(); });
  await page.getByRole("button", { name: "Promote validated surrogate" }).click();
  await expect(page.getByLabel("Retained surrogate evidence", { exact: true })).toBeDisabled();
  complete();
  await expect(page.getByText("Surrogate promoted", { exact: true })).toBeVisible();
  await expect(page.getByLabel("Retained surrogate evidence", { exact: true })).toBeEnabled();
});

test("distribution fit seed is explicit, bounded and passed unchanged to the retained request", async ({ page }) => {
  await installMockApi(page, { authenticated: true, projects: [project] });
  let body: Record<string, unknown> | undefined;
  page.on("request", (request) => {
    if (/\/datasets\/[^/]+\/fits$/.test(request.url()) && request.method() === "POST") body = request.postDataJSON();
  });
  await page.goto("/studies/project-1/data-lab");
  await expect(page.getByLabel("Fit seed")).toHaveValue("42");
  await page.getByLabel("Fit seed").fill("");
  await expect(page.getByRole("button", { name: "Rank candidate fits" })).toBeDisabled();
  await page.getByLabel("Fit seed").fill("73");
  await page.getByRole("button", { name: "Rank candidate fits" }).click();
  await expect(page.getByLabel("Selected marginal for temperature")).toBeVisible();
  expect(body?.seed).toBe(73);
  await expect(page.getByLabel("Fit seed")).toHaveValue("73");
});
