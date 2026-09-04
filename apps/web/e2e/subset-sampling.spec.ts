import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Page } from "@playwright/test";
import type { ModelVersion } from "@uncertaintycat/contracts";
import { analysisResult, installMockApi, makeReport, makeRun, project, savedModel } from "./fixtures";

async function openSubsetComposer(page: Page, model: ModelVersion = savedModel) {
  await installMockApi(page, { authenticated: true, projects: [project] });
  await page.route("**/api/v1/projects/*/models", (route) => route.fulfill({
    contentType: "application/json", body: JSON.stringify(
      route.request().method() === "POST" ? { modelVersion: model } : { modelVersions: [model] },
    ),
  }));
  await page.goto("/studies/project-1/workspace");
  await page.getByLabel("Search reference models").fill("Ishigami");
  await page.locator(".example-card").click();
  await page.getByRole("button", { name: "Validate & Assess" }).click();
  await expect(page.getByText("Model validated", { exact: true })).toBeVisible();
  const checked = page.locator(".analysis-option input:checked");
  while (await checked.count()) await checked.first().uncheck();
  await page.locator(".analysis-option", { hasText: "Reliability Analysis" }).getByRole("checkbox").check();
}

test("bounded subset UI defaults at maximum dimension match the core resource envelope", async ({ page }, testInfo) => {
  const model = structuredClone(savedModel);
  model.metadata.input_dimension = 20;
  model.assessment!.profile.input_dimension = 20;
  model.assessment!.profile.continuous_marginals = 20;
  await openSubsetComposer(page, model);
  await page.getByLabel("Reliability method").selectOption("SUBSET_SAMPLING");
  await expect(page.getByLabel("Subset samples per level")).toHaveValue("2000");
  await expect(page.getByLabel("Maximum evaluations")).toHaveValue("20000");
  await expect(page.getByLabel("Target coefficient of variation")).toHaveCount(0);
  await expect(page.getByText(/coefficient of variation is a diagnostic/)).toBeVisible();
  await page.getByLabel("Maximum evaluations").fill("50001");
  await expect(page.getByRole("button", { name: "Run analyses" })).toBeDisabled();
  await page.getByLabel("Maximum evaluations").fill("20000");
  await page.getByLabel("Subset samples per level").fill("101");
  await expect(page.getByRole("button", { name: "Run analyses" })).toBeDisabled();
  await page.getByLabel("Subset samples per level").fill("2000");
  const a11y = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]).analyze();
  expect(a11y.violations.filter((item) => ["serious", "critical"].includes(item.impact ?? ""))).toEqual([]);
  const methodBox = (await page.getByLabel("Reliability method").boundingBox())!;
  const budgetBox = (await page.getByLabel("Maximum evaluations").boundingBox())!;
  const populationBox = (await page.getByLabel("Subset samples per level").boundingBox())!;
  expect(methodBox.x + methodBox.width).toBeLessThan(budgetBox.x);
  expect(budgetBox.x + budgetBox.width).toBeLessThan(populationBox.x);
  await page.locator(".subset-studio").screenshot({ path: testInfo.outputPath("subset-composer.png") });
  let body: Record<string, unknown> | undefined;
  await page.route("**/api/v1/runs", (route) => {
    body = route.request().postDataJSON();
    return route.fulfill({ contentType: "application/json", body: JSON.stringify({ run: makeRun("queued") }) });
  });
  await page.getByRole("button", { name: "Run analyses" }).click();
  await expect(page).toHaveURL(/\/runs\/run-1$/);
  expect(body?.analyses).toEqual([{ analysisKey: "reliability", outputTargets: [0], config: {
    method: "SUBSET_SAMPLING", threshold: 0, operator: ">", maximum_evaluations: 20000, subset_sample_size: 2000,
  } }]);
});

test("subset method honors core assessment incompatibility without disabling Monte Carlo", async ({ page }) => {
  const model = structuredClone(savedModel);
  model.assessment!.recommendations.push({ capability: "reliability", status: "available", priority: 4,
    rationale_codes: [], compatibility_warnings: [], safe_config: { subset_sampling_available: false,
      subset_sampling_incompatibility: "Subset sampling requires continuous inputs for its standard-space transformation." } });
  await openSubsetComposer(page, model);
  await expect(page.locator('option[value="SUBSET_SAMPLING"]')).toHaveAttribute("disabled", "");
  await expect(page.getByText("Subset sampling requires continuous inputs for its standard-space transformation.")).toBeVisible();
  await page.getByLabel("Reliability method").selectOption("MONTE_CARLO");
  await expect(page.getByRole("button", { name: "Run analyses" })).toBeEnabled();
});

test("subset report preserves exact level evidence and approximate uncertainty labels", async ({ page }, testInfo) => {
  const report = makeReport();
  const result = analysisResult("reliability");
  result.plugin_version = "3.0.0";
  result.warnings = ["The nominal 95% Normal interval is not an exact confidence guarantee."];
  result.payload = {
    metrics: { event_probability: 0.000222, model_evaluations: 8000 },
    tables: { subset_levels: { columns: ["Level", "Output Threshold", "Cumulative Probability Estimate"], rows: [[1, 3.21, 0.1], [2, 1.76, 0.01], [3, 0.66, 0.001], [4, 0, 0.000222]], row_count: 4, truncated: false } },
    series: {}, matrices: {}, artifacts: [],
    facts: { stopping_reason: "requested event threshold reached", history_interpretation: "Only the final row estimates the requested event; not a convergence trace." },
  };
  report.sections = [{ key: "reliability", status: "succeeded", result }];
  await installMockApi(page, { authenticated: true, projects: [project], report });
  await page.goto("/reports/report-1");
  const section = page.locator("#section-reliability");
  await expect(section.getByRole("columnheader", { name: "Cumulative Probability Estimate" })).toBeVisible();
  await expect(section.getByText("requested event threshold reached")).toBeVisible();
  await expect(section.getByText(/not an exact confidence guarantee/)).toBeVisible();
  await expect(section.locator("tbody tr")).toHaveCount(4);
  await expect(section.locator(".echart")).toHaveCount(0);
  await expect(page.getByRole("link", { name: "Data bundle" })).toHaveAttribute("href", "/api/v1/reports/report-1/export");
  const a11y = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]).analyze();
  expect(a11y.violations.filter((item) => ["serious", "critical"].includes(item.impact ?? ""))).toEqual([]);
  await section.screenshot({ path: testInfo.outputPath("subset-report.png") });
});
