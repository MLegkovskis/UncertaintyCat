import { expect, test } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  await page.route("**/api/auth/get-session", (route) => route.fulfill({
    contentType: "application/json",
    body: "null",
  }));
  await page.route("**/api/v1/session", (route) => route.fulfill({
    contentType: "application/json",
    body: JSON.stringify({ identity: { ownerId: "guest:test", authenticated: false }, providers: [] }),
  }));
  await page.route("**/api/v1/analyses/catalog", (route) => route.fulfill({
    contentType: "application/json",
    body: JSON.stringify({ analyses: [{
      key: "monte_carlo", version: "1.0.0", name: "Monte Carlo Propagation",
      category: "Propagation", description: "Propagate the declared input distribution.",
      assumptions: [], supports_dependent_inputs: true, supports_multi_output: true,
      resource_class: "standard", config_schema: {},
    }] }),
  }));
  await page.route("**/api/v1/projects", (route) => route.fulfill({
    contentType: "application/json",
    body: JSON.stringify({ projects: [] }),
  }));
});

test("overview and workspace onboarding are navigable", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /Turn uncertain inputs/ })).toBeVisible();
  await expect(page.getByText("1", { exact: true }).first()).toBeVisible();
  await page.getByRole("link", { name: /Open the workspace/ }).click();
  await expect(page.getByRole("heading", { name: "Start with a durable project." })).toBeVisible();
  await expect(page.getByRole("button", { name: /Create project/ })).toBeEnabled();
});

test("shared reports render native series and matrix evidence", async ({ page }) => {
  await page.route("**/api/v1/shared-reports/demo-token", (route) => route.fulfill({
    contentType: "application/json",
    body: JSON.stringify({ report: {
      id: "report-1", runId: "run-1", title: "Verification report", status: "succeeded",
      generatedAt: "2026-08-19T12:00:00Z",
      model: { source_hash: "abcdef0123456789", input_dimension: 2, output_dimension: 1, inputs: [], outputs: [], openturns_version: "1.25", batch_evaluation_supported: true, validation_runtime_ms: 1, warnings: [] },
      sections: [{ key: "demo", status: "succeeded", result: {
        analysis_key: "demo", plugin_version: "1.0.0", result_schema_version: "1.0.0", model_hash: "abcdef", seed: 42, uq_core_version: "0.2.0", openturns_version: "1.25", status: "succeeded", started_at: "2026-08-19T12:00:00Z", completed_at: "2026-08-19T12:00:01Z", runtime: { duration_ms: 10, model_evaluations: 3 }, warnings: [], assumptions: ["Demo assumption"], payload: {
          metrics: { mean: 2 }, tables: {}, facts: { strongest_input: "x1" },
          series: { running_mean: { name: "Running mean", x: [1, 2, 3], y: [1, 1.5, 2], x_label: "N", y_label: "Y" } },
          matrices: { correlation: { row_labels: ["Y"], column_labels: ["x1", "x2"], values: [[0.8, -0.2]] } },
        },
      } }],
    } }),
  }));
  await page.goto("/shared/demo-token");
  await expect(page.getByRole("heading", { name: "Verification report" })).toBeVisible();
  await expect(page.locator("svg.series-chart")).toBeVisible();
  await expect(page.locator(".matrix-cell")).toHaveCount(2);
  await expect(page.getByRole("button", { name: "PDF" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Share" })).toHaveCount(0);
});
