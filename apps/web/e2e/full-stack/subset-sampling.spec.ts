import { expect, test } from "@playwright/test";
import { strFromU8, unzipSync } from "fflate";

test("bounded subset sampling crosses the real compute boundary and exports retained levels", async ({ page, request }) => {
  await page.route("**/api/auth/get-session", (route) => route.fulfill({ contentType: "application/json", body: JSON.stringify({
    session: { id: "e2e-session", expiresAt: "2099-01-01T00:00:00Z" },
    user: { id: "dev-user", name: "E2E Retained User", email: "e2e@uncertaintycat.local" },
  }) }));
  const created = await request.post("http://127.0.0.1:8787/api/v1/projects", { data: { name: `Subset R-S ${Date.now()}` } });
  expect(created.ok()).toBe(true);
  const projectId = (await created.json()).project.id;
  await page.goto(`/studies/${projectId}/workspace`);
  await page.getByLabel("Model name").fill("Independent resistance minus stress");
  await page.getByRole("button", { name: "Guided builder" }).click();
  await page.getByLabel("Variable 1 name").fill("R");
  await page.getByLabel("Variable 2 name").fill("S");
  await page.getByLabel("Variable 2 distribution").selectOption("Normal");
  await page.getByLabel("Variable 1 Mean").fill("7");
  await page.getByLabel("Variable 2 Mean").fill("2");
  await page.getByLabel("Output 1 formula").fill("R-S");
  await page.getByRole("button", { name: "Validate & Assess" }).click();
  await expect(page.getByText("Model validated", { exact: true })).toBeVisible({ timeout: 120000 });
  const checked = page.locator(".analysis-option input:checked");
  while (await checked.count()) await checked.first().uncheck();
  await page.locator(".analysis-option", { hasText: "Reliability Analysis" }).getByRole("checkbox").check();
  await page.getByRole("combobox", { name: /^Reliability method/ }).selectOption("SUBSET_SAMPLING");
  await page.getByRole("combobox", { name: /^Failure event/ }).selectOption("<");
  await page.getByRole("button", { name: "Run analyses" }).click();
  await expect(page).toHaveURL(/\/runs\/[0-9a-f-]+$/);
  const runId = page.url().split("/").at(-1)!;
  await expect(page.getByText("The report is ready.")).toBeVisible({ timeout: 120000 });
  await page.getByRole("link", { name: /Open report/ }).click();
  const section = page.locator("#section-reliability");
  await expect(section.getByRole("columnheader", { name: "Output Threshold" })).toBeVisible();
  await expect(section.getByText("requested event threshold reached")).toBeVisible();
  await expect(section.getByText(/not an exact confidence guarantee/).first()).toBeVisible();
  await page.reload();
  await expect(section.getByRole("columnheader", { name: "Output Threshold" })).toBeVisible();
  const runResponse = await request.get(`http://127.0.0.1:8787/api/v1/runs/${runId}`);
  expect(runResponse.ok()).toBe(true);
  const run = (await runResponse.json()).run;
  const result = run.tasks[0].result;
  expect(result.plugin_version).toBe("3.0.0");
  expect(result.runtime.model_evaluations).toBe(8000);
  expect(result.payload.metrics.event_probability).toBeCloseTo(0.000222, 10);
  expect(result.payload.tables.subset_levels.row_count).toBe(4);
  expect(result.payload.tables.subset_levels.rows.at(-1)[1]).toBe(0);
  expect(result.model_hash).toMatch(/^[a-f0-9]{64}$/);
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("link", { name: "Data bundle" }).click();
  const download = await downloadPromise;
  const stream = await download.createReadStream();
  const chunks: Buffer[] = [];
  for await (const chunk of stream) chunks.push(Buffer.from(chunk));
  const files = unzipSync(Buffer.concat(chunks));
  expect(strFromU8(files["tables/01-reliability--subset_levels.csv"]!)).toContain("Cumulative Probability Estimate");
  const exported = JSON.parse(strFromU8(files["results/01-reliability.json"]!));
  expect(exported.result.payload).toEqual(result.payload);
  expect(Object.keys(files).some((name) => name.startsWith("series/"))).toBe(false);
  // Direct API requests cannot bypass the shared config or core resource limits.
  const denied = await request.post("http://127.0.0.1:8787/api/v1/runs", { data: {
    modelVersionId: run.modelVersionId, analyses: [{ analysisKey: "reliability", config: {
      method: "SUBSET_SAMPLING", threshold: 0, maximum_evaluations: 50001,
    } }],
  } });
  expect(denied.status()).toBe(422);
});
