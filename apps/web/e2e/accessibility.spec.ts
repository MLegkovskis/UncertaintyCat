import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

import { installMockApi, makeReport, makeRun, project } from "./fixtures";

const routes = [
  ["overview", "/"],
  ["workspace", "/workspace"],
  ["activity", "/activity"],
  ["run", "/runs/run-1"],
  ["report", "/reports/report-1"],
  ["shared report", "/shared/share-token"],
] as const;

for (const [name, path] of routes) {
  test(`${name} has no automatically detectable serious accessibility violations`, async ({ page }) => {
    await installMockApi(page, {
      authenticated: true,
      projects: [project],
      runs: [makeRun()],
      report: makeReport(),
    });
    await page.goto(path);
    await expect(page.locator("main")).toBeVisible();
    const results = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .analyze();
    const serious = results.violations.filter((item) =>
      item.impact === "serious" || item.impact === "critical",
    );
    expect(
      serious.map((violation) => ({
        id: violation.id,
        targets: violation.nodes.map((node) => node.target.join(" ")),
      })),
    ).toEqual([]);
  });
}

test("guided builder and expanded account controls remain accessible", async ({ page }) => {
  await installMockApi(page, { authenticated: true, projects: [project] });
  await page.goto("/workspace");
  await page.getByRole("button", { name: "Guided builder" }).click();
  await page.getByRole("button", { name: "Add variable" }).click();
  await page.getByRole("button", { name: /Mark Legkovskis/ }).click();
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
    .analyze();
  expect(
    results.violations
      .filter((item) => item.impact === "serious" || item.impact === "critical")
      .map((item) => item.id),
  ).toEqual([]);
});

test("mobile navigation has no serious accessibility violations", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await installMockApi(page);
  await page.goto("/");
  await page.getByRole("button", { name: "Open navigation" }).click();
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
    .analyze();
  expect(
    results.violations
      .filter((item) => item.impact === "serious" || item.impact === "critical")
      .map((item) => item.id),
  ).toEqual([]);
});
