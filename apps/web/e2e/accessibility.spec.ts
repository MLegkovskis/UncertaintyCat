import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

import { installMockApi, makeReport, makeRun, project } from "./fixtures";

const routes = [
  ["dashboard", "/"],
  ["new analysis", "/new-analysis"],
  ["studies", "/studies"],
  ["study detail", "/studies/project-1"],
  ["data lab", "/data-lab"],
  ["run", "/runs/run-1"],
  ["report", "/reports/report-1"],
  ["shared report", "/shared/share-token"],
] as const;

for (const theme of ["light", "dark"] as const) {
  test(
    `authentication gate has no automatically detectable serious accessibility violations in ${theme} theme`,
    async ({ page }) => {
      await installMockApi(page);
      await page.addInitScript((selectedTheme) => {
        window.localStorage.setItem("uncertaintycat-theme", selectedTheme);
      }, theme);
      await page.goto("/workspace");
      await expect(
        page.getByRole("heading", { name: "Sign in before starting an analysis." }),
      ).toBeVisible();
      const results = await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
        .analyze();
      expect(
        results.violations
          .filter((item) => item.impact === "serious" || item.impact === "critical")
          .map((item) => item.id),
      ).toEqual([]);
    },
  );

  for (const [name, path] of routes) {
    test(`${name} has no automatically detectable serious accessibility violations in ${theme} theme`, async ({ page }) => {
      await installMockApi(page, {
        authenticated: true,
        projects: [project],
        runs: [makeRun()],
        report: makeReport(),
      });
      await page.addInitScript((selectedTheme) => {
        window.localStorage.setItem("uncertaintycat-theme", selectedTheme);
      }, theme);
      await page.goto(path);
      await expect(page.locator("html")).toHaveAttribute("data-theme", theme);
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
          nodes: violation.nodes.map((node) => ({
            target: node.target.join(" "),
            failure: node.failureSummary,
          })),
        })),
      ).toEqual([]);
    });
  }
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
