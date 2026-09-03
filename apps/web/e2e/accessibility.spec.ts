import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

import {
  installMockApi,
  makeOperatorOverview,
  makeOperatorProject,
  makeReport,
  makeRun,
  project,
} from "./fixtures";

const routes = [
  ["projects home", "/"],
  ["studies", "/studies"],
  ["study detail", "/studies/project-1"],
  ["model and analyses", "/studies/project-1/workspace"],
  ["dimension reduction", "/studies/project-1/dimension-reduction"],
  ["calibration studio", "/studies/project-1/calibration"],
  ["surrogate studio", "/studies/project-1/surrogates"],
  ["distribution fitting", "/studies/project-1/data-lab"],
  ["run", "/runs/run-1"],
  ["report", "/reports/report-1"],
  ["shared report", "/shared/share-token"],
] as const;

for (const theme of ["light", "dark"] as const) {
  test(`authentication gate has no automatically detectable serious accessibility violations in ${theme} theme`, async ({
    page,
  }) => {
    await installMockApi(page);
    await page.addInitScript((selectedTheme) => {
      window.localStorage.setItem("uncertaintycat-theme", selectedTheme);
    }, theme);
    await page.goto("/workspace");
    await expect(
      page.getByRole("heading", {
        name: "Sign in before starting an analysis.",
      }),
    ).toBeVisible();
    const results = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .analyze();
    expect(
      results.violations
        .filter(
          (item) => item.impact === "serious" || item.impact === "critical",
        )
        .map((item) => item.id),
    ).toEqual([]);
  });

  for (const [name, path] of routes) {
    test(`${name} has no automatically detectable serious accessibility violations in ${theme} theme`, async ({
      page,
    }) => {
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
      const serious = results.violations.filter(
        (item) => item.impact === "serious" || item.impact === "critical",
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

test("guided builder and expanded account controls remain accessible", async ({
  page,
}) => {
  await installMockApi(page, { authenticated: true, projects: [project] });
  await page.goto("/studies/project-1/workspace");
  await page.getByRole("button", { name: "Guided builder" }).click();
  await page.getByRole("button", { name: "Add variable" }).click();
  await page.getByRole("button", { name: /Mark Legkovskis/ }).click();
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
    .analyze();
  expect(
    results.violations
      .filter((item) => item.impact === "serious" || item.impact === "critical")
      .map((item) => ({
        id: item.id,
        nodes: item.nodes.map((node) => ({
          target: node.target,
          summary: node.failureSummary,
        })),
      })),
  ).toEqual([]);
});

test("operator dashboard has no serious accessibility violations", async ({
  page,
}) => {
  await installMockApi(page, {
    authenticated: true,
    operator: true,
    operatorOverview: makeOperatorOverview(),
  });
  await page.goto("/operator");
  await expect(
    page.getByRole("heading", { name: "Application health." }),
  ).toBeVisible();
  await expect(page.locator(".echart canvas")).toHaveCount(2);
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
    .analyze();
  expect(
    results.violations
      .filter((item) => item.impact === "serious" || item.impact === "critical")
      .map((item) => item.id),
  ).toEqual([]);
});

test("operator project inspection has no serious accessibility violations", async ({
  page,
}) => {
  await installMockApi(page, {
    authenticated: true,
    operator: true,
    operatorProject: makeOperatorProject(),
  });
  await page.goto("/operator/projects/project-1?run=run-1");
  await expect(
    page.getByRole("heading", { name: "Beam study" }),
  ).toBeVisible();
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
    .analyze();
  expect(
    results.violations
      .filter((item) => item.impact === "serious" || item.impact === "critical")
      .map((item) => item.id),
  ).toEqual([]);
});

test("mobile navigation has no serious accessibility violations", async ({
  page,
}) => {
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
      .map((item) => ({
        id: item.id,
        nodes: item.nodes.map((node) => ({
          target: node.target.join(" "),
          failure: node.failureSummary,
        })),
      })),
  ).toEqual([]);
});
