import AxeBuilder from "@axe-core/playwright";
import { expect, test, type Locator, type Page } from "@playwright/test";

import {
  installMockApi,
  makeOperatorOverview,
  makeOperatorProject,
  makeReport,
  makeRun,
  project,
  savedModel,
} from "./fixtures";

async function tabTo(page: Page, control: Locator) {
  for (let index = 0; index < 120; index += 1) {
    if (await control.evaluate((element) => element === document.activeElement)) return;
    await page.keyboard.press("Tab");
  }
  await expect(control).toBeFocused();
}

async function expectHorizontalContainment(page: Page, controls: Locator) {
  const widths = await page.evaluate(() => ({
    viewport: document.documentElement.clientWidth,
    content: document.documentElement.scrollWidth,
  }));
  expect(widths.content).toBeLessThanOrEqual(widths.viewport + 1);
  const bounds = await controls.evaluateAll((elements) => elements
    .filter((element) => element.getClientRects().length > 0)
    .map((element) => ({
      label: element.getAttribute("aria-label") ?? element.textContent?.trim().slice(0, 80),
      left: element.getBoundingClientRect().left,
      right: element.getBoundingClientRect().right,
    })));
  expect(bounds.length).toBeGreaterThan(0);
  for (const boundsOfControl of bounds) {
    expect(boundsOfControl.left, JSON.stringify(boundsOfControl)).toBeGreaterThanOrEqual(0);
    expect(boundsOfControl.right, JSON.stringify(boundsOfControl)).toBeLessThanOrEqual(widths.viewport + 1);
  }
}

async function expectBuilderFieldsFit(page: Page) {
  const checks = await page.locator(".variable-row input, .variable-row select, .output-row input").evaluateAll((elements) => {
    const context = document.createElement("canvas").getContext("2d")!;
    return elements.map((element) => {
      const control = element as HTMLInputElement | HTMLSelectElement;
      const style = getComputedStyle(control);
      context.font = `${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
      const text = control instanceof HTMLSelectElement ? control.selectedOptions[0]!.text : control.value;
      const available = control.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight)
        - (control instanceof HTMLSelectElement ? 24 : control.type === "number" ? 18 : 0);
      const bounds = control.getBoundingClientRect();
      const authoring = control.closest(".studio-authoring")!.getBoundingClientRect();
      return { name: control.getAttribute("aria-label"), text, textWidth: context.measureText(text).width, available, contained: bounds.left >= authoring.left && bounds.right <= authoring.right };
    });
  });
  expect(checks.length).toBeGreaterThan(0);
  for (const check of checks) {
    expect(check.contained, JSON.stringify(check)).toBe(true);
    expect(check.textWidth, JSON.stringify(check)).toBeLessThanOrEqual(check.available + 1);
  }
  const labels = await page.locator(".variable-row label span").evaluateAll((elements) => elements.map((element) => ({ label: element.textContent, visible: element.scrollWidth <= element.clientWidth + 1 && element.scrollHeight <= element.clientHeight + 1 })));
  expect(labels.every((label) => label.visible), JSON.stringify(labels)).toBe(true);
}

test("200% root-font and fixed-pixel text enlargement preserves keyboard authoring and method controls", async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await installMockApi(page, { authenticated: true, projects: [], models: [{ ...savedModel, projectId: "project-created-1" }] });
  await page.goto("/studies");
  // This exercises root/rem and inherited text enlargement. Fixed-pixel type
  // remains unchanged; it is not a claim of full browser text-only zoom support.
  const rootFontStyle = await page.addStyleTag({ content: "html { font-size: 200% !important; }" });
  await expect(page.locator("html")).toHaveCSS("font-size", "32px");

  await tabTo(page, page.getByRole("button", { name: "New project", exact: true }));
  await page.keyboard.press("Enter");
  await expect(page.getByRole("textbox", { name: "Project name", exact: true })).toBeFocused();
  await page.keyboard.type("Keyboard enlargement study");
  await tabTo(page, page.getByRole("textbox", { name: "Description optional" }));
  await page.keyboard.type("Synthetic keyboard and text-size regression");
  await expectHorizontalContainment(page, page.locator(".project-creator input, .project-creator button"));
  await tabTo(page, page.getByRole("button", { name: "Create project", exact: true }));
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/\/studies\/project-created-1$/);
  await tabTo(page, page.getByRole("link", { name: "New analysis in this project" }));
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/\/studies\/project-created-1\/workspace$/);

  await tabTo(page, page.getByRole("textbox", { name: "Model name", exact: true }));
  await page.keyboard.press("ControlOrMeta+A");
  await page.keyboard.type("Keyboard enlargement model");
  await tabTo(page, page.getByRole("button", { name: "Guided builder", exact: true }));
  await page.keyboard.press("Enter");
  await tabTo(page, page.getByRole("button", { name: "Validate & Assess", exact: true }));
  await page.keyboard.press("Enter");
  const reliability = page.getByRole("checkbox", { name: "Reliability Analysis", exact: true });
  await expect(reliability).toBeEnabled();
  await tabTo(page, reliability);
  await page.keyboard.press("Space");
  await expect(reliability).toBeChecked();
  const threshold = page.getByRole("spinbutton", { name: "Threshold", exact: true });
  await tabTo(page, threshold);
  await page.keyboard.press("ControlOrMeta+A");
  await page.keyboard.type("2.5");
  await expect(threshold).toHaveValue("2.5");
  await expectHorizontalContainment(page, page.locator(".analysis-composer input, .analysis-composer select, .analysis-composer button, .analysis-option"));
  await tabTo(page, page.getByRole("button", { name: "Run analyses", exact: true }));
  await expect(page.getByRole("button", { name: "Run analyses", exact: true })).toBeEnabled();

  // Now independently simulate text-only 200% scaling for fixed-pixel type too.
  // Snapshot every computed value before any write, so nested text is doubled
  // once, not compounded. This is a DOM simulation, not native browser zoom.
  await rootFontStyle.evaluate((element) => element.remove());
  const criticalText = page.locator(".run-button, .sample-budget input, .analysis-option strong");
  const baselineFonts = await criticalText.evaluateAll((elements) => elements.map((element) => parseFloat(getComputedStyle(element).fontSize)));
  await page.evaluate(() => {
    const snapshots = Array.from(document.querySelectorAll<HTMLElement>("html, body, body *"))
      .filter((element) => element instanceof HTMLElement)
      .map((element) => ({ element, fontSize: parseFloat(getComputedStyle(element).fontSize), lineHeight: parseFloat(getComputedStyle(element).lineHeight) }));
    for (const { element, fontSize, lineHeight } of snapshots) {
      if (Number.isFinite(fontSize)) element.style.setProperty("font-size", `${fontSize * 2}px`, "important");
      if (Number.isFinite(lineHeight)) element.style.setProperty("line-height", `${lineHeight * 2}px`, "important");
    }
  });
  const enlargedFonts = await criticalText.evaluateAll((elements) => elements.map((element) => parseFloat(getComputedStyle(element).fontSize)));
  expect(enlargedFonts).toEqual(baselineFonts.map((value) => value * 2));
  await expectBuilderFieldsFit(page);
  await expectHorizontalContainment(page, page.locator(".analysis-composer input, .analysis-composer select, .analysis-composer button, .analysis-option"));
  for (const card of await page.locator(".analysis-option").all()) {
    expect(await card.evaluate((element) => element.scrollWidth <= element.clientWidth + 1)).toBe(true);
  }
  await tabTo(page, threshold);
  await page.keyboard.press("ArrowUp");
  await expect(threshold).toHaveValue("3.5");
  await tabTo(page, page.getByRole("button", { name: "Run analyses", exact: true }));
  const enlargedScreenshot = testInfo.outputPath("simulated-200-percent-text-composer.png");
  await page.screenshot({ path: enlargedScreenshot, fullPage: true });
  await testInfo.attach("simulated-200-percent-text-composer", { path: enlargedScreenshot, contentType: "image/png" });
});

test("guided builder fields reflow within the assessed model pane at common viewport widths", async ({ page }) => {
  await installMockApi(page, { authenticated: true, projects: [project] });
  await page.goto("/studies/project-1/workspace");
  await page.getByLabel("Model name", { exact: true }).fill("Synthetic field containment");
  await page.getByRole("button", { name: "Guided builder", exact: true }).click();
  await page.getByRole("button", { name: "Validate & Assess", exact: true }).click();
  await expect(page.getByRole("button", { name: "Run analyses", exact: true })).toBeEnabled();
  for (const width of [1280, 1440, 1920, 390]) {
    await page.setViewportSize({ width, height: 900 });
    await expectBuilderFieldsFit(page);
    await expectHorizontalContainment(page, page.locator(".variable-row, .output-row"));
  }
});

test("200% root-font enlargement doubles real rem progress text without horizontal overflow", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await installMockApi(page, { authenticated: true, projects: [project], runs: [makeRun("running")] });
  await page.goto("/runs/run-1");
  const progress = page.locator(".task-progress-meta").first();
  await expect(progress).toBeVisible();
  const baseline = await progress.evaluate((element) => parseFloat(getComputedStyle(element).fontSize));
  await page.addStyleTag({ content: "html { font-size: 200% !important; }" });
  expect(await progress.evaluate((element) => parseFloat(getComputedStyle(element).fontSize))).toBeCloseTo(baseline * 2);
  await expectHorizontalContainment(page, page.locator(".task-progress-meta, .task-row, main button"));
});

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
