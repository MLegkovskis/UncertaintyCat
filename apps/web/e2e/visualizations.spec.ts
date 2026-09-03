import { expect, test } from "@playwright/test";

import {
  catalog,
  installMockApi,
  makeVisualizationAuditReport,
  project,
} from "./fixtures";

test.describe("analysis visualization hardening", () => {
  test("renders and captures every registered analysis without clipped controls", async ({
    page,
  }, testInfo) => {
    const report = makeVisualizationAuditReport();
    report.model.equations = [
      {
        output_name: "River slope and depth",
        latex:
          "\\alpha=\\max\\left(\\frac{Z_m-Z_v}{L},0\\right),\\qquadH=\\begin{cases}\\left(\\frac{Q}{K_sB\\sqrt{\\alpha}}\\right)^{0.6},&Q,K_s,\\alpha>0\\\\0,&\\text{otherwise}\\end{cases}",
        representation: "closed_form",
      },
    ];
    await installMockApi(page, {
      authenticated: true,
      projects: [project],
      report,
    });

    await page.setViewportSize({ width: 1440, height: 1050 });
    await page.goto("/reports/report-1");
    await expect(page.locator(".report-section")).toHaveCount(catalog.length);
    await expect(page.locator(".katex-error")).toHaveCount(0);
    await testInfo.attach("retained-equation", {
      body: await page.locator(".model-definition-section").screenshot({
        animations: "disabled",
      }),
      contentType: "image/png",
    });
    await expect(
      page.getByRole("img", { name: /shaded 95% confidence band/ }),
    ).toBeVisible();
    await expect(
      page.getByRole("img", { name: /every input label is shown/ }).first(),
    ).toBeVisible();

    for (const entry of catalog) {
      const section = page.locator(`#section-${entry.key}`);
      await section.scrollIntoViewIfNeeded();
      await expect(section.locator(".echart").first()).toBeVisible();
      const plotPanels = section.locator(".plot-panel");
      const plotCount = await plotPanels.count();
      expect(plotCount, `${entry.key} needs a visual`).toBeGreaterThan(0);
      for (let index = 0; index < plotCount; index += 1) {
        const panelBox = await plotPanels.nth(index).boundingBox();
        const chartBox = await plotPanels.nth(index).locator(".echart").boundingBox();
        expect(panelBox, `${entry.key} panel ${index}`).not.toBeNull();
        expect(chartBox, `${entry.key} chart ${index}`).not.toBeNull();
        expect(chartBox!.x).toBeGreaterThanOrEqual(panelBox!.x);
        expect(chartBox!.x + chartBox!.width).toBeLessThanOrEqual(
          panelBox!.x + panelBox!.width + 1,
        );
      }
      const screenshot = await section.screenshot({ animations: "disabled" });
      await testInfo.attach(`analysis-${entry.key}`, {
        body: screenshot,
        contentType: "image/png",
      });
    }
  });
});
