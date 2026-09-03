import { expect, test } from "@playwright/test";

import {
  analysisResult,
  calibrationSavedModel,
  catalog,
  installMockApi,
  makeOperatorOverview,
  makeReport,
  makeRun,
  project,
  savedModel,
} from "./fixtures";

test.describe("operator telemetry", () => {
  test("hides operations from ordinary users and denies direct navigation", async ({
    page,
  }) => {
    await installMockApi(page, { authenticated: true });
    await page.goto("/studies");
    await expect(page.getByRole("link", { name: "Operations" })).toHaveCount(0);
    await page.goto("/operator");
    await expect(
      page.getByRole("heading", { name: "This view is restricted." }),
    ).toBeVisible();
  });

  test("shows an operator the current D1 snapshot and actionable issue links", async ({
    page,
  }) => {
    await installMockApi(page, {
      authenticated: true,
      operator: true,
      operatorOverview: makeOperatorOverview(),
      runs: [makeRun()],
    });
    await page.goto("/operator");
    await expect(
      page.getByRole("heading", { name: "Application health." }),
    ).toBeVisible();
    await expect(page.getByRole("link", { name: "Operations" })).toBeVisible();
    await expect(page.getByText("3", { exact: true }).first()).toBeVisible();
    await expect(
      page.getByRole("heading", { name: "Errors and stale work" }),
    ).toBeVisible();
    await expect(
      page.getByText("The bounded compute task did not complete."),
    ).toBeVisible();
    await expect(page.getByRole("link", { name: "Open run" })).toHaveAttribute(
      "href",
      "/runs/run-1",
    );
    await page.getByLabel("Reporting window").selectOption("24");
    await expect(page.getByRole("button", { name: "Refresh" })).toBeEnabled();
  });
});

test.describe("application shell and identity", () => {
  test("the public shell advertises a crawlable cat favicon", async ({
    page,
    request,
  }) => {
    await installMockApi(page);
    await page.goto("/");

    await expect(page.locator('link[rel="icon"]')).toHaveAttribute(
      "href",
      "/favicon-96x96.png",
    );
    await expect(page.locator('link[rel="icon"]')).toHaveAttribute(
      "sizes",
      "96x96",
    );

    const favicon = await request.get("/favicon-96x96.png");
    expect(favicon.ok()).toBe(true);
    expect(favicon.headers()["content-type"]).toContain("image/png");
    expect((await favicon.body()).subarray(0, 8)).toEqual(
      Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    );

    const legacyFavicon = await request.get("/favicon.ico");
    expect(legacyFavicon.ok()).toBe(true);
    expect(["image/x-icon", "image/vnd.microsoft.icon"]).toContain(
      legacyFavicon.headers()["content-type"]?.split(";")[0],
    );
    expect((await legacyFavicon.body()).subarray(0, 4)).toEqual(
      Buffer.from([0x00, 0x00, 0x01, 0x00]),
    );

    const robots = await request.get("/robots.txt");
    expect(robots.ok()).toBe(true);
    expect(await robots.text()).toContain("Allow: /");
  });

  test("the public overview, private-route gate, mobile drawer, and Cloudflare sign-in are operable", async ({
    page,
  }) => {
    await installMockApi(page);
    await page.route(
      "https://uncertaintycat.cloudflareaccess.com/**",
      (route) =>
        route.fulfill({
          contentType: "text/html",
          body: "<h1>Cloudflare identity</h1>",
        }),
    );
    await page.goto("/");

    await expect(
      page.getByRole("heading", { name: /Understand what uncertainty does/ }),
    ).toBeVisible();
    await expect(
      page.getByRole("navigation", { name: "Primary navigation" }),
    ).toBeVisible();
    await expect(
      page.getByRole("link", { name: "New analysis", exact: true }),
    ).toHaveCount(0);
    await expect(
      page.getByRole("heading", { name: "Sensitivity analysis" }),
    ).toBeVisible();
    await page.goto("/studies/project-1/workspace");
    await expect(
      page.getByRole("heading", {
        name: "Sign in before starting an analysis.",
      }),
    ).toBeVisible();
    await expect(
      page.getByRole("button", { name: "Continue with Cloudflare" }),
    ).toBeVisible();
    await page.goto("/studies/project-1/calibration");
    await expect(
      page.getByRole("heading", {
        name: "Sign in before starting an analysis.",
      }),
    ).toBeVisible();
    await expect(
      page.getByText("Official OpenTURNS exponential example loaded"),
    ).toHaveCount(0);
    await page.goto("/");

    await page.setViewportSize({ width: 390, height: 844 });
    await page.getByRole("button", { name: "Open navigation" }).click();
    await expect(page.locator(".sidebar")).toHaveClass(/sidebar-open/);
    await page
      .getByRole("button", { name: "Close navigation overlay" })
      .click({ position: { x: 330, y: 100 } });
    await expect(page.locator(".sidebar")).not.toHaveClass(/sidebar-open/);

    await page.getByRole("button", { name: "Not signed in Sign in" }).click();
    await expect(page.getByText("Sign in to use the workspace")).toBeVisible();
    await page
      .getByRole("menuitem", { name: "Continue with Cloudflare" })
      .click();
    await expect(page).toHaveURL(/uncertaintycat\.cloudflareaccess\.com/);
    await expect(
      page.getByRole("heading", { name: "Cloudflare identity" }),
    ).toBeVisible();
  });

  test("a retained user sees account details and can sign out", async ({
    page,
  }) => {
    await installMockApi(page, { authenticated: true, projects: [project] });
    let signedOut = false;
    await page.route("**/api/v1/session", (route) =>
      route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          identity: signedOut
            ? { ownerId: "", authenticated: false }
            : {
                ownerId: "user-1",
                authenticated: true,
                name: "Mark Legkovskis",
                email: "mlegkovskis@gmail.com",
              },
          providers: ["cloudflare"],
        }),
      }),
    );
    await page.route("**/api/auth/sign-out", async (route) => {
      signedOut = true;
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({ success: true }),
      });
    });
    await page.goto("/");
    await expect(
      page.getByRole("heading", { name: "Your projects." }),
    ).toBeVisible();
    await expect(page.getByText("Recent executions")).toHaveCount(0);
    const accountButton = page.getByRole("button", { name: /Mark Legkovskis/ });
    await accountButton.click();
    await expect(page.getByText("mlegkovskis@gmail.com")).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(page.getByRole("menuitem", { name: "Sign out" })).toHaveCount(
      0,
    );
    await expect(accountButton).toBeFocused();
    await accountButton.click();
    await page.getByRole("menuitem", { name: "Sign out" }).click();
    await expect.poll(() => signedOut).toBe(true);
    await expect(page).toHaveURL(/\/$/);
    await expect(
      page.getByRole("button", { name: /Not signed in/ }),
    ).toBeVisible();
    await expect(page.getByRole("link", { name: "Projects" })).toHaveCount(0);
  });

  test("blank profile names fall back to email initials and theme choice persists", async ({
    page,
  }) => {
    await installMockApi(page, { authenticated: true, projects: [project] });
    await page.route("**/api/v1/session", (route) =>
      route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          identity: {
            ownerId: "user-1",
            authenticated: true,
            name: "   ",
            email: "m.legkovskis@gmail.com",
          },
          providers: ["cloudflare"],
        }),
      }),
    );
    await page.goto("/");
    await expect(page.locator(".avatar")).toHaveText("ML");
    await expect(
      page.getByRole("button", { name: /m\.legkovskis@gmail\.com/ }),
    ).toContainText("Signed in");
    await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
    await page.getByRole("button", { name: "Switch to dark theme" }).click();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
    await page.reload();
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  });
});

test.describe("model studio", () => {
  test("unifies all 24 examples with an editable Python model", async ({
    page,
  }) => {
    await installMockApi(page, { authenticated: true, projects: [project] });
    await page.goto("/studies/project-1/workspace");
    await expect(page.locator(".example-card")).toHaveCount(24);
    await expect(page.getByLabel("Model name")).toHaveValue("");
    await expect(
      page.getByRole("textbox", { name: "Python model source" }),
    ).toBeVisible();
    await page.getByLabel("Search reference models").fill("rocket");
    await expect(page.locator(".example-card")).toHaveCount(1);
    await page.locator(".example-card").click();
    await expect(page.locator(".example-card.selected")).toContainText(
      "Rocket",
    );
    await expect(
      page.getByRole("textbox", { name: "Python model source" }),
    ).toBeVisible();
    await expect(page.getByLabel("Model name")).toHaveValue(
      "Rocket trajectory",
    );
    await page.getByLabel("Model name").fill("My adapted rocket");
    await page.getByLabel("Search reference models").fill("beam");
    await page.locator(".example-card").first().click();
    await expect(page.getByLabel("Model name")).toHaveValue(
      "My adapted rocket",
    );
  });

  test("creates a new project from the projects page", async ({ page }) => {
    await installMockApi(page, { authenticated: true, projects: [project] });
    await page.goto("/studies");
    await page.getByRole("button", { name: "New project" }).click();
    await page.getByLabel("Project name").fill("Fresh engineering project");
    await page
      .getByRole("button", { name: "Create project", exact: true })
      .click();
    await expect(
      page.getByRole("heading", { name: "Fresh engineering project" }),
    ).toBeVisible();
  });

  test("requires exact confirmation before permanently deleting a project", async ({
    page,
  }) => {
    await installMockApi(page, { authenticated: true, projects: [project] });
    await page.goto("/studies");
    await page.getByRole("button", { name: `Delete ${project.name}` }).click();
    await expect(
      page.getByRole("dialog", { name: `Delete “${project.name}”?` }),
    ).toBeVisible();
    const remove = page.getByRole("button", {
      name: "Delete project permanently",
    });
    await expect(remove).toBeDisabled();
    await page
      .getByLabel("Project name confirmation")
      .fill(`${project.name} typo`);
    await expect(remove).toBeDisabled();
    await page.getByLabel("Project name confirmation").fill(project.name);
    await remove.click();
    await expect(page.getByText(project.name)).toHaveCount(0);
    await expect(page.getByText("No projects yet")).toBeVisible();
  });

  test("keeps the validated understanding header inside its panel", async ({
    page,
  }, testInfo) => {
    await installMockApi(page, { authenticated: true, projects: [project] });
    await page.setViewportSize({ width: 1920, height: 1200 });
    await page.goto("/studies/project-1/workspace");
    await page.getByLabel("Search reference models").fill("Ishigami");
    await page.locator(".example-card").click();
    await page.getByRole("button", { name: "Validate & Assess" }).click();
    await expect(
      page.getByRole("heading", { name: "Model Understanding" }),
    ).toBeVisible();
    await expect(
      page.getByRole("heading", { name: /model equation/i }),
    ).toBeVisible();
    await expect(page.locator(".understanding-pane .katex")).toBeVisible();
    await page.locator(".validated-studio").evaluate((element) => {
      window.scrollTo({
        top: element.getBoundingClientRect().top + window.scrollY - 86,
        behavior: "auto",
      });
    });
    const workspaceBoxes = () =>
      page.locator(".validated-studio").evaluate((studio) => {
        const rectangle = (selector: string) => {
          const element = studio.querySelector(selector);
          if (!element) return null;
          const { x, y, width, height } = element.getBoundingClientRect();
          return { x, y, width, height };
        };
        return {
          pane: rectangle(".understanding-pane"),
          authoring: rectangle(".studio-authoring"),
          header: rectangle(".understanding-pane > header"),
        };
      });
    await expect
      .poll(
        async () => {
          const { pane, authoring } = await workspaceBoxes();
          if (!pane || !authoring) return Number.POSITIVE_INFINITY;
          const topDelta = Math.abs(pane.y - authoring.y);
          const bottomDelta = Math.abs(
            pane.y + pane.height - (authoring.y + authoring.height),
          );
          return Math.max(topDelta, bottomDelta);
        },
        {
          message:
            "wait for fonts and the validated workspace grid to finish laying out",
          timeout: 5_000,
        },
      )
      .toBeLessThanOrEqual(2);
    const { pane, authoring, header } = await workspaceBoxes();
    expect(pane).not.toBeNull();
    expect(authoring).not.toBeNull();
    expect(header).not.toBeNull();
    expect(header!.x).toBeGreaterThanOrEqual(pane!.x);
    expect(header!.x + header!.width).toBeLessThanOrEqual(
      pane!.x + pane!.width + 1,
    );
    expect(header!.y).toBeGreaterThanOrEqual(pane!.y);
    expect(Math.abs(pane!.y - authoring!.y)).toBeLessThanOrEqual(2);
    expect(
      Math.abs(pane!.y + pane!.height - (authoring!.y + authoring!.height)),
    ).toBeLessThanOrEqual(2);
    await expect(page.locator(".understanding-pane")).toHaveCSS(
      "overflow-y",
      "auto",
    );
    await expect(page.locator(".understanding-pane > header")).toHaveCSS(
      "position",
      "static",
    );
    const screenshot = await page.screenshot({ fullPage: true });
    await testInfo.attach("validated-workspace", {
      body: screenshot,
      contentType: "image/png",
    });
    if (process.env.UI_REVIEW_PATH) {
      await page.screenshot({
        path: process.env.UI_REVIEW_PATH,
        fullPage: true,
      });
    }
  });

  test("polls one in-flight Model Understanding generation without submitting retries", async ({
    page,
  }) => {
    let reads = 0;
    let writes = 0;
    await installMockApi(page, {
      authenticated: true,
      projects: [project],
      modelUnderstanding: async (route) => {
        if (route.request().method() === "POST") {
          writes += 1;
          await route.fulfill({
            status: 202,
            contentType: "application/json",
            body: JSON.stringify({ understanding: null }),
          });
          return;
        }
        reads += 1;
        const succeeded = reads >= 3;
        await route.fulfill({
          contentType: "application/json",
          body: JSON.stringify({
            understanding: {
              id: "understanding-1",
              modelVersionId: "model-1",
              modelHash: "abcdef",
              promptVersion: "1.3.0",
              aiModelId: "@cf/meta/llama-3.2-3b-instruct",
              status: succeeded ? "succeeded" : "generating",
              content: succeeded
                ? "## Model in brief\n\nThe existing generation completed once."
                : null,
              error: null,
              createdAt: "2026-08-23T12:00:00Z",
              updatedAt: "2026-08-23T12:00:01Z",
            },
          }),
        });
      },
    });
    await page.goto("/studies/project-1/workspace");
    await page.getByLabel("Search reference models").fill("Ishigami");
    await page.locator(".example-card").click();
    await page.getByRole("button", { name: "Validate & Assess" }).click();
    await expect(
      page.getByText("An existing AI generation is finishing…"),
    ).toBeVisible();
    await expect(
      page.getByRole("heading", { name: /model equation/i }),
    ).toBeVisible();
    await expect(page.locator(".validation-equations .katex")).toBeVisible();
    await expect(
      page.getByText("This model is practical to evaluate directly."),
    ).toHaveCount(0);
    await expect(
      page.getByText("The existing generation completed once."),
    ).toBeVisible();
    await expect(
      page.getByText("This model is practical to evaluate directly."),
    ).toBeVisible();
    expect(reads).toBeGreaterThanOrEqual(3);
    expect(writes).toBe(0);
  });

  test("surfaces Model Understanding failures as uncharged and retryable", async ({
    page,
  }) => {
    await installMockApi(page, {
      authenticated: true,
      projects: [project],
      modelUnderstanding: async (route) => {
        if (route.request().method() === "GET") {
          await route.fulfill({
            contentType: "application/json",
            body: JSON.stringify({ understanding: null }),
          });
          return;
        }
        await route.fulfill({
          status: 504,
          contentType: "application/json",
          body: JSON.stringify({
            error: {
              message:
                "The AI provider did not answer in time. Please retry; failed requests are not charged.",
            },
          }),
        });
      },
    });
    await page.goto("/studies/project-1/workspace");
    await page.getByLabel("Search reference models").fill("Ishigami");
    await page.locator(".example-card").click();
    await page.getByRole("button", { name: "Validate & Assess" }).click();
    await expect(
      page.getByText(/failed requests are not charged/i),
    ).toBeVisible();
    await expect(page.getByRole("button", { name: "Retry" })).toBeVisible();
  });

  test("opens validation feedback immediately and locks analyses until validation succeeds", async ({
    page,
  }) => {
    await installMockApi(page, { authenticated: true, projects: [project] });
    let releaseValidation: (() => void) | undefined;
    const validationGate = new Promise<void>((resolve) => {
      releaseValidation = resolve;
    });
    await page.route("**/api/v1/projects/*/models", async (route) => {
      if (route.request().method() === "POST") await validationGate;
      await route.fallback();
    });

    await page.goto("/studies/project-1/workspace");
    await page.getByLabel("Search reference models").fill("Ishigami");
    await page.locator(".example-card").click();

    const propagationChoice = page
      .locator(".analysis-option", { hasText: "Uncertainty Propagation" })
      .locator("input");
    await expect(propagationChoice).toBeDisabled();
    await expect(page.getByLabel("Standard sample budget")).toBeDisabled();

    await page.getByRole("button", { name: "Validate & Assess" }).click();
    await expect(page.getByLabel("Model Understanding")).toBeVisible();
    await expect(
      page.getByText("Your model is being validated…"),
    ).toBeVisible();
    await expect(page.locator(".validation-loader")).toBeVisible();
    await expect(page.locator("#direct-analyses")).toHaveAttribute(
      "aria-busy",
      "true",
    );
    await expect(propagationChoice).toBeDisabled();
    await expect(page.getByLabel("Standard sample budget")).toBeDisabled();

    releaseValidation?.();
    await expect(
      page.getByText("Model validated", { exact: true }),
    ).toBeVisible();
    await expect(page.getByText("Your model is being validated…")).toHaveCount(
      0,
    );
    await expect(propagationChoice).toBeEnabled();
    await expect(page.getByLabel("Standard sample budget")).toBeEnabled();
  });

  test("keeps target HSIC UI defaults within the maximum-dimensional core resource envelope", async ({
    page,
  }, testInfo) => {
    const maximumDimensionalModel = {
      ...savedModel,
      id: "model-20d",
      metadata: {
        ...savedModel.metadata,
        input_dimension: 20,
        inputs: Array.from({ length: 20 }, (_, index) => ({
          ...savedModel.metadata.inputs[0]!,
          index,
          name: `x${index}`,
        })),
      },
      assessment: savedModel.assessment
        ? {
            ...savedModel.assessment,
            profile: {
              ...savedModel.assessment.profile,
              input_dimension: 20,
            },
          }
        : undefined,
    };
    await installMockApi(page, {
      authenticated: true,
      projects: [project],
      models: [maximumDimensionalModel],
    });
    let runBody: Record<string, unknown> | undefined;
    page.on("request", (request) => {
      if (
        request.url().endsWith("/api/v1/runs") &&
        request.method() === "POST"
      ) {
        runBody = request.postDataJSON() as Record<string, unknown>;
      }
    });

    await page.goto("/studies/project-1/workspace");
    await page.getByLabel("Search reference models").fill("Ishigami");
    await page.locator(".example-card").click();
    await page.getByRole("button", { name: "Validate & Assess" }).click();
    await expect(
      page.getByText("Model validated", { exact: true }),
    ).toBeVisible();
    await page.getByLabel("Standard sample budget").fill("20000");
    await page
      .locator(".analysis-option", {
        hasText: "Target-Domain HSIC Sensitivity",
      })
      .locator("input")
      .check();
    await expect(
      page.getByRole("region", { name: "HSIC method comparison" }),
    ).toContainText("Two complementary HSIC questions");
    await expect(
      page.getByText("Whole response", { exact: true }),
    ).toBeVisible();
    await expect(
      page.getByText("Critical region", { exact: true }),
    ).toBeVisible();
    await expect(page.getByLabel("Region direction")).toBeVisible();
    await expect(page.getByLabel("Output threshold")).toBeVisible();
    await expect(page.getByLabel("Permutation replicates")).toHaveValue("100");
    const targetBoxes = await page
      .locator(".target-hsic-studio")
      .evaluate((studio) => {
        const bounds = (selector: string) => {
          const element = studio.querySelector(selector);
          if (!element) return null;
          const { x, y, width, height } = element.getBoundingClientRect();
          return { x, y, width, height };
        };
        const { x, y, width, height } = studio.getBoundingClientRect();
        return {
          studio: { x, y, width, height },
          direction: bounds(".target-domain-fields label:first-child"),
          threshold: bounds(".target-domain-fields label:last-child"),
          permutations: bounds(".target-permutation-card label"),
        };
      });
    expect(targetBoxes.studio).not.toBeNull();
    expect(targetBoxes.direction).not.toBeNull();
    expect(targetBoxes.threshold).not.toBeNull();
    expect(targetBoxes.permutations).not.toBeNull();
    expect(targetBoxes.direction!.width).toBeGreaterThan(250);
    expect(targetBoxes.threshold!.width).toBeGreaterThan(200);
    expect(
      targetBoxes.direction!.x + targetBoxes.direction!.width,
    ).toBeLessThanOrEqual(targetBoxes.threshold!.x);
    expect(targetBoxes.permutations!.width).toBeGreaterThan(180);
    await testInfo.attach("target-hsic-composer", {
      body: await page.screenshot({ fullPage: true }),
      contentType: "image/png",
    });
    await page.getByRole("button", { name: "Run analyses" }).click();

    await expect.poll(() => runBody).toBeTruthy();
    const analyses = runBody?.analyses as Array<{
      analysisKey: string;
      config: Record<string, unknown>;
    }>;
    expect(
      analyses.find((item) => item.analysisKey === "target_hsic")?.config,
    ).toMatchObject({
      sample_size: 250,
      permutations: 100,
    });
  });

  test("caps damped-oscillator HSIC at the validated resource limit and explains the bound", async ({
    page,
  }) => {
    const dampedOscillator = {
      ...savedModel,
      id: "model-damped-oscillator",
      displayName: "Damped oscillator",
      metadata: {
        ...savedModel.metadata,
        input_dimension: 8,
        inputs: Array.from({ length: 8 }, (_, index) => ({
          ...savedModel.metadata.inputs[0]!,
          index,
          name: `x${index + 1}`,
        })),
      },
      assessment: savedModel.assessment
        ? {
            ...savedModel.assessment,
            profile: { ...savedModel.assessment.profile, input_dimension: 8 },
            recommendations: savedModel.assessment.recommendations.map(
              (recommendation) =>
                recommendation.capability === "hsic"
                  ? {
                      ...recommendation,
                      safe_config: {
                        maximum_sample_size: 400,
                        permutations: 100,
                      },
                    }
                  : recommendation,
            ),
          }
        : undefined,
    };
    await installMockApi(page, {
      authenticated: true,
      projects: [project],
      models: [dampedOscillator],
    });
    let runBody: Record<string, unknown> | undefined;
    page.on("request", (request) => {
      if (
        request.url().endsWith("/api/v1/runs") &&
        request.method() === "POST"
      ) {
        runBody = request.postDataJSON() as Record<string, unknown>;
      }
    });

    await page.goto("/studies/project-1/workspace");
    await page.getByLabel("Search reference models").fill("Ishigami");
    await page.locator(".example-card").click();
    await page.getByRole("button", { name: "Validate & Assess" }).click();
    await expect(
      page.getByText("Model validated", { exact: true }),
    ).toBeVisible();
    const hsicOption = page.locator(".analysis-option", {
      hasText: "HSIC Dependence",
    });
    await expect(hsicOption).toContainText("at most 400 samples");
    await page.getByLabel("Standard sample budget").fill("1000");
    await hsicOption.locator("input").check();
    await page.getByRole("button", { name: "Run analyses" }).click();

    await expect.poll(() => runBody).toBeTruthy();
    const hsic = (
      runBody?.analyses as Array<{
        analysisKey: string;
        config: Record<string, unknown>;
      }>
    ).find((analysis) => analysis.analysisKey === "hsic");
    expect(hsic?.config).toMatchObject({ sample_size: 400, permutations: 100 });
  });

  test("greys incompatible dependent-input analyses and states the scientific reason", async ({
    page,
  }) => {
    const dependentModel = {
      ...savedModel,
      id: "model-dependent",
      metadata: {
        ...savedModel.metadata,
        dependent_inputs: true,
        copula: "NormalCopula",
      },
      assessment: savedModel.assessment
        ? {
            ...savedModel.assessment,
            profile: {
              ...savedModel.assessment.profile,
              dependent_inputs: true,
              copula: "NormalCopula",
            },
            recommendations: [
              {
                capability: "sobol",
                status: "incompatible" as const,
                priority: 3,
                rationale_codes: ["INDEPENDENT_INPUTS_REQUIRED"],
                compatibility_warnings: [
                  "Sobol Sensitivity Analysis requires independent inputs; this model declares a dependent copula.",
                ],
              },
              {
                capability: "ancova",
                status: "recommended" as const,
                priority: 2,
                rationale_codes: ["DEPENDENT_INPUT_VARIANCE_DECOMPOSITION"],
                compatibility_warnings: [],
              },
            ],
          }
        : undefined,
    };
    await installMockApi(page, {
      authenticated: true,
      projects: [project],
      models: [dependentModel],
    });

    await page.goto("/studies/project-1/workspace");
    await page.getByLabel("Search reference models").fill("Ishigami");
    await page.locator(".example-card").click();
    await page.getByRole("button", { name: "Validate & Assess" }).click();
    const sobol = page.locator(".analysis-option", {
      hasText: "Sobol Sensitivity",
    });
    await expect(sobol.locator("input")).toBeDisabled();
    await expect(sobol).toHaveClass(/incompatible/);
    await expect(sobol).toContainText("requires independent inputs");
    await expect(
      page
        .locator(".analysis-option", {
          hasText: "ANCOVA Dependent-Input Sensitivity",
        })
        .locator("input"),
    ).toBeEnabled();
  });

  test("creates a project, authors with the guided builder, configures all direct plugins, and queues the suite", async ({
    page,
  }) => {
    await installMockApi(page, { authenticated: true });
    let runBody: Record<string, unknown> | undefined;
    page.on("request", (request) => {
      if (
        request.url().endsWith("/api/v1/runs") &&
        request.method() === "POST"
      ) {
        runBody = request.postDataJSON() as Record<string, unknown>;
      }
    });

    await page.goto("/studies");
    await page.getByRole("button", { name: "Create first project" }).click();
    await page.getByLabel("Project name").fill("Complete browser study");
    await page
      .getByRole("button", { name: "Create project", exact: true })
      .click();
    await expect(
      page.getByRole("heading", { name: "Complete browser study" }),
    ).toBeVisible();
    await page
      .getByRole("link", { name: /New analysis in this project/ })
      .click();

    await expect(
      page.getByRole("button", { name: /Run analyses/ }),
    ).toBeDisabled();
    await page.getByLabel("Model name").fill("Guided browser model");
    await page.getByRole("button", { name: "Guided builder" }).click();
    await page.getByRole("button", { name: "Add variable" }).click();
    await page.getByLabel("Variable 3 name").fill("pressure");
    await page.getByLabel("Output 1 formula").fill("x1 + x2^2 + pressure");
    await page.getByRole("button", { name: "Validate & Assess" }).click();
    await expect(
      page.getByText("Model validated", { exact: true }),
    ).toBeVisible();

    const checkboxes = page.locator(".analysis-option input[type=checkbox]");
    const directCatalog = catalog.filter(
      (item) =>
        !["calibration_nlls", "morris", "pce", "gpr"].includes(item.key),
    );
    await expect(checkboxes).toHaveCount(directCatalog.length);
    const incompatibleAncova = page.locator(".analysis-option", {
      hasText: "ANCOVA Dependent-Input Sensitivity",
    });
    await expect(incompatibleAncova.locator("input")).toBeDisabled();
    await expect(incompatibleAncova).toContainText(
      "ANCOVA requires two to ten continuous inputs with a dependent copula.",
    );
    const enabledCheckboxes = page.locator(
      ".analysis-option input[type=checkbox]:enabled",
    );
    for (let index = 0; index < (await enabledCheckboxes.count()); index += 1) {
      await enabledCheckboxes.nth(index).check();
    }
    await expect(page.getByText("10 analysis tasks")).toBeVisible();
    await page.getByLabel("Standard sample budget").fill("512");
    await page.getByLabel("Reliability method").selectOption("MONTE_CARLO");
    await page.getByLabel("Failure event").selectOption("<");
    await page
      .getByRole("spinbutton", { name: "Threshold", exact: true })
      .fill("-2.5");
    await page.getByLabel("Region direction").selectOption("<=");
    await page.getByLabel("Output threshold").fill("4.5");
    await page.getByLabel("Permutation replicates").fill("40");
    await page.getByRole("button", { name: "Run analyses" }).click();

    await expect(page).toHaveURL(/\/runs\/run-1$/);
    await expect.poll(() => runBody).toBeTruthy();
    const analyses = runBody?.analyses as Array<{
      analysisKey: string;
      config: Record<string, unknown>;
    }>;
    expect(analyses.map((item) => item.analysisKey).sort()).toEqual(
      directCatalog
        .filter((item) => !item.requires_dependent_inputs)
        .map((item) => item.key)
        .sort(),
    );
    expect(
      analyses.find((item) => item.analysisKey === "reliability")?.config,
    ).toMatchObject({
      method: "MONTE_CARLO",
      operator: "<",
      threshold: -2.5,
    });
    expect(
      analyses.find((item) => item.analysisKey === "target_hsic")?.config,
    ).toMatchObject({
      sample_size: 250,
      operator: "<=",
      threshold: 4.5,
      permutations: 40,
    });
    expect(
      analyses.some((item) =>
        ["calibration_nlls", "morris", "pce", "gpr"].includes(item.analysisKey),
      ),
    ).toBe(false);
  });

  test("surfaces model validation and catalog failures without enabling a run", async ({
    page,
  }) => {
    await installMockApi(page, { authenticated: true, projects: [project] });
    await page.route("**/api/v1/analyses/catalog", (route) =>
      route.fulfill({
        status: 503,
        contentType: "application/json",
        body: JSON.stringify({ error: { message: "offline" } }),
      }),
    );
    await page.route("**/api/v1/projects/*/models", (route) =>
      route.fulfill({
        status: 422,
        contentType: "application/json",
        body: JSON.stringify({
          error: {
            code: "invalid_model",
            message: "Model must define model and distribution.",
          },
        }),
      }),
    );
    await page.goto("/studies/project-1/workspace");
    await page.getByLabel("Search reference models").fill("Ishigami");
    await page.locator(".example-card").click();
    await page.getByRole("button", { name: "Validate & Assess" }).click();
    await expect(
      page.getByText("Model must define model and distribution."),
    ).toBeVisible();
    await expect(page.getByText("Catalog unavailable")).toBeVisible();
    await expect(
      page.getByRole("button", { name: "Run analyses" }),
    ).toBeDisabled();
  });

  test("keeps surrogate construction in its studio and passes a promoted surrogate explicitly", async ({
    page,
  }) => {
    await installMockApi(page, { authenticated: true, projects: [project] });
    let runBody: Record<string, unknown> | undefined;
    page.on("request", (request) => {
      if (request.url().endsWith("/api/v1/runs") && request.method() === "POST")
        runBody = request.postDataJSON() as Record<string, unknown>;
    });
    await page.goto("/studies/project-1/surrogates");
    await page.getByRole("button", { name: "Build GPR candidate" }).click();
    await expect(page.getByText("Meets default")).toBeVisible();
    await page
      .getByRole("button", { name: "Promote validated surrogate" })
      .click();
    await expect(
      page.getByRole("link", {
        name: /Start a new project with this surrogate/,
      }),
    ).toHaveAttribute("href", /sourceModel=model-1.*surrogate=surrogate-1/);
    await page
      .getByRole("link", { name: /Start a new analysis with this surrogate/ })
      .click();
    await expect(
      page.getByText("Promoted surrogate selected in Surrogate Studio"),
    ).toBeVisible();
    await page.getByRole("button", { name: "Run analyses" }).click();
    await expect.poll(() => runBody?.surrogateModelId).toBe("surrogate-1");
  });

  test("copies a promoted surrogate and its exact source model into a new project", async ({
    page,
  }) => {
    await installMockApi(page, { authenticated: true, projects: [project] });
    await page.goto("/studies/project-1/surrogates");
    await page.getByRole("button", { name: "Build GPR candidate" }).click();
    await page
      .getByRole("button", { name: "Promote validated surrogate" })
      .click();
    await page
      .getByRole("link", { name: /Start a new project with this surrogate/ })
      .click();
    await expect(
      page.getByRole("heading", {
        name: "Start a new project with this surrogate.",
      }),
    ).toBeVisible();
    await expect(page.getByLabel("Project name")).toHaveValue(
      /surrogate study/,
    );
    await page
      .getByRole("button", { name: "Create project with surrogate" })
      .click();
    await expect(page).toHaveURL(
      /\/studies\/project-created-2\/workspace\?sourceModel=.*&surrogate=surrogate-copied/,
    );
    await expect(
      page.getByText("Promoted surrogate selected in Surrogate Studio"),
    ).toBeVisible();
  });

  test("builds and validates a Gaussian-process surrogate from empirical data", async ({
    page,
  }) => {
    await installMockApi(page, { authenticated: true, projects: [project] });
    await page.goto("/studies/project-1/surrogates");
    await page.getByRole("tab", { name: "From empirical data" }).click();
    await expect(
      page.getByRole("heading", {
        name: "Choose a dataset with input and output columns.",
      }),
    ).toBeVisible();
    await expect(page.getByLabel("Output column")).toHaveValue("pressure");
    await expect(
      page.getByRole("group").getByText("temperature", { exact: true }),
    ).toBeVisible();
    await page.getByRole("button", { name: "Build data-driven GPR" }).click();
    await expect(page.getByText("Data-driven GPR retained")).toBeVisible();
    await expect(page.getByText("0.98200")).toBeVisible();
    await expect(
      page.locator(".data-surrogate-evidence .echart canvas"),
    ).toHaveCount(1);
  });

  test("runs dimensionality screening from the dedicated studio", async ({
    page,
  }) => {
    await installMockApi(page, { authenticated: true, projects: [project] });
    let runBody: Record<string, unknown> | undefined;
    page.on("request", (request) => {
      if (request.url().endsWith("/api/v1/runs") && request.method() === "POST")
        runBody = request.postDataJSON() as Record<string, unknown>;
    });
    await page.goto("/studies/project-1/dimension-reduction");
    await page.getByRole("button", { name: "Run Morris screening" }).click();
    await expect(page).toHaveURL(/\/runs\/run-1$/);
    expect(
      (runBody?.analyses as Array<{ analysisKey: string }>)[0]?.analysisKey,
    ).toBe("morris");
  });

  test("runs the official named calibration setup from its project studio", async ({
    page,
  }) => {
    await installMockApi(page, {
      authenticated: true,
      projects: [project],
      models: [calibrationSavedModel],
    });
    let runBody: Record<string, unknown> | undefined;
    page.on("request", (request) => {
      if (
        request.url().endsWith("/api/v1/runs") &&
        request.method() === "POST"
      ) {
        runBody = request.postDataJSON() as Record<string, unknown>;
      }
    });

    await page.goto("/studies/project-1/calibration");
    await expect(
      page.getByRole("heading", {
        name: "Estimate model parameters from observations.",
      }),
    ).toBeVisible();
    await expect(
      page.getByText("Official OpenTURNS exponential example loaded"),
    ).toBeVisible();
    await expect(page.getByLabel("Starting value for a")).toHaveValue("1");
    await expect(page.getByLabel("Starting value for b")).toHaveValue("1");
    await expect(page.getByLabel("Starting value for c")).toHaveValue("1");
    await expect(page.getByLabel("Calibration observation CSV")).toHaveValue(
      /^x,y/,
    );
    await expect(page.getByText("10 / 250")).toBeVisible();
    await page
      .getByRole("button", { name: "Run nonlinear least-squares calibration" })
      .click();
    await expect(page).toHaveURL(/\/runs\/run-1$/);

    const analysis = (
      runBody?.analyses as Array<{
        analysisKey: string;
        config: Record<string, unknown>;
        outputTargets: number[];
      }>
    )[0]!;
    expect(analysis.analysisKey).toBe("calibration_nlls");
    expect(analysis.outputTargets).toEqual([0]);
    expect(analysis.config).toMatchObject({
      parameter_indices: [0, 1, 2],
      starting_values: [1, 1, 1],
      observed_input_names: ["x"],
      observed_output_name: "y",
      maximum_calls: 250,
    });
    expect(analysis.config.observed_inputs).toHaveLength(10);
    expect(analysis.config.observed_outputs).toHaveLength(10);
  });
});

test.describe("run lifecycle", () => {
  test("renders live task progress and sends cancellation", async ({
    page,
  }) => {
    await installMockApi(page, {
      authenticated: true,
      projects: [project],
      runs: [makeRun("running")],
    });
    let cancelRequested = false;
    await page.route("**/api/v1/runs/run-1/cancel", async (route) => {
      cancelRequested = true;
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({ status: "cancelled" }),
      });
    });
    await page.goto("/runs/run-1");
    await expect(page.getByText("1 of 3 tasks complete")).toBeVisible();
    await expect(page.locator(".task-row")).toHaveCount(3);
    const activeHsic = page.locator('[data-analysis-key="hsic"]');
    await expect(activeHsic).toContainText(
      "OpenTURNS is evaluating 100 permutation replicates.",
    );
    await expect(
      activeHsic.getByRole("progressbar", { name: "Global HSIC progress" }),
    ).not.toHaveAttribute("aria-valuenow");
    const queuedCorrelation = page.locator('[data-analysis-key="correlation"]');
    await expect(queuedCorrelation).toContainText(
      "Waiting for isolated compute capacity.",
    );
    await expect(
      queuedCorrelation.getByRole("progressbar", {
        name: "correlation progress",
      }),
    ).toBeVisible();
    await page.getByRole("button", { name: "Cancel" }).click();
    await expect.poll(() => cancelRequested).toBe(true);
  });

  test("opens the report from a terminal run", async ({ page }) => {
    await installMockApi(page, {
      authenticated: true,
      projects: [project],
      runs: [makeRun("succeeded")],
    });
    await page.goto("/runs/run-1");
    await expect(page.getByText("The report is ready.")).toBeVisible();
    await page.getByRole("link", { name: /Open report/ }).click();
    await expect(
      page.getByRole("heading", { name: "Verification report" }),
    ).toBeVisible();
  });
});

test.describe("reports and grounded chat", () => {
  test("renders equation metadata retained from an authenticated custom Python model", async ({
    page,
  }) => {
    const report = makeReport();
    report.modelVersion.sourceKind = "python";
    report.model.equations = [
      {
        output_name: "User-defined deflection",
        latex: "y=\\frac{F L^3}{3 E I}",
        representation: "closed_form",
      },
    ];
    await installMockApi(page, {
      authenticated: true,
      projects: [project],
      report,
    });

    await page.goto("/reports/report-1");
    await expect(page.getByText("User-defined deflection")).toBeVisible();
    await expect(
      page.locator(".model-definition-section .katex"),
    ).toBeVisible();
    await expect(page.locator(".model-definition-section")).not.toContainText(
      "response = x1 + x2^2",
    );
  });

  test("renders table-only FAST evidence as sensitivity visuals with exact rows retained", async ({
    page,
  }) => {
    const fast = analysisResult("fast");
    fast.payload.metrics = {
      sample_size: 1000,
      model_evaluations: 7000,
    };
    fast.payload.tables = {
      indices: {
        columns: ["Variable", "First Order", "Total Order", "Interaction"],
        rows: [
          ["x1", 0.31, 0.55, 0.24],
          ["x2", 0.44, 0.46, 0.02],
          ["x3", 0.01, 0.18, 0.17],
        ],
        row_count: 3,
        truncated: false,
      },
    };
    fast.payload.series = {};
    fast.payload.matrices = {};
    const report = makeReport();
    report.status = "succeeded";
    report.sections = [{ key: "fast", status: "succeeded", result: fast }];
    await installMockApi(page, {
      authenticated: true,
      projects: [project],
      report,
    });

    await page.goto("/reports/report-1");
    await expect(
      page.getByRole("img", {
        name: /FAST first-order and total-order indices/,
      }),
    ).toBeVisible();
    await expect(
      page.getByRole("img", { name: /FAST total-order decomposition/ }),
    ).toBeVisible();
    await expect(
      page.getByRole("table").filter({ hasText: "x3" }),
    ).toBeVisible();
  });

  test("renders every evidence type and operates export, share, print, and streaming chat", async ({
    page,
  }, testInfo) => {
    await installMockApi(page, {
      authenticated: true,
      projects: [project],
      runs: [makeRun()],
      report: makeReport(),
    });
    await page.goto("/reports/report-1");

    await expect(
      page.getByRole("heading", { name: "Verification report" }),
    ).toBeVisible();
    await expect(page.locator(".metric")).toHaveCount(2);
    await expect(page.getByRole("table").first()).toBeVisible();
    await expect(page.getByText("Rendered equations")).toBeVisible();
    await expect(
      page.getByLabel("Exact immutable Python model source"),
    ).toBeVisible();
    await expect(
      page.locator(".model-definition-section .cm-line span").first(),
    ).toHaveText("import");
    await expect(page.locator(".echart")).toHaveCount(2);
    await expect(page.getByText("Exact heatmap values")).toBeVisible();
    const chartPanel = page
      .locator(".plot-panel")
      .filter({ hasText: "Exact chart data" })
      .first();
    const chartSummary = chartPanel.getByText("Exact chart data", {
      exact: true,
    });
    await expect(chartSummary).toHaveCSS("display", "flex");
    const chartPanelBox = await chartPanel.boundingBox();
    const chartSummaryBox = await chartSummary.boundingBox();
    expect(chartPanelBox).not.toBeNull();
    expect(chartSummaryBox).not.toBeNull();
    expect(chartSummaryBox!.x).toBeGreaterThanOrEqual(chartPanelBox!.x);
    await expect(
      page.getByText("Deliberate partial-failure evidence."),
    ).toBeVisible();
    await page.getByText("Method assumptions and provenance").click();
    await expect(page.getByText(/Core 0.2.0/)).toBeVisible();

    await expect(
      page.getByRole("link", { name: "Data bundle" }),
    ).toHaveAttribute("href", "/api/v1/reports/report-1/export");

    await page.getByRole("button", { name: "Share" }).click();
    await expect(
      page.getByRole("checkbox", { name: "Include model definition" }),
    ).not.toBeChecked();
    await page.getByRole("button", { name: "Create share link" }).click();
    await expect(page.getByText(/Share link copied:/)).toBeVisible();
    const downloadPromise = page.waitForEvent("download");
    await page.getByRole("button", { name: "Download PDF" }).click();
    const download = await downloadPromise;
    expect(download.suggestedFilename()).toMatch(/-report\.pdf$/);

    await page
      .getByRole("button", { name: "Which input has the greatest influence?" })
      .click();
    await expect(page.getByLabel("Question about report")).toHaveValue(
      /greatest influence/,
    );
    await page.getByRole("button", { name: "Send question" }).click();
    await expect(page.getByText(/x1 is greatest/)).toBeVisible();
    await expect(
      page.getByText("Source: Monte Carlo · Strongest Input"),
    ).toBeVisible();
    await expect(
      page.getByText(/analysis\.fact:monte_carlo\.strongest_input/),
    ).toHaveCount(0);
    await testInfo.attach("report-evidence-ui", {
      body: await page.screenshot({ fullPage: true }),
      contentType: "image/png",
    });
  });

  test("shared reports are read-only and hide owner chat", async ({ page }) => {
    await installMockApi(page, { authenticated: true, report: makeReport() });
    await page.goto("/shared/share-token");
    await expect(
      page.getByRole("heading", { name: "Verification report" }),
    ).toBeVisible();
    await expect(page.getByRole("button", { name: "Share" })).toHaveCount(0);
    await expect(page.getByRole("link", { name: "Data bundle" })).toHaveCount(
      0,
    );
    await expect(page.getByText("Ask this report")).toHaveCount(0);
    await expect(
      page.getByRole("button", { name: "Download PDF" }),
    ).toBeVisible();
  });

  test("chat failures are explicit and preserve the user's question", async ({
    page,
  }) => {
    await installMockApi(page, { authenticated: true, report: makeReport() });
    await page.route("**/api/v1/reports/report-1/chat", async (route) => {
      if (route.request().method() === "GET") {
        await route.fulfill({
          contentType: "application/json",
          body: JSON.stringify({ messages: [] }),
        });
      } else {
        await route.fulfill({
          status: 429,
          contentType: "application/json",
          body: JSON.stringify({
            error: { message: "Daily report-chat quota exceeded." },
          }),
        });
      }
    });
    await page.goto("/reports/report-1");
    await page.getByLabel("Question about report").fill("Summarise the result");
    await page.getByLabel("Question about report").press("Enter");
    await expect(
      page.getByText("Daily report-chat quota exceeded."),
    ).toBeVisible();
    await expect(page.getByText("Summarise the result")).toBeVisible();
  });

  test("an unauthenticated visitor cannot open any report", async ({
    page,
  }) => {
    await installMockApi(page, { report: makeReport() });
    await page.goto("/reports/report-1");
    await expect(
      page.getByRole("heading", {
        name: "Sign in before starting an analysis.",
      }),
    ).toBeVisible();
    await expect(
      page.getByRole("heading", { name: "Verification report" }),
    ).toHaveCount(0);
    await expect(page.getByText("Ask this report")).toHaveCount(0);
  });
});

test.describe("distribution data lab", () => {
  test("uploads retained data, ranks marginals, and generates an explicit problem draft", async ({
    page,
  }) => {
    await installMockApi(page, { authenticated: true, projects: [project] });
    await page.goto("/studies/project-1/data-lab");
    await expect(
      page
        .getByText("Or paste comma-separated data")
        .locator("..")
        .getByRole("textbox"),
    ).toHaveValue(/^E,F,L,I/);
    await expect(
      page.getByRole("button", { name: /Fixture observations\.csv/ }),
    ).toBeVisible();
    await expect(page.getByRole("table").first()).toContainText("temperature");
    await page.getByRole("button", { name: "Rank candidate fits" }).click();
    await expect(
      page.getByRole("heading", { name: "temperature" }),
    ).toBeVisible();
    await expect(
      page.locator(".distribution-chart-grid .echart canvas"),
    ).toHaveCount(6);
    await page
      .getByLabel("Selected marginal for temperature")
      .selectOption("Normal");
    await page
      .getByLabel("Selected marginal for pressure")
      .selectOption("Normal");
    await page
      .getByRole("button", { name: "Generate problem definition" })
      .click();
    await expect(page.getByText("generated_problem.py")).toBeVisible();
    await page.getByRole("button", { name: "Prepare model draft" }).click();
    await expect(page).toHaveURL(
      /\/studies\/project-1\/workspace\?dataFit=fit-2/,
    );
    await expect(
      page.getByText(/Distribution draft from retained fit/),
    ).toBeVisible();
  });
});

test.describe("dimensionality screening", () => {
  test("requires explicit fixed values before creating an immutable derived version", async ({
    page,
  }) => {
    const report = makeReport();
    const morris = analysisResult("morris");
    morris.plugin_version = "2.0.0";
    morris.payload.metrics = { candidate_threshold_fraction: 0.05 };
    morris.payload.tables = {
      effects: {
        columns: [
          "Variable",
          "Signed Mean Effect",
          "Mean Absolute Effect",
          "Effect Dispersion",
          "Rank",
          "Candidate Retained",
        ],
        rows: [
          ["x1", 0.8, 0.8, 0.1, 1, true],
          ["x2", 0.2, 0.2, 0.05, 2, true],
          ["x3", 0.01, 0.01, 0.02, 3, false],
        ],
        row_count: 3,
        truncated: false,
      },
    };
    report.sections = [{ key: "morris", status: "succeeded", result: morris }];
    await installMockApi(page, {
      authenticated: true,
      projects: [project],
      runs: [makeRun()],
      report,
    });
    await page.goto("/reports/report-1");
    await expect(
      page.getByRole("heading", { name: "Confirm active and fixed variables" }),
    ).toBeVisible();
    await page.getByLabel("Fixed value for x3").fill("0.25");
    await page.getByText(/I confirm these explicit fixed values/).click();
    await page.getByRole("button", { name: "Create derived version" }).click();
    await expect(page.getByText("Reduced model created")).toBeVisible();
    await expect(
      page.getByRole("link", {
        name: /Start a new analysis with the reduced model/,
      }),
    ).toHaveAttribute(
      "href",
      "/studies/project-1/workspace?sourceModel=model-reduced",
    );
    await expect(
      page.getByRole("link", {
        name: /Start a new project with the reduced model/,
      }),
    ).toHaveAttribute("href", /\/studies\?new=1&sourceModel=model-reduced/);
  });
});

test.describe("durable activity", () => {
  test("shows empty states and later links retained projects and runs", async ({
    page,
  }) => {
    await installMockApi(page, { authenticated: true });
    await page.goto("/activity");
    await expect(page.getByText("No projects yet")).toBeVisible();

    const secondPage = await page.context().newPage();
    await installMockApi(secondPage, {
      authenticated: true,
      projects: [project],
      runs: [makeRun()],
    });
    await secondPage.goto("/activity");
    await expect(secondPage.getByText(project.name)).toBeVisible();
    await secondPage.locator(".project-row").click();
    await expect(secondPage).toHaveURL(/\/studies\/project-1$/);
    await secondPage.close();
  });
});
