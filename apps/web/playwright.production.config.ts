import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e/production",
  fullyParallel: false,
  workers: 1,
  timeout: 120_000,
  expect: { timeout: 20_000 },
  retries: process.env.CI ? 2 : 0,
  reporter: process.env.CI
    ? [["github"], ["html", { open: "never", outputFolder: "playwright-report-production" }]]
    : [["list"], ["html", { open: "never", outputFolder: "playwright-report-production" }]],
  use: {
    baseURL: process.env.E2E_BASE_URL ?? "https://uncertaintycat.com",
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
    video: "retain-on-failure",
  },
  projects: [{ name: "production-chromium", use: { ...devices["Desktop Chrome"] } }],
});
