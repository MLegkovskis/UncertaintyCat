import { defineConfig, devices } from "@playwright/test";

const stateDirectory = `.wrangler/e2e-${process.pid}`;
const wrangler =
  "npx --yes node@22 apps/api/node_modules/wrangler/bin/wrangler.js";

export default defineConfig({
  testDir: "./e2e/full-stack",
  fullyParallel: false,
  workers: 1,
  timeout: 8 * 60_000,
  expect: { timeout: 30_000 },
  retries: 0,
  reporter: process.env.CI
    ? [["github"], ["html", { open: "never", outputFolder: "playwright-report-full-stack" }]]
    : [["list"], ["html", { open: "never", outputFolder: "playwright-report-full-stack" }]],
  use: {
    baseURL: "http://127.0.0.1:4174",
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
    video: "retain-on-failure",
  },
  projects: [{ name: "full-stack-chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: [
    {
      command: "uv run uvicorn services.compute.main:app --host 127.0.0.1 --port 8080",
      cwd: "../..",
      url: "http://127.0.0.1:8080/health",
      reuseExistingServer: false,
      timeout: 120_000,
    },
    {
      command: `${wrangler} d1 migrations apply uncertaintycat-local --local --persist-to ${stateDirectory} --config apps/api/wrangler.jsonc && ${wrangler} dev --local --port 8787 --persist-to ${stateDirectory} --config apps/api/wrangler.jsonc`,
      cwd: "../..",
      url: "http://127.0.0.1:8787/health",
      reuseExistingServer: false,
      timeout: 120_000,
    },
    {
      command: "npm run dev --workspace @uncertaintycat/web -- --port 4174",
      cwd: "../..",
      url: "http://127.0.0.1:4174",
      reuseExistingServer: false,
      timeout: 120_000,
    },
  ],
});
