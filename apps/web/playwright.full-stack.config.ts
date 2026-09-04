import { defineConfig, devices } from "@playwright/test";

// Resolve Wrangler through npm's workspace contract. npm is free to hoist the
// executable to the repository root when regenerating the lockfile, so tests
// must not depend on a particular node_modules layout.
const wrangler = "npm exec --workspace @uncertaintycat/api -- wrangler";
// npm exec runs the command from the selected workspace, so these paths are
// deliberately relative to apps/api rather than the Playwright/root process.
const stateDirectory = `../../.wrangler/e2e-${process.pid}`;
const wranglerConfig = "wrangler.no-ai.jsonc";

export default defineConfig({
  testDir: "./e2e/full-stack",
  fullyParallel: false,
  workers: 1,
  timeout: 8 * 60_000,
  expect: { timeout: 30_000 },
  retries: 0,
  reporter: process.env.CI
    ? [
        ["github"],
        [
          "html",
          { open: "never", outputFolder: "playwright-report-full-stack" },
        ],
      ]
    : [
        ["list"],
        [
          "html",
          { open: "never", outputFolder: "playwright-report-full-stack" },
        ],
      ],
  use: {
    baseURL: "http://127.0.0.1:4174",
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
    video: "retain-on-failure",
  },
  projects: [
    { name: "full-stack-chromium", use: { ...devices["Desktop Chrome"] } },
  ],
  webServer: [
    {
      command: "node e2e/full-stack/groq-fixture.mjs",
      cwd: ".",
      url: "http://127.0.0.1:8790/health",
      reuseExistingServer: false,
      timeout: 30_000,
    },
    {
      command:
        "uv run uvicorn services.compute.main:app --host 127.0.0.1 --port 8080",
      cwd: "../..",
      url: "http://127.0.0.1:8080/health",
      reuseExistingServer: false,
      timeout: 120_000,
    },
    {
      command: `${wrangler} d1 migrations apply uncertaintycat-local --local --persist-to ${stateDirectory} --config ${wranglerConfig} && ${wrangler} dev --local --port 8787 --persist-to ${stateDirectory} --config ${wranglerConfig} --var AI_PROVIDER:groq --var GROQ_API_KEY:e2e-local-only --var GROQ_BASE_URL:http://127.0.0.1:8790/openai/v1`,
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
