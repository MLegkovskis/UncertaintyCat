import { expect, test } from "@playwright/test";
import { installMockApi, project } from "./fixtures";

test("a failed sign-out retains account context and offers a working retry", async ({ page }) => {
  await installMockApi(page, { authenticated: true, projects: [project] });
  let signedOut = false;
  let fail = true;
  await page.route("**/api/v1/session", (route) => route.fulfill({ json: {
    identity: signedOut ? { authenticated: false, ownerId: "" } : { authenticated: true, ownerId: "test-owner", name: "Audit User" },
    providers: ["cloudflare"],
  } }));
  await page.route("**/api/auth/sign-out", async (route) => {
    if (fail) await route.fulfill({ status: 503, json: { message: "Unavailable" } });
    else { signedOut = true; await route.fulfill({ json: { success: true } }); }
  });
  await page.goto("/studies");
  await page.getByRole("button", { name: "Signed in Audit User" }).click();
  await page.getByRole("menuitem", { name: "Sign out" }).click();
  await expect(page.getByRole("alert")).toContainText("You are still signed in");
  await expect(page).toHaveURL(/\/studies$/);
  await expect(page.getByRole("link", { name: "Projects", exact: true })).toBeVisible();
  fail = false;
  await page.getByRole("menuitem", { name: "Sign out" }).click();
  await expect(page.getByRole("button", { name: "Not signed in Sign in" })).toBeVisible();
  await expect(page).toHaveURL(/\/$/);
});

test("session failure keeps private content locked and can recover in place", async ({ page }) => {
  await installMockApi(page, { authenticated: true, projects: [project] });
  let fail = true;
  const privateReads: string[] = [];
  page.on("request", (request) => {
    if (/\/api\/v1\/projects/.test(request.url())) privateReads.push(request.url());
  });
  await page.route("**/api/v1/session", (route) => fail
    ? route.fulfill({ status: 503, json: { error: { message: "Unavailable" } } })
    : route.fallback());
  await page.goto("/studies");
  await expect(page.getByRole("alert")).toContainText("workspace stays locked");
  expect(privateReads).toEqual([]);
  fail = false;
  await page.getByRole("button", { name: "Retry session check" }).click();
  await expect(page.getByRole("heading", { name: "Your projects." })).toBeVisible();
});
