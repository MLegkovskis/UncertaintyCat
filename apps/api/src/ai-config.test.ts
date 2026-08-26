import { describe, expect, it } from "vitest";

import {
  generationFailure,
  generationLeaseIsActive,
  MODEL_UNDERSTANDING_LEASE_MS,
  runSequentialFallback,
} from "./ai-config";

describe("AI runtime policy", () => {
  it("treats only a recent generating row as an active single-flight lease", () => {
    const current = Date.parse("2026-08-23T20:00:00.000Z");
    expect(
      generationLeaseIsActive(
        "generating",
        new Date(current - MODEL_UNDERSTANDING_LEASE_MS + 1).toISOString(),
        current,
      ),
    ).toBe(true);
    expect(
      generationLeaseIsActive(
        "generating",
        new Date(current - MODEL_UNDERSTANDING_LEASE_MS).toISOString(),
        current,
      ),
    ).toBe(false);
    expect(
      generationLeaseIsActive("succeeded", new Date(current).toISOString(), current),
    ).toBe(false);
    expect(generationLeaseIsActive("generating", "not-a-date", current)).toBe(false);
  });

  it("returns a user-safe timeout while retaining a bounded diagnostic", () => {
    const failure = generationFailure(new Error("request timeout at upstream"));
    expect(failure).toMatchObject({
      code: "model_understanding_timeout",
      status: 504,
    });
    expect(failure.message).toContain("failed requests are not charged");
    expect(failure.diagnostic).toBe("request timeout at upstream");
  });

  it("uses the first successful fallback attempt", async () => {
    const attempts: string[] = [];
    const generation = await runSequentialFallback(
      ["primary", "fallback"],
      async (attempt) => {
        attempts.push(attempt);
        if (attempt === "primary") throw new Error("primary timeout");
        return "grounded explanation";
      },
    );
    expect(attempts).toEqual(["primary", "fallback"]);
    expect(generation).toEqual({
      result: "grounded explanation",
      attempt: "fallback",
      index: 1,
    });
  });

  it("returns the final error when every configured model fails", async () => {
    await expect(
      runSequentialFallback(["primary", "fallback"], async (attempt) => {
        throw new Error(`${attempt} failed`);
      }),
    ).rejects.toThrow("fallback failed");
  });
});
