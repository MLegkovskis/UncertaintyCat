import { describe, expect, it } from "vitest";

import {
  generationFailure,
  generationLeaseIsActive,
  LOW_LATENCY_AI_SETTINGS,
  MODEL_UNDERSTANDING_LEASE_MS,
} from "./ai-config";

describe("Workers AI runtime policy", () => {
  it("disables unnecessary hidden reasoning for latency-sensitive explanations", () => {
    expect(LOW_LATENCY_AI_SETTINGS).toEqual({
      reasoning_effort: null,
      chat_template_kwargs: { enable_thinking: false },
    });
  });

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
});
