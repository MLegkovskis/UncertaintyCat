import { describe, expect, it } from "vitest";

import {
  aiProviderOptions,
  aiRuntime,
  AI_PROVIDER_DEFINITIONS,
  localGroqBaseUrl,
  modelUnderstandingCacheVersion,
  resolveAiProvider,
} from "./ai-provider";
import type { Env } from "./env";

function env(overrides: Partial<Env> = {}): Env {
  return overrides as Env;
}

describe("deployment-selectable AI provider", () => {
  it("defaults to Groq and uses current GPT-OSS production models", () => {
    expect(resolveAiProvider(undefined)).toBe("groq");
    expect(AI_PROVIDER_DEFINITIONS.groq.modelUnderstanding.modelId).toBe(
      "openai/gpt-oss-20b",
    );
    expect(AI_PROVIDER_DEFINITIONS.groq.modelUnderstanding.reviewerModelId).toBe(
      "openai/gpt-oss-120b",
    );
    expect(AI_PROVIDER_DEFINITIONS.groq.reportChat.modelId).toBe(
      "openai/gpt-oss-120b",
    );
  });

  it("requires only the selected provider's credential or binding", () => {
    expect(aiRuntime(env({ GROQ_API_KEY: "gsk_test" })).configured).toBe(true);
    expect(aiRuntime(env()).configured).toBe(false);
    expect(
      aiRuntime(env({ AI_PROVIDER: "cloudflare", AI: {} as Ai })).configured,
    ).toBe(true);
  });

  it("rejects unknown deployment configuration", () => {
    expect(() => resolveAiProvider("other-provider")).toThrow(
      "Unsupported AI_PROVIDER",
    );
  });

  it("keeps cached explanations separate across providers and models", () => {
    expect(
      modelUnderstandingCacheVersion(
        env({ AI_PROVIDER: "groq", GROQ_API_KEY: "gsk_test" }),
        "1.4.0",
      ),
    ).toBe(
      "1.4.0:groq:openai/gpt-oss-20b:openai/gpt-oss-120b",
    );
    expect(
      modelUnderstandingCacheVersion(
        env({ AI_PROVIDER: "cloudflare", AI: {} as Ai }),
        "1.4.0",
      ),
    ).toContain("1.4.0:cloudflare:@cf/");
  });

  it("disables parallel tool calls and uses low reasoning on Groq", () => {
    expect(aiProviderOptions("groq", "reportChat")).toEqual({
      groq: {
        reasoningEffort: "low",
        parallelToolCalls: false,
        structuredOutputs: true,
        strictJsonSchema: true,
      },
    });
    expect(aiProviderOptions("cloudflare", "reportChat")).toBeUndefined();
  });

  it("allows a Groq-compatible fixture only on loopback in local bypass mode", () => {
    expect(
      localGroqBaseUrl(
        env({
          DEV_AUTH_BYPASS: "true",
          GROQ_BASE_URL: "http://127.0.0.1:8790/openai/v1/",
        }),
      ),
    ).toBe("http://127.0.0.1:8790/openai/v1");
    expect(() =>
      localGroqBaseUrl(
        env({ GROQ_BASE_URL: "http://127.0.0.1:8790/openai/v1" }),
      ),
    ).toThrow("restricted to local development");
    expect(() =>
      localGroqBaseUrl(
        env({
          DEV_AUTH_BYPASS: "true",
          GROQ_BASE_URL: "https://example.com/openai/v1",
        }),
      ),
    ).toThrow("HTTP on a loopback host");
  });
});
