import { createGroq } from "@ai-sdk/groq";
import { createWorkersAI } from "workers-ai-provider";

import type { Env } from "./env";

export type AiProviderName = "groq" | "cloudflare";
export type AiPurpose = "modelUnderstanding" | "reportChat";

interface AiModelDefinition {
  modelId: string;
  label: string;
}

interface AiProviderDefinition {
  modelUnderstanding: AiModelDefinition & {
    fallbackModelId: string;
    reviewerModelId: string;
  };
  reportChat: AiModelDefinition;
}

export const DEFAULT_AI_PROVIDER: AiProviderName = "groq";

export const AI_PROVIDER_DEFINITIONS: Record<
  AiProviderName,
  AiProviderDefinition
> = {
  groq: {
    modelUnderstanding: {
      modelId: "openai/gpt-oss-20b",
      fallbackModelId: "openai/gpt-oss-120b",
      reviewerModelId: "openai/gpt-oss-120b",
      label: "Groq · GPT-OSS 20B + 120B equation review",
    },
    reportChat: {
      modelId: "openai/gpt-oss-120b",
      label: "Groq · GPT-OSS 120B",
    },
  },
  cloudflare: {
    modelUnderstanding: {
      modelId: "@cf/meta/llama-3.2-3b-instruct",
      fallbackModelId: "@cf/meta/llama-3.2-1b-instruct",
      reviewerModelId: "@cf/meta/llama-3.2-3b-instruct",
      label: "Cloudflare Workers AI · Llama 3.2 3B with equation review",
    },
    reportChat: {
      modelId: "@cf/zai-org/glm-4.7-flash",
      label: "Cloudflare Workers AI · GLM-4.7-Flash",
    },
  },
};

export function resolveAiProvider(value: string | undefined): AiProviderName {
  const normalized = value?.trim().toLowerCase();
  if (!normalized) return DEFAULT_AI_PROVIDER;
  if (normalized === "groq" || normalized === "cloudflare") return normalized;
  throw new Error(`Unsupported AI_PROVIDER value: ${value}`);
}

export function aiRuntime(env: Env) {
  const provider = resolveAiProvider(env.AI_PROVIDER);
  const models = AI_PROVIDER_DEFINITIONS[provider];
  const configured =
    provider === "groq" ? Boolean(env.GROQ_API_KEY?.trim()) : Boolean(env.AI);
  return { provider, models, configured };
}

export function localGroqBaseUrl(env: Env): string | undefined {
  const configured = env.GROQ_BASE_URL?.trim();
  if (!configured) return undefined;
  if (env.DEV_AUTH_BYPASS !== "true") {
    throw new Error("GROQ_BASE_URL is restricted to local development.");
  }

  let parsed: URL;
  try {
    parsed = new URL(configured);
  } catch {
    throw new Error("GROQ_BASE_URL must be a valid loopback URL.");
  }
  if (
    parsed.protocol !== "http:" ||
    !["127.0.0.1", "localhost", "[::1]"].includes(parsed.hostname)
  ) {
    throw new Error("GROQ_BASE_URL must use HTTP on a loopback host.");
  }
  return configured.replace(/\/+$/, "");
}

export function createAiLanguageModel(
  env: Env,
  modelId: string,
  sessionAffinity: string,
  purpose: AiPurpose,
) {
  const provider = resolveAiProvider(env.AI_PROVIDER);
  if (provider === "groq") {
    if (!env.GROQ_API_KEY?.trim())
      throw new Error("GROQ_API_KEY is not configured.");
    const baseURL = localGroqBaseUrl(env);
    return createGroq({
      apiKey: env.GROQ_API_KEY,
      ...(baseURL ? { baseURL } : {}),
    })(modelId);
  }
  if (!env.AI) throw new Error("Cloudflare Workers AI is not configured.");
  return createWorkersAI({ binding: env.AI })(modelId, {
    sessionAffinity,
    ...(purpose === "reportChat"
      ? {
          reasoning_effort: null,
          chat_template_kwargs: { enable_thinking: false },
        }
      : {}),
  });
}

export function aiProviderOptions(
  provider: AiProviderName,
  purpose: AiPurpose,
) {
  if (provider === "groq") {
    return {
      groq: {
        reasoningEffort: "low",
        parallelToolCalls: false,
        structuredOutputs: true,
        strictJsonSchema: true,
      },
    } as const;
  }
  if (purpose === "reportChat") return undefined;
  return undefined;
}

export function modelUnderstandingCacheVersion(env: Env, promptVersion: string) {
  const runtime = aiRuntime(env);
  return `${promptVersion}:${runtime.provider}:${runtime.models.modelUnderstanding.modelId}:${runtime.models.modelUnderstanding.reviewerModelId}`;
}
