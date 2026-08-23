export const MODEL_UNDERSTANDING_PROMPT_VERSION = "1.2.0";
export const MODEL_UNDERSTANDING_TIMEOUT_MS = 15_000;
export const REPORT_CHAT_TIMEOUT_MS = 45_000;
export const MODEL_UNDERSTANDING_LEASE_MS = 30_000;

export const LOW_LATENCY_AI_SETTINGS = {
  reasoning_effort: null,
  chat_template_kwargs: {
    enable_thinking: false,
  },
} as const;

export function generationLeaseIsActive(
  status: string | undefined,
  updatedAt: string | undefined,
  currentTimeMs = Date.now(),
): boolean {
  if (status !== "generating" || !updatedAt) return false;
  const updatedAtMs = Date.parse(updatedAt);
  return (
    Number.isFinite(updatedAtMs) &&
    currentTimeMs - updatedAtMs < MODEL_UNDERSTANDING_LEASE_MS
  );
}

export function generationFailure(error: unknown) {
  const raw = error instanceof Error ? error.message : String(error);
  const timedOut = /abort|deadline|timed?\s*out|timeout/i.test(raw);
  return {
    code: timedOut ? "model_understanding_timeout" : "model_understanding_failed",
    message: timedOut
      ? "Workers AI did not answer within 15 seconds. Please retry; failed requests are not charged."
      : "Workers AI could not create the explanation. Please retry; failed requests are not charged.",
    diagnostic: raw.slice(0, 2_000),
    status: timedOut ? 504 : 502,
  } as const;
}
