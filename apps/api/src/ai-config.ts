export const MODEL_UNDERSTANDING_PROMPT_VERSION = "1.4.0";
export const MODEL_UNDERSTANDING_PRIMARY_TIMEOUT_MS = 12_000;
export const MODEL_UNDERSTANDING_FALLBACK_TIMEOUT_MS = 15_000;
export const REPORT_CHAT_TIMEOUT_MS = 45_000;
export const MODEL_UNDERSTANDING_LEASE_MS = 30_000;

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
      ? "The AI provider did not answer in time. Please retry; failed requests are not charged."
      : "The AI provider could not create the explanation. Please retry; failed requests are not charged.",
    diagnostic: raw.slice(0, 2_000),
    status: timedOut ? 504 : 502,
  } as const;
}

export async function runSequentialFallback<T, TAttempt>(
  attempts: readonly TAttempt[],
  operation: (attempt: TAttempt, index: number) => Promise<T>,
): Promise<{ result: T; attempt: TAttempt; index: number }> {
  if (attempts.length === 0) throw new Error("No AI generation attempts configured.");
  let lastError: unknown;
  for (let index = 0; index < attempts.length; index += 1) {
    const attempt = attempts[index]!;
    try {
      return { result: await operation(attempt, index), attempt, index };
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError;
}
