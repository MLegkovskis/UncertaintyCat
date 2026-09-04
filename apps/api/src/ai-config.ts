export const MODEL_UNDERSTANDING_PROMPT_VERSION = "2.1.0";
export const MODEL_UNDERSTANDING_PRIMARY_TIMEOUT_MS = 12_000;
export const MODEL_UNDERSTANDING_FALLBACK_TIMEOUT_MS = 15_000;
export const MODEL_UNDERSTANDING_REVIEW_TIMEOUT_MS = 15_000;
export const REPORT_CHAT_TIMEOUT_MS = 45_000;
export const MODEL_UNDERSTANDING_LEASE_MS = 60_000;

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
  const statusCode =
    typeof error === "object" &&
    error !== null &&
    "statusCode" in error &&
    typeof error.statusCode === "number"
      ? error.statusCode
      : undefined;
  const timedOut = /abort|deadline|timed?\s*out|timeout/i.test(raw);
  const rateLimited = statusCode === 429;
  const authenticationFailed = statusCode === 401 || statusCode === 403;
  const invalidRequest = statusCode === 400 || statusCode === 422;
  const invalidResponse =
    /no object generated|schema|invalid brief|invalid model understanding/i.test(
      raw,
    );
  const diagnostic = timedOut
    ? "upstream_timeout"
    : rateLimited
      ? "upstream_rate_limited"
      : authenticationFailed
        ? "upstream_authentication_failed"
        : invalidRequest
          ? "upstream_invalid_request"
          : invalidResponse
            ? "upstream_response_invalid"
            : statusCode !== undefined && statusCode >= 500
              ? "upstream_server_error"
              : "upstream_generation_failed";
  return {
    code: timedOut
      ? "model_understanding_timeout"
      : rateLimited
        ? "model_understanding_rate_limited"
        : "model_understanding_failed",
    message: timedOut
      ? "The model explanation did not finish in time. Please retry."
      : rateLimited
        ? "The model explanation service is temporarily busy. Please retry shortly."
        : "The model explanation could not be completed. Please retry.",
    diagnostic,
    providerStatusCode: statusCode,
    status: timedOut ? 504 : rateLimited ? 503 : 502,
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
