import { getSandbox } from "@cloudflare/sandbox";

import { parseProgressLine } from "./compute-progress";
import type { Env } from "./env";

type ComputeOperation =
  | "catalog"
  | "validate"
  | "execute"
  | "inspect-data"
  | "fit-data"
  | "fit-data-surrogate"
  | "serialize-surrogate"
  | "execute-surrogate";

interface SandboxEnvelope {
  status: number;
  body: unknown;
}

export interface ComputeProgress {
  phase: string;
  percent: number;
  message: string;
  indeterminate: boolean;
}

interface ComputeFetchOptions {
  onProgress?: (progress: ComputeProgress) => void;
}

function operationFor(path: string): ComputeOperation {
  if (path.endsWith("/catalog")) return "catalog";
  if (path.endsWith("/validate")) return "validate";
  if (path.endsWith("/execute")) return "execute";
  if (path.endsWith("/data/inspect")) return "inspect-data";
  if (path.endsWith("/data/fit")) return "fit-data";
  if (path.endsWith("/data/surrogate")) return "fit-data-surrogate";
  if (path.endsWith("/surrogates/serialize")) return "serialize-surrogate";
  if (path.endsWith("/surrogates/execute")) return "execute-surrogate";
  throw new Error(`Unsupported compute path: ${path}`);
}

function executionRunId(init?: RequestInit): string | null {
  if (!init?.body) return null;
  try {
    const runId = (JSON.parse(String(init.body)) as { run_id?: unknown })
      .run_id;
    return typeof runId === "string" && /^[0-9a-f-]{36}$/i.test(runId)
      ? runId
      : null;
  } catch {
    return null;
  }
}

async function sandboxFetch(
  env: Env,
  path: string,
  init?: RequestInit,
  options: ComputeFetchOptions = {},
): Promise<Response> {
  if (!env.SANDBOX)
    throw new Error("Cloudflare Sandbox binding is unavailable.");
  const operation = operationFor(path);
  const runId = ["execute", "execute-surrogate"].includes(operation)
    ? executionRunId(init)
    : null;
  const sandbox = getSandbox(
    env.SANDBOX,
    runId ? `uq-run-${runId}` : `uq-${crypto.randomUUID()}`,
    {
      enableDefaultSession: false,
    },
  );
  const inputPath = `/workspace/request-${crypto.randomUUID()}.json`;
  let completed = false;
  try {
    if (init?.body) await sandbox.writeFile(inputPath, String(init.body));
    const command =
      `cd /app && /app/.venv/bin/python -m services.compute.cli ${operation}` +
      (init?.body ? ` ${inputPath}` : "");
    let stderrBuffer = "";
    const result = await sandbox.exec(command, {
      timeout: 180_000,
      stream: Boolean(options.onProgress),
      onOutput: (stream, data) => {
        if (stream !== "stderr" || !options.onProgress) return;
        stderrBuffer += data;
        const lines = stderrBuffer.split("\n");
        stderrBuffer = lines.pop() ?? "";
        for (const line of lines) {
          const progress = parseProgressLine(line.trim());
          if (progress) options.onProgress(progress);
        }
      },
    });
    if (stderrBuffer && options.onProgress) {
      const progress = parseProgressLine(stderrBuffer.trim());
      if (progress) options.onProgress(progress);
    }
    if (!result.success) {
      console.error(
        JSON.stringify({
          event: "isolated_compute_failed",
          operation,
          exitCode: result.exitCode,
          stderrBytes: new TextEncoder().encode(result.stderr).byteLength,
        }),
      );
      throw new Error(
        `Isolated compute exited with status ${result.exitCode}.`,
      );
    }
    const envelope = JSON.parse(result.stdout) as SandboxEnvelope;
    if (
      !Number.isInteger(envelope.status) ||
      envelope.status < 100 ||
      envelope.status > 599
    ) {
      throw new Error("Isolated compute returned an invalid response status.");
    }
    completed = true;
    return Response.json(envelope.body, { status: envelope.status });
  } finally {
    if (!runId || !completed) await sandbox.destroy().catch(() => undefined);
  }
}

export async function computeFetch(
  env: Env,
  path: string,
  init?: RequestInit,
  options: ComputeFetchOptions = {},
): Promise<Response> {
  if (env.SANDBOX) return sandboxFetch(env, path, init, options);
  if (!env.COMPUTE_SERVICE_URL)
    throw new Error("No compute backend is configured.");
  options.onProgress?.({
    phase: "openturns",
    percent: 20,
    message: "OpenTURNS computation is active.",
    indeterminate: true,
  });
  return fetch(`${env.COMPUTE_SERVICE_URL}${path}`, {
    ...init,
    headers: {
      ...init?.headers,
      ...(env.UNCERTAINTYCAT_INTERNAL_TOKEN
        ? { Authorization: `Bearer ${env.UNCERTAINTYCAT_INTERNAL_TOKEN}` }
        : {}),
    },
  });
}

export async function destroyRunSandbox(
  env: Env,
  runId: string,
): Promise<void> {
  if (!env.SANDBOX) return;
  const sandbox = getSandbox(env.SANDBOX, `uq-run-${runId}`, {
    enableDefaultSession: false,
  });
  await sandbox.destroy().catch(() => undefined);
}
