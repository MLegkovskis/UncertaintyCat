import { getSandbox } from "@cloudflare/sandbox";

import type { Env } from "./env";

type ComputeOperation = "catalog" | "validate" | "execute";

interface SandboxEnvelope {
  status: number;
  body: unknown;
}

function operationFor(path: string): ComputeOperation {
  if (path.endsWith("/catalog")) return "catalog";
  if (path.endsWith("/validate")) return "validate";
  if (path.endsWith("/execute")) return "execute";
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
): Promise<Response> {
  if (!env.SANDBOX)
    throw new Error("Cloudflare Sandbox binding is unavailable.");
  const operation = operationFor(path);
  const runId = operation === "execute" ? executionRunId(init) : null;
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
    const result = await sandbox.exec(command, { timeout: 180_000 });
    if (!result.success) {
      console.error(
        JSON.stringify({
          event: "isolated_compute_failed",
          operation,
          exitCode: result.exitCode,
          stderr: result.stderr.slice(0, 2_000),
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
): Promise<Response> {
  if (env.SANDBOX) return sandboxFetch(env, path, init);
  if (!env.COMPUTE_SERVICE_URL)
    throw new Error("No compute backend is configured.");
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
