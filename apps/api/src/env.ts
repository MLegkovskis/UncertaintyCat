import type { Sandbox } from "@cloudflare/sandbox";

export interface Env {
  DB: D1Database;
  ARTIFACTS: R2Bucket;
  RUN_QUEUE: Queue<RunTaskMessage>;
  AI?: Ai;
  ASSETS?: Fetcher;
  SANDBOX?: DurableObjectNamespace<Sandbox>;
  COMPUTE_SERVICE_URL?: string;
  PUBLIC_WEB_ORIGIN?: string;
  UNCERTAINTYCAT_INTERNAL_TOKEN?: string;
  BETTER_AUTH_SECRET?: string;
  BETTER_AUTH_URL: string;
  DEV_AUTH_BYPASS?: string;
  CLOUDFLARE_ACCESS_CLIENT_ID?: string;
  CLOUDFLARE_ACCESS_CLIENT_SECRET?: string;
  CLOUDFLARE_ACCESS_ISSUER?: string;
}

export interface RunTaskMessage {
  taskId: string;
  runId: string;
  attempt: number;
}

export interface Identity {
  ownerId: string;
  authenticated: boolean;
  name?: string;
  email?: string;
}
