import { drizzleAdapter } from "better-auth/adapters/drizzle";
import { betterAuth } from "better-auth/minimal";
import { genericOAuth } from "better-auth/plugins";
import { drizzle } from "drizzle-orm/d1";
import type { Context } from "hono";

import { authSchema } from "./auth-schema";
import type { Env, Identity } from "./env";

export function createAuth(env: Env) {
  const cloudflareProvider =
    env.CLOUDFLARE_ACCESS_CLIENT_ID &&
    env.CLOUDFLARE_ACCESS_CLIENT_SECRET &&
    env.CLOUDFLARE_ACCESS_ISSUER
      ? genericOAuth({
          config: [
            {
              providerId: "cloudflare",
              name: "Cloudflare",
              discoveryUrl: `${env.CLOUDFLARE_ACCESS_ISSUER.replace(/\/$/, "")}/.well-known/openid-configuration`,
              clientId: env.CLOUDFLARE_ACCESS_CLIENT_ID,
              clientSecret: env.CLOUDFLARE_ACCESS_CLIENT_SECRET,
              scopes: ["openid", "email", "profile"],
              pkce: true,
              requireIdTokenVerification: true,
            },
          ],
        })
      : null;
  return betterAuth({
    database: drizzleAdapter(drizzle(env.DB), {
      provider: "sqlite",
      schema: authSchema,
    }),
    secret:
      env.BETTER_AUTH_SECRET ??
      "local-development-secret-must-not-be-used-in-production",
    baseURL: env.BETTER_AUTH_URL,
    plugins: cloudflareProvider ? [cloudflareProvider] : [],
    trustedOrigins: [
      env.BETTER_AUTH_URL,
      ...(env.PUBLIC_WEB_ORIGIN ? [env.PUBLIC_WEB_ORIGIN] : []),
      "http://127.0.0.1:5173",
      "http://localhost:5173",
    ],
  });
}

export async function identityFor(
  c: Context<{ Bindings: Env; Variables: { requestId: string } }>,
): Promise<Identity> {
  if (c.env.DEV_AUTH_BYPASS === "true") {
    return {
      ownerId: "dev-user",
      authenticated: true,
      name: "Local retained user",
      email: "developer@localhost",
    };
  }
  try {
    const session = await createAuth(c.env).api.getSession({
      headers: c.req.raw.headers,
    });
    if (session?.user) {
      return {
        ownerId: session.user.id,
        authenticated: true,
        name: session.user.name,
        email: session.user.email,
      };
    }
  } catch {
    // Treat invalid and absent sessions identically. Private API middleware
    // rejects both before any application resource is read or mutated.
  }
  return { ownerId: "", authenticated: false };
}
