import { useQuery } from "@tanstack/react-query";
import { Cloud, ShieldCheck } from "lucide-react";
import type { PropsWithChildren } from "react";

import { api } from "../api";
import { authClient } from "../auth-client";

export function AuthenticatedRoute({ children }: PropsWithChildren) {
  const session = useQuery({ queryKey: ["session-policy"], queryFn: api.session });

  if (session.isPending) {
    return <div className="route-loading">Checking your secure session…</div>;
  }

  if (session.data?.identity.authenticated) {
    return children;
  }

  return (
    <div className="page auth-required-page">
      <section className="auth-required-card">
        <span className="catalog-icon"><ShieldCheck /></span>
        <span className="section-kicker">Authentication required</span>
        <h1>Sign in before starting an analysis.</h1>
        <p>
          Models, datasets, executions, reports, shared evidence, and AI
          conversations are private to an authenticated UncertaintyCat account.
        </p>
        {session.isError ? (
          <p className="error-copy">The session service is unavailable. Please try again.</p>
        ) : (
          <button
            className="button primary"
            type="button"
            onClick={() =>
              authClient.signIn.social({
                provider: "cloudflare",
                callbackURL: window.location.href,
              })
            }
          >
            <Cloud /> Continue with Cloudflare
          </button>
        )}
      </section>
    </div>
  );
}
