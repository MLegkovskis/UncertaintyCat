import { useQuery } from "@tanstack/react-query";
import { Cloud, ShieldCheck } from "lucide-react";
import type { PropsWithChildren } from "react";

import { api } from "../api";
import { useCloudflareSignIn } from "../auth-client";

export function AuthenticatedRoute({ children }: PropsWithChildren) {
  const session = useQuery({ queryKey: ["session-policy"], queryFn: api.session });
  const { beginSignIn, isSigningIn, signInError } = useCloudflareSignIn();

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
            onClick={() => void beginSignIn(`${window.location.origin}/`)}
            disabled={isSigningIn}
          >
            <Cloud /> {isSigningIn ? "Connecting to Cloudflare…" : "Continue with Cloudflare"}
          </button>
        )}
        {signInError && <p className="error-copy" role="alert">{signInError}</p>}
      </section>
    </div>
  );
}
