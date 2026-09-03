import { useQuery } from "@tanstack/react-query";
import { ShieldAlert } from "lucide-react";
import type { PropsWithChildren } from "react";

import { api } from "../api";

export function OperatorRoute({ children }: PropsWithChildren) {
  const session = useQuery({
    queryKey: ["session-policy"],
    queryFn: api.session,
  });

  if (session.isPending) {
    return <div className="route-loading">Verifying operator access…</div>;
  }
  if (session.data?.identity.operator) return children;

  return (
    <div className="page auth-required-page">
      <section className="auth-required-card">
        <span className="catalog-icon">
          <ShieldAlert />
        </span>
        <span className="section-kicker">Operator access required</span>
        <h1>This view is restricted.</h1>
        <p>
          Operational telemetry contains account and execution metadata. Only
          explicitly configured UncertaintyCat operators can inspect it.
        </p>
      </section>
    </div>
  );
}
