import {
  Activity,
  BookOpen,
  Cat,
  Cloud,
  FolderKanban,
  Github,
  LogOut,
  Menu,
  X,
} from "lucide-react";
import { useEffect, useState, type PropsWithChildren } from "react";
import { NavLink } from "react-router-dom";

import { authClient } from "../auth-client";

export function Shell({ children }: PropsWithChildren) {
  const [open, setOpen] = useState(false);
  const [accountOpen, setAccountOpen] = useState(false);
  const [providers, setProviders] = useState<Array<"cloudflare">>([]);
  const session = authClient.useSession();
  useEffect(() => {
    void fetch("/api/v1/session", { credentials: "include" })
      .then(
        (response) =>
          response.json() as Promise<{ providers?: Array<"cloudflare"> }>,
      )
      .then((body) => setProviders(body.providers ?? []))
      .catch(() => undefined);
  }, []);
  return (
    <div className="app-shell">
      <aside className={`sidebar ${open ? "sidebar-open" : ""}`}>
        <div className="brand">
          <span className="brand-mark">
            <Cat size={22} />
          </span>
          <span>UncertaintyCat</span>
          <button
            className="icon-button mobile-close"
            onClick={() => setOpen(false)}
            aria-label="Close navigation"
          >
            <X />
          </button>
        </div>
        <nav aria-label="Primary navigation">
          <NavLink to="/" end>
            <BookOpen size={18} /> Overview
          </NavLink>
          <NavLink to="/workspace">
            <FolderKanban size={18} /> Workspace
          </NavLink>
          <NavLink to="/activity">
            <Activity size={18} /> Activity
          </NavLink>
        </nav>
        <div className="sidebar-footer">
          <div className="runtime-pill">
            <span /> OpenTURNS engine
          </div>
          <a
            href="https://github.com/MLegkovskis/UncertaintyCat"
            target="_blank"
            rel="noreferrer"
          >
            <Github size={16} /> Source
          </a>
          <small>
            Numerical results are computed deterministically. AI narrative is
            labelled separately.
          </small>
        </div>
      </aside>
      <div className="main-column">
        <header className="topbar">
          <button
            className="icon-button mobile-menu"
            onClick={() => setOpen(true)}
            aria-label="Open navigation"
          >
            <Menu />
          </button>
          <div className="topbar-context">
            <strong>Scientific workspace</strong>
            <span>Reproducible · inspectable · exportable</span>
          </div>
          <div className="account-menu">
            <button
              className="account-chip"
              onClick={() => setAccountOpen((value) => !value)}
            >
              {session.data?.user.name ?? "Guest workspace"}{" "}
              <span className="avatar">
                {session.data?.user.name?.slice(0, 2).toUpperCase() ?? "UC"}
              </span>
            </button>
            {accountOpen && (
              <div className="account-popover">
                {session.data?.user ? (
                  <>
                    <strong>{session.data.user.name}</strong>
                    <small>{session.data.user.email}</small>
                    <button onClick={() => authClient.signOut()}>
                      <LogOut /> Sign out
                    </button>
                  </>
                ) : (
                  <>
                    <strong>Keep custom models private</strong>
                    <small>
                      Sign in to keep projects, executions, reports, and report
                      conversations across devices.
                    </small>
                    {providers.includes("cloudflare") && (
                      <button
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
                    {providers.length === 0 && (
                      <small>
                        Sign-in is awaiting production identity configuration.
                      </small>
                    )}
                  </>
                )}
              </div>
            )}
          </div>
        </header>
        <main>{children}</main>
      </div>
      {open && (
        <button
          className="scrim"
          onClick={() => setOpen(false)}
          aria-label="Close navigation overlay"
        />
      )}
    </div>
  );
}
