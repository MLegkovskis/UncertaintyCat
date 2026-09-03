import {
  BookOpen,
  Cat,
  Cloud,
  FolderKanban,
  GitBranch,
  LogOut,
  Menu,
  Moon,
  Gauge,
  Sun,
  User,
  X,
} from "lucide-react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState, type PropsWithChildren } from "react";
import { Link, NavLink, useLocation, useNavigate } from "react-router-dom";

import { api } from "../api";
import { authClient } from "../auth-client";
import { useTheme } from "./Theme";

interface IdentityClaims {
  authenticated: boolean;
  name?: string | null;
  email?: string | null;
}

export function formatIdentity(claims?: IdentityClaims | null) {
  if (!claims?.authenticated) {
    return { initials: "UC", label: "Sign in", fallbackIcon: false };
  }
  const name = claims.name?.trim() ?? "";
  const email = claims.email?.trim() ?? "";
  if (name) {
    const words = name.split(/\s+/).filter(Boolean);
    const first = words[0] ?? "";
    const initials =
      words.length === 1
        ? first.slice(0, 2)
        : `${first[0] ?? ""}${words.at(-1)?.[0] ?? ""}`;
    return {
      initials: initials.toUpperCase(),
      label: name,
      fallbackIcon: false,
    };
  }
  if (email) {
    const local = (email.split("@", 1)[0] ?? "").replace(/[^a-z0-9]/gi, "");
    if (local) {
      return {
        initials: local.slice(0, 2).toUpperCase(),
        label: email,
        fallbackIcon: false,
      };
    }
  }
  return { initials: "", label: "Account", fallbackIcon: true };
}

export function Shell({ children }: PropsWithChildren) {
  const [open, setOpen] = useState(false);
  const [accountOpen, setAccountOpen] = useState(false);
  const [signingOut, setSigningOut] = useState(false);
  const accountMenu = useRef<HTMLDivElement>(null);
  const accountButton = useRef<HTMLButtonElement>(null);
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const location = useLocation();
  const session = useQuery({
    queryKey: ["session-policy"],
    queryFn: api.session,
  });
  const { theme, setTheme } = useTheme();
  useEffect(() => {
    if (!accountOpen) return;
    const dismiss = (event: KeyboardEvent | PointerEvent) => {
      if (event instanceof KeyboardEvent && event.key === "Escape") {
        setAccountOpen(false);
        accountButton.current?.focus();
        return;
      }
      if (
        event instanceof PointerEvent &&
        !accountMenu.current?.contains(event.target as Node)
      ) {
        setAccountOpen(false);
      }
    };
    document.addEventListener("keydown", dismiss);
    document.addEventListener("pointerdown", dismiss);
    return () => {
      document.removeEventListener("keydown", dismiss);
      document.removeEventListener("pointerdown", dismiss);
    };
  }, [accountOpen]);

  const claims: IdentityClaims = signingOut
    ? { authenticated: false }
    : (session.data?.identity ?? { authenticated: false });
  const formatted = formatIdentity(claims);
  const signedIn = claims.authenticated;
  const sessionLoading = session.isPending;
  const providers = session.data?.providers ?? [];
  const context = !signedIn
    ? ["Uncertainty quantification", "OpenTURNS, made interactive"]
    : location.pathname === "/studies"
      ? ["Projects", "Models, studies, and retained results"]
      : location.pathname === "/operator"
        ? ["Operations", "Application health and execution telemetry"]
        : location.pathname.startsWith("/studies/")
          ? ["Project workspace", "Model, methods, and numerical evidence"]
          : ["Scientific report", "Reproducible numerical evidence"];

  const handleSignOut = async () => {
    setSigningOut(true);
    setAccountOpen(false);
    try {
      await authClient.signOut();
      queryClient.setQueryData(["session-policy"], {
        identity: {
          ownerId: "anonymous",
          authenticated: false,
          operator: false,
        },
        providers,
      });
      queryClient.removeQueries({
        predicate: (query) => query.queryKey[0] !== "session-policy",
      });
      navigate("/", { replace: true });
      await queryClient.invalidateQueries({ queryKey: ["session-policy"] });
    } finally {
      setSigningOut(false);
    }
  };
  return (
    <div className="app-shell">
      <aside className={`sidebar ${open ? "sidebar-open" : ""}`}>
        <div className="brand-row">
          <Link
            className="brand"
            to="/"
            onClick={() => setOpen(false)}
            aria-label="UncertaintyCat home"
          >
            <span className="brand-mark">
              <Cat size={22} />
            </span>
            <span>UncertaintyCat</span>
          </Link>
          <button
            className="icon-button mobile-close"
            onClick={() => setOpen(false)}
            aria-label="Close navigation"
          >
            <X />
          </button>
        </div>
        <nav aria-label="Primary navigation">
          {signedIn ? (
            <>
              <NavLink to="/studies">
                <FolderKanban size={18} /> Projects
              </NavLink>
              {session.data?.identity.operator && (
                <NavLink to="/operator">
                  <Gauge size={18} /> Operations
                </NavLink>
              )}
            </>
          ) : (
            <NavLink to="/" end>
              <BookOpen size={18} /> Overview
            </NavLink>
          )}
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
            <GitBranch size={16} /> Source
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
            <strong>{context[0]}</strong>
            <span>{context[1]}</span>
          </div>
          <button
            className="icon-button theme-toggle"
            onClick={() => setTheme(theme === "light" ? "dark" : "light")}
            aria-label={`Switch to ${theme === "light" ? "dark" : "light"} theme`}
            title={`Switch to ${theme === "light" ? "dark" : "light"} theme`}
          >
            {theme === "light" ? <Moon /> : <Sun />}
          </button>
          <div className="account-menu" ref={accountMenu}>
            <button
              ref={accountButton}
              className="account-chip"
              onClick={() => setAccountOpen((value) => !value)}
              aria-expanded={accountOpen}
              aria-controls="account-popover"
              aria-haspopup="menu"
              aria-busy={sessionLoading}
              aria-label={
                sessionLoading
                  ? "Checking session"
                  : `${signedIn ? "Signed in" : "Not signed in"} ${formatted.label}`
              }
            >
              <span className="account-copy">
                <small>
                  {sessionLoading
                    ? "Checking session"
                    : signedIn
                      ? "Signed in"
                      : "Not signed in"}
                </small>
                <strong>{sessionLoading ? "Loading…" : formatted.label}</strong>
              </span>
              <span className="avatar">
                {formatted.fallbackIcon ? (
                  <User aria-hidden="true" />
                ) : (
                  formatted.initials
                )}
              </span>
            </button>
            {accountOpen && (
              <div className="account-popover" id="account-popover" role="menu">
                {signedIn ? (
                  <>
                    <strong>{formatted.label}</strong>
                    {claims.name?.trim() && claims.email?.trim() && (
                      <small>{claims.email.trim()}</small>
                    )}
                    <button
                      role="menuitem"
                      onClick={() => void handleSignOut()}
                      disabled={signingOut}
                    >
                      <LogOut /> {signingOut ? "Signing out…" : "Sign out"}
                    </button>
                  </>
                ) : (
                  <>
                    <strong>Sign in to use the workspace</strong>
                    <small>
                      Authentication is required for every model, dataset,
                      execution, report, share link, and AI conversation.
                    </small>
                    {providers.includes("cloudflare") && (
                      <button
                        role="menuitem"
                        onClick={() =>
                          authClient.signIn.social({
                            provider: "cloudflare",
                            callbackURL: `${window.location.origin}/`,
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
