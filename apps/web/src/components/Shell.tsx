import {
  BookOpen,
  Cat,
  Cloud,
  Database,
  FolderKanban,
  Github,
  LogOut,
  Menu,
  Moon,
  PlusCircle,
  ScanSearch,
  Sun,
  User,
  Waves,
  X,
} from "lucide-react";
import {
  useEffect,
  useRef,
  useState,
  type PropsWithChildren,
} from "react";
import { NavLink } from "react-router-dom";

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
    return { initials: initials.toUpperCase(), label: name, fallbackIcon: false };
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
  const [providers, setProviders] = useState<Array<"cloudflare">>([]);
  const [identity, setIdentity] = useState<IdentityClaims | null>(null);
  const [policyLoading, setPolicyLoading] = useState(true);
  const accountMenu = useRef<HTMLDivElement>(null);
  const accountButton = useRef<HTMLButtonElement>(null);
  const session = authClient.useSession();
  const { theme, setTheme } = useTheme();
  useEffect(() => {
    void fetch("/api/v1/session", { credentials: "include" })
      .then(
        (response) =>
          response.json() as Promise<{
            identity?: IdentityClaims;
            providers?: Array<"cloudflare">;
          }>,
      )
      .then((body) => {
        setProviders(body.providers ?? []);
        setIdentity(body.identity ?? null);
      })
      .catch(() => undefined)
      .finally(() => setPolicyLoading(false));
  }, []);
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

  const sessionUser = session.data?.user;
  const claims: IdentityClaims = sessionUser
    ? {
        authenticated: true,
        name: sessionUser.name,
        email: sessionUser.email,
      }
    : (identity ?? { authenticated: false });
  const formatted = formatIdentity(claims);
  const signedIn = claims.authenticated;
  const sessionLoading = session.isPending || policyLoading;
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
            <BookOpen size={18} /> {signedIn ? "Dashboard" : "Overview"}
          </NavLink>
          {signedIn && (
            <>
              <NavLink to="/new-analysis">
                <PlusCircle size={18} /> New analysis
              </NavLink>
              <NavLink to="/studies">
                <FolderKanban size={18} /> Projects
              </NavLink>
              <NavLink to="/dimension-reduction">
                <ScanSearch size={18} /> Dimension reduction
              </NavLink>
              <NavLink to="/surrogates">
                <Waves size={18} /> Surrogate Studio
              </NavLink>
              <NavLink to="/data-lab">
                <Database size={18} /> Distribution fitting
              </NavLink>
            </>
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
                <small>{sessionLoading ? "Checking session" : signedIn ? "Signed in" : "Not signed in"}</small>
                <strong>{sessionLoading ? "Loading…" : formatted.label}</strong>
              </span>
              <span className="avatar">
                {formatted.fallbackIcon ? <User aria-hidden="true" /> : formatted.initials}
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
                    <button role="menuitem" onClick={() => authClient.signOut()}>
                      <LogOut /> Sign out
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
