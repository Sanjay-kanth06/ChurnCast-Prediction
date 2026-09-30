"use client";

import { usePathname, useRouter } from "next/navigation";
import { MODES } from "../../lib/api";

/* Navigation icons.
 *
 * Inline SVG rather than an icon package: the project has exactly three
 * dependencies (next, react, react-dom) and five icons do not justify a fourth.
 * All share one 24-unit grid, 1.75 stroke, round caps and `currentColor`, so
 * they inherit the existing active and hover styling with no extra CSS. */
function Icon({ name }) {
  const common = {
    width: 18,
    height: 18,
    viewBox: "0 0 24 24",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.75,
    strokeLinecap: "round",
    strokeLinejoin: "round",
    "aria-hidden": true,
    focusable: false,
  };
  switch (name) {
    case "dashboard": // layout panels
      return (
        <svg {...common}>
          <rect x="3" y="3" width="7.5" height="8.5" rx="1.5" />
          <rect x="13.5" y="3" width="7.5" height="5" rx="1.5" />
          <rect x="13.5" y="11" width="7.5" height="10" rx="1.5" />
          <rect x="3" y="14.5" width="7.5" height="6.5" rx="1.5" />
        </svg>
      );
    case "predict": // scan-search
      return (
        <svg {...common}>
          <path d="M3 7V5a2 2 0 0 1 2-2h2" />
          <path d="M17 3h2a2 2 0 0 1 2 2v2" />
          <path d="M21 17v2a2 2 0 0 1-2 2h-2" />
          <path d="M7 21H5a2 2 0 0 1-2-2v-2" />
          <circle cx="11.5" cy="11.5" r="3.5" />
          <path d="m14.2 14.2 2.3 2.3" />
        </svg>
      );
    case "subscribers": // users-round
      return (
        <svg {...common}>
          <circle cx="10" cy="8" r="3.5" />
          <path d="M3.5 20a6.5 6.5 0 0 1 13 0" />
          <path d="M16.5 4.6a3.5 3.5 0 0 1 0 6.8" />
          <path d="M18.5 14.4a6.5 6.5 0 0 1 3 5.6" />
        </svg>
      );
    case "model": // brain-circuit
      return (
        <svg {...common}>
          <path d="M12 4.5a2.5 2.5 0 0 0-4.9-.6A2.5 2.5 0 0 0 4.4 7a2.5 2.5 0 0 0-.4 4.4A2.5 2.5 0 0 0 5 16a2.5 2.5 0 0 0 3.5 2.6A2.5 2.5 0 0 0 12 19.5z" />
          <path d="M12 4.5v15" />
          <path d="M16 7.5h1.5a2 2 0 0 1 2 2v.5" />
          <path d="M16 16.5h1.5a2 2 0 0 0 2-2V14" />
          <circle cx="15.5" cy="7.5" r="1.4" />
          <circle cx="15.5" cy="16.5" r="1.4" />
          <circle cx="20.2" cy="12" r="1.4" />
        </svg>
      );
    case "about": // info
      return (
        <svg {...common}>
          <circle cx="12" cy="12" r="9" />
          <path d="M12 11v5" />
          <path d="M12 8h.01" />
        </svg>
      );
    default:
      return null;
  }
}

const NAV = [
  { href: "/", label: "Dashboard", icon: "dashboard" },
  { href: "/predict", label: "Predict", icon: "predict" },
  { href: "/subscribers", label: "Subscribers", icon: "subscribers" },
  { href: "/model", label: "Model", icon: "model" },
  { href: "/about", label: "About", icon: "about" },
];

function isActive(pathname, href) {
  if (href === "/") return pathname === "/";
  return pathname === href || pathname.startsWith(`${href}/`);
}

/* The single data-mode selector. One instance on desktop (sidebar bottom, in
 * the SYSTEM utility area) and one compact instance in the mobile header. Both
 * read and write the same ModeContext state passed down from Shell -- there is
 * no second source of truth. */
function ModeControl({ mode, setMode, compact = false }) {
  return (
    <div className={`mode-control${compact ? " compact" : ""}`}>
      <div className="mode-switch" role="group" aria-label="Select data mode">
        <button
          type="button"
          className={`mode-option${mode === "synthetic" ? " selected" : ""}`}
          onClick={() => setMode("synthetic")}
          aria-pressed={mode === "synthetic"}
        >
          <span className="mode-dot" />
          SYNTHETIC
        </button>
        <button
          type="button"
          className={`mode-option${mode === "real" ? " selected" : ""}`}
          onClick={() => setMode("real")}
          aria-pressed={mode === "real"}
        >
          <span className="mode-dot" />
          REAL
        </button>
      </div>
    </div>
  );
}

export default function Sidebar({ mode, setMode, status }) {
  const pathname = usePathname();
  const router = useRouter();
  const active = status?.[mode] ?? {};

  const items = NAV.map((n) => {
    const on = isActive(pathname, n.href);
    return (
      <div
        key={n.href}
        className={`nav-item${on ? " active" : ""}`}
        onClick={() => router.push(n.href)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            router.push(n.href);
          }
        }}
        role="link"
        tabIndex={0}
        aria-current={on ? "page" : undefined}
      >
        <span className="ico">
          <Icon name={n.icon} />
        </span>
        <span className="label">{n.label}</span>
      </div>
    );
  });

  const population = mode === "real" ? active.subscribers ?? 970960 : 7500;
  const model =
    mode === "real"
      ? active.modelName ?? MODES.real.modelName
      : MODES.synthetic.modelName;

  return (
    <>
      <nav className="sidebar">
        <div className="brand">
          <img
            src="/logo.png"
            alt="ChurnCast — Subscriber Churn Intelligence"
            className="brand-logo"
          />
        </div>

        <div className="nav-container">
          <div className="nav-list">{items}</div>
        </div>

        <div className="sidebar-foot">
          <div className="system-label">SYSTEM</div>
          <div className="system-card">
            <div className="system-row">
              <span
                className={`system-dot ${active.online ? "online" : "offline"}`}
              />
              <span>{active.online ? "API Online" : "Offline"}</span>
            </div>
            <div className="system-value">
              {model}
              {active.version ? ` · v${active.version}` : ""}
            </div>
            <div className="system-value">
              {population.toLocaleString("en-US")} subscribers
            </div>
            {mode === "real" && active.cutoff ? (
              <div className="system-value">Cutoff {active.cutoff}</div>
            ) : (
              <div className="system-value">Demo population</div>
            )}
          </div>

          <div className="system-label mode-label">DATA MODE</div>
          <ModeControl mode={mode} setMode={setMode} />
          <div className="mode-caption">{MODES[mode].dataLabel}</div>
        </div>
      </nav>

      <header className="mobile-head">
        <div className="mobile-top">
          <img src="/logo.png" alt="ChurnCast Logo" className="mobile-logo" />
          <ModeControl mode={mode} setMode={setMode} compact />
        </div>
        <div className="mobile-nav">{items}</div>
      </header>
    </>
  );
}
