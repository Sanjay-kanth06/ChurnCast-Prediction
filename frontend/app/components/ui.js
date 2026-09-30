"use client";

import { DASH } from "../../lib/api";


export function StatIcon({ name }) {
  const paths = {
    users: (
      <>
        <circle cx="9" cy="8" r="2.5" />
        <path d="M4.5 17c.4-2.4 2-3.8 4.5-3.8S13.1 14.6 13.5 17" />
        <path d="M14.5 6.5a2.1 2.1 0 1 1 0 3.9" />
        <path d="M15 13.6c1.9.4 3 1.5 3.3 3.4" />
      </>
    ),
    activity: (
      <>
        <path d="M3 12h3l2-5 3.5 10 2-5H21" />
      </>
    ),
    warning: (
      <>
        <path d="M12 3 21 19H3L12 3Z" />
        <path d="M12 9v4" />
        <path d="M12 16h.01" />
      </>
    ),
    risk: (
      <>
        <path d="M5 18V9" />
        <path d="M12 18V5" />
        <path d="M19 18v-7" />
        <path d="M4 19h16" />
      </>
    ),
    renew: (
      <>
        <path d="M19 8a7 7 0 0 0-12-2L5 8" />
        <path d="M5 4v4h4" />
        <path d="M5 16a7 7 0 0 0 12 2l2-2" />
        <path d="M19 20v-4h-4" />
      </>
    ),
  };
  return (
    <span className="stat-icon" aria-hidden="true">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
        {paths[name] ?? paths.activity}
      </svg>
    </span>
  );
}

/* Shared primitives. None of these derive a value: each renders what it is
 * given, and renders an explicit dash when given nothing. */

export function TopBar({ title, subtitle, right }) {
  return (
    <div className="topbar">
      <div>
        <h1>{title}</h1>
        {subtitle ? <p>{subtitle}</p> : null}
      </div>
      {right ? <div className="topbar-right">{right}</div> : null}
    </div>
  );
}

export function Card({
  title,
  subtitle,
  right,
  children,
  pad = true,
  className = "",
  hover = false,
  style,
}) {
  const cls = ["card", className, hover ? "card-hover" : ""]
    .filter(Boolean)
    .join(" ");
  return (
    <section className={cls} style={style}>
      {title ? (
        <div className="card-head">
          <div>
            <div className="card-title">{title}</div>
            {subtitle ? <div className="card-sub">{subtitle}</div> : null}
          </div>
          {right ?? null}
        </div>
      ) : null}
      {pad ? <div className="card-pad">{children}</div> : children}
    </section>
  );
}

export function Kpi({ label, value, foot, small, icon, className = "" }) {
  return (
    <div className={`card kpi card-hover ${className}`.trim()}>
      <div className="kpi-head" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
        <div className="kpi-label">{label}</div>
        {icon && <div className="kpi-icon" style={{ fontSize: '18px', color: 'var(--a-bright)', opacity: 0.8 }}>{icon}</div>}
      </div>
      <div className={`kpi-value${small ? " sm" : ""}`}>{value ?? DASH}</div>
      {foot ? <div className="kpi-foot">{foot}</div> : null}
    </div>
  );
}

export function Badge({ level, children, plain }) {
  return (
    <span className={`badge ${level ?? "neutral"}${plain ? " plain" : ""}`}>
      {children ?? level}
    </span>
  );
}

export function Notice({ kind = "", title, children }) {
  return (
    <div
      className={`notice ${kind}`.trim()}
      role={kind === "err" ? "alert" : undefined}
    >
      {title ? <strong>{title}</strong> : null}
      {children}
    </div>
  );
}

export function Empty({ ico = "◉", title, children }) {
  return (
    <div className="empty">
      <div className="empty-ico" aria-hidden="true">
        {ico}
      </div>
      <h3>{title}</h3>
      <p>{children}</p>
    </div>
  );
}

export function Row({ k, v }) {
  return (
    <div className="dl-row">
      <span className="dl-k">{k}</span>
      <span className="dl-v">{v ?? DASH}</span>
    </div>
  );
}

export function Skeleton({ w = "100%", h = 13, mb = 10 }) {
  return (
    <div className="skel" style={{ width: w, height: h, marginBottom: mb }} />
  );
}

export function LoadingCard({ rows = 4 }) {
  return (
    <div className="card card-pad">
      <Skeleton w="34%" h={15} mb={18} />
      {Array.from({ length: rows }, (_, i) => (
        <Skeleton key={i} w={`${92 - i * 11}%`} />
      ))}
    </div>
  );
}

/** Metric tiles for genuinely measured values. Missing reads as a dash. */
export function Metrics({ items }) {
  return (
    <div className="metrics">
      {items.map((m) => (
        <div className="metric" key={m.feature ?? m.label}>
          <div className="metric-k">{m.label}</div>
          <div className="metric-v">{m.value ?? DASH}</div>
        </div>
      ))}
    </div>
  );
}
