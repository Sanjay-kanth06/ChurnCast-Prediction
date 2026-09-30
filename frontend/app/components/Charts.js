"use client";

import { int } from "../../lib/api";

/**
 * Chart primitives, hand-drawn as SVG.
 *
 * No charting library: these are four fixed shapes over small, already
 * aggregated arrays, and a library would add a bundle and a theming layer to
 * draw rectangles. Every component takes real counts and renders exactly them
 * — none of them smooth, interpolate, or synthesise a data point.
 *
 * Risk is distinguished by brightness and saturation inside the blue system
 * plus an explicit label, never by colour alone.
 */

const RISK_FILL = {
  LOW: "var(--low)",
  MEDIUM: "var(--med)",
  HIGH: "var(--high)",
};

/* Non-risk donuts (auto-renew state) need their own ramp. Three distinct
 * brightness steps on the locked blue axis, so the segments separate without
 * introducing a colour outside the palette. */
const STATE_FILL = {
  Enabled: "var(--c-ink)",      // #DCEAF7 - brightest
  Disabled: "var(--a-solid)",   // #4A74A7 - mid
  Unknown: "var(--l4)",         // #1b3a5c - darkest, derived from #142A44
};

/** A segment names itself by `level` (risk) or `label` (any other category). */
const segKey = (s) => s.level ?? s.label;
const segFill = (s) =>
  RISK_FILL[segKey(s)] ?? STATE_FILL[segKey(s)] ?? "var(--a-solid)";

/* ------------------------------------------------------------------ donut */

export function Donut({ segments, centerValue, centerLabel, size = 210 }) {
  const total = segments.reduce((a, s) => a + s.count, 0);
  const stroke = 26;
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;

  let offset = 0;
  const arcs = segments.map((s) => {
    const share = total ? s.count / total : 0;
    const len = share * c;
    const arc = {
      ...s,
      share,
      dash: `${len} ${c - len}`,
      offset: -offset,
    };
    offset += len;
    return arc;
  });

  return (
    <div className="donut-wrap">
      <div className="donut-chart" style={{ width: size, height: size }}>
        <svg
          width={size}
          height={size}
          viewBox={`0 0 ${size} ${size}`}
          role="img"
          aria-label={`${centerLabel ?? "Distribution"}: ${segments
            .map((s) => `${segKey(s)} ${s.count}`)
            .join(", ")}`}
        >
          <circle
            cx={size / 2}
            cy={size / 2}
            r={r}
            fill="none"
            stroke="rgb(8 20 38 / 0.85)"
            strokeWidth={stroke}
          />
          {arcs.map((a) => (
            <circle
              key={segKey(a)}
              cx={size / 2}
              cy={size / 2}
              r={r}
              fill="none"
              stroke={segFill(a)}
              strokeWidth={stroke}
              strokeDasharray={a.dash}
              strokeDashoffset={a.offset}
              transform={`rotate(-90 ${size / 2} ${size / 2})`}
              strokeLinecap="butt"
            >
              <title>{`${segKey(a)}: ${int(a.count)} (${(a.share * 100).toFixed(2)}%)`}</title>
            </circle>
          ))}
        </svg>
        <div className="donut-center">
          <div className="donut-value">{centerValue}</div>
          <div className="donut-label">{centerLabel}</div>
        </div>
      </div>

      <ul className="donut-legend">
        {arcs.map((a) => (
          <li key={segKey(a)} title={`${segKey(a)}: ${int(a.count)} (${(a.share * 100).toFixed(2)}%)`}>
            <span
              className="legend-dot"
              style={{ background: segFill(a) }}
              aria-hidden="true"
            />
            <span className="legend-name">{segKey(a)}</span>
            <span className="legend-num">
              {int(a.count)}
              <span className="legend-pct">
                {" "}
                ({(a.share * 100).toFixed(2)}%)
              </span>
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/* -------------------------------------------------------------- histogram */

export function Histogram({ buckets, height }) {
  const max = buckets.reduce((a, b) => Math.max(a, b.count), 0) || 1;
  return (
    <div className="hist-wrap">
      <div className="hist" style={height ? { height } : undefined}>
        {buckets.map((b) => {
          // Bucket midpoint decides the band, so the bar is coloured by the
          // risk it actually represents rather than an arbitrary gradient.
          const mid = (b.from + b.to) / 2;
          const level = mid >= 0.7 ? "HIGH" : mid >= 0.4 ? "MEDIUM" : "LOW";
          return (
            <div
              className="hist-col"
              key={b.from}
              title={`${(b.from * 100).toFixed(0)}–${(b.to * 100).toFixed(
                0
              )}%: ${int(b.count)} subscribers (${(b.share * 100).toFixed(2)}%)`}
            >
              <div
                className="hist-bar"
                style={{
                  height: `${(b.count / max) * 100}%`,
                  background: RISK_FILL[level],
                }}
              />
              <span className="hist-x">{(b.from * 100).toFixed(0)}</span>
            </div>
          );
        })}
      </div>
      <div className="hist-axis">
        <span>Churn probability (%)</span>
        <span>Peak bucket {int(max)}</span>
      </div>
    </div>
  );
}

/* --------------------------------------------------------------- bar list */

export function BarList({ items, accent = "var(--a-bright)", showShare }) {
  const max = items.reduce((a, i) => Math.max(a, i.count), 0) || 1;
  return (
    <div className="barlist">
      {items.map((i) => (
        <div className="bl-row" key={i.label}>
          <span className="bl-label" title={i.label}>
            {i.label}
          </span>
          <div className="bl-track">
            <div
              className="bl-fill"
              style={{
                width: `${(i.count / max) * 100}%`,
                background: i.color ?? accent,
              }}
            />
          </div>
          <span className="bl-num">
            {int(i.count)}
            {showShare && i.share !== undefined ? (
              <span className="legend-pct"> ({(i.share * 100).toFixed(2)}%)</span>
            ) : null}
          </span>
        </div>
      ))}
    </div>
  );
}

/* ----------------------------------------------------------- stacked bars */

export function StackedRisk({ groups }) {
  return (
    <div className="stacked">
      {groups.map((g) => {
        const total = g.LOW + g.MEDIUM + g.HIGH;
        return (
          <div className="st-row" key={g.label}>
            <div className="st-head">
              <span className="st-name">{g.label}</span>
              <span className="st-total">{int(total)}</span>
            </div>
            <div className="st-track" role="img" aria-label={`${g.label}: LOW ${g.LOW}, MEDIUM ${g.MEDIUM}, HIGH ${g.HIGH}`}>
              {["LOW", "MEDIUM", "HIGH"].map((lvl) =>
                g[lvl] > 0 ? (
                  <span
                    key={lvl}
                    className="st-seg"
                    style={{
                      width: `${total ? (g[lvl] / total) * 100 : 0}%`,
                      background: RISK_FILL[lvl],
                    }}
                    title={`${lvl}: ${int(g[lvl])} (${(
                      (g[lvl] / total) *
                      100
                    ).toFixed(2)}%)`}
                  />
                ) : null
              )}
            </div>
            <div className="st-legend">
              {["LOW", "MEDIUM", "HIGH"].map((lvl) => (
                <span key={lvl}>
                  <span
                    className="legend-dot sm"
                    style={{ background: RISK_FILL[lvl] }}
                    aria-hidden="true"
                  />
                  {lvl} {int(g[lvl])}
                  {total ? ` (${((g[lvl] / total) * 100).toFixed(2)}%)` : ""}
                </span>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
