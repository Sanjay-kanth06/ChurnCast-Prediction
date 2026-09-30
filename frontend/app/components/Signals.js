"use client";

import { Card, Notice } from "./ui";

/**
 * Model signals as a centered diverging axis.
 *
 * The bar length encodes the magnitude of the model's log-odds contribution
 * and the side encodes its direction — left of zero pushes the estimate down,
 * right pushes it up. It deliberately does NOT encode the feature's own value,
 * which is what the previous "level bar" implied and which made a binary flag
 * look like it was "50% on".
 *
 * Bars are scaled against the strongest contribution in this view, so the
 * chart compares signals with each other rather than claiming an absolute
 * effect size.
 */
export default function Signals({ signals, autoRenew }) {
  if (!signals || signals.length === 0) {
    return (
      <Card
        title="Model signals"
        subtitle="How individual features shift the model's estimated churn risk"
      >
        <Notice kind="warn" title="Signal attribution unavailable">
          This model does not expose per-feature contributions, so no breakdown
          can be shown for this prediction.
        </Notice>
      </Card>
    );
  }

  const max = signals.reduce(
    (a, s) => Math.max(a, Math.abs(s.contribution ?? 0)),
    0
  );

  return (
    <Card
      title="Model signals"
      subtitle="How individual features shift the model's estimated churn risk"
    >
      {autoRenew ? <AutoRenew signal={autoRenew} /> : null}

      <div className="sig" style={{ marginTop: autoRenew ? 20 : 0 }}>
        <div className="sig-axis" aria-hidden="true">
          <span>&larr; Lower churn risk</span>
          <span className="zero">0</span>
          <span className="r">Higher churn risk &rarr;</span>
        </div>

        {signals.map((s) => {
          const c = s.contribution ?? 0;
          const up = c > 0;
          const width = max ? (Math.abs(c) / max) * 100 : 0;
          return (
            <div className="sig-row" key={s.feature}>
              <div className="sig-top">
                <span className="sig-name">{s.label ?? s.feature}</span>
                <span className="sig-val">{s.value}</span>
              </div>

              <div
                className="sig-bar"
                role="img"
                aria-label={`${s.label ?? s.feature}: ${
                  up ? "increases" : "decreases"
                } estimated risk by ${Math.abs(c).toFixed(3)} log-odds`}
              >
                <span className="sig-half left">
                  {!up ? (
                    <span
                      className="sig-fill down"
                      style={{ width: `${width}%` }}
                    />
                  ) : null}
                </span>
                <span className="sig-mid" />
                <span className="sig-half right">
                  {up ? (
                    <span
                      className="sig-fill up"
                      style={{ width: `${width}%` }}
                    />
                  ) : null}
                </span>
              </div>

              <div className="sig-note">
                <span>
                  {up ? "Increases" : "Decreases"} estimated risk
                </span>
                <span className="sig-lo">
                  {c > 0 ? "+" : ""}
                  {c.toFixed(3)} log-odds
                </span>
              </div>
            </div>
          );
        })}
      </div>

      <div style={{ marginTop: 18 }}>
        <Notice>
          Contribution values describe model behavior, not causal effects. They
          are the fitted model&apos;s own weights applied to this
          subscriber&apos;s feature values, and they explain how the estimate
          was reached — not why the person might leave.
        </Notice>
      </div>
    </Card>
  );
}

/**
 * Auto-renewal as a status, not a level.
 *
 * A binary flag gets a badge plus its actual model contribution. There is no
 * proportional bar, because there is no proportion to show.
 */
function AutoRenew({ signal }) {
  const c = signal.contribution ?? 0;
  const enabled = String(signal.value).toLowerCase() === "enabled";
  const lowers = c < 0;

  return (
    <div className="renew">
      <div>
        <div className="renew-k">Auto-renewal</div>
        <div style={{ marginTop: 7 }}>
          <span className={`badge ${enabled ? "LOW" : "HIGH"}`}>
            {enabled ? "Enabled" : "Disabled"}
          </span>
        </div>
      </div>
      <div>
        <div className="renew-k">Model contribution</div>
        <div className="renew-contrib" style={{ marginTop: 7 }}>
          {lowers ? "↓" : "↑"} {c > 0 ? "+" : ""}
          {c.toFixed(3)} log-odds
        </div>
      </div>
      <div className="renew-note">
        Currently associated with {lowers ? "lower" : "higher"} predicted churn
        for this subscriber. This is model interpretation, not causal proof.
      </div>
    </div>
  );
}
