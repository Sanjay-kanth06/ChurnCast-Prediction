"use client";

import { Card, Notice, Badge, Metrics } from "./ui";
import Signals from "./Signals";
import Plan from "./Plan";
import { DASH } from "../../lib/api";

export default function Analysis({ data }) {
  const p = data.prediction ?? {};
  const level = p.risk_level ?? "neutral";
  const prob = typeof p.churn_probability === "number" ? p.churn_probability : null;
  const signals = data.signals ?? [];
  const profile = data.profile ?? [];
  const behaviour = data.behaviour ?? [];
  const features = data.features ?? [];
  const isReal = data.data_mode === "real_v2";

  const renewFeatures = ["current_auto_renew", "is_auto_renew"];
  const autoRenew = signals.find((s) => renewFeatures.includes(s.feature));
  const otherSignals = signals.filter((s) => s !== autoRenew);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 32 }}>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: '#081426', padding: '16px 24px', borderRadius: '12px', border: '1px solid #142a44' }}>
        <div>
          <div style={{ fontSize: '11px', color: 'var(--t3)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>CUSTOMER CHURN ANALYSIS</div>
          <div style={{ fontSize: '20px', fontWeight: '700', color: 'var(--t1)', marginTop: '4px', display: 'flex', alignItems: 'center', gap: '12px' }}>
            <span className="mono">{data.msno}</span>
            <button className="btn-ghost" style={{ padding: '4px 8px', fontSize: '12px' }} onClick={() => navigator.clipboard.writeText(data.msno)}>Copy</button>
            <Badge level={level}>{level} RISK</Badge>
          </div>
        </div>
      </div>

      <div className="grid g12" style={{ gap: '32px' }}>
        <Card title="CHURN PROBABILITY" className="chart-card" pad={true}>
          <div style={{ textAlign: "center", padding: '40px 0' }}>
            <div className={`prob-num ${level}`} style={{ fontSize: '80px', fontWeight: '800', textShadow: '0 0 40px rgba(var(--rgb-accent), 0.3)' }}>
              {prob === null ? DASH : `${(prob * 100).toFixed(2)}%`}
            </div>
            <div className="prob-cap" style={{ fontSize: '16px', color: 'var(--t1)', marginTop: '16px' }}>{level.toUpperCase()} RISK</div>
            <div style={{ marginTop: '24px', background: '#081426', padding: '12px', borderRadius: '8px', fontSize: '13px', color: 'var(--t2)', border: '1px solid #142a44' }}>
              {data.interpretation ?? DASH}
            </div>
          </div>
        </Card>

        <Card title="CUSTOMER PROFILE" className="table-card">
          <div className="dl" style={{ padding: '24px' }}>
            {profile.map((f) => (
              <div className="dl-row" key={f.label} style={{ padding: '16px 0', borderBottom: '1px solid #142a44' }}>
                <span className="dl-k" style={{ color: 'var(--t3)', fontSize: '14px' }}>{f.label}</span>
                <span className="dl-v" style={{ color: '#fff', fontSize: '15px', fontWeight: '600' }}>{f.value}</span>
              </div>
            ))}
          </div>
        </Card>
      </div>

      <Plan plan={data.plan} />

      <Card title="LISTENING ACTIVITY" className="table-card">
        <div style={{ padding: '24px' }}>
          <Metrics items={behaviour.length > 0 ? behaviour : features.filter(f => ["active_days_last_30d", "total_secs_last_30d", "unique_songs_last_30d", "completion_ratio", "days_since_last_listen", "engagement_trend", "txns_last_30d", "transaction_frequency", "recency_days", "prior_cancellations"].includes(f.feature)).map(f => ({ label: f.label, value: f.value, feature: f.feature }))} />
        </div>
      </Card>

      <Card title="BEHAVIOR SIGNALS (WHY THIS CUSTOMER IS AT RISK)" className="table-card">
        <div style={{ padding: '24px' }}>
          <Signals signals={otherSignals} autoRenew={autoRenew} />
        </div>
      </Card>

      <Card title="TECHNICAL FEATURE VECTOR" className="chart-card">
        <details className="feat" style={{ padding: '16px 24px' }}>
          <summary style={{ color: 'var(--a-bright)', fontSize: '14px', fontWeight: '600' }}>Show full {data.feature_count ?? features.length}-feature vector</summary>
          <div className="feat-grid" style={{ marginTop: '16px', background: '#081426' }}>
            {features.map((f) => (
              <div className="feat-cell" key={f.feature} style={{ background: '#0d1d31', border: '1px solid #142a44' }}>
                <div className="feat-k">{f.label}</div>
                <div className="feat-v">{f.value}</div>
              </div>
            ))}
          </div>
        </details>
      </Card>
    </div>
  );
}
