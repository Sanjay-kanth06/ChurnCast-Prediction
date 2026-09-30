"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  getPopulationAnalytics,
  getPopulationAtRisk,
  int,
  pct,
  DASH,
} from "../../lib/api";
import { TopBar, Card, Kpi, Notice, LoadingCard, Badge, StatIcon } from "./ui";
import { Donut, Histogram, BarList, StackedRisk } from "./Charts";

export default function RealDashboard() {
  const router = useRouter();
  const [a, setA] = useState(null);
  const [risky, setRisky] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    getPopulationAnalytics()
      .then(setA)
      .catch((e) => setError(e.message));
    getPopulationAtRisk(10)
      .then((d) => setRisky(d.rows ?? []))
      .catch(() => setRisky([]));
  }, []);

  const header = (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', paddingBottom: '24px', borderBottom: '1px solid var(--line)', marginBottom: '32px' }}>
      <div>
        <h1 style={{ fontSize: '28px', fontWeight: '700', color: 'var(--t1)', margin: 0, letterSpacing: '-0.02em' }}>CHURNCAST</h1>
        <div style={{ fontSize: '18px', fontWeight: '600', color: 'var(--a-bright)', marginTop: '4px' }}>Subscriber Churn Intelligence</div>
        <p style={{ color: 'var(--t3)', fontSize: '14px', marginTop: '8px' }}>Monitor subscriber activity, churn exposure and retention opportunities.</p>
      </div>
      <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
        <span className="chip accent">{a?.provenance?.model ?? "ChurnCastRealV2"} {a?.provenance?.model_version ? `v${a.provenance.model_version}` : ""}</span>
        <span className="chip">Last updated: {a?.provenance?.observation_cutoff || DASH}</span>
      </div>
    </div>
  );

  if (error) {
    return (
      <div className="content">
        {header}
        <Notice kind="err" title="Population analytics unavailable">{error}</Notice>
      </div>
    );
  }

  if (!a) {
    return (
      <div className="content">
        {header}
        <div className="grid g2">
          <LoadingCard />
          <LoadingCard />
        </div>
      </div>
    );
  }

  const k = a.kpi;
  const prov = a.provenance;
  const risk = a.risk_distribution ?? [];
  const highShare = k.high_risk_share ?? 0;

  const genderItems = [
    { label: "Female", count: a.gender?.female ?? 0 },
    { label: "Male", count: a.gender?.male ?? 0 },
    { label: "Unknown / missing", count: a.gender?.unknown ?? 0 },
  ].map((g) => ({
    ...g,
    share: k.total_subscribers ? g.count / k.total_subscribers : 0,
  }));

  const renewDistribution = [
    { label: "Enabled", count: a.auto_renew?.enabled ?? 0, LOW: 0, MEDIUM: 0, HIGH: 0 },
    { label: "Disabled", count: a.auto_renew?.disabled ?? 0, LOW: 0, MEDIUM: 0, HIGH: 0 },
    { label: "Unknown", count: a.auto_renew?.unknown ?? 0, LOW: 0, MEDIUM: 0, HIGH: 0 }
  ];

  const highRiskCount = k.high_risk_subscribers;

  return (
    <div className="content">
      {header}

      {/* KPI ROW */}
      <div className="grid g3" style={{ marginBottom: '32px' }}>
        <Kpi label="TOTAL SUBSCRIBERS" value={int(k.total_subscribers)} icon={<StatIcon name="users" />} className="kpi-premium" foot="Total tracked population" />
        <Kpi label="ACTIVE LISTENERS" value={int(k.active_listeners)} icon={<StatIcon name="activity" />} foot={`${pct(k.active_listeners_share)} of base`} className="kpi-premium" />
        <Kpi label="LIKELY TO CHURN" value={int(k.likely_to_churn)} icon={<StatIcon name="warning" />} foot={`${pct(k.likely_to_churn_share)} above threshold`} className="kpi-premium" />
        <Kpi label="HIGH RISK" value={int(k.high_risk_subscribers)} icon={<StatIcon name="risk" />} foot={`${pct(k.high_risk_share)} of base`} className="kpi-premium alert" />
        <Kpi label="AVG CHURN PROBABILITY" value={pct(k.average_churn_probability, 2)} icon={<StatIcon name="activity" />} foot="Population mean" className="kpi-premium" />
        <Kpi label="AUTO-RENEW ENABLED" value={int(k.auto_renew_enabled)} icon={<StatIcon name="renew" />} foot={`${pct(k.auto_renew_enabled_share)} of base`} className="kpi-premium" />
      </div>

      {/* MAIN ANALYTICS ROW */}
      <div className="grid g2" style={{ marginBottom: '32px' }}>
        <Card title="RISK DISTRIBUTION" className="insight chart-card" pad={true}>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
            <Donut segments={risk} centerValue={int(highRiskCount)} centerLabel="High Risk" />
          </div>
        </Card>

        <Card title="CHURN PROBABILITY DISTRIBUTION" className="chart-card">
          <Histogram buckets={a.probability_histogram ?? []} />
        </Card>
      </div>

      {/* SECOND ANALYTICS ROW */}
      <div className="grid g2" style={{ marginBottom: '32px' }}>
        <Card title="SUBSCRIBER GENDER" className="chart-card">
          <BarList items={genderItems} showShare accent="var(--a-bright)" />
        </Card>

        <Card title="AUTO-RENEW STATUS" className="chart-card">
          <Donut segments={renewDistribution} centerValue={int(a.auto_renew?.enabled ?? 0)} centerLabel="Enabled" />
        </Card>
      </div>

      {/* BEHAVIOR ANALYTICS */}
      <div style={{ marginBottom: '32px' }}>
        <Card title="SUBSCRIBER ACTIVITY" className="chart-card">
          <BarList
            items={(a.activity_segmentation ?? []).map((s) => ({
              label: s.band,
              count: s.count,
              share: s.share,
            }))}
            showShare
          />
        </Card>

        <div className="insights-panel" style={{ marginTop: '24px' }}>
          <h3 style={{ fontSize: '13px', color: 'var(--t3)', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: '16px' }}>EXECUTIVE INSIGHTS</h3>
          <div className="grid g3">
            <div className="insight-card">
              <div className="ins-metric">{pct(highShare)}</div>
              <div className="ins-text">of subscribers are classified as HIGH RISK.</div>
            </div>
            <div className="insight-card">
              <div className="ins-metric">{pct(k.likely_to_churn_share)}</div>
              <div className="ins-text">of subscribers cross the churn probability decision threshold.</div>
            </div>
            <div className="insight-card">
              <div className="ins-metric">{pct(k.auto_renew_enabled_share)}</div>
              <div className="ins-text">have auto-renew enabled, stabilizing short-term retention.</div>
            </div>
          </div>
        </div>
      </div>

      {/* AT-RISK TABLE */}
      <div style={{ marginBottom: '32px' }}>
        <Card title="AT-RISK SUBSCRIBERS" pad={false} className="table-card">
          <div className="tbl-wrap">
            <table className="tbl">
              <thead>
                <tr>
                  <th>Subscriber ID</th>
                  <th>Risk</th>
                  <th className="num">Churn Probability</th>
                  <th>Top Risk Factor</th>
                  <th>Auto-Renew</th>
                  <th className="num">Activity (Days)</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {risky.map((r) => (
                  <tr key={r.msno} onClick={() => router.push(`/predict?msno=${encodeURIComponent(r.msno)}`)}>
                    <td className="mono"><span className="trunc">{r.msno}</span></td>
                    <td><Badge level={r.risk_level} /></td>
                    <td className="num mono">{(r.churn_probability * 100).toFixed(2)}%</td>
                    <td style={{ whiteSpace: "normal" }}>{r.top_signal_label ?? DASH}</td>
                    <td>{r.auto_renew === null ? DASH : r.auto_renew === 1 ? "Enabled" : "Disabled"}</td>
                    <td className="num">{r.active_days_30d === null ? DASH : r.active_days_30d}</td>
                    <td><button className="btn-ghost" style={{ padding: '4px 8px', fontSize: '11px' }}>Analyze &rarr;</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      </div>

      {/* RETENTION OPPORTUNITIES */}
      <div style={{ marginBottom: '32px' }}>
        <Card title="RETENTION OPPORTUNITIES">
          <div className="grid g2">
            {(a.severity_distribution ?? []).map((s) => (
              <div key={s.severity} className="opp-card">
                <div>
                  <div className="opp-title">{s.severity.replace('_', ' ')}</div>
                  <div className="opp-count">{int(s.count)} <span className="opp-share">({pct(s.share)})</span></div>
                </div>
                <div className="opp-desc">Targeted retention intervention pool based on predictive scores.</div>
              </div>
            ))}
          </div>
        </Card>
      </div>

    </div>
  );
}
