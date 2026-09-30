"use client";

import { Card, Notice } from "./ui";

export default function Plan({ plan }) {
  if (!plan) return null;

  const { severity, severity_label, risk_level, heading, intervention } = plan;
  const actions = plan.actions ?? [];
  const flagged = plan.flagged_because ?? [];
  const urgent = severity === "HIGH_PRIORITY" || severity === "CRITICAL";

  return (
    <Card className="chart-card" pad={false} style={{ border: intervention ? '2px solid var(--a-bright)' : '1px solid #142a44', overflow: 'hidden' }}>
      <div style={{ background: intervention ? 'linear-gradient(90deg, #142a44, #0d1d31)' : '#0d1d31', padding: '32px' }}>
        <div className="plan-head" style={{ borderBottom: 'none' }}>
          <div>
            <div style={{ fontSize: '12px', color: 'var(--a-bright)', fontWeight: '700', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: '8px' }}>PERSONALIZED RETENTION PLAN</div>
            <div className="plan-title" style={{ fontSize: '24px' }}>{heading}</div>
            <div className="plan-sub" style={{ fontSize: '15px' }}>
              {intervention ? "WHAT SHOULD WE DO TO PREVENT THIS CUSTOMER FROM CHURNING?" : "No targeted retention intervention is indicated."}
            </div>
          </div>
          <div className="plan-states">
            <div>
              <div className="plan-state-k">Priority</div>
              <span className={`badge plain ${urgent ? "HIGH" : "accent"}`} style={{ fontSize: '14px', padding: '6px 16px' }}>
                {severity_label}
              </span>
            </div>
          </div>
        </div>

        {actions.length > 0 && (
          <div style={{ marginTop: '32px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
            {actions.map((a, i) => (
              <div key={a.key} style={{ background: '#081426', borderRadius: '16px', padding: '24px', border: '1px solid #142a44', display: 'flex', gap: '20px' }}>
                <div style={{ background: 'var(--a-solid)', color: '#fff', width: '48px', height: '48px', borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '20px', fontWeight: 'bold', flexShrink: 0 }}>
                  0{i + 1}
                </div>
                <div>
                  <div style={{ fontSize: '18px', fontWeight: '700', color: '#fff', textTransform: 'uppercase' }}>{a.title}</div>
                  <div style={{ fontSize: '14px', color: 'var(--a-bright)', marginTop: '4px', fontWeight: '600' }}>Reason: {a.rationale}</div>

                  {a.evidence?.length > 0 && (
                    <div style={{ marginTop: '16px', background: '#0d1d31', padding: '12px', borderRadius: '8px' }}>
                      <div style={{ fontSize: '11px', color: 'var(--t4)', textTransform: 'uppercase', marginBottom: '8px' }}>Supporting Evidence</div>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                        {a.evidence.map((e) => (
                          <div key={e.feature} style={{ background: '#142a44', padding: '6px 12px', borderRadius: '6px', fontSize: '12px', color: '#dceaf7' }}>
                            {e.label ?? e.feature}: <strong style={{ color: '#fff' }}>{e.value}</strong>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  <div style={{ marginTop: '16px', background: 'rgba(74, 116, 167, 0.1)', padding: '16px', borderRadius: '8px', borderLeft: '4px solid var(--a-bright)' }}>
                    <strong style={{ color: 'var(--t1)', display: 'block', marginBottom: '4px' }}>Recommended action:</strong>
                    <span style={{ color: '#dceaf7', fontSize: '14px' }}>{a.next_step}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </Card>
  );
}
