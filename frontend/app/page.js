"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  getDashboardSummary,
  num4,
  int,
  modelName,
  MODES,
  DASH,
} from "../lib/api";
import { TopBar, Card, Kpi, Notice, LoadingCard } from "./components/ui";
import { useMode } from "./components/Shell";
import RealDashboard from "./components/RealDashboard";

const PIPELINE = [
  "Raw data",
  "Validation",
  "Features",
  "Temporal split",
  "Training",
  "Evaluation",
  "MLflow",
  "Registry",
  "API",
  "Prediction",
];

const QUICK = [
  {
    href: "/predict",
    ico: "◉",
    t: "Predict a subscriber",
    d: "Score one ID and get a personalized retention plan",
  },
  {
    href: "/subscribers",
    ico: "☰",
    t: "Browse subscribers",
    d: "Search, filter and sort the scored population",
  },
  {
    href: "/model",
    ico: "◩",
    t: "Model intelligence",
    d: "Metrics, features and registry state",
  },
];

export default function DashboardPage() {
  const router = useRouter();
  const { mode, status } = useMode();
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (mode !== "synthetic") return;
    getDashboardSummary()
      .then(setData)
      .catch((e) => setError(e.message));
  }, [mode]);

  const header = (
    <TopBar
      title="Dashboard"
      subtitle="Subscriber churn intelligence overview"
      right={
        <>
          <span className="chip">
            <span className="dot up" />
            {status?.[mode]?.online ? "Online" : "Offline"}
          </span>
          <span className="chip accent">{MODES[mode].dataLabel}</span>
        </>
      }
    />
  );

  const quickAccess = (
    <>
      <div className="sec-title">Quick access</div>
      <div className="grid g3">
        {QUICK.map((q) => (
          <div
            key={q.href}
            className="card card-hover qa"
            onClick={() => router.push(q.href)}
            onKeyDown={(e) => {
              if (e.key === "Enter") router.push(q.href);
            }}
            role="link"
            tabIndex={0}
          >
            <div className="qa-ico" aria-hidden="true">
              {q.ico}
            </div>
            <div>
              <div className="qa-t">{q.t}</div>
              <div className="qa-d">{q.d}</div>
            </div>
          </div>
        ))}
      </div>
    </>
  );

  // Real mode renders the company-level intelligence view, backed entirely
  // by the precomputed population artifact.
  if (mode === "real") return <RealDashboard />;

  if (error) {
    return (
      <>
        {header}
        <div className="content">
          <Notice kind="err" title="Unable to load the dashboard">
            {error} No figures are shown, because none could be retrieved from
            the backend.
          </Notice>
        </div>
      </>
    );
  }

  if (!data) {
    return (
      <>
        {header}
        <div className="content grid g2">
          <LoadingCard />
          <LoadingCard />
        </div>
      </>
    );
  }

  const m = data.model ?? {};
  const dist = data.risk_distribution ?? [];
  const th = data.risk_thresholds ?? {};

  return (
    <>
      {header}
      <div className="content">
        <div className="grid g4">
          <Kpi
            label="Subscribers scored"
            value={int(data.total_subscribers)}
            foot="Entire population, scored by the registered model"
          />
          <Kpi
            label="Registered model"
            value={m.version ? `v${m.version}` : DASH}
            foot={modelName(m.algorithm) ?? "Algorithm unavailable"}
            small
          />
          <Kpi
            label="Test ROC-AUC"
            value={num4(m.test_roc_auc)}
            foot={
              m.validation_roc_auc
                ? `Validation ${num4(m.validation_roc_auc)}`
                : "Validation unavailable"
            }
          />
          <Kpi
            label="Features"
            value={int(data.feature_count)}
            foot="Leakage-aware, built before the observation cutoff"
          />
        </div>

        <div className="sec-title">Risk distribution</div>
        <div className="grid g21">
          <Card
            className="insight"
            title="Scored population by risk band"
            subtitle={`Counted from ${int(
              data.total_subscribers
            )} model predictions, not sampled or estimated`}
          >
            <div className="riskbar">
              {dist.map((d) => (
                <div className="rb-row" key={d.level}>
                  <span className={`badge ${d.level}`}>{d.level}</span>
                  <div className="rb-track">
                    <div
                      className={`rb-fill ${d.level}`}
                      style={{ width: `${(d.share * 100).toFixed(2)}%` }}
                    />
                  </div>
                  <span className="rb-meta">
                    {int(d.count)} &middot; {(d.share * 100).toFixed(2)}%
                  </span>
                </div>
              ))}
            </div>

            <div style={{ marginTop: 22 }}>
              <Notice>
                Bands come from the configured thresholds: <b>LOW</b> below{" "}
                {th.low_below ?? DASH}, <b>MEDIUM</b> from{" "}
                {th.low_below ?? DASH} to {th.high_at_or_above ?? DASH},{" "}
                <b>HIGH</b> at or above {th.high_at_or_above ?? DASH}.
              </Notice>
            </div>
          </Card>

          <Card title="Model health">
            <div className="dl">
              <div className="dl-row">
                <span className="dl-k">Name</span>
                <span className="dl-v mono">{m.name ?? DASH}</span>
              </div>
              <div className="dl-row">
                <span className="dl-k">Version</span>
                <span className="dl-v">
                  {m.version ? `v${m.version}` : DASH}
                </span>
              </div>
              <div className="dl-row">
                <span className="dl-k">Algorithm</span>
                <span className="dl-v">{modelName(m.algorithm) ?? DASH}</span>
              </div>
              <div className="dl-row">
                <span className="dl-k">State</span>
                <span className="dl-v">
                  <span className="badge LOW">{m.status ?? DASH}</span>
                </span>
              </div>
              <div className="dl-row">
                <span className="dl-k">Cohorts</span>
                <span className="dl-v">
                  {data.cohorts?.length ? data.cohorts.length : DASH}
                </span>
              </div>
            </div>
            <button
              className="btn-ghost"
              style={{ width: "100%", marginTop: 18 }}
              onClick={() => router.push("/model")}
            >
              View full model card
            </button>
          </Card>
        </div>

        {quickAccess}

        <div className="sec-title">Pipeline</div>
        <Card>
          <div className="pipe">
            {PIPELINE.map((s, i) => (
              <span key={s} style={{ display: "inline-flex", gap: 8 }}>
                <span
                  className={`pipe-step${
                    i === PIPELINE.length - 1 ? " hl" : ""
                  }`}
                >
                  {s}
                </span>
                {i < PIPELINE.length - 1 ? (
                  <span className="pipe-arrow" aria-hidden="true">
                    &rarr;
                  </span>
                ) : null}
              </span>
            ))}
          </div>
          <div style={{ marginTop: 20 }}>
            <Notice kind="warn" title="Prototype on synthetic data">
              Every figure on this page is computed from a generated
              KKBOX-like dataset, not the official KKBOX competition data. The
              metrics are real measurements of a real model, but they describe
              performance on synthetic subscribers. Switch to Real mode for the
              KKBOX-trained model.
            </Notice>
          </div>
        </Card>
      </div>
    </>
  );
}
