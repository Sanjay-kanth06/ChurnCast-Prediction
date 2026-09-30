"use client";

import { useEffect, useState } from "react";
import {
  getModelDetails,
  getRealModelDetails,
  num4,
  int,
  modelName,
  MODES,
  DASH,
} from "../../lib/api";
import {
  TopBar,
  Card,
  Kpi,
  Notice,
  LoadingCard,
  Badge,
} from "../components/ui";
import { useMode } from "../components/Shell";

const METRICS = [
  ["roc_auc", "ROC-AUC", "Ranking quality across all thresholds"],
  ["pr_auc", "PR-AUC", "Precision/recall trade-off on the positive class"],
  ["accuracy", "Accuracy", "Correct predictions overall"],
  ["precision", "Precision", "Of those flagged as churn, how many churned"],
  ["recall", "Recall", "Of those who churned, how many were flagged"],
  ["f1", "F1", "Harmonic mean of precision and recall"],
];

export default function ModelPage() {
  const { mode } = useMode();
  const [d, setD] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    setD(null);
    setError(null);
    const load = mode === "real" ? getRealModelDetails : getModelDetails;
    load()
      .then(setD)
      .catch((e) => setError(e.message));
  }, [mode]);

  const header = (
    <TopBar
      title="Model Intelligence"
      subtitle="Model card, measured metrics and registry state"
      right={
        <>
          <span className="chip accent mono">
            {d?.model_name ?? MODES[mode].modelName}
          </span>
          <span className="chip">
            {d?.version ? `Version ${d.version}` : "Version unavailable"}
          </span>
          <span className="chip">{MODES[mode].dataLabel}</span>
        </>
      }
    />
  );

  if (error) {
    return (
      <>
        {header}
        <div className="content">
          <Notice kind="err" title="Unable to load model details">
            {error}
          </Notice>
        </div>
      </>
    );
  }

  if (!d) {
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

  return (
    <>
      {header}
      <div className="content">
        {mode === "real" ? (
          <RealModel d={d} />
        ) : (
          <SyntheticModel d={d} />
        )}
      </div>
    </>
  );
}

/* ------------------------------------------------------------- real model */

function RealModel({ d }) {
  const test = d.test ?? {};
  const val = d.validation ?? {};
  const split = d.split?.sizes ?? {};
  const sens = d.expiry_sensitivity ?? null;
  const cm = test.confusion_matrix ?? {};

  return (
    <>
      <div className="grid g4">
        <Kpi
          label="Algorithm"
          value={modelName(d.algorithm) ?? DASH}
          foot="Selected by held-out performance"
          small
        />
        <Kpi
          label="Test ROC-AUC"
          value={num4(test.roc_auc)}
          foot={`Validation ${num4(val.roc_auc)}`}
        />
        <Kpi
          label="Test PR-AUC"
          value={num4(test.pr_auc)}
          foot="Against a highly imbalanced positive class"
        />
        <Kpi
          label="Features"
          value={int(d.feature_count)}
          foot={`Cutoff ${d.observation_cutoff}`}
        />
      </div>

      {!d.metrics_available ? (
        <div style={{ marginTop: 18 }}>
          <Notice kind="warn" title="Training metrics unavailable">
            {d.detail ?? "The registration artifact was not found."}
          </Notice>
        </div>
      ) : null}

      <div className="sec-title">Measured performance</div>
      <Card
        className="insight"
        title="Validation and test metrics"
        subtitle="Read from the registered run, not recomputed"
        pad={false}
      >
        <div className="tbl-wrap">
          <table className="tbl">
            <thead>
              <tr>
                <th>Metric</th>
                <th className="num">Validation</th>
                <th className="num">Test</th>
              </tr>
            </thead>
            <tbody>
              {METRICS.map(([key, label, help]) => (
                <tr key={key} style={{ cursor: "default" }}>
                  <td>
                    <div style={{ fontWeight: 600, color: "var(--t1)" }}>
                      {label}
                    </div>
                    <div style={{ fontSize: 12, color: "var(--t3)" }}>
                      {help}
                    </div>
                  </td>
                  <td className="num mono">{num4(val[key])}</td>
                  <td
                    className="num mono"
                    style={{ fontWeight: 650, color: "var(--t1)" }}
                  >
                    {num4(test[key])}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="sec-title">Training data</div>
      <div className="grid g21">
        <Card title="Split" subtitle="Stratified, with the positive rate held constant">
          <div className="dl">
            {Object.entries(split).map(([name, s]) => (
              <div className="dl-row" key={name}>
                <span className="dl-k" style={{ textTransform: "capitalize" }}>
                  {name}
                </span>
                <span className="dl-v mono">
                  {int(s.rows)} &middot; {(s.positive_rate * 100).toFixed(2)}%
                  positive
                </span>
              </div>
            ))}
            <div className="dl-row">
              <span className="dl-k">Observation cutoff</span>
              <span className="dl-v mono">{d.observation_cutoff}</span>
            </div>
            <div className="dl-row">
              <span className="dl-k">Label window</span>
              <span className="dl-v">{d.dataset?.label_window ?? DASH}</span>
            </div>
          </div>
          {cm.tn !== undefined ? (
            <div style={{ marginTop: 18 }}>
              <div className="metrics">
                <div className="metric">
                  <div className="metric-k">True negatives</div>
                  <div className="metric-v">{int(cm.tn)}</div>
                </div>
                <div className="metric">
                  <div className="metric-k">False positives</div>
                  <div className="metric-v">{int(cm.fp)}</div>
                </div>
                <div className="metric">
                  <div className="metric-k">False negatives</div>
                  <div className="metric-v">{int(cm.fn)}</div>
                </div>
                <div className="metric">
                  <div className="metric-k">True positives</div>
                  <div className="metric-v">{int(cm.tp)}</div>
                </div>
              </div>
            </div>
          ) : null}
        </Card>

        <Card title="Registry">
          <div className="dl">
            <div className="dl-row">
              <span className="dl-k">Name</span>
              <span className="dl-v mono">{d.model_name}</span>
            </div>
            <div className="dl-row">
              <span className="dl-k">Version</span>
              <span className="dl-v">v{d.version}</span>
            </div>
            <div className="dl-row">
              <span className="dl-k">State</span>
              <span className="dl-v">
                <Badge level="LOW">{d.status}</Badge>
              </span>
            </div>
            <div className="dl-row">
              <span className="dl-k">Experiment</span>
              <span className="dl-v mono" style={{ fontSize: 12 }}>
                {d.experiment ?? DASH}
              </span>
            </div>
            <div className="dl-row">
              <span className="dl-k">Subscribers</span>
              <span className="dl-v">{int(d.subscribers)}</span>
            </div>
          </div>
        </Card>
      </div>

      {sens ? (
        <>
          <div className="sec-title">Leakage sensitivity</div>
          <Card>
            <Notice
              kind={sens.suspicious_inflation_detected ? "warn" : "accent"}
              title={
                sens.suspicious_inflation_detected
                  ? "Suspicious inflation detected"
                  : "No suspicious inflation detected"
              }
            >
              {sens.rationale}
            </Notice>
            <div style={{ marginTop: 14 }} className="metrics">
              <div className="metric">
                <div className="metric-k">Max ROC-AUC delta</div>
                <div className="metric-v">{num4(sens.max_roc_auc_delta)}</div>
              </div>
              <div className="metric">
                <div className="metric-k">Max importance share</div>
                <div className="metric-v">
                  {(sens.max_expiry_importance_share * 100).toFixed(2)}%
                </div>
              </div>
            </div>
          </Card>
        </>
      ) : null}

      <div className="sec-title">Feature importance</div>
      <Card
        title={`Top contributors of ${d.feature_count ?? DASH} features`}
        subtitle="Gain-based ranking from the registered model"
      >
        <div style={{ display: "flex", flexDirection: "column", gap: 11 }}>
          {(d.top_features ?? []).slice(0, 12).map((f) => (
            <div className="rb-row" key={f.feature}>
              <span
                style={{ fontSize: 12.5, color: "var(--t2)" }}
                className="trunc"
                title={f.feature}
              >
                {f.feature}
              </span>
              <div className="rb-track">
                <div
                  className="rb-fill"
                  style={{
                    width: `${Math.min(100, f.share * 100 * 3.2)}%`,
                    background:
                      "linear-gradient(90deg, var(--a-2), var(--a-bright))",
                  }}
                />
              </div>
              <span className="rb-meta">{(f.share * 100).toFixed(2)}%</span>
            </div>
          ))}
        </div>
      </Card>

      <div style={{ marginTop: 20 }}>
        <Notice kind="accent" title="Trained on real KKBOX data">
          This model is supervised on the official train_v2 labels with an
          enforced {d.observation_cutoff} observation cutoff. It is not a
          recreation of the competition&apos;s hidden cohort-selection
          artifact, which the published files cannot reproduce.
        </Notice>
      </div>
    </>
  );
}

/* -------------------------------------------------------- synthetic model */

function SyntheticModel({ d }) {
  const best = d.best_model ?? null;
  const val = d.validation ?? {};
  const test = d.test ?? {};
  const split = d.split ?? {};
  const candidates = Object.keys(val);
  const th = d.risk_thresholds ?? {};

  return (
    <>
      {!d.metrics_available ? (
        <div style={{ marginBottom: 18 }}>
          <Notice kind="warn" title="Training metrics unavailable">
            reports/model_metrics.json was not found, so no evaluation figures
            are shown. Run the training pipeline to produce it.
          </Notice>
        </div>
      ) : null}

      <div className="grid g4">
        <Kpi
          label="Algorithm"
          value={modelName(d.algorithm ?? best) || DASH}
          foot="Selected by validation ROC-AUC"
          small
        />
        <Kpi
          label="Test ROC-AUC"
          value={best ? num4(test[best]?.roc_auc) : DASH}
          foot="Held-out final cohort"
        />
        <Kpi
          label="Features"
          value={int(d.feature_count)}
          foot={`${d.numeric_features?.length ?? DASH} numeric, ${
            d.categorical_features?.length ?? DASH
          } categorical`}
        />
        <Kpi
          label="Training subscribers"
          value={int(d.n_subscribers)}
          foot={
            split.n_train
              ? `${int(split.n_train)} / ${int(split.n_val)} / ${int(
                  split.n_test
                )}`
              : "Split unavailable"
          }
        />
      </div>

      {d.metrics_available && candidates.length > 0 ? (
        <>
          <div className="sec-title">Measured performance</div>
          <Card
            className="insight"
            title="Validation and test metrics"
            subtitle="Read directly from the training run output"
            pad={false}
          >
            <div className="tbl-wrap">
              <table className="tbl">
                <thead>
                  <tr>
                    <th>Metric</th>
                    {candidates.map((c) => (
                      <th key={c} className="num">
                        {modelName(c)} (val)
                      </th>
                    ))}
                    <th className="num">
                      {best ? `${modelName(best)} (test)` : "Test"}
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {METRICS.map(([key, label, help]) => (
                    <tr key={key} style={{ cursor: "default" }}>
                      <td>
                        <div style={{ fontWeight: 600, color: "var(--t1)" }}>
                          {label}
                        </div>
                        <div style={{ fontSize: 12, color: "var(--t3)" }}>
                          {help}
                        </div>
                      </td>
                      {candidates.map((c) => (
                        <td key={c} className="num mono">
                          {num4(val[c]?.[key])}
                        </td>
                      ))}
                      <td
                        className="num mono"
                        style={{ fontWeight: 650, color: "var(--t1)" }}
                      >
                        {best ? num4(test[best]?.[key]) : DASH}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </>
      ) : null}

      <div className="sec-title">Temporal split</div>
      <div className="grid g21">
        <Card
          title="Cohort-based split"
          subtitle="Subscribers are divided by cohort month, never at random"
        >
          {split.train_cohorts ? (
            <div className="dl">
              <div className="dl-row">
                <span className="dl-k">Train cohorts</span>
                <span className="dl-v mono">
                  {split.train_cohorts.join(", ")}
                </span>
              </div>
              <div className="dl-row">
                <span className="dl-k">Validation cohorts</span>
                <span className="dl-v mono">
                  {split.val_cohorts?.join(", ") ?? DASH}
                </span>
              </div>
              <div className="dl-row">
                <span className="dl-k">Test cohorts</span>
                <span className="dl-v mono">
                  {split.test_cohorts?.join(", ") ?? DASH}
                </span>
              </div>
              <div className="dl-row">
                <span className="dl-k">Random seed</span>
                <span className="dl-v mono">{d.seed ?? DASH}</span>
              </div>
            </div>
          ) : (
            <Notice>Split details are unavailable.</Notice>
          )}
        </Card>

        <Card title="Thresholds">
          <div className="dl">
            <div className="dl-row">
              <span className="dl-k">Decision threshold</span>
              <span className="dl-v mono">{d.decision_threshold ?? DASH}</span>
            </div>
            <div className="dl-row">
              <span className="dl-k">
                <Badge level="LOW">LOW</Badge>
              </span>
              <span className="dl-v mono">&lt; {th.low_below ?? DASH}</span>
            </div>
            <div className="dl-row">
              <span className="dl-k">
                <Badge level="MEDIUM">MEDIUM</Badge>
              </span>
              <span className="dl-v mono">
                {th.low_below ?? DASH} &ndash; {th.high_at_or_above ?? DASH}
              </span>
            </div>
            <div className="dl-row">
              <span className="dl-k">
                <Badge level="HIGH">HIGH</Badge>
              </span>
              <span className="dl-v mono">
                &ge; {th.high_at_or_above ?? DASH}
              </span>
            </div>
          </div>
        </Card>
      </div>

      <div className="sec-title">Features</div>
      <Card
        title={`${d.feature_count ?? DASH} model inputs`}
        subtitle="Every feature is computed from data available before the observation cutoff"
      >
        <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
          <div>
            <div className="act-sub" style={{ margin: "0 0 10px" }}>
              Numeric ({d.numeric_features?.length ?? 0})
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              {(d.numeric_features ?? []).map((f) => (
                <span className="chip mono" key={f}>
                  {f}
                </span>
              ))}
            </div>
          </div>
          <div>
            <div className="act-sub" style={{ margin: "0 0 10px" }}>
              Categorical ({d.categorical_features?.length ?? 0})
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              {(d.categorical_features ?? []).map((f) => (
                <span className="chip mono" key={f}>
                  {f}
                </span>
              ))}
            </div>
          </div>
        </div>
      </Card>

      <div style={{ marginTop: 20 }}>
        <Notice kind="warn" title="What these numbers do and do not mean">
          All metrics above are genuine measurements from the training run, but
          the data is a generated KKBOX-like dataset rather than the official
          competition data. Switch to Real mode for the KKBOX-trained model.
        </Notice>
      </div>
    </>
  );
}
