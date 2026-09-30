"use client";

import { useEffect, useState } from "react";
import {
  getDashboardSummary,
  getRealModelDetails,
  getPopulationAnalytics,
  int,
  modelName,
  DASH,
} from "../../lib/api";
import { TopBar, Card, Notice, Kpi } from "../components/ui";
import { useMode } from "../components/Shell";

const COMMON_STACK = [
  ["Tracking", "MLflow", "Experiment tracking and the model registry serving the API"],
  ["Orchestration", "Apache Airflow", "Scheduled retraining and promotion workflow"],
  ["Drift", "Evidently", "Feature and target drift reporting"],
  ["API", "FastAPI", "Typed request validation and server-side inference"],
  ["Interface", "Next.js", "This application, rendering API responses only"],
  ["Runtime", "Docker Compose", "Containerized application services"],
];

const REAL_STAGES = [
  ["Ingest", "Load the published KKBOX subscriber, transaction and activity tables."],
  ["Validate", "Check schemas, ranges and subscriber-key coverage."],
  ["Engineer features", "Build 57 leakage-aware features from records available by 2017-02-28."],
  ["Split", "Use the documented held-out subscriber split for model evaluation."],
  ["Train", "Fit candidate models and select the registered XGBoost model."],
  ["Evaluate", "Measure ROC-AUC, PR-AUC, accuracy, precision, recall and F1."],
  ["Track", "Log parameters, metrics and artifacts to MLflow."],
  ["Serve", "Load the registered model and score subscribers through FastAPI."],
];

const SYN_STAGES = [
  ["Ingest", "Load the generated KKBOX-like subscriber and activity tables."],
  ["Validate", "Check schemas, ranges and key coverage."],
  ["Engineer features", "Build the synthetic feature table without target leakage."],
  ["Split", "Evaluate on the documented held-out synthetic cohort."],
  ["Train", "Fit the synthetic candidate models."],
  ["Evaluate", "Measure the recorded validation and test metrics."],
  ["Track", "Log parameters, metrics and artifacts to MLflow."],
  ["Serve", "Load ChurnCastModel v1 through FastAPI."],
];

export default function AboutPage() {
  const { mode } = useMode();
  const [state, setState] = useState({ data: null, analytics: null, error: null });

  useEffect(() => {
    let alive = true;
    setState({ data: null, analytics: null, error: null });

    if (mode === "real") {
      Promise.all([getRealModelDetails(), getPopulationAnalytics()])
        .then(([data, analytics]) => {
          if (alive) setState({ data, analytics, error: null });
        })
        .catch((e) => {
          if (alive) setState({ data: null, analytics: null, error: e.message });
        });
    } else {
      getDashboardSummary()
        .then((data) => {
          if (alive) setState({ data, analytics: null, error: null });
        })
        .catch((e) => {
          if (alive) setState({ data: null, analytics: null, error: e.message });
        });
    }

    return () => {
      alive = false;
    };
  }, [mode]);

  const d = state.data ?? {};
  const a = state.analytics ?? {};
  const real = mode === "real";
  const population = real ? a.kpi?.total_subscribers ?? d.subscribers : d.total_subscribers;
  const features = real ? d.feature_count : d.feature_count;
  const model = real ? d.model_name : d.model?.name;
  const version = real ? d.version : d.model?.version;
  const algorithm = real ? d.algorithm : d.model?.algorithm;
  const cutoff = real ? d.observation_cutoff : null;

  return (
    <div className="content">
      <TopBar
        title="About ChurnCast"
        subtitle="System architecture, deployment context and model boundaries."
        right={<span className="chip accent">{real ? "REAL KKBOX" : "SYNTHETIC DEMO"}</span>}
      />

      <Card className="hero-card">
        <div className="about-hero">
          <div>
            <div className="eyebrow">SUBSCRIBER CHURN INTELLIGENCE</div>
            <h2>From subscriber signals to retention action.</h2>
            <p>
              ChurnCast estimates an individual subscriber&apos;s churn probability from
              the information available before the observation cutoff, then turns the
              model output into a deterministic retention plan.
            </p>
          </div>
          <div className="hero-badge">
            <span className="hero-badge-dot" />
            {real ? "REAL DEPLOYMENT" : "DEMO DEPLOYMENT"}
          </div>
        </div>
      </Card>

      <div className="sec-title">This deployment</div>
      <div className="grid g4">
        <Kpi label="Subscribers" value={int(population)} foot={real ? "Official train_v2 labeled population" : "Generated KKBOX-like population"} />
        <Kpi label="Features" value={int(features)} foot={real ? `Built before ${cutoff ?? DASH}` : "Leakage-aware synthetic feature set"} />
        <Kpi label="Serving model" value={version ? `v${version}` : DASH} foot={modelName(algorithm) ?? "Algorithm unavailable"} small />
        <Kpi label="Data mode" value={real ? "REAL" : "DEMO"} foot={real ? "KKBOX train_v2 labels" : "Synthetic demonstration"} small />
      </div>

      <div className="sec-title">How a prediction is produced</div>
      <Card pad={false}>
        <div style={{ padding: "6px 22px 10px" }}>
          {(real ? REAL_STAGES : SYN_STAGES).map(([name, text], i) => (
            <div className="stage-row" key={name}>
              <div className="rec-n">{i + 1}</div>
              <div>
                <div className="rec-title">{name}</div>
                <div className="rec-detail">{text}</div>
              </div>
            </div>
          ))}
        </div>
      </Card>

      <div className="sec-title">Technology</div>
      <Card pad={false}>
        <div className="tbl-wrap">
          <table className="tbl">
            <thead><tr><th>Layer</th><th>Technology</th><th>Role</th></tr></thead>
            <tbody>
              {(real
                ? [["Model", "XGBoost", "Registered real_v2 churn model"], ...COMMON_STACK]
                : [["Model", "scikit-learn / gradient boosting", "Registered synthetic churn model"], ...COMMON_STACK]
              ).map(([layer, tech, role]) => (
                <tr key={layer}>
                  <td style={{ fontWeight: 650, color: "var(--t1)" }}>{layer}</td>
                  <td>{tech}</td>
                  <td style={{ whiteSpace: "normal" }}>{role}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {state.error ? (
        <Notice kind="err" title="Deployment details unavailable">{state.error}</Notice>
      ) : null}

      <div className="sec-title">Guardrails and limitations</div>
      <div className="grid g2">
        <Card title="What the data is">
          {real ? (
            <Notice kind="accent" title="REAL KKBOX deployment">
              This mode uses the official published train_v2 labels and the project&apos;s
              documented 2017-02-28 observation cutoff. The published files do not
              reproduce the hidden competition cohort-selection artifact, so this UI
              does not claim to recreate that hidden step.
            </Notice>
          ) : (
            <Notice kind="warn" title="Synthetic demonstration">
              This mode runs on a generated KKBOX-like dataset. Its metrics are
              measurements of the synthetic model and should not be interpreted as
              performance on the official KKBOX data.
            </Notice>
          )}
          <div style={{ marginTop: 14 }}>
            <Notice>
              Subscriber records expose coded identifiers rather than names or contact
              details. The UI therefore uses subscriber IDs as the primary identity.
            </Notice>
          </div>
        </Card>

        <Card title="What the model cannot do">
          <ul className="about-list">
            <li>It estimates association, not causation.</li>
            <li>Retention suggestions are deterministic rules, not predicted treatment effects.</li>
            <li>Model performance can drift as subscriber behaviour changes.</li>
            <li>A subscriber absent from the feature table cannot be scored.</li>
          </ul>
        </Card>
      </div>

      <Notice>
        Probabilities, metrics and distributions shown in the interface are returned by
        the backend; the browser does not retrain or recompute the model.
      </Notice>
    </div>
  );
}
