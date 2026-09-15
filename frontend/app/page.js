"use client";

import { useCallback, useEffect, useState } from "react";
import {
  ApiError,
  getHealth,
  getModelInfo,
  getSampleSubscribers,
  getSubscriberFeatures,
  predictByMsno,
} from "../lib/api";

/** Features surfaced in the read-only profile panel, with display labels. */
const PROFILE_FIELDS = [
  ["recency_days", "Recency", (v) => `${Math.round(v)} days`],
  ["active_days_last_30d", "Active days (30d)", (v) => `${Math.round(v)}`],
  ["txns_last_90d", "Transactions (90d)", (v) => `${Math.round(v)}`],
  ["total_secs_last_30d", "Listening (30d)", (v) => `${Math.round(v / 3600)} hrs`],
  ["engagement_trend", "Engagement trend", (v) => v.toFixed(1)],
  ["plan_price", "Plan price", (v) => `${Math.round(v)}`],
  ["is_auto_renew", "Auto renew", (v) => (String(v) === "1" ? "Yes" : "No")],
  ["tenure_days", "Tenure", (v) => `${Math.round(v)} days`],
];

export default function Home() {
  const [msno, setMsno] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [profile, setProfile] = useState(null);
  const [error, setError] = useState(null);
  const [samples, setSamples] = useState([]);
  const [health, setHealth] = useState(null);
  const [info, setInfo] = useState(null);

  // Service status and sample IDs come from the backend, never hard-coded.
  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth(null));
    getModelInfo().then(setInfo).catch(() => setInfo(null));
    getSampleSubscribers(3)
      .then((d) => setSamples(d.subscribers || []))
      .catch(() => setSamples([]));
  }, []);

  const handlePredict = useCallback(async () => {
    const id = msno.trim();
    if (!id) {
      setError({ title: "Invalid input", body: "Please enter a valid Subscriber ID." });
      setResult(null);
      setProfile(null);
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);
    setProfile(null);

    try {
      const prediction = await predictByMsno(id);
      setResult(prediction);
      // profile is supplementary; a failure here must not hide the prediction
      try {
        const f = await getSubscriberFeatures(id);
        setProfile(f.features);
      } catch {
        setProfile(null);
      }
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) {
        setError({
          title: "Subscriber not found",
          body: "Please check the Member ID and try again.",
        });
      } else if (e instanceof ApiError && e.status === 422) {
        setError({ title: "Invalid input", body: e.message });
      } else if (e instanceof ApiError && e.status === 503) {
        setError({
          title: "Service unavailable",
          body: "The model is not loaded. Run the pipeline, then try again.",
        });
      } else {
        setError({
          title: "Something went wrong",
          body:
            e instanceof ApiError
              ? e.message
              : "Prediction service is currently unavailable. Please try again.",
        });
      }
    } finally {
      setLoading(false);
    }
  }, [msno]);

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !loading) handlePredict();
  };

  const pct = result ? (result.churn_probability * 100).toFixed(1) : null;

  return (
    <main className="wrap">
      <header className="header">
        <h1 className="title">CHURNCAST</h1>
        <p className="subtitle">Subscriber Churn Prediction</p>
        <p className="tagline">
          Predict whether a subscriber is likely to churn within 30 days.
        </p>
        <span className="badge">Prototype — Synthetic Data</span>
      </header>

      <section className="card">
        <label className="field-label" htmlFor="msno-input">
          Subscriber ID
        </label>
        <input
          id="msno-input"
          type="text"
          value={msno}
          placeholder="SYN000001"
          autoComplete="off"
          spellCheck={false}
          disabled={loading}
          onChange={(e) => setMsno(e.target.value)}
          onKeyDown={onKeyDown}
          aria-label="Subscriber ID"
        />

        <button
          className="primary"
          onClick={handlePredict}
          disabled={loading}
          aria-busy={loading}
        >
          {loading ? "Analyzing subscriber…" : "Predict Churn"}
        </button>

        {samples.length > 0 && (
          <div className="samples">
            <span className="samples-label">Try a sample:</span>
            {samples.map((s) => (
              <button
                key={s}
                className="chip"
                disabled={loading}
                onClick={() => setMsno(s)}
                type="button"
              >
                {s}
              </button>
            ))}
          </div>
        )}
      </section>

      {error && (
        <div className="message error" role="alert">
          <strong>{error.title}</strong>
          <p>{error.body}</p>
        </div>
      )}

      {result && (
        <section className="card" aria-live="polite">
          <div className="result-head">Prediction Result</div>

          <div className="result-grid">
            <div className="result-item">
              <div className="item-label">Subscriber ID</div>
              <div className="item-value mono">{result.msno}</div>
            </div>
            <div className="result-item">
              <div className="item-label">Churn Probability</div>
              <div className="prob">{pct}%</div>
            </div>
            <div className="result-item">
              <div className="item-label">Prediction</div>
              <div className="item-value">
                {result.predicted_label === 1 ? "LIKELY TO CHURN" : "LIKELY TO STAY"}
              </div>
            </div>
            <div className="result-item">
              <div className="item-label">Risk Level</div>
              <div>
                <span className={`pill ${result.risk_level}`}>{result.risk_level}</span>
              </div>
            </div>
            <div className="result-item">
              <div className="item-label">Model Version</div>
              <div className="item-value mono">{result.model_version}</div>
            </div>
          </div>

          <div className="meter" aria-hidden="true">
            <div
              className={`meter-fill ${result.risk_level}`}
              style={{ width: `${Math.min(100, result.churn_probability * 100)}%` }}
            />
          </div>

          {profile && (
            <details className="features">
              <summary>View Subscriber Features</summary>
              <div className="feature-grid">
                {PROFILE_FIELDS.map(([key, label, fmt]) => {
                  const raw = profile[key];
                  if (raw === null || raw === undefined) return null;
                  let shown;
                  try {
                    shown = fmt(typeof raw === "string" ? raw : Number(raw));
                  } catch {
                    shown = String(raw);
                  }
                  return (
                    <div className="feature-cell" key={key}>
                      <div className="feature-name">{label}</div>
                      <div className="feature-val">{shown}</div>
                    </div>
                  );
                })}
              </div>
              <p className="readonly-note">
                Read-only values retrieved from the feature table for this subscriber.
              </p>
            </details>
          )}
        </section>
      )}

      <div className="status-bar">
        <div className="status-item">
          <span className={`dot ${health?.status === "ok" ? "up" : "down"}`} />
          API <b>{health ? (health.status === "ok" ? "Healthy" : "Degraded") : "Offline"}</b>
        </div>
        <div className="status-item">
          <span className={`dot ${health?.model_available ? "up" : "down"}`} />
          Model <b>{health?.model_available ? "Available" : "Unavailable"}</b>
        </div>
        <div className="status-item">
          Version <b>{health?.model_version ?? "—"}</b>
        </div>
        <div className="status-item">
          Type <b>{info?.model_type ?? "—"}</b>
        </div>
        <div className="status-item">
          Subscribers <b>{health?.subscribers?.toLocaleString() ?? "—"}</b>
        </div>
      </div>

      <footer className="note">
        ChurnCast runs on deterministic <strong>KKBOX-like synthetic data</strong>.
        <br />
        It demonstrates the MLOps architecture; results are not KKBOX benchmark figures.
      </footer>
    </main>
  );
}
