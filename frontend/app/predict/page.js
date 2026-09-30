"use client";

import { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import {
  getAnalysisFor,
  getSampleFor,
  MODES,
  searchSubscribersFor,
  probPct,
} from "../../lib/api";
import { TopBar, Card, Notice, Skeleton } from "../components/ui";
import Analysis from "../components/Analysis";
import { useMode } from "../components/Shell";

const DEFAULT_IDS = {
  real: "ugx0CjOMzazClkFzU2xasmDZaoIqOUAZPsH1q0teWCg=",
  synthetic: "SYN000001",
};

export default function PredictPage() {
  return (
    <Suspense
      fallback={
        <div className="content">
          <div className="card card-pad">
            <Skeleton w="26%" h={15} mb={16} />
            <Skeleton w="70%" h={38} />
          </div>
        </div>
      }
    >
      <PredictView />
    </Suspense>
  );
}

function PredictView() {
  const params = useSearchParams();
  const preset = params.get("msno") ?? "";
  const { mode, status } = useMode();

  const [msno, setMsno] = useState(preset);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [suggestions, setSuggestions] = useState([]);
  const [samples, setSamples] = useState([]);

  const modeInfo = MODES[mode];
  const ready = status?.[mode]?.ready;

  async function run(id) {
    const clean = (id ?? "").trim();
    if (!clean) {
      setError("Enter a subscriber ID to run a prediction.");
      setResult(null);
      return;
    }
    setBusy(true);
    setError(null);
    setSuggestions([]);
    try {
      setResult(await getAnalysisFor(clean, mode));
    } catch (e) {
      setError(e.message);
      setResult(null);
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    let alive = true;
    setResult(null);
    setError(null);
    setSuggestions([]);
    const fallback = DEFAULT_IDS[mode];

    getSampleFor(mode, 20)
      .then((d) => {
        if (!alive) return;
        const ids = d?.subscribers ?? [];
        setSamples(ids);
        if (!preset) {
          setMsno(fallback);
          run(fallback);
        }
      })
      .catch(() => {
        if (!alive) return;
        setSamples([]);
        if (!preset) {
          setMsno(fallback);
          run(fallback);
        }
      });

    return () => {
      alive = false;
    };
    // Intentionally keyed only to mode/preset to make switching modes load a
    // clean sample once rather than triggering duplicate prediction requests.
  }, [mode, preset]);

  useEffect(() => {
    const q = msno.trim();
    if (q.length < 3) {
      setSuggestions([]);
      return undefined;
    }

    let alive = true;
    const timer = setTimeout(async () => {
      try {
        const remote = await searchSubscribersFor(mode, q, 6);
        const merged = [...remote, ...samples.filter((s) => s.toLowerCase().includes(q.toLowerCase()))];
        const unique = [...new Set(merged)].slice(0, 6);
        if (alive) setSuggestions(unique);
      } catch {
        if (alive) setSuggestions(
          samples.filter((s) => s.toLowerCase().includes(q.toLowerCase())).slice(0, 6)
        );
      }
    }, 180);

    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [msno, mode, samples]);

  return (
    <div className="content">
      <TopBar
        title="Predict Customer Churn"
        subtitle="Investigate one subscriber and generate a personalized retention plan."
        right={
          <>
            <span className="chip accent">{modeInfo.modelName}</span>
            <span className="chip">{modeInfo.dataLabel}</span>
          </>
        }
      />

      {ready === false ? (
        <Notice kind="warn" title={`${modeInfo.modelName} is not loaded`}>
          Predictions in {modeInfo.label} mode are unavailable.
        </Notice>
      ) : null}

      <Card title="Subscriber lookup" subtitle="Paste an ID, type a prefix, or try the verified sample." className="chart-card">
        <div className="predict-search">
          <div className="predict-input-wrap">
            <input
              id="msno"
              type="text"
              value={msno}
              placeholder={mode === "real" ? "Start typing a KKBOX subscriber hash…" : "SYN000001…"}
              autoComplete="off"
              spellCheck={false}
              onChange={(e) => {
                setMsno(e.target.value);
                if (error) setError(null);
              }}
              disabled={busy}
              aria-label="Subscriber ID"
            />
            {suggestions.length > 0 ? (
              <div className="predict-suggestions">
                {suggestions.map((s) => (
                  <button
                    type="button"
                    key={s}
                    className="suggestion"
                    onClick={() => {
                      setMsno(s);
                      run(s);
                    }}
                  >
                    <span className="mono trunc">{s}</span>
                    <span>Analyze →</span>
                  </button>
                ))}
              </div>
            ) : null}
          </div>

          <button
            type="button"
            className="btn-primary predict-button"
            disabled={busy || !msno.trim()}
            onClick={() => run(msno)}
          >
            {busy ? "Scoring…" : "Analyze subscriber"}
          </button>
        </div>

        <div className="sample-strip">
          <div>
            <div className="sample-kicker">VERIFIED SAMPLE</div>
            <div className="mono sample-id">{DEFAULT_IDS[mode]}</div>
          </div>
          <button
            type="button"
            className="btn-ghost"
            disabled={busy}
            onClick={() => {
              setMsno(DEFAULT_IDS[mode]);
              run(DEFAULT_IDS[mode]);
            }}
          >
            Try sample →
          </button>
        </div>
      </Card>

      {error ? (
        <div style={{ marginTop: 20 }}>
          <Notice kind="err" title="Prediction failed">{error}</Notice>
        </div>
      ) : null}

      {busy ? (
        <div style={{ marginTop: 24 }} className="chart-card card-pad">
          <Skeleton w="30%" h={15} mb={18} />
          <Skeleton w="100%" h={120} mb={18} />
          <Skeleton w="88%" />
          <Skeleton w="74%" />
        </div>
      ) : null}

      {result && !busy ? (
        <div style={{ marginTop: 24 }}>
          <Analysis data={result} />
        </div>
      ) : null}

      {!busy && !error && !result ? (
        <div className="predict-empty">
          <div className="predict-empty-icon">◉</div>
          <h3>Ready to analyze</h3>
          <p>Choose a subscriber ID above or use the verified sample.</p>
        </div>
      ) : null}
    </div>
  );
}
