"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { listSubscribers, getFilters, int, MODES, DASH, probPct } from "../../lib/api";
import { TopBar, Card, Notice, Empty, Badge, Skeleton } from "../components/ui";
import { useMode } from "../components/Shell";
import RealSubscribers from "../components/RealSubscribers";

const PAGE = 25;

export default function SubscribersPage() {
  const router = useRouter();
  const { mode } = useMode();

  const [query, setQuery] = useState("");
  const [risk, setRisk] = useState("");
  const [autoRenew, setAutoRenew] = useState("");
  const [city, setCity] = useState("");
  const [offset, setOffset] = useState(0);

  const [opts, setOpts] = useState(null);
  const [data, setData] = useState(null);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    getFilters()
      .then(setOpts)
      .catch(() => setOpts(null));
  }, []);

  const load = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const d = await listSubscribers({
        query,
        risk,
        autoRenew,
        city,
        limit: PAGE,
        offset,
      });
      setData(d);
    } catch (e) {
      setError(e.message);
      setData(null);
    } finally {
      setBusy(false);
    }
  }, [query, risk, autoRenew, city, offset]);

  // Debounced so typing in the search box does not fire a request per keystroke.
  useEffect(() => {
    const t = setTimeout(load, 220);
    return () => clearTimeout(t);
  }, [load]);

  // Any filter change returns to the first page.
  const change = (setter) => (v) => {
    setter(v);
    setOffset(0);
  };

  // Real mode has its own operational table over the batch-scored population.
  if (mode === "real") return <RealSubscribers />;

  const rows = data?.rows ?? [];
  const total = data?.total ?? 0;
  const shownFrom = total === 0 ? 0 : offset + 1;
  const shownTo = Math.min(offset + PAGE, total);
  const filtered = Boolean(query || risk || autoRenew || city);

  return (
    <>
      <TopBar
        title="Subscribers"
        subtitle="Browse the scored subscriber population"
        right={
          <>
            <span className="chip">
              {busy && !data
                ? "Loading…"
                : `${int(total)} ${filtered ? "matching" : "subscribers"}`}
            </span>
            <span className="chip accent">SYNTHETIC DEMO</span>
          </>
        }
      />

      <div className="content">
        <Card className="insight" title="Filters" subtitle="Values are taken from the dataset itself">
          <div className="filters-row">
            <div>
              <label className="fl" htmlFor="q">
                Subscriber ID
              </label>
              <input
                id="q"
                type="text"
                value={query}
                placeholder="SYN0000"
                autoComplete="off"
                spellCheck={false}
                onChange={(e) => change(setQuery)(e.target.value)}
              />
            </div>
            <div>
              <label className="fl" htmlFor="risk">
                Risk band
              </label>
              <select
                id="risk"
                value={risk}
                onChange={(e) => change(setRisk)(e.target.value)}
              >
                <option value="">All bands</option>
                {(opts?.risk_levels ?? []).map((r) => (
                  <option key={r} value={r}>
                    {r}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="fl" htmlFor="ar">
                Auto-renewal
              </label>
              <select
                id="ar"
                value={autoRenew}
                onChange={(e) => change(setAutoRenew)(e.target.value)}
              >
                <option value="">Any</option>
                {(opts?.auto_renew ?? []).map((a) => (
                  <option key={a.value} value={a.value}>
                    {a.label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="fl" htmlFor="city">
                City code
              </label>
              <select
                id="city"
                value={city}
                onChange={(e) => change(setCity)(e.target.value)}
              >
                <option value="">All cities</option>
                {(opts?.cities ?? []).map((c) => (
                  <option key={c} value={c}>
                    Code {c}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {filtered ? (
            <button
              className="btn-link"
              style={{ marginTop: 14 }}
              onClick={() => {
                setQuery("");
                setRisk("");
                setAutoRenew("");
                setCity("");
                setOffset(0);
              }}
            >
              Clear all filters
            </button>
          ) : null}
        </Card>

        <div style={{ marginTop: 18 }}>
          {error ? (
            <Notice kind="err" title="Unable to load subscribers">
              {error}
            </Notice>
          ) : (
            <Card
              title="Scored subscribers"
              subtitle="Every probability below was produced by the registered model"
              pad={false}
            >
              {busy && !data ? (
                <div className="card-pad">
                  {Array.from({ length: 8 }, (_, i) => (
                    <Skeleton key={i} h={17} mb={13} />
                  ))}
                </div>
              ) : rows.length === 0 ? (
                <Empty title="No subscribers match these filters">
                  Adjust or clear the filters above to see results.
                </Empty>
              ) : (
                <>
                  <div className="tbl-wrap">
                    <table className="tbl">
                      <thead>
                        <tr>
                          <th>Subscriber ID</th>
                          <th>City</th>
                          <th>Channel</th>
                          <th>Auto-renewal</th>
                          <th className="num">Tenure (days)</th>
                          <th className="num">Active days (30d)</th>
                          <th className="num">Churn probability</th>
                          <th>Risk</th>
                        </tr>
                      </thead>
                      <tbody>
                        {rows.map((r) => (
                          <tr
                            key={r.msno}
                            onClick={() =>
                              router.push(
                                `/subscribers/${encodeURIComponent(r.msno)}`
                              )
                            }
                            title={`Open ${r.msno}`}
                          >
                            <td className="mono">{r.msno}</td>
                            <td>{r.city}</td>
                            <td>{r.registered_via}</td>
                            <td>{r.auto_renew}</td>
                            <td className="num">
                              {r.tenure_days === null ? DASH : int(r.tenure_days)}
                            </td>
                            <td className="num">
                              {r.active_days_last_30d === null
                                ? DASH
                                : r.active_days_last_30d}
                            </td>
                            <td className="num mono">
                              {probPct(r.churn_probability)}
                            </td>
                            <td>
                              <Badge level={r.risk_level} />
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  <div className="pager" style={{ borderTop: "1px solid var(--line)" }}>
                    <span className="pager-info">
                      Showing {int(shownFrom)}&ndash;{int(shownTo)} of{" "}
                      {int(total)}
                    </span>
                    <span style={{ display: "flex", gap: 9 }}>
                      <button
                        className="btn-ghost"
                        disabled={offset === 0 || busy}
                        onClick={() => setOffset(Math.max(0, offset - PAGE))}
                      >
                        Previous
                      </button>
                      <button
                        className="btn-ghost"
                        disabled={shownTo >= total || busy}
                        onClick={() => setOffset(offset + PAGE)}
                      >
                        Next
                      </button>
                    </span>
                  </div>
                </>
              )}
            </Card>
          )}
        </div>

        <div style={{ marginTop: 18 }}>
          <Notice>
            Select any row to open the full analysis for that subscriber. City,
            channel and payment method are integer codes in this dataset, not
            names.
          </Notice>
        </div>
      </div>
    </>
  );
}
