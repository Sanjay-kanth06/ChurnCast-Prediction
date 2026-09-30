"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  listPopulation,
  getPopulationFilters,
  int,
  DASH,
  probPct,
} from "../../lib/api";
import { TopBar, Card, Notice, Empty, Badge, Skeleton } from "./ui";

/**
 * Operational table over the scored real population.
 *
 * Filtering and pagination run server-side against the batch artifact, so the
 * browser never holds 970,960 rows and no scoring happens per keystroke. The
 * "top signal" column is the model's own strongest risk-raising contribution
 * for that subscriber, computed once during the batch pass.
 */

const PAGE = 25;

const PROBABILITY_RANGES = [
  { value: "", label: "Any probability" },
  { value: "0:0.4", label: "Below 40%" },
  { value: "0.4:0.7", label: "40% to 70%" },
  { value: "0.7:0.9", label: "70% to 90%" },
  { value: "0.9:1", label: "90% and above" },
];

export default function RealSubscribers() {
  const router = useRouter();

  const [query, setQuery] = useState("");
  const [risk, setRisk] = useState("");
  const [autoRenew, setAutoRenew] = useState("");
  const [activity, setActivity] = useState("");
  const [range, setRange] = useState("");
  const [sort, setSort] = useState("probability_desc");
  const [offset, setOffset] = useState(0);

  const [opts, setOpts] = useState(null);
  const [data, setData] = useState(null);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    getPopulationFilters()
      .then(setOpts)
      .catch(() => setOpts(null));
  }, []);

  const load = useCallback(async () => {
    setBusy(true);
    setError(null);
    const [lo, hi] = range ? range.split(":") : [null, null];
    try {
      setData(
        await listPopulation({
          query,
          risk,
          autoRenew,
          activity,
          minProbability: lo,
          maxProbability: hi,
          sort,
          limit: PAGE,
          offset,
        })
      );
    } catch (e) {
      setError(e.message);
      setData(null);
    } finally {
      setBusy(false);
    }
  }, [query, risk, autoRenew, activity, range, sort, offset]);

  // Debounced so typing does not fire a request per keystroke.
  useEffect(() => {
    const t = setTimeout(load, 240);
    return () => clearTimeout(t);
  }, [load]);

  const change = (setter) => (v) => {
    setter(v);
    setOffset(0);
  };

  const rows = data?.rows ?? [];
  const total = data?.total ?? 0;
  const from = total === 0 ? 0 : offset + 1;
  const to = Math.min(offset + PAGE, total);
  const filtered = Boolean(query || risk || autoRenew || activity || range);

  const open = (msno) =>
    router.push(`/predict?msno=${encodeURIComponent(msno)}`);

  return (
    <>
      <TopBar
        title="Subscribers"
        subtitle="Operational view over the scored real population"
        right={
          <>
            <span className="chip">
              {busy && !data
                ? "Loading…"
                : `${int(total)} ${filtered ? "matching" : "subscribers"}`}
            </span>
            <span className="chip accent">Data mode: REAL KKBOX</span>
          </>
        }
      />

      <div className="content">
        <Card
          className="insight"
          title="Filters"
          subtitle="Values present in the scored population"
        >
          <div className="filters-row">
            <div>
              <label className="fl" htmlFor="q">
                Subscriber ID
              </label>
              <input
                id="q"
                type="text"
                value={query}
                placeholder="Search by hash"
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
              <label className="fl" htmlFor="prob">
                Probability
              </label>
              <select
                id="prob"
                value={range}
                onChange={(e) => change(setRange)(e.target.value)}
              >
                {PROBABILITY_RANGES.map((r) => (
                  <option key={r.value} value={r.value}>
                    {r.label}
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
              <label className="fl" htmlFor="act">
                Activity
              </label>
              <select
                id="act"
                value={activity}
                onChange={(e) => change(setActivity)(e.target.value)}
              >
                <option value="">All activity</option>
                {(opts?.activity_bands ?? []).map((b) => (
                  <option key={b} value={b}>
                    {b}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="fl" htmlFor="sort">
                Sort
              </label>
              <select
                id="sort"
                value={sort}
                onChange={(e) => change(setSort)(e.target.value)}
              >
                <option value="probability_desc">Highest risk first</option>
                <option value="probability_asc">Lowest risk first</option>
              </select>
            </div>
          </div>

          {filtered ? (
            <button
              className="btn-link"
              style={{ marginTop: 16 }}
              onClick={() => {
                setQuery("");
                setRisk("");
                setAutoRenew("");
                setActivity("");
                setRange("");
                setOffset(0);
              }}
            >
              Clear all filters
            </button>
          ) : null}
        </Card>

        <div style={{ marginTop: 20 }}>
          {error ? (
            <Notice kind="err" title="Unable to load subscribers">
              {error}
            </Notice>
          ) : (
            <Card
              title="Scored subscribers"
              subtitle="Probabilities from the offline batch scoring pass"
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
                          <th>Subscriber</th>
                          <th className="num">Probability</th>
                          <th>Risk</th>
                          <th>Activity</th>
                          <th>Auto-renew</th>
                          <th>Top signal</th>
                        </tr>
                      </thead>
                      <tbody>
                        {rows.map((r) => (
                          <tr
                            key={r.msno}
                            onClick={() => open(r.msno)}
                            tabIndex={0}
                            onKeyDown={(e) => {
                              if (e.key === "Enter") open(r.msno);
                            }}
                            title={`Analyze ${r.msno}`}
                          >
                            <td className="mono">
                              <span className="trunc">{r.msno}</span>
                            </td>
                            <td className="num mono">
                              {probPct(r.churn_probability)}
                            </td>
                            <td>
                              <Badge level={r.risk_level} />
                            </td>
                            <td>
                              {r.activity_band || DASH}
                              {r.active_days_30d !== null ? (
                                <span className="legend-pct">
                                  {" "}
                                  ({r.active_days_30d}d)
                                </span>
                              ) : null}
                            </td>
                            <td>
                              {r.auto_renew === null
                                ? DASH
                                : r.auto_renew === 1
                                ? "Enabled"
                                : "Disabled"}
                            </td>
                            <td style={{ whiteSpace: "normal" }}>
                              {r.top_signal_label ?? DASH}
                              {r.top_signal_contribution ? (
                                <span
                                  className="legend-pct mono"
                                  style={{ marginLeft: 6 }}
                                >
                                  +{r.top_signal_contribution.toFixed(2)}
                                </span>
                              ) : null}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  <div
                    className="pager"
                    style={{ borderTop: "1px solid var(--line)" }}
                  >
                    <span className="pager-info">
                      Showing {int(from)}&ndash;{int(to)} of {int(total)}
                    </span>
                    <span style={{ display: "flex", gap: 10 }}>
                      <button
                        className="btn-ghost"
                        disabled={offset === 0 || busy}
                        onClick={() => setOffset(Math.max(0, offset - PAGE))}
                      >
                        Previous
                      </button>
                      <button
                        className="btn-ghost"
                        disabled={to >= total || busy}
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

        <div style={{ marginTop: 20 }}>
          <Notice>
            Select any row to open the full analysis and personalized
            prevention plan for that subscriber. Top signal is the model&apos;s
            strongest risk-raising contribution for that individual, not a
            causal explanation.
          </Notice>
        </div>
      </div>
    </>
  );
}
