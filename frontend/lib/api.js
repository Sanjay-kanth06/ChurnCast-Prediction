/**
 * Thin client for the ChurnCast FastAPI backend.
 *
 * All model inference happens server-side. This module only moves JSON; it
 * never computes a prediction, and it never duplicates feature engineering.
 */

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

/** Error carrying the HTTP status so callers can distinguish 404 from 503. */
export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(`${API_URL}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch {
    throw new ApiError(
      "Prediction service is currently unavailable. Please try again.",
      0
    );
  }

  let body = null;
  try {
    body = await res.json();
  } catch {
    /* some errors carry no JSON body */
  }

  if (!res.ok) {
    const detail =
      typeof body?.detail === "string"
        ? body.detail
        : Array.isArray(body?.detail)
        ? body.detail[0]?.msg ?? "Invalid request."
        : `Request failed (${res.status}).`;
    throw new ApiError(detail, res.status);
  }
  return body;
}

export const getHealth = () => request("/health");
export const getModelInfo = () => request("/model-info");
export const getSampleSubscribers = (n = 3) =>
  request(`/subscribers/sample?n=${n}`);
export const getSubscriberFeatures = (msno) =>
  request(`/subscriber/${encodeURIComponent(msno)}/features`);
export const predictByMsno = (msno) =>
  request(`/predict/${encodeURIComponent(msno)}`, { method: "POST" });

/* ---------------------------------------------------------------- UI views
 * Endpoints backing the five-page interface. Each one returns values the
 * backend derived from the feature table, the registered model or
 * reports/model_metrics.json. Nothing here computes or substitutes a value.
 */

export const getDashboardSummary = () => request("/dashboard/summary");
export const getFilters = () => request("/filters");
export const getModelDetails = () => request("/model-details");
export const getSubscriberAnalysis = (msno) =>
  request(`/subscriber/${encodeURIComponent(msno)}/analysis`);

export function listSubscribers({
  query = "",
  risk = "",
  autoRenew = "",
  city = "",
  limit = 25,
  offset = 0,
} = {}) {
  const qs = new URLSearchParams();
  if (query) qs.set("query", query);
  if (risk) qs.set("risk", risk);
  if (autoRenew) qs.set("auto_renew", autoRenew);
  if (city) qs.set("city", city);
  qs.set("limit", String(limit));
  qs.set("offset", String(offset));
  return request(`/subscribers?${qs.toString()}`);
}

/* ---------------------------------------------------------------- format
 * Display helpers. `nullish` values render as an explicit dash rather than
 * as zero, so a missing metric is never shown as a real measurement.
 */

export const DASH = "\u2014";

// Percentages render with exactly two decimals everywhere in the UI.
export const pct = (v, digits = 2) =>
  v === null || v === undefined || Number.isNaN(v)
    ? DASH
    : `${(v * 100).toFixed(digits)}%`;

export const num4 = (v) =>
  v === null || v === undefined || Number.isNaN(v) ? DASH : Number(v).toFixed(4);

export const int = (v) =>
  v === null || v === undefined || Number.isNaN(v)
    ? DASH
    : Number(v).toLocaleString("en-US");

/** Display name for a model key. Unknown keys are shown unchanged rather than
 *  guessed at, so a new algorithm never renders under the wrong label. */
const MODEL_NAMES = {
  logistic_regression: "Logistic Regression",
  gradient_boosting: "Gradient Boosting",
  xgboost: "XGBoost",
};
export const modelName = (k) => (k ? MODEL_NAMES[k] ?? k : null);

/* ------------------------------------------------------------- data modes
 * Two completely separate populations and models. They are never mixed: a
 * synthetic msno cannot be scored by the real model and vice versa, and the
 * backend returns 404 either way. Mode selects which endpoints are used.
 */

export const MODES = {
  synthetic: {
    id: "synthetic",
    label: "Synthetic",
    dataLabel: "SYNTHETIC DEMO",
    modelName: "ChurnCastModel",
  },
  real: {
    id: "real",
    label: "Real",
    dataLabel: "REAL KKBOX",
    modelName: "ChurnCastRealV2",
  },
};

export const getRealHealth = () => request("/health/real");
export const getRealSample = (n = 3) => request(`/subscribers-real/sample?n=${n}`);
// Query-parameter form: KKBOX ids are base64 and roughly half contain "/",
// which a path segment cannot carry even percent-encoded.
export const getRealAnalysis = (msno) =>
  request(`/subscriber-real/analysis?msno=${encodeURIComponent(msno)}`);
export const predictReal = (msno) =>
  request(`/predict-real?msno=${encodeURIComponent(msno)}`, { method: "POST" });

/** Analysis for whichever mode is active. */
export const getAnalysisFor = (msno, mode) =>
  mode === "real" ? getRealAnalysis(msno) : getSubscriberAnalysis(msno);

/** Sample IDs for whichever mode is active. */
export const getSampleFor = (mode, n = 3) =>
  mode === "real" ? getRealSample(n) : getSampleSubscribers(n);

/** Partial subscriber search for the Predict experience. */
export async function searchSubscribersFor(mode, query, limit = 6) {
  const q = String(query ?? "").trim();
  if (q.length < 3) return [];
  if (mode === "real") {
    const d = await listPopulation({ query: q, limit, offset: 0, sort: "probability_desc" });
    return (d?.rows ?? []).map((r) => r.msno).filter(Boolean);
  }
  const d = await listSubscribers({ query: q, limit, offset: 0 });
  return (d?.rows ?? []).map((r) => r.msno).filter(Boolean);
}

export const probPct = (v) =>
  v === null || v === undefined || Number.isNaN(v) ? DASH : `${(v * 100).toFixed(2)}%`;

/** Seconds -> a compact human duration. Input is a real measured value. */
export const secs = (v) => {
  if (v === null || v === undefined || Number.isNaN(v)) return DASH;
  const h = v / 3600;
  if (Math.abs(h) >= 1) return `${h.toFixed(1)} h`;
  const m = v / 60;
  if (Math.abs(m) >= 1) return `${m.toFixed(0)} min`;
  return `${Number(v).toFixed(0)} s`;
};

export const getRealModelDetails = () => request("/model-details/real");

/* --------------------------------------------- real population analytics
 * Served from the offline batch artifact. No inference happens per request.
 */
export const getPopulationAnalytics = () => request("/population/analytics");
export const getPopulationFilters = () => request("/population/filters");
export const getPopulationAtRisk = (limit = 12) =>
  request(`/population/at-risk?limit=${limit}`);

export function listPopulation({
  query = "",
  risk = "",
  autoRenew = "",
  activity = "",
  minProbability = null,
  maxProbability = null,
  sort = "probability_desc",
  limit = 25,
  offset = 0,
} = {}) {
  const qs = new URLSearchParams();
  if (query) qs.set("query", query);
  if (risk) qs.set("risk", risk);
  if (autoRenew) qs.set("auto_renew", autoRenew);
  if (activity) qs.set("activity", activity);
  if (minProbability !== null && minProbability !== "")
    qs.set("min_probability", String(minProbability));
  if (maxProbability !== null && maxProbability !== "")
    qs.set("max_probability", String(maxProbability));
  qs.set("sort", sort);
  qs.set("limit", String(limit));
  qs.set("offset", String(offset));
  return request(`/population/subscribers?${qs.toString()}`);
}
