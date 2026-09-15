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
