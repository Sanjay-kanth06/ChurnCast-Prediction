"use client";

import { createContext, useContext, useEffect, useState } from "react";
import Sidebar from "./Sidebar";
import { getHealth, getRealHealth } from "../../lib/api";

/**
 * Application frame and data-mode context.
 *
 * Mode is global and sticky because the two populations share nothing: a
 * synthetic subscriber ID is meaningless to the real model and vice versa.
 * Keeping it in one place is what stops a page rendering a real probability
 * against a synthetic profile.
 *
 * Both services are polled independently, so a page can tell the difference
 * between "the API is down" and "the real model is not registered yet".
 */

const ModeContext = createContext({
  mode: "synthetic",
  setMode: () => {},
  status: null,
});

export const useMode = () => useContext(ModeContext);

const STORAGE_KEY = "churncast.mode";

export default function Shell({ children }) {
  const [mode, setModeState] = useState("synthetic");
  const [status, setStatus] = useState({
    checked: false,
    synthetic: { online: false, ready: false, version: null },
    real: { online: false, ready: false, version: null, modelName: null },
  });

  // Restore the previous choice. Wrapped because storage throws in some
  // privacy modes, and a remembered tab is not worth a crashed shell.
  useEffect(() => {
    try {
      const saved = window.localStorage.getItem(STORAGE_KEY);
      if (saved === "real" || saved === "synthetic") setModeState(saved);
    } catch {
      /* ignore */
    }
  }, []);

  const setMode = (next) => {
    setModeState(next);
    try {
      window.localStorage.setItem(STORAGE_KEY, next);
    } catch {
      /* ignore */
    }
  };

  useEffect(() => {
    let cancelled = false;

    const check = async () => {
      const [syn, real] = await Promise.allSettled([getHealth(), getRealHealth()]);
      if (cancelled) return;
      setStatus({
        checked: true,
        synthetic:
          syn.status === "fulfilled"
            ? {
                online: true,
                ready: syn.value.model_available === true,
                version: syn.value.model_version ?? null,
              }
            : { online: false, ready: false, version: null },
        real:
          real.status === "fulfilled"
            ? {
                online: true,
                ready: real.value.model_available === true,
                version: real.value.model_version ?? null,
                modelName: real.value.model_name ?? null,
                subscribers: real.value.subscribers ?? null,
                cutoff: real.value.observation_cutoff ?? null,
              }
            : { online: false, ready: false, version: null, modelName: null },
      });
    };

    check();
    const timer = setInterval(check, 30000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  return (
    <ModeContext.Provider value={{ mode, setMode, status }}>
      <div className="app">
        <Sidebar mode={mode} setMode={setMode} status={status} />
        <div className="main">{children}</div>
      </div>
    </ModeContext.Provider>
  );
}
