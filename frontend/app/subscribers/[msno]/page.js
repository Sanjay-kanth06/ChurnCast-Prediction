"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { getAnalysisFor, MODES } from "../../../lib/api";
import { TopBar, Notice, Skeleton } from "../../components/ui";
import Analysis from "../../components/Analysis";
import { useMode } from "../../components/Shell";

export default function SubscriberDetailPage() {
  const { msno } = useParams();
  const router = useRouter();
  const { mode } = useMode();
  const id = decodeURIComponent(Array.isArray(msno) ? msno[0] : msno ?? "");

  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!id) return;
    setData(null);
    setError(null);
    getAnalysisFor(id, mode)
      .then(setData)
      .catch((e) => setError(e.message));
  }, [id, mode]);

  return (
    <>
      <TopBar
        title={
          <span className="mono" style={{ wordBreak: "break-all" }}>
            {id || "Subscriber"}
          </span>
        }
        subtitle="Full model analysis and personalized retention plan"
        right={
          <>
            <span className="chip">{MODES[mode].dataLabel}</span>
            <button
              className="btn-ghost"
              onClick={() => router.push("/subscribers")}
            >
              &larr; Back
            </button>
          </>
        }
      />

      <div className="content">
        {error ? (
          <Notice kind="err" title="No analysis available">
            {error}
          </Notice>
        ) : !data ? (
          <div className="card card-pad">
            <Skeleton w="28%" h={15} mb={18} />
            <Skeleton w="58%" h={44} mb={18} />
            <Skeleton w="90%" />
            <Skeleton w="76%" />
            <Skeleton w="82%" />
          </div>
        ) : (
          <Analysis data={data} />
        )}
      </div>
    </>
  );
}
