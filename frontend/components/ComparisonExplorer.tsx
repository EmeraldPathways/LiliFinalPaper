"use client";

import { useEffect, useState } from "react";

import { AgentProcessPanel } from "@/components/AgentProcessPanel";
import { RecommendationTable } from "@/components/RecommendationTable";
import type { RecommendationComparison } from "@/lib/api";
import { getRecommendationComparison } from "@/lib/api";

type ComparisonExplorerProps = {
  userIds: string[];
  benchmarkLabel: string;
};

export function ComparisonExplorer({ userIds, benchmarkLabel }: ComparisonExplorerProps) {
  const [selectedUser, setSelectedUser] = useState(userIds[0] ?? "");
  const [comparison, setComparison] = useState<RecommendationComparison | null>(null);
  const [error, setError] = useState<string>("");
  const displayedStages = comparison?.agentic_process.slice(0, 3) ?? [];

  useEffect(() => {
    if (!selectedUser) {
      return;
    }
    getRecommendationComparison(selectedUser)
      .then((data) => {
        setComparison(data);
        setError("");
      })
      .catch((fetchError) => {
        setComparison(null);
        setError(fetchError instanceof Error ? fetchError.message : "Unable to fetch recommendations.");
      });
  }, [selectedUser]);

  if (!userIds.length) {
    return <section className="card muted">Run the backend experiment to populate sample users.</section>;
  }

  return (
    <div className="stack">
      <div className="controls">
        <select value={selectedUser} onChange={(event) => setSelectedUser(event.target.value)}>
          {userIds.map((userId) => (
            <option key={userId} value={userId}>
              {userId}
            </option>
          ))}
        </select>
      </div>
      {error ? <section className="card muted">{error}</section> : null}
      {comparison ? (
        <div className="stack">
          <div className="grid two">
            <RecommendationTable
              title={benchmarkLabel}
              userId={comparison.user_id}
              items={comparison.cf_recommendations}
            />
            <RecommendationTable
              title="3-Agent Agentic AI Recommendation Framework"
              userId={comparison.user_id}
              items={comparison.agentic_recommendations}
              showReasons
            />
          </div>
          <section className="card">
            <span className="eyebrow">3-Agent Trace</span>
            <h3 className="section-title">What the formal agentic pipeline returned for this user</h3>
            <p className="muted">
              This trace exposes the saved intention, retrieval, and ranking steps behind the final
              agentic recommendations for the formal evaluation run.
            </p>
          </section>
          <AgentProcessPanel stages={displayedStages} />
        </div>
      ) : null}
    </div>
  );
}
