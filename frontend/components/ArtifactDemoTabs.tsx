"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";

import {
  getWorkflowCases,
  type ArtifactDemoMethodItem,
  type ArtifactDemoValueCount,
  type ArtifactDemoWorkflowCase,
} from "@/lib/api";

type ArtifactDemoTabsProps = {
  workflowCases: ArtifactDemoWorkflowCase[];
};

type TabId = "svd" | "agentic" | "hybrid";

const tabs: Array<{ id: TabId; label: string }> = [
  { id: "svd", label: "SVD" },
  { id: "agentic", label: "3-Agent" },
  { id: "hybrid", label: "Hybrid" },
];

function formatNumber(value: number | null | undefined, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "Not available";
  }
  return Number.isInteger(value) ? String(value) : value.toFixed(digits);
}

function formatBoolean(value: boolean | null | undefined) {
  if (value === null || value === undefined) {
    return "Not available";
  }
  return value ? "Yes" : "No";
}

function compactValueCounts(items: ArtifactDemoValueCount[]) {
  if (!items.length) {
    return "Not available";
  }
  return items.map((item) => `${item.value} (${item.count})`).join(", ");
}

function calculatePercentages(items: ArtifactDemoValueCount[]) {
  if (!items.length) {
    return [];
  }
  const maxCount = Math.max(...items.map((item) => item.count));
  if (!maxCount) {
    return items.map((item) => ({ ...item, percentage: 0 }));
  }
  return items.map((item) => ({
    ...item,
    percentage: Math.round((item.count / maxCount) * 100),
  }));
}

function MetadataDefinition({
  label,
  value,
}: {
  label: string;
  value: string | null | undefined;
}) {
  return (
    <div className="artifact-definition">
      <span className="artifact-definition-label">{label}</span>
      <span className="artifact-definition-value">{value || "Not available"}</span>
    </div>
  );
}

function AgenticMetaBox({
  label,
  value,
}: {
  label: string;
  value: string | null | undefined;
}) {
  return (
    <div className="artifact-agentic-meta-box">
      <span className="artifact-agentic-meta-label">{label}</span>
      <strong className="artifact-agentic-meta-value">{value || "Not available"}</strong>
    </div>
  );
}

function ScorePill({
  label,
  value,
}: {
  label: string;
  value: number | null | undefined;
}) {
  return (
    <div className="artifact-score-pill">
      <span>{label}</span>
      <strong>{formatNumber(value)}</strong>
    </div>
  );
}

function PreferenceChipList({
  label,
  values,
}: {
  label: string;
  values: string[];
}) {
  return (
    <article className="artifact-profile-panel">
      <span className="artifact-profile-label">{label}</span>
      <div className="artifact-chip-row">
        {values.length ? (
          values.map((value) => (
            <span key={`${label}-${value}`} className="artifact-chip">
              {value}
            </span>
          ))
        ) : (
          <span className="artifact-chip artifact-chip-muted">Not available</span>
        )}
      </div>
    </article>
  );
}

function UserSummaryPanel({
  workflowCase,
  method,
  compact = false,
}: {
  workflowCase: ArtifactDemoWorkflowCase;
  method: TabId;
  compact?: boolean;
}) {
  const methodHit =
    method === "svd"
      ? workflowCase.method_hits.svd
      : method === "agentic"
        ? workflowCase.method_hits.agentic
        : workflowCase.method_hits.hybrid;

  const methodLabel = method === "svd" ? "SVD" : method === "agentic" ? "3-Agent" : "Hybrid";

  return (
    <article className={compact ? "artifact-summary-card artifact-summary-card-compact" : "card artifact-shell"}>
      <span className="eyebrow">{compact ? "Selected User" : "Selected Real Demo User"}</span>
      <strong className="artifact-user-name">{workflowCase.demo_user.label}</strong>
      <div className={compact ? "artifact-compact-grid" : "artifact-context-grid"}>
        <MetadataDefinition label="Customer ID" value={workflowCase.demo_user.customer_id_short} />
        <MetadataDefinition
          label="Training History"
          value={String(workflowCase.leave_one_out.training_history_count)}
        />
        <MetadataDefinition label="Ground Truth" value={workflowCase.leave_one_out.ground_truth_article_id} />
        <MetadataDefinition
          label="Candidate Pool"
          value={String(workflowCase.leave_one_out.candidate_pool_size)}
        />
      </div>
      <div className="artifact-badge-row">
        <span className="artifact-badge">
          Ground truth in pool: {formatBoolean(workflowCase.leave_one_out.ground_truth_in_candidate_pool)}
        </span>
        <span className={methodHit ? "artifact-badge artifact-badge-hit" : "artifact-badge"}>
          {methodLabel} {methodHit ? "Hit at Top-10" : "Miss at Top-10"}
        </span>
      </div>
    </article>
  );
}

function AgenticExplanation({
  score,
  reason,
  fallback,
}: {
  score?: number | null;
  reason?: string | null;
  fallback: string;
}) {
  if (!reason) {
    return <p className="muted artifact-card-copy">{fallback}</p>;
  }

  return (
    <p className="muted artifact-card-copy">
      {score !== null && score !== undefined ? `Score ${formatNumber(score)}. ` : ""}
      {reason}
    </p>
  );
}

function EvidenceProductCard({ item }: { item: ArtifactDemoMethodItem }) {
  return (
    <article className="artifact-rec-card artifact-rec-card-agentic artifact-rec-card-evidence">
      <div className="artifact-rec-topline artifact-rec-topline-start">
        <div className="artifact-agentic-rank-stack">
          <span className="artifact-rank">Evidence Candidate</span>
          <h4 className="artifact-rec-title">
            {item.product_type_name || "Not available"} · {item.article_id}
          </h4>
        </div>
        <div className="artifact-match-badge artifact-match-badge-emerald">
          <span>Matches</span>
          <strong>{item.matched_evidence.length}</strong>
        </div>
      </div>
      <div className="artifact-agentic-meta-grid">
        <AgenticMetaBox label="Type" value={item.product_type_name || "Not available"} />
        <AgenticMetaBox label="Group" value={item.product_group_name} />
        <AgenticMetaBox label="Colour" value={item.colour_group_name} />
        <AgenticMetaBox label="Appearance" value={item.graphical_appearance_name} />
      </div>
      <AgenticExplanation reason={item.reason} fallback="Evidence explanation not available" />
      {item.matched_evidence.length ? (
        <div className="artifact-chip-row artifact-chip-row-spaced">
          {item.matched_evidence.map((evidence) => (
            <span key={`${item.article_id}-${evidence}`} className="artifact-chip artifact-chip-evidence">
              {evidence}
            </span>
          ))}
        </div>
      ) : (
        <p className="muted artifact-card-copy">No matched evidence fields available.</p>
      )}
    </article>
  );
}

function AgenticEvidenceCard({ item }: { item: ArtifactDemoMethodItem }) {
  return <EvidenceProductCard item={item} />;
}

function DecisionRecommendationCard({ item }: { item: ArtifactDemoMethodItem }) {
  return (
    <article className="artifact-rec-card artifact-rec-card-agentic">
      <div className="artifact-rec-topline artifact-rec-topline-start">
        <div className="artifact-agentic-rank-stack">
          <span className="artifact-rank">Rank #{item.rank ?? "NA"}</span>
          <h4 className="artifact-rec-title">
            {item.product_type_name || "Not available"} · {item.article_id}
          </h4>
        </div>
        {item.score !== null && item.score !== undefined ? (
          <div className="artifact-match-badge">
            <span>Match Score</span>
            <strong>{formatNumber(item.score)}</strong>
          </div>
        ) : null}
      </div>
      <div className="artifact-agentic-meta-grid">
        <AgenticMetaBox label="Type" value={item.product_type_name || "Not available"} />
        <AgenticMetaBox label="Group" value={item.product_group_name} />
        <AgenticMetaBox label="Colour" value={item.colour_group_name} />
        <AgenticMetaBox label="Appearance" value={item.graphical_appearance_name} />
        
      </div>
      <AgenticExplanation score={item.score} reason={item.reason} fallback="Score explanation not available" />
      {item.matched_evidence.length ? (
        <div className="artifact-chip-row artifact-chip-row-spaced">
          {item.matched_evidence.map((evidence) => (
            <span key={`${item.article_id}-${evidence}`} className="artifact-chip artifact-chip-evidence">
              {evidence}
            </span>
          ))}
        </div>
      ) : (
        <p className="muted artifact-card-copy">No matched evidence fields available.</p>
      )}
    </article>
  );
}

function SvdRecommendationCard({ item }: { item: ArtifactDemoMethodItem }) {
  return (
    <article className="artifact-rec-card">
      <div className="artifact-rec-topline">
        <span className="artifact-rank">Rank {item.rank ?? "NA"}</span>
      </div>
      <h4 className="artifact-rec-title">{item.article_id}</h4>
      <div className="artifact-score-row">
        <ScorePill label="SVD score" value={item.score} />
      </div>
      <div className="artifact-meta-grid">
        <MetadataDefinition label="Type" value={item.product_type_name} />
        <MetadataDefinition label="Group" value={item.product_group_name} />
        <MetadataDefinition label="Colour" value={item.colour_group_name} />
        <MetadataDefinition label="Appearance" value={item.graphical_appearance_name} />
      </div>
    </article>
  );
}

function HybridRecommendationCard({
  item,
  hasAudit,
}: {
  item: ArtifactDemoMethodItem;
  hasAudit: boolean;
}) {
  return (
    <article className={hasAudit ? "artifact-rec-card artifact-rec-card-selected" : "artifact-rec-card"}>
      <div className="artifact-rec-topline">
        <span className="artifact-rank">Rank {item.rank ?? "NA"}</span>
        {item.is_ground_truth ? <span className="artifact-badge artifact-badge-hit">Ground Truth</span> : null}
      </div>
      <h4 className="artifact-rec-title">
        {item.product_type_name || "Not available"} · {item.article_id}
      </h4>
      <div className="artifact-score-row">
        <ScorePill label="Hybrid score" value={item.hybrid_score ?? item.score} />
        <ScorePill label="SVD score" value={item.normalized_svd_score} />
        <ScorePill label="Agentic score" value={item.normalized_agentic_score} />
        <ScorePill label="Diversity bonus" value={item.diversity_bonus} />
      </div>
      <div className="artifact-meta-grid">
        <MetadataDefinition label="Group" value={item.product_group_name} />
        <MetadataDefinition label="Colour" value={item.colour_group_name} />
        <MetadataDefinition label="Appearance" value={item.graphical_appearance_name} />
      </div>
      {item.reason ? <p className="muted artifact-card-copy">{item.reason}</p> : null}
      <span className={hasAudit ? "artifact-badge artifact-badge-hybrid" : "artifact-badge artifact-badge-muted"}>
        {hasAudit ? "Explanation available" : "Hybrid-ranked item"}
      </span>
    </article>
  );
}

function RecommendationSection({
  title,
  eyebrow,
  intro,
  items,
  mode,
  hybridAuditArticleId,
}: {
  title: string;
  eyebrow?: string;
  intro?: string;
  items: ArtifactDemoMethodItem[];
  mode: TabId;
  hybridAuditArticleId?: string | null;
}) {
  return (
    <section className="card artifact-shell">
      <div className="artifact-section-header">
        {eyebrow ? <span className="artifact-eyebrow artifact-eyebrow-fuchsia">{eyebrow}</span> : null}
        <h3>{title}</h3>
        {intro ? <p>{intro}</p> : null}
      </div>
      <div className={mode === "svd" ? "artifact-rec-grid" : "artifact-rec-grid artifact-rec-grid-agentic"}>
        {items.length ? (
          items.map((item) => {
            if (mode === "agentic") {
              return (
                <DecisionRecommendationCard
                  key={`agentic-${item.article_id}-${item.rank ?? 0}`}
                  item={item}
                />
              );
            }
            if (mode === "hybrid") {
              return (
                <HybridRecommendationCard
                  key={`hybrid-${item.article_id}-${item.rank ?? 0}`}
                  item={item}
                  hasAudit={hybridAuditArticleId === item.article_id}
                />
              );
            }
            return (
              <SvdRecommendationCard
                key={`svd-${item.article_id}-${item.rank ?? 0}`}
                item={item}
              />
            );
          })
        ) : (
          <div className="artifact-empty-state">Saved recommendations unavailable.</div>
        )}
      </div>
    </section>
  );
}

function FlowNode({
  title,
  body,
  layer,
  source,
  accent = false,
}: {
  title: string;
  body: string;
  layer: string;
  source?: string;
  accent?: boolean;
}) {
  return (
    <article className={accent ? "artifact-flow-node-box artifact-flow-node-box-accent" : "artifact-flow-node-box"}>
      <div className="artifact-flow-node-head">
        <span className="artifact-flow-layer">{layer}</span>
        {source ? <span className="artifact-flow-source">{source}</span> : null}
      </div>
      <span className="artifact-flow-label">{title}</span>
      <p>{body}</p>
    </article>
  );
}

function MethodHeaderCard({
  eyebrow,
  title,
  description,
  eyebrowColor = "cyan",
}: {
  eyebrow: string;
  title: string;
  description: string;
  eyebrowColor?: "cyan" | "emerald" | "fuchsia" | "warm";
}) {
  const colorClass =
    eyebrowColor === "cyan"
      ? "artifact-eyebrow-cyan"
      : eyebrowColor === "emerald"
        ? "artifact-eyebrow-emerald"
        : eyebrowColor === "fuchsia"
          ? "artifact-eyebrow-fuchsia"
          : "artifact-eyebrow-warm";

  return (
    <article className="card artifact-shell artifact-method-header">
      <div className="artifact-section-header">
        <span className={`artifact-eyebrow ${colorClass}`}>{eyebrow}</span>
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </article>
  );
}

function PreferenceBarChart({
  title,
  items,
}: {
  title: string;
  items: ArtifactDemoValueCount[];
}) {
  const data = useMemo(() => calculatePercentages(items), [items]);

  return (
    <article className="artifact-profile-panel">
      <span className="artifact-profile-label">{title}</span>
      {data.length ? (
        <div className="artifact-bar-chart">
          {data.map((item) => (
            <div key={item.value} className="artifact-bar-item">
              <div className="artifact-bar-labels">
                <span>{item.value}</span>
                <span>{item.percentage}%</span>
              </div>
              <div className="artifact-bar-track">
                <div className="artifact-bar-fill" style={{ width: `${Math.max(item.percentage, 2)}%` }} />
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="artifact-empty-state" style={{ padding: 18 }}>
          No preference data available for this dimension.
        </div>
      )}
    </article>
  );
}

function ArtifactHero({
  selectedCaseId,
  cases,
  activeTab,
  onSelectCase,
  onSelectTab,
}: {
  selectedCaseId: string;
  cases: ArtifactDemoWorkflowCase[];
  activeTab: TabId;
  onSelectCase: (value: string) => void;
  onSelectTab: (tab: TabId) => void;
}) {
  return (
    <section className="artifact-hero">
      <div className="artifact-hero-top">
        <div className="artifact-hero-copy">
          <span className="artifact-eyebrow artifact-eyebrow-cyan">
            OFFLINE H&amp;M RECOMMENDER ARTEFACT
          </span>
          <h1 className="artifact-hero-title">H&amp;M Hybrid SVD + 3-Agent Recommendation Artefact</h1>
        </div>

        <div className="artifact-hero-right">
          <select
            className="artifact-hero-select"
            value={selectedCaseId}
            onChange={(event) => onSelectCase(event.target.value)}
            disabled={!cases.length}
          >
            {cases.map((item) => (
              <option key={item.demo_user.customer_id} value={item.demo_user.customer_id}>
                {item.demo_user.label} - {item.demo_user.customer_id_short}
              </option>
            ))}
          </select>

          <div className="artifact-tabbar" role="tablist" aria-label="Artefact methods">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                type="button"
                role="tab"
                aria-selected={activeTab === tab.id}
                className={activeTab === tab.id ? "artifact-tab artifact-tab-active" : "artifact-tab"}
                onClick={() => onSelectTab(tab.id)}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

function SectionHeader({
  eyebrow,
  title,
  subtitle,
  eyebrowColor = "cyan",
  action,
}: {
  eyebrow: string;
  title: string;
  subtitle?: string;
  eyebrowColor?: "cyan" | "emerald" | "fuchsia" | "warm";
  action?: ReactNode;
}) {
  const colorClass =
    eyebrowColor === "cyan"
      ? "artifact-eyebrow-cyan"
      : eyebrowColor === "emerald"
        ? "artifact-eyebrow-emerald"
        : eyebrowColor === "fuchsia"
          ? "artifact-eyebrow-fuchsia"
          : "artifact-eyebrow-warm";

  return (
    <div className="artifact-profile-header" style={{ marginBottom: 0 }}>
      <div className="artifact-section-header" style={{ marginBottom: 0 }}>
        <span className={`artifact-eyebrow ${colorClass}`}>{eyebrow}</span>
        <h3>{title}</h3>
        {subtitle ? <p>{subtitle}</p> : null}
      </div>
      {action ? <div>{action}</div> : null}
    </div>
  );
}

export function ArtifactDemoTabs({ workflowCases }: ArtifactDemoTabsProps) {
  const [cases, setCases] = useState(workflowCases);
  const [isLoadingCases, setIsLoadingCases] = useState(workflowCases.length === 0);
  const [activeTab, setActiveTab] = useState<TabId>("hybrid");
  const [selectedCaseId, setSelectedCaseId] = useState(workflowCases[0]?.demo_user.customer_id ?? "");

  useEffect(() => {
    setCases(workflowCases);
    setIsLoadingCases(false);
  }, [workflowCases]);

  useEffect(() => {
    if (workflowCases.length) {
      return;
    }

    let cancelled = false;

    async function loadCases() {
      setIsLoadingCases(true);
      const payload = await getWorkflowCases();
      if (cancelled) {
        return;
      }
      setCases(payload?.cases ?? []);
      setIsLoadingCases(false);
    }

    void loadCases();

    return () => {
      cancelled = true;
    };
  }, [workflowCases]);

  useEffect(() => {
    if (!cases.length) {
      setSelectedCaseId("");
      return;
    }
    setSelectedCaseId((current) => {
      if (current && cases.some((item) => item.demo_user.customer_id === current)) {
        return current;
      }
      return cases[0].demo_user.customer_id;
    });
  }, [cases]);

  const selectedCase = useMemo(
    () => cases.find((item) => item.demo_user.customer_id === selectedCaseId) ?? cases[0] ?? null,
    [selectedCaseId, cases],
  );

  const evidencePreviewItems = selectedCase?.agentic_top10.slice(0, 9) ?? [];
  const matchedPreferenceFields = selectedCase?.hybrid_explainability.matched_preference_fields ?? [];
  const selectedHybridAuditItem = useMemo(() => {
    if (!selectedCase) {
      return null;
    }
    return (
      selectedCase.hybrid_top10.find(
        (item) => item.article_id === selectedCase.hybrid_explainability.article_id,
      ) ?? selectedCase.hybrid_top10[0] ?? null
    );
  }, [selectedCase]);
  const selectedSvdItem = useMemo(() => {
    if (!selectedCase || !selectedHybridAuditItem) {
      return null;
    }
    return selectedCase.svd_top10.find((item) => item.article_id === selectedHybridAuditItem.article_id) ?? null;
  }, [selectedCase, selectedHybridAuditItem]);
  const selectedAgenticItem = useMemo(() => {
    if (!selectedCase || !selectedHybridAuditItem) {
      return null;
    }
    return (
      selectedCase.agentic_top10.find((item) => item.article_id === selectedHybridAuditItem.article_id) ??
      null
    );
  }, [selectedCase, selectedHybridAuditItem]);

  return (
    <div className="artifact-page artifact-one-page">
      <ArtifactHero
        selectedCaseId={selectedCaseId}
        cases={cases}
        activeTab={activeTab}
        onSelectCase={setSelectedCaseId}
        onSelectTab={setActiveTab}
      />

      {!selectedCase && isLoadingCases ? (
        <section className="card artifact-shell muted">Loading saved artifact cases...</section>
      ) : null}

      {!selectedCase && !isLoadingCases ? (
        <section className="card artifact-shell muted">
          Saved artefact cases are unavailable. Check that the formal seed99 saved artifacts are present.
        </section>
      ) : null}

      {selectedCase && activeTab === "svd" ? (
        <div className="stack">
          <section className="card artifact-shell artifact-method-header">
            <div className="artifact-section-header">
              <span className="artifact-eyebrow artifact-eyebrow-warm">SVD Baseline</span>
              <h3>SVD Matrix Factorisation Baseline</h3>
              <p>
                SVD is the formal collaborative filtering baseline. It uses user-item purchase interactions
                to produce behavioural recommendation scores for candidate items.
              </p>
            </div>
          </section>

          <div className="grid two">
            <UserSummaryPanel workflowCase={selectedCase} method="svd" compact />
          </div>

          <div className="grid two">
            <article className="card artifact-shell artifact-signal-card">
              <div className="artifact-section-header">
                <span className="artifact-eyebrow artifact-eyebrow-warm">SVD Signal</span>
                <h3>Behavioural Interaction Pattern</h3>
                <p>SVD uses behavioural interaction patterns only. It does not inspect item explanations or agentic evidence.</p>
              </div>
              <div className="artifact-meta-grid artifact-meta-grid-tight">
                <MetadataDefinition label="Ground Truth Article" value={selectedCase.ground_truth.article_id} />
                <MetadataDefinition label="Type" value={selectedCase.ground_truth.product_type_name} />
                <MetadataDefinition label="Group" value={selectedCase.ground_truth.product_group_name} />
                <MetadataDefinition label="Colour" value={selectedCase.ground_truth.colour_group_name} />
              </div>
            </article>
          </div>

          <RecommendationSection
            eyebrow="SVD Top-10"
            title="SVD Top-10 Recommendations"
            intro="Candidate-pool-restricted behavioural recommendations ranked by SVD score."
            items={selectedCase.svd_top10}
            mode="svd"
          />
        </div>
      ) : null}

      {selectedCase && activeTab === "agentic" ? (
        <div className="stack">
          <div className="grid three">
            <MethodHeaderCard
              eyebrow="1. Preference Agent"
              title="User Preference Profile"
              description="Analyze transaction history and produce a preference profile."
              eyebrowColor="cyan"
            />
            <MethodHeaderCard
              eyebrow="2. Evidence Agent"
              title="Candidate Evidence Set"
              description="Retrieve candidate products and explain matched evidence."
              eyebrowColor="emerald"
            />
            <MethodHeaderCard
              eyebrow="3. Decision Agent"
              title="Final Recommendations"
              description="Score, validate constraints, and output top recommendations."
              eyebrowColor="fuchsia"
            />
          </div>

          <div className="grid two">
            <UserSummaryPanel workflowCase={selectedCase} method="agentic" compact />
          </div>

          <section className="card artifact-shell">
            <SectionHeader
              eyebrow="Preference Agent"
              title="User Preference Profile"
              subtitle={selectedCase.preference_agent.inferred_intent || undefined}
              eyebrowColor="cyan"
              action={
                <div className="artifact-profile-stat">
                  <span>Transactions Analysed</span>
                  <strong>{selectedCase.leave_one_out.training_history_count}</strong>
                </div>
              }
            />
            <div className="artifact-chip-row artifact-chip-row-spaced">
              {selectedCase.demo_user.label ? (
                <span className="artifact-tag">{selectedCase.demo_user.label}</span>
              ) : null}
              {selectedCase.preference_agent.preferred_product_types.map((value) => (
                <span key={`type-${value}`} className="artifact-tag">
                  Often buys {value}
                </span>
              ))}
              {selectedCase.preference_agent.preferred_categories.map((value) => (
                <span key={`group-${value}`} className="artifact-tag">
                  {value}
                </span>
              ))}
              {selectedCase.preference_agent.preferred_colours.map((value) => (
                <span key={`colour-${value}`} className="artifact-tag">
                  {value}
                </span>
              ))}
              {selectedCase.preference_agent.preferred_appearance.map((value) => (
                <span key={`appearance-${value}`} className="artifact-tag">
                  {value}
                </span>
              ))}
            </div>
            <div className="artifact-profile-grid" style={{ marginTop: 18 }}>
              <PreferenceBarChart
                title="Preferred Product Types"
                items={selectedCase.preference_agent.frequent_product_types}
              />
              <PreferenceBarChart
                title="Preferred Product Groups"
                items={selectedCase.preference_agent.frequent_product_groups}
              />
              <PreferenceBarChart
                title="Preferred Colours"
                items={selectedCase.preference_agent.frequent_colours}
              />
              <PreferenceBarChart
                title="Preferred Appearances"
                items={selectedCase.preference_agent.frequent_graphical_appearances}
              />
            </div>
          </section>

          <section className="card artifact-shell">
            <div className="artifact-section-header">
              <span className="artifact-eyebrow artifact-eyebrow-emerald">Evidence Agent</span>
              <h3>Candidate Evidence Set</h3>
              <p>
                Retrieved {evidencePreviewItems.length} catalog candidates by matching soft preference signals
                against product metadata.
              </p>
            </div>
            <div className="artifact-evidence-grid">
              {evidencePreviewItems.length ? (
                evidencePreviewItems.map((item) => (
                  <AgenticEvidenceCard key={`evidence-${item.article_id}-${item.rank ?? 0}`} item={item} />
                ))
              ) : (
                <div className="artifact-empty-state">No evidence candidates available.</div>
              )}
            </div>
          </section>

          <RecommendationSection
            eyebrow="Decision Agent"
            title="Final Recommendations"
            intro="Ranked the evidence set with transparent weighted scoring and returned the top 10 items."
            items={selectedCase.agentic_top10}
            mode="agentic"
          />
        </div>
      ) : null}

      {selectedCase && activeTab === "hybrid" ? (
        <div className="stack">
          <section className="card artifact-shell">
            <div className="artifact-section-header">
              <span className="artifact-eyebrow artifact-eyebrow-cyan">End-to-End Path</span>
              <h3>Hybrid Recommendation Path</h3>
              <p>How the selected user moves from history data to explainable Hybrid Top-10 recommendations.</p>
            </div>
            <div className="artifact-flow-grid artifact-flow-grid-six">
              <FlowNode
                layer="Layer 1"
                title="Input"
                body="Selected H&M user, training history, held-out item, and 100-item candidate pool."
                source="Evaluation Base"
              />
              <FlowNode
                layer="Layer 2"
                title="SVD Signal"
                body="Saved SVD score ranks candidate items using behavioural collaborative filtering."
                accent
                source="SVD Artifact"
              />
              <FlowNode
                layer="Layer 3"
                title="3-Agent Signal"
                body="Preference, evidence, and decision signals score items using user history and metadata."
                source="3-Agent Evidence Signal"
              />
              <FlowNode
                layer="Layer 4"
                title="Hybrid Reranker"
                body="Combines SVD score, agentic score, and diversity bonus."
                accent
                source="Hybrid Formula"
              />
              <FlowNode
                layer="Layer 5"
                title="Hybrid Top-10"
                body="Outputs the final ranked Top-10 recommendation list."
                source="Hybrid Top-10 Output"
              />
              <FlowNode
                layer="Layer 6"
                title="Explainability Audit"
                body="Checks groundedness, evidence traceability, rank shift, and score components."
                source="Explainability Audit Layer"
              />
            </div>
          </section>

          <div className="grid two">
            <UserSummaryPanel workflowCase={selectedCase} method="hybrid" compact />
          </div>

          <section className="artifact-hybrid-signal-grid">
            <article className="card artifact-shell">
              <div className="artifact-section-header">
                <span className="artifact-eyebrow artifact-eyebrow-warm">SVD Signal</span>
                <h3>SVD Behavioural Signal</h3>
                <p>This signal comes from saved SVD recommendations and reflects behavioural collaborative filtering relevance.</p>
              </div>
              <div className="artifact-score-stack artifact-score-stack-compact">
                <ScorePill label="SVD Rank" value={selectedCase.hybrid_explainability.svd_rank} />
                <ScorePill
                  label="Normalized SVD"
                  value={selectedCase.hybrid_explainability.score_components.normalized_svd_score}
                />
                <ScorePill label="SVD Score" value={selectedSvdItem?.score} />
              </div>
              <div className="artifact-chip-row artifact-chip-row-spaced">
                <span className="artifact-chip">
                  Ground truth in pool: {formatBoolean(selectedCase.leave_one_out.ground_truth_in_candidate_pool)}
                </span>
                {selectedSvdItem?.product_type_name ? (
                  <span className="artifact-chip">{selectedSvdItem.product_type_name}</span>
                ) : null}
                {selectedSvdItem?.product_group_name ? (
                  <span className="artifact-chip">{selectedSvdItem.product_group_name}</span>
                ) : null}
                {!selectedSvdItem ? <span className="artifact-chip artifact-chip-muted">Not available</span> : null}
              </div>
            </article>

            <article className="card artifact-shell">
              <div className="artifact-section-header">
                <span className="artifact-eyebrow artifact-eyebrow-emerald">3-Agent Signal</span>
                <h3>3-Agent Evidence Signal</h3>
                <p>This signal comes from the structured 3-Agent workflow: Preference Agent, Evidence Agent, and Decision Agent.</p>
              </div>
              <div className="artifact-score-stack artifact-score-stack-compact">
                <ScorePill
                  label="Normalized Agentic"
                  value={selectedCase.hybrid_explainability.score_components.normalized_agentic_score}
                />
                <ScorePill label="Agentic Score" value={selectedAgenticItem?.score} />
              </div>
              <div className="artifact-chip-row artifact-chip-row-spaced">
                {selectedCase.hybrid_explainability.item_metadata.product_type_name ? (
                  <span className="artifact-chip">{selectedCase.hybrid_explainability.item_metadata.product_type_name}</span>
                ) : null}
                {selectedCase.hybrid_explainability.item_metadata.product_group_name ? (
                  <span className="artifact-chip">{selectedCase.hybrid_explainability.item_metadata.product_group_name}</span>
                ) : null}
                {selectedCase.hybrid_explainability.item_metadata.colour_group_name ? (
                  <span className="artifact-chip">{selectedCase.hybrid_explainability.item_metadata.colour_group_name}</span>
                ) : null}
                {selectedCase.hybrid_explainability.item_metadata.graphical_appearance_name ? (
                  <span className="artifact-chip">{selectedCase.hybrid_explainability.item_metadata.graphical_appearance_name}</span>
                ) : null}
                {matchedPreferenceFields.length ? (
                  matchedPreferenceFields.map((field, index) => (
                    <span key={`matched-${index}`} className="artifact-chip">
                      {String(field.field ?? field.item_value ?? "match")}
                    </span>
                  ))
                ) : (
                  <span className="artifact-chip artifact-chip-muted">Not available</span>
                )}
              </div>
              {selectedAgenticItem?.reason ? (
                <p className="muted artifact-card-copy artifact-signal-footnote">{selectedAgenticItem.reason}</p>
              ) : null}
            </article>

            <article className="card artifact-shell artifact-formula-card-dark">
              <div className="artifact-section-header">
                <span className="artifact-eyebrow artifact-eyebrow-cyan">Hybrid Formula</span>
                <h3>Score Combination</h3>
              </div>
              <div className="artifact-formula-stack">
                <strong>hybrid_score =</strong>
                <span>0.70 × normalized_svd_score</span>
                <span>0.25 × normalized_agentic_score</span>
                <span>0.05 × diversity_bonus</span>
              </div>
              <div className="artifact-score-stack artifact-score-stack-compact">
                <ScorePill
                  label="Normalized SVD"
                  value={selectedCase.hybrid_explainability.score_components.normalized_svd_score}
                />
                <ScorePill
                  label="Normalized Agentic"
                  value={selectedCase.hybrid_explainability.score_components.normalized_agentic_score}
                />
                <ScorePill
                  label="Diversity Bonus"
                  value={selectedCase.hybrid_explainability.score_components.diversity_bonus}
                />
                <ScorePill
                  label="Hybrid Score"
                  value={selectedCase.hybrid_explainability.score_components.hybrid_score}
                />
              </div>
            </article>
          </section>

          <RecommendationSection
            eyebrow="Hybrid Top-10"
            title="Hybrid Top-10 Recommendations"
            intro="Final ranked recommendations produced by combining behavioural relevance, agentic evidence, and diversity adjustment."
            items={selectedCase.hybrid_top10}
            mode="hybrid"
            hybridAuditArticleId={selectedCase.hybrid_explainability.article_id}
          />

          <section className="grid two artifact-hybrid-audit-grid">
            <article className="card artifact-shell">
              <div className="artifact-section-header">
                <span className="artifact-eyebrow artifact-eyebrow-fuchsia">Explainability Audit</span>
                <h3>Read-Only Explanation Evidence</h3>
                <p>
                  The explainability layer is a read-only audit of Hybrid recommendations. It does not
                  generate new recommendations and does not change ranking metrics.
                </p>
              </div>
              <div className="artifact-meta-grid">
                <MetadataDefinition label="Selected Article" value={selectedCase.hybrid_explainability.article_id} />
                <MetadataDefinition label="Groundedness" value={selectedCase.hybrid_explainability.groundedness_status} />
                <MetadataDefinition
                  label="Hybrid Rank"
                  value={
                    selectedCase.hybrid_explainability.hybrid_rank !== null &&
                    selectedCase.hybrid_explainability.hybrid_rank !== undefined
                      ? String(selectedCase.hybrid_explainability.hybrid_rank)
                      : "Not available"
                  }
                />
                <MetadataDefinition
                  label="SVD Rank"
                  value={
                    selectedCase.hybrid_explainability.svd_rank !== null &&
                    selectedCase.hybrid_explainability.svd_rank !== undefined
                      ? String(selectedCase.hybrid_explainability.svd_rank)
                      : "Not available"
                  }
                />
                <MetadataDefinition
                  label="Rank Shift"
                  value={
                    selectedCase.hybrid_explainability.rank_shift !== null &&
                    selectedCase.hybrid_explainability.rank_shift !== undefined
                      ? String(selectedCase.hybrid_explainability.rank_shift)
                      : "Not available"
                  }
                />
                <MetadataDefinition
                  label="Hit Status"
                  value={selectedCase.hybrid_explainability.is_ground_truth ? "Ground-truth hit" : "Not available"}
                />
                <MetadataDefinition
                  label="Preference Trace"
                  value={matchedPreferenceFields.length ? "Available" : "Not available"}
                />
                <MetadataDefinition
                  label="Product Name"
                  value={selectedCase.hybrid_explainability.item_metadata.prod_name}
                />
              </div>
              <div className="artifact-chip-row artifact-chip-row-spaced">
                {selectedCase.hybrid_explainability.item_metadata.product_type_name ? (
                  <span className="artifact-chip">{selectedCase.hybrid_explainability.item_metadata.product_type_name}</span>
                ) : null}
                {selectedCase.hybrid_explainability.item_metadata.product_group_name ? (
                  <span className="artifact-chip">{selectedCase.hybrid_explainability.item_metadata.product_group_name}</span>
                ) : null}
                {selectedCase.hybrid_explainability.item_metadata.colour_group_name ? (
                  <span className="artifact-chip">{selectedCase.hybrid_explainability.item_metadata.colour_group_name}</span>
                ) : null}
                {selectedCase.hybrid_explainability.item_metadata.graphical_appearance_name ? (
                  <span className="artifact-chip">{selectedCase.hybrid_explainability.item_metadata.graphical_appearance_name}</span>
                ) : null}
                {matchedPreferenceFields.length ? (
                  matchedPreferenceFields.map((field, index) => (
                    <span key={`${selectedCase.hybrid_explainability.article_id}-${index}`} className="artifact-chip">
                      {String(field.field ?? field.item_value ?? "match")}
                    </span>
                  ))
                ) : (
                  <span className="artifact-chip artifact-chip-muted">Matched preference fields not available</span>
                )}
              </div>
              {selectedCase.hybrid_explainability.explanation_text ? (
                <p className="muted artifact-card-copy artifact-card-copy-roomy">
                  {selectedCase.hybrid_explainability.explanation_text}
                </p>
              ) : null}
            </article>
          </section>

          <section className="card artifact-shell artifact-limitation-card">
            <p className="muted artifact-card-copy-small">
              This is an offline recommendation artefact. It demonstrates ranking behaviour and
              source-grounded explanation evidence using saved H&amp;M experiment artifacts. It does not
              claim CTR, CVR, conversion, live customer engagement, add-to-cart, dwell time, or live
              feedback adaptation.
            </p>
          </section>
        </div>
      ) : null}
    </div>
  );
}
