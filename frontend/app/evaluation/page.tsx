import { ChartCard } from "@/components/ChartCard";
import { MetricCard } from "@/components/MetricCard";
import { ModelComparisonTable } from "@/components/ModelComparisonTable";
import { getMetrics } from "@/lib/api";

export default async function EvaluationPage() {
  const metrics = await getMetrics();

  if (!metrics) {
    return <section className="card muted">Run the backend experiment to generate evaluation metrics.</section>;
  }

  const baseline = metrics.svd_matrix_factorization;
  const metricChart = [
    { label: "Hit Rate@10", value: metrics.agentic_ai_framework.hit_rate_at_10 },
    { label: "NDCG@10", value: metrics.agentic_ai_framework.ndcg_at_10 ?? 0 },
    {
      label: "Intra-list Diversity@10",
      value: metrics.agentic_ai_framework.intra_list_diversity_at_10 ?? 0,
    },
  ];
  const hitRateDelta =
    metrics.agentic_ai_framework.hit_rate_at_10 - (baseline?.hit_rate_at_10 ?? 0);
  const ndcgDelta = (metrics.agentic_ai_framework.ndcg_at_10 ?? 0) - (baseline?.ndcg_at_10 ?? 0);
  const diversityDelta =
    (metrics.agentic_ai_framework.intra_list_diversity_at_10 ?? 0) -
    (baseline?.intra_list_diversity_at_10 ?? 0);

  return (
    <div className="stack">
      <section className="card">
        <span className="eyebrow">Research Result</span>
        <h2 className="section-title">Offline Comparative Evaluation</h2>
        <p className="muted">
          This stage reports an offline comparative experiment on the H&amp;M public dataset. The
          benchmark is traditional collaborative filtering, while the proposed model is an agentic
          AI recommendation framework that transparentises user-need understanding through user
          shopping intention understanding, product retrieval, recommendation reasoning,
          recommendation explanation, and feedback adaptation.
        </p>
        <p className="muted">
          Both methods generate Top-10 recommendations under the same leave-one-out split and
          shared candidate pools. They are compared through Hit Rate@10, NDCG@10, and Intra-list
          Diversity@10 across {metrics.evaluated_users} evaluated users.
        </p>
      </section>
      <section className="grid three">
        <MetricCard
          label="Hit Rate@10"
          value={metrics.business_mapping.hit_rate_at_10}
          detail="Mapped from Hit Rate@10"
        />
        <MetricCard
          label="NDCG@10"
          value={metrics.business_mapping.ndcg_at_10}
          detail="Ranking quality for held-out purchases"
        />
        <MetricCard
          label="ILD@10"
          value={metrics.business_mapping.intra_list_diversity_at_10}
          detail="Assortment breadth inside each Top-10 list"
        />
      </section>
      <ModelComparisonTable metrics={metrics} />
      <section className="grid two">
        <ChartCard title="Agentic AI Metric Profile" data={metricChart} />
        <article className="card">
          <strong>Research Interpretation</strong>
          <p className="muted">
            Hit Rate@10 captures whether the held-out item appears anywhere in the final list.
            NDCG@10 adds rank sensitivity, so higher values indicate better placement of relevant
            items. Intra-list Diversity@10 shows how broad or narrow each Top-10 assortment is.
          </p>
          <p className="muted">
            In the current run, the 3-agent framework is{" "}
            {ndcgDelta > 0 ? "higher" : "not higher"} than SVD on NDCG@10 by{" "}
            {Math.abs(ndcgDelta).toFixed(3)}. That is the clearest signal of whether the agentic
            ranker improves ordering quality rather than only matching on list inclusion.
          </p>
          <p className="muted">
            Hit Rate@10 is {hitRateDelta > 0 ? "higher" : hitRateDelta < 0 ? "lower" : "unchanged"} for
            the agentic framework by {Math.abs(hitRateDelta).toFixed(3)}. Intra-list Diversity@10
            is {diversityDelta < 0 ? "lower" : "higher"} by {Math.abs(diversityDelta).toFixed(3)},
            which shows whether the ranking logic is trading behavioural relevance for narrower or
            broader recommendation lists.
          </p>
          <p className="muted">
            These are offline comparison metrics rather than live business outcomes. They are
            useful for controlled SVD-versus-agentic evaluation, not for claiming production CTR or
            conversion impact on their own.
          </p>
        </article>
      </section>
    </div>
  );
}
