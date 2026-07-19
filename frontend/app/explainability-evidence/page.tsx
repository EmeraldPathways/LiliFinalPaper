import { MetricCard } from "@/components/MetricCard";
import { getExplainabilityEvidence, type ExplainabilityExampleRow } from "@/lib/api";

function formatRate(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatValue(value: unknown) {
  if (value === null || value === undefined || value === "") {
    return "n/a";
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(3);
  }
  return String(value);
}

function parseMatchedEvidence(row: ExplainabilityExampleRow) {
  try {
    const parsed = JSON.parse(row.matched_preference_fields_json) as Array<{
      field?: string;
      item_value?: string;
      matched_user_history_count?: number;
    }>;
    if (!parsed.length) {
      return "No traced match";
    }
    return parsed
      .map((item) => `${item.field}: ${item.item_value} (${item.matched_user_history_count})`)
      .join("; ");
  } catch {
    return "No traced match";
  }
}

export default async function ExplainabilityEvidencePage() {
  const payload = await getExplainabilityEvidence();

  if (!payload) {
    return (
      <section className="card muted">
        Explainability artifacts are missing. Run the backend explainability audit script first.
      </section>
    );
  }

  const metrics = payload.summary.summary_metrics;
  const caseStudy = payload.case_study;

  return (
    <div className="stack">
      <section className="card">
        <span className="eyebrow">Explainability Audit</span>
        <h2 className="section-title">Explainability Evidence</h2>
        <p className="muted">
          This page audits whether the Hybrid SVD + 3-Agent reranker provides source-grounded
          recommendation evidence beyond SVD score-based ranking. It does not evaluate live CTR,
          CVR or customer engagement.
        </p>
        <p className="muted">{payload.summary.interpretation.safe_claim}</p>
      </section>

      <section className="grid three">
        <MetricCard label="Evidence Coverage" value={formatRate(metrics.evidence_coverage_rate)} />
        <MetricCard label="Preference Trace Rate" value={formatRate(metrics.preference_trace_rate)} />
        <MetricCard
          label="Score Component Coverage"
          value={formatRate(metrics.score_component_coverage_rate)}
        />
        <MetricCard label="Groundedness Rate" value={formatRate(metrics.groundedness_rate)} />
        <MetricCard label="Rank-shift Coverage" value={formatRate(metrics.rank_shift_coverage_rate)} />
        <MetricCard
          label="Ungrounded Claim Count"
          value={String(metrics.ungrounded_claim_count)}
          detail="Lower is better"
        />
      </section>

      <section className="card">
        <strong>Trade-off Warning</strong>
        <p className="muted">
          Hybrid improves explainability and remains competitive on ranking quality, but it reduces
          intra-list diversity compared with SVD.
        </p>
      </section>

      <section className="card">
        <strong>Explanation Examples</strong>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Customer</th>
                <th>Article</th>
                <th>Hybrid Rank</th>
                <th>SVD Rank</th>
                <th>Rank Shift</th>
                <th>Product Group</th>
                <th>Colour</th>
                <th>Matched Evidence</th>
                <th>Explanation</th>
              </tr>
            </thead>
            <tbody>
              {payload.examples.map((row) => (
                <tr key={`${row.customer_id}-${row.article_id}-${row.hybrid_rank}`}>
                  <td>{row.customer_id}</td>
                  <td>{row.article_id}</td>
                  <td>{row.hybrid_rank}</td>
                  <td>{formatValue(row.svd_rank)}</td>
                  <td>{formatValue(row.rank_shift)}</td>
                  <td>{row.product_group_name ?? "n/a"}</td>
                  <td>{row.colour_group_name ?? "n/a"}</td>
                  <td>{parseMatchedEvidence(row)}</td>
                  <td>{row.explanation_text}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {caseStudy ? (
        <section className="grid two">
          <article className="card">
            <strong>Case Study</strong>
            <p className="muted">{caseStudy.explanation_text}</p>
            <p className="muted">
              Customer: {caseStudy.customer_id}
              <br />
              Recommended Article: {caseStudy.recommended_article_id}
              <br />
              Ground-truth Hit: {caseStudy.is_ground_truth ? "Yes" : "No"}
              <br />
              SVD Rank: {formatValue(caseStudy.svd_rank)}
              <br />
              Hybrid Rank: {formatValue(caseStudy.hybrid_rank)}
              <br />
              Rank Shift: {formatValue(caseStudy.rank_shift)}
            </p>
            {caseStudy.limitation_note ? <p className="muted">{caseStudy.limitation_note}</p> : null}
          </article>
          <article className="card">
            <strong>Case Study Evidence</strong>
            <pre className="trace-pre">
              {JSON.stringify(
                {
                  user_history_summary: caseStudy.user_history_summary,
                  item_metadata: caseStudy.item_metadata,
                  score_components: caseStudy.score_components,
                  matched_preference_fields: caseStudy.matched_preference_fields,
                },
                null,
                2,
              )}
            </pre>
          </article>
        </section>
      ) : null}

      {payload.warnings.length ? (
        <section className="card">
          <strong>Warnings</strong>
          {payload.warnings.map((warning) => (
            <p className="muted" key={warning}>
              {warning}
            </p>
          ))}
        </section>
      ) : null}
    </div>
  );
}
