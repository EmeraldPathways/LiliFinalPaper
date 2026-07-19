import type { MetricsResponse } from "@/lib/api";

function formatValue(value: number | null | undefined) {
  return typeof value === "number" ? value.toFixed(3) : "n/a";
}

export function ModelComparisonTable({ metrics }: { metrics: MetricsResponse }) {
  const baseline = metrics.svd_matrix_factorization;

  return (
    <section className="card">
      <strong>Comparative Result Table</strong>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Hit Rate@10</th>
              <th>NDCG@10</th>
              <th>Intra-list Diversity@10</th>
              <th>Hits</th>
              <th>Misses</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>SVD Matrix Factorisation</td>
              <td>{formatValue(baseline?.hit_rate_at_10)}</td>
              <td>{formatValue(baseline?.ndcg_at_10)}</td>
              <td>{formatValue(baseline?.intra_list_diversity_at_10)}</td>
              <td>{baseline?.hits_count ?? "n/a"}</td>
              <td>{baseline?.miss_count ?? "n/a"}</td>
            </tr>
            <tr>
              <td>3-Agent Agentic AI Framework</td>
              <td>{formatValue(metrics.agentic_ai_framework.hit_rate_at_10)}</td>
              <td>{formatValue(metrics.agentic_ai_framework.ndcg_at_10)}</td>
              <td>{formatValue(metrics.agentic_ai_framework.intra_list_diversity_at_10)}</td>
              <td>{metrics.agentic_ai_framework.hits_count ?? "n/a"}</td>
              <td>{metrics.agentic_ai_framework.miss_count ?? "n/a"}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  );
}
