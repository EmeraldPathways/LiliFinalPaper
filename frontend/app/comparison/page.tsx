import { ComparisonExplorer } from "@/components/ComparisonExplorer";
import { getSetup } from "@/lib/api";

export default async function ComparisonPage() {
  const setup = await getSetup();
  const userIds = setup?.summary?.evaluated_user_ids ?? setup?.summary?.sample_user_ids ?? [];

  return (
    <div className="stack">
      <section className="card">
        <span className="eyebrow">Model Comparison</span>
        <h2 className="section-title">Collaborative Filtering vs Agentic AI</h2>
        <p className="muted">
          The benchmark returns products from the formal SVD leave-one-out experiment. The agentic
          model applies the saved 3-agent ranking test on the same evaluated users and candidate
          pools.
        </p>
      </section>
      <ComparisonExplorer userIds={userIds} benchmarkLabel={setup?.benchmark ?? "SVD Matrix Factorisation"} />
    </div>
  );
}
