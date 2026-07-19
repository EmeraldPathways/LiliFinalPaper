import { ChartCard } from "@/components/ChartCard";
import { MetricCard } from "@/components/MetricCard";
import { getSetup } from "@/lib/api";

export default async function DataProcessingPage() {
  const setup = await getSetup();
  const summary = setup?.summary;

  return (
    <div className="stack">
      <section className="card">
        <span className="eyebrow">Preprocessing</span>
        <h2 className="section-title">Interaction Sample and Metadata Profile</h2>
        <p className="muted">
          Transactions are merged with article metadata, filtered for usable interactions, sampled,
          and split into historical train data and future test data.
        </p>
      </section>

      <section className="grid three">
        <MetricCard label="Interactions" value={String(summary?.sample_size ?? 0)} />
        <MetricCard label="Users" value={String(summary?.distinct_users ?? 0)} />
        <MetricCard label="Products" value={String(summary?.distinct_products ?? 0)} />
      </section>

      <section className="grid three">
        <MetricCard label="Train Size" value={String(summary?.train_size ?? 0)} />
        <MetricCard label="Test Size" value={String(summary?.test_size ?? 0)} />
        <MetricCard
          label="Split Boundary"
          value={summary?.split_boundary_date ?? "Not available"}
          detail="Last date included in training data"
        />
      </section>

      <section className="grid three">
        <MetricCard
          label="Repeat-User Ratio"
          value={summary ? `${(summary.repeat_user_ratio * 100).toFixed(1)}%` : "0%"}
          detail="Share of sampled users with at least two interactions"
        />
        <MetricCard
          label="Avg/User"
          value={String(summary?.average_interactions_per_user ?? 0)}
          detail="Average interactions per sampled user"
        />
        <MetricCard
          label="Evaluated Users"
          value={String(summary?.evaluated_users ?? 0)}
          detail="Users included in the offline comparison cohort"
        />
      </section>

      <section className="grid two">
        <ChartCard title="Top Product Groups" data={summary?.top_product_groups ?? []} />
        <ChartCard title="Top Product Types" data={summary?.top_product_types ?? []} color="#826f4f" />
      </section>

      <section className="grid two">
        <ChartCard title="Top Colours" data={summary?.top_colours ?? []} color="#c46e3b" />
        <ChartCard title="Top Appearances" data={summary?.top_appearances ?? []} color="#4d7b94" />
      </section>

      <section className="grid two">
        <article className="card">
          <strong>EDA interpretation</strong>
          <p className="muted">
            This page shows whether the sampled offline environment is dense enough to support both
            the CF baseline and the agentic layer. Repeat-user coverage and average interactions
            per user matter because both models depend on historical behaviour rather than isolated
            one-off transactions.
          </p>
          <p className="muted">
            The product-group, product-type, colour, and appearance charts also make the agentic
            feature space visible, which helps explain how the system infers user intention and
            reasons over candidates later in the pipeline.
          </p>
        </article>
        <article className="card">
          <strong>Experimental traceability</strong>
          <p className="muted">
            The preprocessing stage merges transaction history with article metadata, removes sparse
            noise, preserves time order, and then applies a leave-one-out split per evaluated
            customer. That keeps the formal SVD versus 3-agent comparison aligned with a realistic
            next-item recommendation setting and avoids information leakage.
          </p>
        </article>
      </section>
    </div>
  );
}
