import { MetricCard } from "@/components/MetricCard";
import { getSetup } from "@/lib/api";

export default async function ResearchSetupPage() {
  const setup = await getSetup();

  return (
    <div className="stack">
      <section className="card">
        <span className="eyebrow">Experiment Design</span>
        <h2 className="section-title">Offline Comparative Research Setup</h2>
        <p className="muted">
          The benchmark is collaborative filtering. The proposed method is an agentic AI
          recommendation framework implemented as a structured decision-making layer above the
          recommendation system.
        </p>
      </section>
      <section className="grid three">
        <MetricCard label="Dataset" value={setup?.dataset ?? "Unavailable"} />
        <MetricCard label="Benchmark" value={setup?.benchmark ?? "Collaborative Filtering"} />
        <MetricCard
          label="Proposed Model"
          value={setup?.proposed_framework ?? "Agentic AI Recommendation Framework"}
        />
      </section>
      <section className="grid two">
        <article className="card">
          <strong>Core Design</strong>
          <p className="muted">Task: Top-N recommendation on the configured offline evaluation split.</p>
          <p className="muted">Split: {setup?.split_method ?? "Unavailable"}.</p>
          <p className="muted">
            Metrics: {(setup?.evaluation_metrics ?? ["Hit Rate@10", "Preference Alignment", "Diversity"]).join(", ")}
          </p>
        </article>
        <article className="card">
          <strong>Academic Framing</strong>
          <p className="muted">
            Higher Hit Rate@10 suggests stronger CTR potential. Higher preference alignment suggests
            stronger CVR potential. Higher diversity suggests deeper engagement potential.
          </p>
        </article>
      </section>
    </div>
  );
}
