type MetricCardProps = {
  label: string;
  value: string;
  detail?: string;
};

export function MetricCard({ label, value, detail }: MetricCardProps) {
  return (
    <article className="card">
      <span className="eyebrow">{label}</span>
      <div className="metric-value">{value}</div>
      {detail ? <p className="muted">{detail}</p> : null}
    </article>
  );
}

