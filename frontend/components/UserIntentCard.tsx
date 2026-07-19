import type { UserIntent } from "@/lib/api";

function Pills({ values }: { values: string[] }) {
  return (
    <div>
      {values.map((value) => (
        <span className="pill" key={value}>
          {value}
        </span>
      ))}
    </div>
  );
}

export function UserIntentCard({ intent }: { intent: UserIntent }) {
  return (
    <section className="card stack">
      <div>
        <span className="eyebrow">User Profile</span>
        <h2 className="section-title">{intent.user_id}</h2>
        <p className="muted">{intent.inferred_intent}</p>
      </div>
      <div>
        <strong>Preferred Categories</strong>
        <Pills values={intent.preferred_categories} />
      </div>
      <div>
        <strong>Preferred Product Types</strong>
        <Pills values={intent.preferred_product_types} />
      </div>
      <div>
        <strong>Preferred Colours</strong>
        <Pills values={intent.preferred_colours} />
      </div>
      <div>
        <strong>Preferred Appearance</strong>
        <Pills values={intent.preferred_appearance} />
      </div>
      <div>
        <strong>Shopping Context</strong>
        <p className="muted">{intent.shopping_context}</p>
      </div>
    </section>
  );
}

