"use client";

import type { ReactNode } from "react";

import type { AgentProcessStage } from "@/lib/api";

function formatLabel(value: string) {
  return value.replaceAll("_", " ");
}

function renderValue(value: unknown): ReactNode {
  if (Array.isArray(value)) {
    if (!value.length) {
      return <span className="muted">No items</span>;
    }
    if (typeof value[0] === "object" && value[0] !== null) {
      return (
        <div className="stack">
          {value.map((item, index) => (
            <pre className="trace-pre" key={index}>
              {JSON.stringify(item, null, 2)}
            </pre>
          ))}
        </div>
      );
    }
    return <p className="muted">{value.map((item) => String(item)).join(", ")}</p>;
  }

  if (value && typeof value === "object") {
    return <pre className="trace-pre">{JSON.stringify(value, null, 2)}</pre>;
  }

  return <p className="muted">{value == null ? "Not available" : String(value)}</p>;
}

export function AgentProcessPanel({ stages }: { stages: AgentProcessStage[] }) {
  if (!stages.length) {
    return <section className="card muted">Run the experiment to inspect the 3-agent process trace.</section>;
  }

  return (
    <section className="stack">
      {stages.map((stage) => (
        <article className="card stack" key={`${stage.agent}-${stage.title}`}>
          <div>
            <span className="eyebrow">{stage.agent}</span>
            <h3 className="section-title">{stage.title}</h3>
            <p className="muted">{stage.summary}</p>
          </div>
          <div className="grid two">
            {Object.entries(stage.payload).map(([key, value]) => (
              <div className="trace-block" key={key}>
                <strong>{formatLabel(key)}</strong>
                {renderValue(value)}
              </div>
            ))}
          </div>
        </article>
      ))}
    </section>
  );
}
