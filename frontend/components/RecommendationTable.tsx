"use client";

import { useState } from "react";

import type { AgenticRecommendationItem, RecommendationItem } from "@/lib/api";
import { sendFeedback } from "@/lib/api";

type RecommendationTableProps = {
  title: string;
  userId: string;
  items: RecommendationItem[] | AgenticRecommendationItem[];
  showReasons?: boolean;
};

const feedbackOptions = [
  ["click", "Click"],
  ["add_to_cart", "Add to Cart"],
  ["ignore", "Ignore"],
  ["purchase", "Purchase"],
] as const;

export function RecommendationTable({
  title,
  userId,
  items,
  showReasons = false,
}: RecommendationTableProps) {
  const [status, setStatus] = useState<string>("");

  async function handleFeedback(articleId: string, feedbackType: (typeof feedbackOptions)[number][0]) {
    try {
      const response = await sendFeedback(userId, articleId, feedbackType);
      setStatus(response.message);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Unable to save feedback.");
    }
  }

  return (
    <section className="card">
      <strong>{title}</strong>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Product</th>
              <th>Type</th>
              <th>Score</th>
              {showReasons ? <th>Reason</th> : null}
              {showReasons ? <th>Feedback</th> : null}
            </tr>
          </thead>
          <tbody>
            {items.map((item) => {
              const reason = "reason" in item ? item.reason : "";
              return (
                <tr key={item.article_id}>
                  <td>
                    <strong>{item.product_name}</strong>
                    <div className="muted">{item.article_id}</div>
                  </td>
                  <td>
                    {item.product_type}
                    <div className="muted">{item.colour} | {item.appearance}</div>
                  </td>
                  <td>{item.score.toFixed(3)}</td>
                  {showReasons ? <td>{reason}</td> : null}
                  {showReasons ? (
                    <td>
                      <div className="controls">
                        {feedbackOptions.map(([value, label]) => (
                          <button key={value} type="button" onClick={() => handleFeedback(item.article_id, value)}>
                            {label}
                          </button>
                        ))}
                      </div>
                    </td>
                  ) : null}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {status ? <p className="muted">{status}</p> : null}
    </section>
  );
}
