from __future__ import annotations

import pandas as pd

from app.services.evaluation_service import EvaluationService


def test_evaluation_metrics_from_fixture(isolated_env, sample_interactions: pd.DataFrame):
    train_df = sample_interactions.iloc[:9].copy()
    test_df = sample_interactions.iloc[9:].copy()
    cf_recommendations = {
        "u1": [
            {
                "article_id": "a4",
                "product_name": "Oxford Shirt",
                "product_type": "Shirt",
                "product_group": "Garment Upper body",
                "colour": "Blue",
                "appearance": "Solid",
                "score": 0.8,
                "model": "collaborative_filtering",
            }
        ]
    }
    agentic_recommendations = {
        "u1": [
            {
                "article_id": "a4",
                "product_name": "Oxford Shirt",
                "product_type": "Shirt",
                "product_group": "Garment Upper body",
                "colour": "Blue",
                "appearance": "Solid",
                "score": 0.91,
                "model": "agentic_ai_framework",
                "reason": "Recommended because the blue shirt fits the user's upper-body preference and solid casual style.",
            }
        ]
    }

    metrics = EvaluationService(isolated_env).evaluate(
        train_df, test_df, cf_recommendations, agentic_recommendations
    )

    assert metrics["collaborative_filtering"]["hit_rate_at_10"] == 1.0
    assert metrics["agentic_ai_framework"]["preference_alignment"] >= 0.75

