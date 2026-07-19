from __future__ import annotations

import json

import pandas as pd

from app.services.agentic_service import AgenticRecommendationService


def test_agentic_scoring_uses_expected_fields(isolated_env, sample_interactions: pd.DataFrame):
    train_df = sample_interactions.iloc[:9].copy()
    service = AgenticRecommendationService(isolated_env)
    user_profile = {
        "user_id": "u1",
        "inferred_intent": "casual daily clothing",
        "preferred_categories": ["Garment Upper body"],
        "preferred_product_types": ["Shirt", "Top"],
        "preferred_colours": ["White", "Beige", "Blue"],
        "preferred_appearance": ["Solid", "Plain"],
        "shopping_context": "daily wear",
    }
    candidates = service.retrieve_candidate_products(user_profile, train_df)

    scored = service.score_candidates(user_profile, candidates, train_df)

    assert scored
    assert {"intent_match", "preference_alignment", "product_relevance", "diversity", "behavioural_signal"} <= set(scored[0])
    assert scored[0]["model"] == "agentic_ai_framework"


def test_feedback_updates_weights(isolated_env):
    service = AgenticRecommendationService(isolated_env)
    weights = service.adapt_from_feedback("u1", "a1", "add_to_cart")

    assert weights["product_type_weight"] > 1.0
    assert weights["appearance_weight"] > 1.0


def test_formal_agentic_top10_experiment_respects_candidate_pool_and_train_exclusions(isolated_env):
    processed = pd.DataFrame(
        [
            ["u1", "a1", "2024-01-01", 0.1, 1, "Shirt", "Upper", "Solid", "Blue", "Knitwear", "Dept", "Section", "Index", ""],
            ["u1", "a2", "2024-01-02", 0.1, 1, "Top", "Upper", "Plain", "White", "Jersey", "Dept", "Section", "Index", "Soft cotton top"],
            ["u2", "a3", "2024-01-03", 0.1, 1, "Dress", "Full", "Patterned", "Red", "Dresses", "Dept", "Section", "Index", ""],
            ["u2", "a4", "2024-01-04", 0.1, 1, "Shirt", "Upper", "Solid", "Blue", "Knitwear", "Dept", "Section", "Index", "Oxford shirt"],
            ["u2", "a5", "2024-01-05", 0.1, 1, "Top", "Upper", "Plain", "White", "Jersey", "Dept", "Section", "Index", "Sleeveless top"],
            ["u2", "a6", "2024-01-06", 0.1, 1, "Trousers", "Lower", "Solid", "Black", "Trousers", "Dept", "Section", "Index", "Relaxed trousers"],
            ["u2", "a7", "2024-01-07", 0.1, 1, "Blouse", "Upper", "Plain", "White", "Blouses", "Dept", "Section", "Index", "Light blouse"],
            ["u2", "a8", "2024-01-08", 0.1, 1, "Cardigan", "Upper", "Melange", "Grey", "Knitwear", "Dept", "Section", "Index", "Grey cardigan"],
            ["u2", "a9", "2024-01-09", 0.1, 1, "Skirt", "Lower", "Solid", "Black", "Skirts", "Dept", "Section", "Index", "Black skirt"],
            ["u2", "a10", "2024-01-10", 0.1, 1, "Jacket", "Upper", "Solid", "Blue", "Outdoor", "Dept", "Section", "Index", "Blue jacket"],
            ["u2", "a11", "2024-01-11", 0.1, 1, "Vest top", "Upper", "Plain", "White", "Jersey", "Dept", "Section", "Index", "White vest top"],
            ["u2", "a12", "2024-01-12", 0.1, 1, "Hoodie", "Upper", "Solid", "Blue", "Sweats", "Dept", "Section", "Index", "Blue hoodie"],
        ],
        columns=[
            "customer_id",
            "article_id",
            "t_dat",
            "price",
            "sales_channel_id",
            "product_type_name",
            "product_group_name",
            "graphical_appearance_name",
            "colour_group_name",
            "garment_group_name",
            "department_name",
            "section_name",
            "index_name",
            "detail_desc",
        ],
    )
    processed.to_csv(isolated_env.processed_interactions_with_articles_csv_path, index=False)
    isolated_env.evaluation_base_table_svd_top10_100_json_path.write_text(
        json.dumps(
            [
                {
                    "customer_id": "u1",
                    "train_article_ids": ["a1", "a2"],
                    "ground_truth_article_id": "a3",
                    "candidate_pool_article_ids": ["a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10", "a11", "a12"],
                    "candidate_pool_size": 10,
                }
            ]
        ),
        encoding="utf-8",
    )

    report = AgenticRecommendationService(isolated_env).build_top10_formal_experiment()
    payload = json.loads(isolated_env.agentic_recommendations_top10_100_json_path.read_text(encoding="utf-8"))

    assert report["users_with_agentic_recommendations"] == 1
    assert report["users_with_full_top_10"] == 1
    assert report["users_where_top_10_all_inside_candidate_pool"] == 1
    assert report["users_where_top_10_contains_training_items"] == 0
    assert report["users_where_detail_desc_missing_but_handled"] == 1
    assert payload[0]["method"] == "3_agent"
    assert payload[0]["recommendation_count"] == 10
    assert all(
        item["article_id"] in {"a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10", "a11", "a12"}
        for item in payload[0]["top_10_recommendations"]
    )

