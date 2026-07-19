from __future__ import annotations

import json

import pandas as pd

from app.services.agentic_service import AgenticRecommendationService
from app.services.cf_service import CollaborativeFilteringService
from app.services.hybrid_service import HybridRecommendationService


def test_hybrid_service_builds_valid_top10_outputs(isolated_env):
    processed = pd.DataFrame(
        [
            ["u1", "a1", "2024-01-01", 0.1, 1, "Shirt", "Upper", "Solid", "Blue", "Knitwear", "Dept", "Section", "Index", "desc"],
            ["u1", "a2", "2024-01-02", 0.1, 1, "Top", "Upper", "Plain", "White", "Jersey", "Dept", "Section", "Index", "desc"],
            ["u2", "a3", "2024-01-03", 0.1, 1, "Dress", "Full", "Patterned", "Red", "Dresses", "Dept", "Section", "Index", "desc"],
            ["u2", "a4", "2024-01-04", 0.1, 1, "Skirt", "Lower", "Solid", "Black", "Skirts", "Dept", "Section", "Index", "desc"],
            ["u2", "a5", "2024-01-05", 0.1, 1, "Blouse", "Upper", "Plain", "White", "Blouses", "Dept", "Section", "Index", "desc"],
            ["u2", "a6", "2024-01-06", 0.1, 1, "Trousers", "Lower", "Check", "Grey", "Trousers", "Dept", "Section", "Index", "desc"],
            ["u2", "a7", "2024-01-07", 0.1, 1, "Vest top", "Upper", "Solid", "Black", "Jersey", "Dept", "Section", "Index", "desc"],
            ["u2", "a8", "2024-01-08", 0.1, 1, "Cardigan", "Upper", "Melange", "Grey", "Knitwear", "Dept", "Section", "Index", "desc"],
            ["u2", "a9", "2024-01-09", 0.1, 1, "Jacket", "Upper", "Solid", "Blue", "Outerwear", "Dept", "Section", "Index", "desc"],
            ["u2", "a10", "2024-01-10", 0.1, 1, "Shorts", "Lower", "Plain", "White", "Trousers", "Dept", "Section", "Index", "desc"],
            ["u2", "a11", "2024-01-11", 0.1, 1, "Hoodie", "Upper", "Solid", "Green", "Sweats", "Dept", "Section", "Index", "desc"],
            ["u2", "a12", "2024-01-12", 0.1, 1, "T-shirt", "Upper", "Print", "Yellow", "Jersey", "Dept", "Section", "Index", "desc"],
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
    isolated_env.evaluation_base_table_svd_top10_json_path(1).write_text(
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
    isolated_env.top_n = 10

    service = HybridRecommendationService(
        isolated_env,
        CollaborativeFilteringService(isolated_env),
        AgenticRecommendationService(isolated_env),
    )
    report = service.build_hybrid_svd_agentic_reranker(subset_size=1, random_state=42)
    payload = json.loads(isolated_env.hybrid_svd_agentic_recommendations_top10_json_path(1).read_text(encoding="utf-8"))

    assert report["users_evaluated"] == 1
    assert report["users_with_hybrid_recommendations"] == 1
    assert report["users_with_full_top_10"] == 1
    assert report["users_where_top_10_all_inside_candidate_pool"] == 1
    assert report["users_where_top_10_contains_training_items"] == 0
    assert payload[0]["method"] == "hybrid_svd_agentic_reranker"
    assert payload[0]["recommendation_count"] == 10
    assert all("normalized_svd_score" in item for item in payload[0]["top_10_recommendations"])
