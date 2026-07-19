from __future__ import annotations

import json

import pandas as pd

from app.services.cf_service import CollaborativeFilteringService


def test_cf_excludes_user_history_and_returns_top_n(isolated_env, sample_interactions: pd.DataFrame):
    train_df = sample_interactions.iloc[:9].copy()
    service = CollaborativeFilteringService(isolated_env)
    matrix = service._build_user_item_matrix(train_df)
    metadata = service._article_metadata(train_df)
    user_history = train_df.groupby("customer_id")["article_id"].agg(set).to_dict()

    results = service.recommend_for_user("u1", matrix, metadata, user_history)

    assert results
    assert all(item["article_id"] not in {"a1", "a2", "a3"} for item in results)
    assert results[0]["model"] == "collaborative_filtering"


def test_build_svd_top10_debug_baseline_outputs_valid_ranked_recommendations(isolated_env):
    isolated_env.top_n = 2
    service = CollaborativeFilteringService(isolated_env)
    processed = pd.DataFrame(
        [
            ["u1", "a1", "2024-01-01", 10.0, 1, "Top", "Upper", "Solid", "Black", "Jersey", "Dept", "Sec", "Idx", None],
            ["u1", "a2", "2024-01-02", 10.0, 1, "Top", "Upper", "Solid", "Black", "Jersey", "Dept", "Sec", "Idx", None],
            ["u1", "a3", "2024-01-03", 10.0, 1, "Top", "Upper", "Solid", "Black", "Jersey", "Dept", "Sec", "Idx", None],
            ["u2", "a2", "2024-01-01", 10.0, 1, "Top", "Upper", "Solid", "Black", "Jersey", "Dept", "Sec", "Idx", None],
            ["u2", "a3", "2024-01-02", 10.0, 1, "Top", "Upper", "Solid", "Black", "Jersey", "Dept", "Sec", "Idx", None],
            ["u2", "a4", "2024-01-03", 10.0, 1, "Dress", "Full", "Patterned", "Red", "Dresses", "Dept", "Sec", "Idx", None],
            ["u3", "a1", "2024-01-01", 10.0, 1, "Top", "Upper", "Solid", "Black", "Jersey", "Dept", "Sec", "Idx", None],
            ["u3", "a4", "2024-01-02", 10.0, 1, "Dress", "Full", "Patterned", "Red", "Dresses", "Dept", "Sec", "Idx", None],
            ["u3", "a5", "2024-01-03", 10.0, 1, "Shirt", "Upper", "Solid", "Blue", "Jersey", "Dept", "Sec", "Idx", None],
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
    subset = [
        {
            "customer_id": "u1",
            "train_article_ids": ["a1", "a2"],
            "ground_truth_article_id": "a3",
            "candidate_pool_article_ids": ["a3", "a4", "a5"],
            "candidate_pool_size": 3,
        },
        {
            "customer_id": "u2",
            "train_article_ids": ["a2", "a3"],
            "ground_truth_article_id": "a4",
            "candidate_pool_article_ids": ["a1", "a4", "a5"],
            "candidate_pool_size": 3,
        },
    ]
    isolated_env.evaluation_base_table_svd_top10_100_json_path.write_text(
        json.dumps(subset, indent=2),
        encoding="utf-8",
    )

    report = service.build_svd_top10_debug_baseline(n_components=5, random_state=42)

    assert report["evaluated_users_requested"] == 2
    assert report["users_with_svd_recommendations"] == 2
    assert report["users_with_full_top_10"] == 2
    assert report["users_where_top_10_all_inside_candidate_pool"] == 2
    assert report["users_where_top_10_contains_training_items"] == 0

    payload = json.loads(isolated_env.svd_recommendations_top10_100_json_path.read_text(encoding="utf-8"))
    assert len(payload) == 2
    assert payload[0]["method"] == "svd_matrix_factorisation"
    assert payload[0]["top_10_all_inside_candidate_pool"] is True
    assert payload[0]["top_10_contains_training_items"] is False
