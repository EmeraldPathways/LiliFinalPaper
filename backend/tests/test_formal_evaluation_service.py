from __future__ import annotations

import json

import pandas as pd

from app.services.evaluation_service import EvaluationService


def test_formal_top10_metrics_include_hit_rate_ndcg_and_intra_list_diversity(
    isolated_env, sample_interactions: pd.DataFrame
):
    train_df = sample_interactions.iloc[:9].copy()
    test_df = sample_interactions.iloc[9:].copy()
    svd_recommendations = {
        "u1": [
            {
                "article_id": "a4",
                "product_name": "Oxford Shirt",
                "product_type": "Shirt",
                "product_group": "Garment Upper body",
                "colour": "Blue",
                "appearance": "Solid",
                "score": 0.92,
                "model": "svd_matrix_factorization",
            },
            {
                "article_id": "a5",
                "product_name": "Jersey Top",
                "product_type": "Top",
                "product_group": "Garment Upper body",
                "colour": "White",
                "appearance": "Patterned",
                "score": 0.64,
                "model": "svd_matrix_factorization",
            },
        ]
    }
    agentic_recommendations = {
        "u1": [
            {
                "article_id": "a5",
                "product_name": "Jersey Top",
                "product_type": "Top",
                "product_group": "Garment Upper body",
                "colour": "White",
                "appearance": "Patterned",
                "score": 0.95,
                "model": "agentic_ai_framework",
                "reason": "Recommended because the patterned top extends the user's upper-body wardrobe.",
            },
            {
                "article_id": "a4",
                "product_name": "Oxford Shirt",
                "product_type": "Shirt",
                "product_group": "Garment Upper body",
                "colour": "Blue",
                "appearance": "Solid",
                "score": 0.76,
                "model": "agentic_ai_framework",
                "reason": "Recommended because the blue shirt matches the user's solid upper-body preferences.",
            },
        ]
    }

    metrics = EvaluationService(isolated_env).evaluate_top10_experiment(
        train_df, test_df, svd_recommendations, agentic_recommendations
    )

    assert metrics["svd_matrix_factorization"]["hit_rate_at_10"] == 1.0
    assert metrics["svd_matrix_factorization"]["ndcg_at_10"] == 1.0
    assert metrics["agentic_ai_framework"]["intra_list_diversity_at_10"] > 0.0


def test_compute_formal_top10_metrics_from_saved_artifacts_writes_requested_outputs(isolated_env):
    processed = pd.DataFrame(
        [
            ["u1", "a1", "2024-01-01", 0.1, 1, "Shirt", "Upper", "Solid", "Blue", "Knitwear", "Dept", "Section", "Index", "desc"],
            ["u1", "a2", "2024-01-02", 0.1, 1, "Top", "Upper", "Plain", "White", "Jersey", "Dept", "Section", "Index", "desc"],
            ["u1", "a3", "2024-01-03", 0.1, 1, "Dress", "Full", "Patterned", "Red", "Dresses", "Dept", "Section", "Index", "desc"],
            ["u2", "a4", "2024-01-04", 0.1, 1, "Skirt", "Lower", "Solid", "Black", "Skirts", "Dept", "Section", "Index", "desc"],
            ["u2", "a5", "2024-01-05", 0.1, 1, "Blouse", "Upper", "Plain", "White", "Blouses", "Dept", "Section", "Index", "desc"],
            ["u2", "a6", "2024-01-06", 0.1, 1, "Trousers", "Lower", "Check", "Grey", "Trousers", "Dept", "Section", "Index", "desc"],
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
                    "ground_truth_article_id": "a3",
                    "candidate_pool_article_ids": ["a3", "a4", "a5", "a6"],
                    "candidate_pool_size": 4,
                }
            ]
        ),
        encoding="utf-8",
    )
    isolated_env.svd_recommendations_top10_100_json_path.write_text(
        json.dumps(
            [
                {
                    "customer_id": "u1",
                    "ground_truth_article_id": "a3",
                    "top_10_recommendations": [
                        {
                            "rank": 1,
                            "article_id": "a4",
                            "product_group_name": "Lower",
                            "colour_group_name": "Black",
                            "graphical_appearance_name": "Solid",
                        },
                        {
                            "rank": 2,
                            "article_id": "a3",
                            "product_group_name": "Full",
                            "colour_group_name": "Red",
                            "graphical_appearance_name": "Patterned",
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    isolated_env.agentic_recommendations_top10_100_json_path.write_text(
        json.dumps(
            [
                {
                    "customer_id": "u1",
                    "ground_truth_article_id": "a3",
                    "top_10_recommendations": [
                        {
                            "rank": 1,
                            "article_id": "a5",
                            "product_group_name": "Upper",
                            "colour_group_name": "White",
                            "graphical_appearance_name": "Plain",
                        },
                        {
                            "rank": 2,
                            "article_id": "a6",
                            "product_group_name": "Lower",
                            "colour_group_name": "Grey",
                            "graphical_appearance_name": "Check",
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    result = EvaluationService(isolated_env).compute_formal_top10_metrics_from_saved_artifacts(
        subset_size=100,
        bootstrap_samples=8,
        random_seed=42,
    )
    per_user = json.loads(isolated_env.per_user_metrics_top10_100_json_path.read_text(encoding="utf-8"))
    summary = json.loads(isolated_env.metric_summary_top10_100_json_path.read_text(encoding="utf-8"))
    validation = json.loads(isolated_env.metric_validation_report_top10_100_path.read_text(encoding="utf-8"))

    assert result["summary"]["svd"]["hit_rate_at_10"] == 1.0
    assert result["summary"]["agentic"]["hit_rate_at_10"] == 0.0
    assert per_user[0]["svd_ground_truth_rank"] == 2
    assert per_user[0]["svd_ndcg_at_10"] > 0.0
    assert per_user[0]["agentic_ground_truth_rank"] is None
    assert summary["evaluation_scope"]["valid_evaluated_users"] == 1
    assert validation["users_with_svd_metrics"] == 1
    assert validation["users_with_agentic_metrics"] == 1
    ci_summary = json.loads(isolated_env.metric_summary_top10_with_ci_json_path(100).read_text(encoding="utf-8"))
    bootstrap_report = json.loads(isolated_env.bootstrap_ci_report_top10_json_path(100).read_text(encoding="utf-8"))
    assert "confidence_intervals" in ci_summary
    assert bootstrap_report["bootstrap_samples"] == 8


def test_compute_three_method_top10_metrics_from_saved_artifacts_writes_hybrid_outputs(isolated_env):
    processed = pd.DataFrame(
        [
            ["u1", "a1", "2024-01-01", 0.1, 1, "Shirt", "Upper", "Solid", "Blue", "Knitwear", "Dept", "Section", "Index", "desc"],
            ["u1", "a2", "2024-01-02", 0.1, 1, "Top", "Upper", "Plain", "White", "Jersey", "Dept", "Section", "Index", "desc"],
            ["u1", "a3", "2024-01-03", 0.1, 1, "Dress", "Full", "Patterned", "Red", "Dresses", "Dept", "Section", "Index", "desc"],
            ["u2", "a4", "2024-01-04", 0.1, 1, "Skirt", "Lower", "Solid", "Black", "Skirts", "Dept", "Section", "Index", "desc"],
            ["u2", "a5", "2024-01-05", 0.1, 1, "Blouse", "Upper", "Plain", "White", "Blouses", "Dept", "Section", "Index", "desc"],
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
                    "ground_truth_article_id": "a3",
                    "candidate_pool_article_ids": ["a3", "a4", "a5"],
                    "candidate_pool_size": 3,
                }
            ]
        ),
        encoding="utf-8",
    )
    isolated_env.svd_recommendations_top10_json_path(1).write_text(
        json.dumps(
            [{"customer_id": "u1", "ground_truth_article_id": "a3", "top_10_recommendations": [{"rank": 1, "article_id": "a3", "product_group_name": "Full", "colour_group_name": "Red", "graphical_appearance_name": "Patterned"}]}]
        ),
        encoding="utf-8",
    )
    isolated_env.agentic_recommendations_top10_json_path(1).write_text(
        json.dumps(
            [{"customer_id": "u1", "ground_truth_article_id": "a3", "top_10_recommendations": [{"rank": 1, "article_id": "a4", "product_group_name": "Lower", "colour_group_name": "Black", "graphical_appearance_name": "Solid"}]}]
        ),
        encoding="utf-8",
    )
    isolated_env.hybrid_svd_agentic_recommendations_top10_json_path(1).write_text(
        json.dumps(
            [{"customer_id": "u1", "ground_truth_article_id": "a3", "top_10_recommendations": [{"rank": 1, "article_id": "a3", "product_group_name": "Full", "colour_group_name": "Red", "graphical_appearance_name": "Patterned"}]}]
        ),
        encoding="utf-8",
    )

    result = EvaluationService(isolated_env).compute_three_method_top10_metrics_from_saved_artifacts(
        subset_size=1,
        bootstrap_samples=8,
        random_seed=42,
    )
    summary = json.loads(isolated_env.metric_summary_top10_three_methods_json_path(1).read_text(encoding="utf-8"))
    ci_report = json.loads(isolated_env.bootstrap_ci_report_top10_three_methods_json_path(1).read_text(encoding="utf-8"))

    assert result["summary"]["hybrid"]["hit_rate_at_10"] == 1.0
    assert summary["agentic"]["hit_rate_at_10"] == 0.0
    assert "difference_hybrid_minus_svd_hit_rate_at_10" in ci_report["confidence_intervals"]
