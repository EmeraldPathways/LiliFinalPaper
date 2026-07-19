from __future__ import annotations

import inspect
import json

import pandas as pd
import pytest

from app.services.evaluation_service import EvaluationService
from app.services.hybrid_service import HybridRecommendationService


ARTIFACT_PREFIX = "seed99_robustness"


def test_default_and_prefixed_artifact_paths_are_distinct(isolated_env):
    assert isolated_env.evaluation_base_table_svd_top10_json_path(1000).name == "evaluation_base_table_svd_top10_1000.json"
    assert (
        isolated_env.evaluation_base_table_svd_top10_json_path(1000, artifact_prefix=ARTIFACT_PREFIX).name
        == "seed99_robustness_evaluation_base_table_top10_1000.json"
    )
    assert (
        isolated_env.candidate_pool_validation_report_svd_top10_path(1000, artifact_prefix=ARTIFACT_PREFIX).name
        == "seed99_robustness_candidate_pools_top10_1000.json"
    )
    assert (
        isolated_env.hybrid_svd_agentic_audit_report_top10_path(1000, artifact_prefix=ARTIFACT_PREFIX).name
        == "seed99_robustness_experiment_report_top10_1000.md"
    )
    assert (
        isolated_env.validation_report_top10_three_methods_json_path(1000, artifact_prefix=ARTIFACT_PREFIX).name
        == "seed99_robustness_validation_report_top10_1000_three_methods.json"
    )
    assert isolated_env.metric_summary_top10_three_methods_json_path(1000).name != isolated_env.metric_summary_top10_three_methods_json_path(
        1000,
        artifact_prefix=ARTIFACT_PREFIX,
    ).name


def test_prefixed_output_path_refuses_overwrite_by_default(isolated_env):
    target = isolated_env.svd_recommendations_top10_json_path(1000, artifact_prefix=ARTIFACT_PREFIX)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError):
        isolated_env.ensure_output_path(target, artifact_prefix=ARTIFACT_PREFIX)

    assert isolated_env.ensure_output_path(target, artifact_prefix=ARTIFACT_PREFIX, allow_overwrite=True) == target


def test_three_method_prefixed_metrics_keep_metric_keys_and_write_prefixed_outputs(isolated_env):
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
    isolated_env.evaluation_base_table_svd_top10_json_path(1, artifact_prefix=ARTIFACT_PREFIX).write_text(
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
    isolated_env.svd_recommendations_top10_json_path(1, artifact_prefix=ARTIFACT_PREFIX).write_text(
        json.dumps(
            [{"customer_id": "u1", "ground_truth_article_id": "a3", "top_10_recommendations": [{"rank": 1, "article_id": "a3", "product_group_name": "Full", "colour_group_name": "Red", "graphical_appearance_name": "Patterned"}]}]
        ),
        encoding="utf-8",
    )
    isolated_env.agentic_recommendations_top10_json_path(1, artifact_prefix=ARTIFACT_PREFIX).write_text(
        json.dumps(
            [{"customer_id": "u1", "ground_truth_article_id": "a3", "top_10_recommendations": [{"rank": 1, "article_id": "a4", "product_group_name": "Lower", "colour_group_name": "Black", "graphical_appearance_name": "Solid"}]}]
        ),
        encoding="utf-8",
    )
    isolated_env.hybrid_svd_agentic_recommendations_top10_json_path(1, artifact_prefix=ARTIFACT_PREFIX).write_text(
        json.dumps(
            [{"customer_id": "u1", "ground_truth_article_id": "a3", "top_10_recommendations": [{"rank": 1, "article_id": "a3", "product_group_name": "Full", "colour_group_name": "Red", "graphical_appearance_name": "Patterned"}]}]
        ),
        encoding="utf-8",
    )

    result = EvaluationService(isolated_env).compute_three_method_top10_metrics_from_saved_artifacts(
        subset_size=1,
        bootstrap_samples=8,
        random_seed=42,
        artifact_prefix=ARTIFACT_PREFIX,
    )

    summary = result["summary"]
    for method_key in ("svd", "agentic", "hybrid"):
        assert "hit_rate_at_10" in summary[method_key]
        assert "ndcg_at_10" in summary[method_key]
        assert "intra_list_diversity_at_10" in summary[method_key]

    assert isolated_env.metric_summary_top10_three_methods_json_path(1, artifact_prefix=ARTIFACT_PREFIX).exists()
    assert isolated_env.validation_report_top10_three_methods_json_path(1, artifact_prefix=ARTIFACT_PREFIX).exists()
    assert isolated_env.hybrid_svd_agentic_audit_report_top10_path(1, artifact_prefix=ARTIFACT_PREFIX).exists()


def test_hybrid_formula_source_is_unchanged():
    source = inspect.getsource(HybridRecommendationService._build_hybrid_result_for_user)

    assert "0.70 * normalized_svd_scores.get(article_id, 0.0)" in source
    assert "0.25 * normalized_agentic_scores.get(article_id, 0.0)" in source
    assert "0.05 * diversity_bonus" in source
