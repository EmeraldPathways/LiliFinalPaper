from __future__ import annotations

import json

import pandas as pd

from tests.test_explainability_service import _write_explainability_source_artifacts
from tests.conftest import write_processed_artifacts


def _write_workflow_case_artifacts(settings) -> None:
    settings.evaluation_base_table_svd_top10_json_path(1000, artifact_prefix="seed99_robustness").write_text(
        json.dumps(
            [
                {
                    "customer_id": "u1",
                    "train_article_ids": ["a1", "a2", "a3"],
                    "ground_truth_article_id": "a4",
                    "ground_truth_product_type_name": "Dress",
                    "ground_truth_product_group_name": "Garment Full body",
                    "ground_truth_colour_group_name": "Black",
                    "ground_truth_graphical_appearance_name": "Solid",
                    "ground_truth_garment_group_name": "Dresses Ladies",
                    "candidate_pool_article_ids": ["a4", "a5", "a6"],
                },
                {
                    "customer_id": "u2",
                    "train_article_ids": ["b1", "b2"],
                    "ground_truth_article_id": "b3",
                    "ground_truth_product_type_name": "Top",
                    "ground_truth_product_group_name": "Garment Upper body",
                    "ground_truth_colour_group_name": "White",
                    "ground_truth_graphical_appearance_name": "Solid",
                    "ground_truth_garment_group_name": "Jersey Basic",
                    "candidate_pool_article_ids": ["b3", "b4", "b5"],
                },
            ]
        ),
        encoding="utf-8",
    )
    settings.svd_recommendations_top10_json_path(1000, artifact_prefix="seed99_robustness").write_text(
        json.dumps(
            [
                {
                    "customer_id": "u1",
                    "top_10_recommendations": [
                        {
                            "rank": 1,
                            "article_id": "a4",
                            "score": 0.91,
                            "product_type_name": "Dress",
                            "product_group_name": "Garment Full body",
                            "colour_group_name": "Black",
                            "graphical_appearance_name": "Solid",
                        }
                    ],
                },
                {
                    "customer_id": "u2",
                    "top_10_recommendations": [
                        {
                            "rank": 1,
                            "article_id": "b4",
                            "score": 0.51,
                            "product_type_name": "Top",
                            "product_group_name": "Garment Upper body",
                            "colour_group_name": "White",
                            "graphical_appearance_name": "Solid",
                        }
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )
    settings.agentic_recommendations_top10_json_path(1000, artifact_prefix="seed99_robustness").write_text(
        json.dumps(
            [
                {
                    "customer_id": "u1",
                    "preference_profile": {
                        "user_id": "u1",
                        "inferred_intent": "occasionwear",
                        "preferred_categories": ["Garment Full body"],
                        "preferred_product_types": ["Dress"],
                        "preferred_colours": ["Black"],
                        "preferred_appearance": ["Solid"],
                        "shopping_context": "offline historical preference evaluation",
                    },
                    "top_10_recommendations": [
                        {
                            "rank": 1,
                            "article_id": "a5",
                            "score": 0.73,
                            "product_type_name": "Dress",
                            "product_group_name": "Garment Full body",
                            "colour_group_name": "Black",
                            "graphical_appearance_name": "Solid",
                            "garment_group_name": "Dresses Ladies",
                            "recommendation_reason": "Matches the user's saved preference profile.",
                            "matched_evidence": ["product_group_name=Garment Full body"],
                        }
                    ],
                },
                {
                    "customer_id": "u2",
                    "preference_profile": {
                        "user_id": "u2",
                        "inferred_intent": "daily tops",
                        "preferred_categories": ["Garment Upper body"],
                        "preferred_product_types": ["Top"],
                        "preferred_colours": ["White"],
                        "preferred_appearance": ["Solid"],
                        "shopping_context": "offline historical preference evaluation",
                    },
                    "top_10_recommendations": [
                        {
                            "rank": 1,
                            "article_id": "b3",
                            "score": 0.66,
                            "product_type_name": "Top",
                            "product_group_name": "Garment Upper body",
                            "colour_group_name": "White",
                            "graphical_appearance_name": "Solid",
                            "garment_group_name": "Jersey Basic",
                            "recommendation_reason": "Matches the user's saved preference profile.",
                            "matched_evidence": ["product_group_name=Garment Upper body"],
                        }
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )
    settings.hybrid_svd_agentic_recommendations_top10_json_path(1000, artifact_prefix="seed99_robustness").write_text(
        json.dumps(
            [
                {
                    "customer_id": "u1",
                    "top_10_recommendations": [
                        {
                            "rank": 1,
                            "article_id": "a4",
                            "hybrid_score": 0.95,
                            "normalized_svd_score": 1.0,
                            "normalized_agentic_score": 0.8,
                            "diversity_bonus": 0.1,
                            "product_type_name": "Dress",
                            "product_group_name": "Garment Full body",
                            "colour_group_name": "Black",
                            "graphical_appearance_name": "Solid",
                            "garment_group_name": "Dresses Ladies",
                            "recommendation_reason": "Hybrid reranking promoted a known preference match.",
                        }
                    ],
                },
                {
                    "customer_id": "u2",
                    "top_10_recommendations": [
                        {
                            "rank": 1,
                            "article_id": "b3",
                            "hybrid_score": 0.88,
                            "normalized_svd_score": 0.7,
                            "normalized_agentic_score": 0.9,
                            "diversity_bonus": 0.05,
                            "product_type_name": "Top",
                            "product_group_name": "Garment Upper body",
                            "colour_group_name": "White",
                            "graphical_appearance_name": "Solid",
                            "garment_group_name": "Jersey Basic",
                            "recommendation_reason": "Hybrid reranking promoted a known preference match.",
                        }
                    ],
                },
            ]
        ),
        encoding="utf-8",
    )
    settings.per_user_metrics_top10_three_methods_json_path(1000, artifact_prefix="seed99_robustness").write_text(
        json.dumps(
            [
                {
                    "customer_id": "u1",
                    "ground_truth_article_id": "a4",
                    "svd_hit_rate_at_10": 1,
                    "svd_ground_truth_rank": 1,
                    "svd_ndcg_at_10": 1.0,
                    "svd_ild_at_10": 0.8,
                    "agentic_hit_rate_at_10": 0,
                    "agentic_ground_truth_rank": None,
                    "agentic_ndcg_at_10": 0.0,
                    "agentic_ild_at_10": 0.5,
                    "hybrid_hit_rate_at_10": 1,
                    "hybrid_ground_truth_rank": 1,
                    "hybrid_ndcg_at_10": 1.0,
                    "hybrid_ild_at_10": 0.7,
                },
                {
                    "customer_id": "u2",
                    "ground_truth_article_id": "b3",
                    "svd_hit_rate_at_10": 0,
                    "svd_ground_truth_rank": None,
                    "svd_ndcg_at_10": 0.0,
                    "svd_ild_at_10": 0.9,
                    "agentic_hit_rate_at_10": 1,
                    "agentic_ground_truth_rank": 1,
                    "agentic_ndcg_at_10": 1.0,
                    "agentic_ild_at_10": 0.6,
                    "hybrid_hit_rate_at_10": 1,
                    "hybrid_ground_truth_rank": 1,
                    "hybrid_ndcg_at_10": 1.0,
                    "hybrid_ild_at_10": 0.72,
                },
            ]
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "customer_id": "u1",
                "article_id": "a4",
                "hybrid_rank": 1,
                "svd_rank": 2,
                "rank_shift": 1,
                "is_ground_truth": True,
                "normalized_svd_score": 1.0,
                "normalized_agentic_score": 0.8,
                "diversity_bonus": 0.1,
                "hybrid_score": 0.95,
                "product_type_name": "Dress",
                "product_group_name": "Garment Full body",
                "graphical_appearance_name": "Solid",
                "colour_group_name": "Black",
                "garment_group_name": "Dresses Ladies",
                "department_name": "",
                "section_name": "",
                "index_name": "",
                "prod_name": "",
                "detail_desc": "",
                "matched_preference_fields_json": "[]",
                "explanation_text": "Grounded hybrid explanation for u1.",
                "grounded_claim_count": 2,
                "ungrounded_claim_count": 0,
            },
            {
                "customer_id": "u2",
                "article_id": "b3",
                "hybrid_rank": 1,
                "svd_rank": "",
                "rank_shift": "",
                "is_ground_truth": True,
                "normalized_svd_score": 0.7,
                "normalized_agentic_score": 0.9,
                "diversity_bonus": 0.05,
                "hybrid_score": 0.88,
                "product_type_name": "Top",
                "product_group_name": "Garment Upper body",
                "graphical_appearance_name": "Solid",
                "colour_group_name": "White",
                "garment_group_name": "Jersey Basic",
                "department_name": "",
                "section_name": "",
                "index_name": "",
                "prod_name": "",
                "detail_desc": "",
                "matched_preference_fields_json": "[]",
                "explanation_text": "Grounded hybrid explanation for u2.",
                "grounded_claim_count": 2,
                "ungrounded_claim_count": 0,
            },
        ]
    ).to_csv(settings.explainability_examples_csv_path("seed99_full_retry"), index=False)
    settings.explainability_audit_path("seed99_full_retry").write_text(
        json.dumps(
            {
                "case_study_preview": {
                    "customer_id": "u1",
                    "ground_truth_article_id": "a4",
                    "recommended_article_id": "a4",
                    "is_ground_truth": True,
                    "user_history_summary": {"recent_product_types": ["Dress"]},
                    "item_metadata": {"article_id": "a4"},
                    "svd_rank": 2,
                    "hybrid_rank": 1,
                    "rank_shift": 1,
                    "score_components": {
                        "normalized_svd_score": 1.0,
                        "normalized_agentic_score": 0.8,
                        "diversity_bonus": 0.1,
                        "hybrid_score": 0.95,
                    },
                    "matched_preference_fields": [{"field": "product_group_name", "item_value": "Garment Full body"}],
                    "explanation_text": "Grounded hybrid explanation for u1.",
                    "limitation_note": None,
                }
            }
        ),
        encoding="utf-8",
    )


def test_setup_endpoint(client, isolated_env, sample_interactions: pd.DataFrame):
    write_processed_artifacts(isolated_env, sample_interactions)

    response = client.get("/experiment/setup")

    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset"] == isolated_env.dataset_name
    assert payload["summary"]["sample_size"] == len(sample_interactions)
    assert payload["summary"]["evaluated_user_ids"] == ["u1"]


def test_metrics_endpoint(client, isolated_env):
    isolated_env.metrics_path.write_text(
        json.dumps(
            {
                "collaborative_filtering": {
                    "hit_rate_at_10": 0.1,
                    "preference_alignment": 0.5,
                    "diversity": 0.4,
                    "explanation_quality": None,
                    "feedback_adaptability": None,
                },
                "agentic_ai_framework": {
                    "hit_rate_at_10": 0.2,
                    "preference_alignment": 0.6,
                    "diversity": 0.5,
                    "explanation_quality": 0.8,
                    "feedback_adaptability": 0.75,
                },
                "business_mapping": {
                    "hit_rate_at_10": "Potential CTR improvement",
                    "preference_alignment": "Potential CVR improvement",
                    "diversity": "Potential engagement depth improvement",
                },
                "evaluated_users": 1,
                "generated_at": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.json()["agentic_ai_framework"]["diversity"] == 0.5


def test_metrics_endpoint_reads_formal_svd_summary_when_requested(client, isolated_env):
    isolated_env.metric_summary_top10_100_json_path.write_text(
        json.dumps(
            {
                "evaluation_scope": {
                    "valid_evaluated_users": 100,
                    "candidate_pool_size": 100,
                    "top_k": 10,
                    "split_strategy": "leave_one_out",
                    "baseline": "SVD Matrix Factorisation",
                    "comparison_method": "3-Agent Agentic AI",
                },
                "svd": {
                    "hit_rate_at_10": 0.35,
                    "hits_count": 35,
                    "miss_count": 65,
                    "ndcg_at_10": 0.187798,
                    "intra_list_diversity_at_10": 0.75,
                },
                "agentic": {
                    "hit_rate_at_10": 0.38,
                    "hits_count": 38,
                    "miss_count": 62,
                    "ndcg_at_10": 0.196679,
                    "intra_list_diversity_at_10": 0.549185,
                },
            }
        ),
        encoding="utf-8",
    )

    response = client.get("/metrics?mode=svd_top10_experiment")

    assert response.status_code == 200
    payload = response.json()
    assert payload["svd_matrix_factorization"]["ndcg_at_10"] == 0.187798
    assert payload["agentic_ai_framework"]["intra_list_diversity_at_10"] == 0.549185
    assert payload["evaluated_users"] == 100


def test_formal_user_intent_endpoint_reads_saved_preference_profile(client, isolated_env):
    isolated_env.agentic_recommendations_top10_100_json_path.write_text(
        json.dumps(
            [
                {
                    "customer_id": "u1",
                    "preference_profile": {
                        "user_id": "u1",
                        "inferred_intent": "casual tops",
                        "preferred_categories": ["Garment Upper body"],
                        "preferred_product_types": ["Top"],
                        "preferred_colours": ["White"],
                        "preferred_appearance": ["Solid"],
                        "shopping_context": "offline historical preference evaluation",
                    },
                    "top_10_recommendations": [],
                }
            ]
        ),
        encoding="utf-8",
    )

    response = client.get("/users/u1/intent?mode=svd_top10_experiment")

    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == "u1"
    assert payload["inferred_intent"] == "casual tops"


def test_formal_recommendation_comparison_reads_saved_top10_artifacts(client, isolated_env):
    isolated_env.svd_recommendations_top10_100_json_path.write_text(
        json.dumps(
            [
                {
                    "customer_id": "u1",
                    "top_10_recommendations": [
                        {
                            "article_id": "a1",
                            "score": 0.42,
                            "product_type_name": "Top",
                            "product_group_name": "Garment Upper body",
                            "colour_group_name": "White",
                            "graphical_appearance_name": "Solid",
                        }
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
                    "top_10_recommendations": [
                        {
                            "article_id": "a2",
                            "score": 0.91,
                            "product_type_name": "Blouse",
                            "product_group_name": "Garment Upper body",
                            "colour_group_name": "Blue",
                            "graphical_appearance_name": "Patterned",
                            "recommendation_reason": "Matches the user's saved preference profile.",
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    response = client.get("/recommendations/compare/u1?mode=svd_top10_experiment")

    assert response.status_code == 200
    payload = response.json()
    assert payload["cf_recommendations"][0]["article_id"] == "a1"
    assert payload["cf_recommendations"][0]["product_type"] == "Top"
    assert payload["agentic_recommendations"][0]["article_id"] == "a2"
    assert payload["agentic_recommendations"][0]["reason"] == "Matches the user's saved preference profile."
    assert payload["agentic_process"] == []


def test_recommendation_comparison_includes_agent_process(client, isolated_env, sample_interactions: pd.DataFrame):
    write_processed_artifacts(isolated_env, sample_interactions)

    response = client.get("/recommendations/compare/u1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["agentic_recommendations"][0]["article_id"] == "a5"
    assert payload["agentic_process"][0]["agent"] == "Agent 1"


def test_explainability_endpoint_returns_generated_payload(client, isolated_env):
    _write_explainability_source_artifacts(isolated_env)
    from app.services.explainability_service import ExplainabilityService

    ExplainabilityService(isolated_env).generate_explainability_artifacts(
        artifact_prefix="seed99_robustness",
        output_prefix="seed99",
        sample_size=2,
    )

    response = client.get(
        "/metrics/explainability?mode=svd_top10_experiment&artifact_prefix=seed99_robustness&output_prefix=seed99"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["summary_metrics"]["evidence_coverage_rate"] >= 0
    assert len(payload["examples"]) == 4
    assert payload["case_study"]["customer_id"] == "u1"
    assert payload["limitations"][0].startswith("This explainability audit is offline")


def test_explainability_endpoint_returns_controlled_missing_artifact_message(client, isolated_env):
    response = client.get(
        "/metrics/explainability?mode=svd_top10_experiment&artifact_prefix=seed99_robustness&output_prefix=seed99"
    )

    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == "Explainability artifacts not found. Run python -m backend.scripts.run_explainability_audit "
        "--artifact-prefix seed99_robustness --sample-size 100 --output-prefix seed99 first."
    )


def test_workflow_cases_endpoint_returns_saved_artifact_cases(client, isolated_env):
    _write_workflow_case_artifacts(isolated_env)

    response = client.get(
        "/demo/workflow-cases?artifact_prefix=seed99_robustness&explainability_prefix=seed99_full_retry"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["artifact_prefix"] == "seed99_robustness"
    assert payload["explainability_prefix"] == "seed99_full_retry"
    assert len(payload["cases"]) == 2
    assert payload["cases"][0]["demo_user"]["label"] == "Demo User 1"
    assert payload["cases"][0]["demo_user"]["customer_id"] == "u1"
    assert payload["cases"][0]["demo_user"]["customer_id_short"] == "u1"
    assert payload["cases"][0]["leave_one_out"]["training_history_count"] == 3
    assert payload["cases"][0]["leave_one_out"]["ground_truth_article_id"] == "a4"
    assert payload["cases"][0]["leave_one_out"]["candidate_pool_size"] == 3
    assert payload["cases"][0]["leave_one_out"]["ground_truth_in_candidate_pool"] is True
    assert payload["cases"][0]["method_hits"] == {"svd": True, "agentic": False, "hybrid": True}
    assert payload["cases"][0]["preference_agent"]["availability"]["has_preference_summary"] is True
    assert payload["cases"][0]["evidence_agent"]["availability"]["has_item_metadata"] is True
    assert payload["cases"][0]["decision_agent"]["selected_article_id"] == "a5"
    assert payload["cases"][0]["decision_agent"]["selected_score"] == 0.73
    assert payload["cases"][0]["hybrid_explainability"]["article_id"] == "a4"
    assert payload["cases"][0]["hybrid_explainability"]["score_components"]["hybrid_score"] == 0.95
    assert payload["cases"][0]["hybrid_explainability"]["availability"]["has_score_breakdown"] is True
    assert payload["cases"][0]["hybrid_explainability"]["availability"]["prod_name_available"] is False
    assert payload["cases"][0]["hybrid_explainability"]["groundedness_status"] == "Grounded"
    assert payload["cases"][0]["hybrid_explainability"]["rank_shift"] == 1
    assert payload["cases"][0]["svd_top10"][0]["article_id"] == "a4"
    assert payload["cases"][0]["agentic_top10"][0]["article_id"] == "a5"
    assert payload["cases"][0]["hybrid_top10"][0]["article_id"] == "a4"
    assert payload["cases"][0]["hybrid_top10"][0]["hybrid_score"] == 0.95
    assert payload["cases"][1]["demo_user"]["label"] == "Demo User 2"


def test_workflow_cases_endpoint_returns_missing_artifact_error(client, isolated_env):
    response = client.get(
        "/demo/workflow-cases?artifact_prefix=seed99_robustness&explainability_prefix=seed99_full_retry"
    )

    assert response.status_code == 400
    assert "Missing required walkthrough artifacts" in response.json()["detail"]
