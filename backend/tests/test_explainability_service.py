from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.services.explainability_service import ExplainabilityService


def _write_explainability_source_artifacts(settings) -> None:
    processed = pd.DataFrame(
        [
            {
                "customer_id": "u1",
                "article_id": "a1",
                "t_dat": "2024-01-01",
                "price": 0.1,
                "sales_channel_id": 1,
                "product_type_name": "Top",
                "product_group_name": "Garment Upper body",
                "graphical_appearance_name": "Solid",
                "colour_group_name": "Black",
                "garment_group_name": "Jersey Basic",
                "department_name": "Tops",
                "section_name": "Women",
                "index_name": "Ladieswear",
                "detail_desc": "Black jersey top.",
            },
            {
                "customer_id": "u1",
                "article_id": "a2",
                "t_dat": "2024-01-02",
                "price": 0.1,
                "sales_channel_id": 1,
                "product_type_name": "Top",
                "product_group_name": "Garment Upper body",
                "graphical_appearance_name": "Solid",
                "colour_group_name": "Black",
                "garment_group_name": "Jersey Basic",
                "department_name": "Tops",
                "section_name": "Women",
                "index_name": "Ladieswear",
                "detail_desc": "Another black top.",
            },
            {
                "customer_id": "u1",
                "article_id": "a3",
                "t_dat": "2024-01-03",
                "price": 0.1,
                "sales_channel_id": 1,
                "product_type_name": "Dress",
                "product_group_name": "Garment Full body",
                "graphical_appearance_name": "Patterned",
                "colour_group_name": "Blue",
                "garment_group_name": "Dresses Ladies",
                "department_name": "Dresses",
                "section_name": "Women",
                "index_name": "Ladieswear",
                "detail_desc": "Blue patterned dress.",
            },
            {
                "customer_id": "u2",
                "article_id": "a4",
                "t_dat": "2024-02-01",
                "price": 0.1,
                "sales_channel_id": 1,
                "product_type_name": "Shirt",
                "product_group_name": "Garment Upper body",
                "graphical_appearance_name": "Solid",
                "colour_group_name": "White",
                "garment_group_name": "Shirts",
                "department_name": "Shirts",
                "section_name": "Men",
                "index_name": "Menswear",
                "detail_desc": "White shirt.",
            },
            {
                "customer_id": "u2",
                "article_id": "a5",
                "t_dat": "2024-02-02",
                "price": 0.1,
                "sales_channel_id": 1,
                "product_type_name": "Shirt",
                "product_group_name": "Garment Upper body",
                "graphical_appearance_name": "Solid",
                "colour_group_name": "White",
                "garment_group_name": "Shirts",
                "department_name": "Shirts",
                "section_name": "Men",
                "index_name": "Menswear",
                "detail_desc": "Second white shirt.",
            },
            {
                "customer_id": "u2",
                "article_id": "a6",
                "t_dat": "2024-02-03",
                "price": 0.1,
                "sales_channel_id": 1,
                "product_type_name": "Trousers",
                "product_group_name": "Garment Lower body",
                "graphical_appearance_name": "Solid",
                "colour_group_name": "Black",
                "garment_group_name": "Trousers",
                "department_name": "Trousers",
                "section_name": "Men",
                "index_name": "Menswear",
                "detail_desc": "Black trousers.",
            },
        ]
    )
    processed.to_csv(settings.processed_interactions_with_articles_csv_path, index=False)

    evaluation_rows = [
        {
            "customer_id": "u1",
            "train_article_ids": ["a1", "a2"],
            "ground_truth_article_id": "a3",
            "candidate_pool_article_ids": ["a3", "a7", "a8"],
        },
        {
            "customer_id": "u2",
            "train_article_ids": ["a4", "a5"],
            "ground_truth_article_id": "a6",
            "candidate_pool_article_ids": ["a6", "a9", "a10"],
        },
    ]
    svd_rows = [
        {
            "customer_id": "u1",
            "ground_truth_article_id": "a3",
            "top_10_recommendations": [
                {
                    "rank": 1,
                    "article_id": "a7",
                    "score": 0.8,
                    "product_group_name": "Garment Upper body",
                    "colour_group_name": "Black",
                    "graphical_appearance_name": "Solid",
                },
                {
                    "rank": 2,
                    "article_id": "a3",
                    "score": 0.7,
                    "product_group_name": "Garment Full body",
                    "colour_group_name": "Blue",
                    "graphical_appearance_name": "Patterned",
                },
                {
                    "rank": 3,
                    "article_id": "a8",
                    "score": 0.6,
                    "product_group_name": "Garment Lower body",
                    "colour_group_name": "Grey",
                    "graphical_appearance_name": "Solid",
                },
            ],
        },
        {
            "customer_id": "u2",
            "ground_truth_article_id": "a6",
            "top_10_recommendations": [
                {
                    "rank": 1,
                    "article_id": "a9",
                    "score": 0.8,
                    "product_group_name": "Garment Upper body",
                    "colour_group_name": "White",
                    "graphical_appearance_name": "Solid",
                },
                {
                    "rank": 2,
                    "article_id": "a10",
                    "score": 0.7,
                    "product_group_name": "Garment Upper body",
                    "colour_group_name": "White",
                    "graphical_appearance_name": "Solid",
                },
            ],
        },
    ]
    hybrid_rows = [
        {
            "customer_id": "u1",
            "ground_truth_article_id": "a3",
            "top_10_recommendations": [
                {
                    "rank": 1,
                    "article_id": "a3",
                    "hybrid_score": 0.9,
                    "normalized_svd_score": 0.8,
                    "normalized_agentic_score": 0.7,
                    "diversity_bonus": 0.1,
                    "product_type_name": "Dress",
                    "product_group_name": "Garment Full body",
                    "colour_group_name": "Blue",
                    "graphical_appearance_name": "Patterned",
                    "garment_group_name": "Dresses Ladies",
                },
                {
                    "rank": 2,
                    "article_id": "a7",
                    "hybrid_score": 0.7,
                    "normalized_svd_score": 0.6,
                    "normalized_agentic_score": 0.5,
                    "diversity_bonus": 0.0,
                    "product_type_name": "Top",
                    "product_group_name": "Garment Upper body",
                    "colour_group_name": "Black",
                    "graphical_appearance_name": "Solid",
                    "garment_group_name": "Jersey Basic",
                },
            ],
        },
        {
            "customer_id": "u2",
            "ground_truth_article_id": "a6",
            "top_10_recommendations": [
                {
                    "rank": 1,
                    "article_id": "a6",
                    "hybrid_score": 0.85,
                    "normalized_svd_score": 0.55,
                    "normalized_agentic_score": 0.9,
                    "diversity_bonus": 0.2,
                    "product_type_name": "Trousers",
                    "product_group_name": "Garment Lower body",
                    "colour_group_name": "Black",
                    "graphical_appearance_name": "Solid",
                    "garment_group_name": "Trousers",
                },
                {
                    "rank": 2,
                    "article_id": "a10",
                    "hybrid_score": 0.5,
                    "normalized_svd_score": None,
                    "normalized_agentic_score": None,
                    "diversity_bonus": None,
                    "product_type_name": None,
                    "product_group_name": "Garment Upper body",
                    "colour_group_name": None,
                    "graphical_appearance_name": "Solid",
                    "garment_group_name": "Shirts",
                },
            ],
        },
    ]

    settings.evaluation_base_table_svd_top10_json_path(2, artifact_prefix="seed99_robustness").write_text(
        json.dumps(evaluation_rows, indent=2),
        encoding="utf-8",
    )
    settings.svd_recommendations_top10_json_path(2, artifact_prefix="seed99_robustness").write_text(
        json.dumps(svd_rows, indent=2),
        encoding="utf-8",
    )
    settings.hybrid_svd_agentic_recommendations_top10_json_path(2, artifact_prefix="seed99_robustness").write_text(
        json.dumps(hybrid_rows, indent=2),
        encoding="utf-8",
    )


def test_explainability_service_generates_grounded_deterministic_outputs(isolated_env):
    _write_explainability_source_artifacts(isolated_env)
    service = ExplainabilityService(isolated_env)

    result = service.generate_explainability_artifacts(
        artifact_prefix="seed99_robustness",
        output_prefix="seed99",
        sample_size=2,
    )

    assert result["summary"]["run_context"]["users_included"] == 2
    assert result["summary"]["run_context"]["recommendations_explained"] == 4
    assert result["summary"]["summary_metrics"]["ungrounded_claim_count"] == 0

    examples_path = isolated_env.explainability_examples_csv_path("seed99")
    examples = pd.read_csv(examples_path, dtype={"customer_id": "string", "article_id": "string"})
    assert examples["customer_id"].tolist() == ["u1", "u1", "u2", "u2"]
    assert examples["article_id"].tolist() == ["a3", "a7", "a6", "a10"]
    assert examples.loc[0, "rank_shift"] == 1
    assert pd.isna(examples.loc[2, "svd_rank"])
    assert "Blue" in examples.loc[0, "explanation_text"]
    assert "CTR" not in examples.loc[0, "explanation_text"]
    assert "dwell time" not in examples.loc[0, "explanation_text"]

    audit = json.loads(isolated_env.explainability_audit_path("seed99").read_text(encoding="utf-8"))
    assert "prod_name" in audit["missing_optional_fields"]
    assert audit["score_component_availability"]["normalized_svd_score"]["present_count"] == 3
    assert audit["diversity_concentration_summary"]["user_count_with_concentration"] >= 1

    first_run = {
        path.name: path.read_text(encoding="utf-8")
        for path in (
            isolated_env.explainability_summary_path("seed99"),
            isolated_env.explainability_audit_path("seed99"),
            isolated_env.explainability_examples_csv_path("seed99"),
            isolated_env.rank_shift_analysis_csv_path("seed99"),
            isolated_env.explainability_case_studies_path("seed99"),
        )
    }
    second_result = service.generate_explainability_artifacts(
        artifact_prefix="seed99_robustness",
        output_prefix="seed99",
        sample_size=2,
        allow_overwrite=True,
    )
    second_run = {
        path.name: path.read_text(encoding="utf-8")
        for path in (
            isolated_env.explainability_summary_path("seed99"),
            isolated_env.explainability_audit_path("seed99"),
            isolated_env.explainability_examples_csv_path("seed99"),
            isolated_env.rank_shift_analysis_csv_path("seed99"),
            isolated_env.explainability_case_studies_path("seed99"),
        )
    }
    assert second_result["summary"]["summary_metrics"] == result["summary"]["summary_metrics"]
    assert first_run == second_run


def test_explainability_service_does_not_modify_source_artifacts(isolated_env):
    _write_explainability_source_artifacts(isolated_env)
    service = ExplainabilityService(isolated_env)
    source_path = isolated_env.hybrid_svd_agentic_recommendations_top10_json_path(2, artifact_prefix="seed99_robustness")
    before = source_path.read_text(encoding="utf-8")

    service.generate_explainability_artifacts(
        artifact_prefix="seed99_robustness",
        output_prefix="seed99",
        sample_size=1,
    )

    after = source_path.read_text(encoding="utf-8")
    assert before == after


def test_explainability_service_reports_missing_required_artifacts(isolated_env):
    service = ExplainabilityService(isolated_env)

    try:
        service.generate_explainability_artifacts(
            artifact_prefix="seed99_robustness",
            output_prefix="seed99",
            sample_size=1,
        )
    except FileNotFoundError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")

    assert "seed99_robustness" in message
    assert "evaluation_base_table_top10" in message


def test_explainability_service_precomputes_filtered_history_summaries(isolated_env):
    _write_explainability_source_artifacts(isolated_env)
    service = ExplainabilityService(isolated_env)

    filtered = service._load_filtered_processed_rows(
        selected_customer_ids={"u1"},
        needed_article_ids={"a1", "a2", "a3"},
    )
    summaries = service._build_user_history_summaries(
        processed=filtered,
        train_article_ids_by_customer={"u1": ["a1", "a2"]},
    )

    assert sorted(filtered["customer_id"].dropna().unique().tolist()) == ["u1"]
    assert summaries["u1"]["frequent_product_groups"] == [{"value": "Garment Upper body", "count": 2}]
    assert summaries["u1"]["frequent_colours"] == [{"value": "Black", "count": 2}]
