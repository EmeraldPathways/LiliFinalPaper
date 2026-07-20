from __future__ import annotations

import inspect
import json
import math
import shutil
from pathlib import Path

import pandas as pd
import pytest

from app.services.agentic_service import AgenticRecommendationService
from app.services.cf_service import CollaborativeFilteringService
from app.services.hybrid_service import HybridRecommendationService


REPO_PROCESSED_DIR = Path(__file__).resolve().parents[1] / "app" / "data" / "processed"
EXPECTED_ORIGINAL_HR = 0.498000
EXPECTED_ORIGINAL_NDCG = 0.304552


def _build_service(settings) -> HybridRecommendationService:
    return HybridRecommendationService(
        settings,
        CollaborativeFilteringService(settings),
        AgenticRecommendationService(settings),
    )


def _write_minimal_processed_dataset(settings) -> None:
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
    processed.to_csv(settings.processed_interactions_with_articles_csv_path, index=False)
    settings.top_n = 10


def _write_subset(settings, prefix: str | None, customer_id: str) -> None:
    settings.evaluation_base_table_svd_top10_json_path(1, artifact_prefix=prefix).write_text(
        json.dumps(
            [
                {
                    "customer_id": customer_id,
                    "train_article_ids": ["a1", "a2"],
                    "ground_truth_article_id": "a3",
                    "candidate_pool_article_ids": ["a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10", "a11", "a12"],
                    "candidate_pool_size": 10,
                }
            ]
        ),
        encoding="utf-8",
    )


def _compute_hr_and_ndcg(
    evaluation_rows: list[dict[str, object]],
    recommendation_rows: list[dict[str, object]],
) -> tuple[float, float]:
    evaluation_lookup = {str(row["customer_id"]): row for row in evaluation_rows}
    recommendation_lookup = {str(row["customer_id"]): row for row in recommendation_rows}

    assert set(evaluation_lookup) == set(recommendation_lookup)

    hits: list[float] = []
    ndcg_values: list[float] = []

    for customer_id, evaluation_row in evaluation_lookup.items():
        ground_truth = str(evaluation_row["ground_truth_article_id"])
        recommendation_ids = [
            str(item["article_id"])
            for item in recommendation_lookup[customer_id]["top_10_recommendations"]
        ]
        rank = next(
            (position for position, article_id in enumerate(recommendation_ids, start=1) if article_id == ground_truth),
            None,
        )
        if rank is None:
            hits.append(0.0)
            ndcg_values.append(0.0)
        else:
            hits.append(1.0)
            ndcg_values.append(1.0 / math.log2(rank + 1))

    return sum(hits) / len(hits), sum(ndcg_values) / len(ndcg_values)


def test_signature_exposes_default_weights_and_input_prefix():
    signature = inspect.signature(HybridRecommendationService.build_hybrid_svd_agentic_reranker)

    assert signature.parameters["input_artifact_prefix"].default is None
    assert signature.parameters["svd_weight"].default == 0.70
    assert signature.parameters["agentic_weight"].default == 0.25
    assert signature.parameters["diversity_weight"].default == 0.05


def test_existing_calls_remain_backward_compatible(isolated_env):
    _write_minimal_processed_dataset(isolated_env)
    _write_subset(isolated_env, None, "u1")

    service = _build_service(isolated_env)
    report = service.build_hybrid_svd_agentic_reranker(subset_size=1, random_state=42)

    payload = json.loads(isolated_env.hybrid_svd_agentic_recommendations_top10_json_path(1).read_text(encoding="utf-8"))
    validation = json.loads(isolated_env.hybrid_svd_agentic_validation_report_top10_path(1).read_text(encoding="utf-8"))

    assert report["users_evaluated"] == 1
    assert payload[0]["customer_id"] == "u1"
    assert validation["svd_weight"] == 0.70
    assert validation["agentic_weight"] == 0.25
    assert validation["diversity_weight"] == 0.05
    assert validation["input_artifact_prefix"] is None


def test_invalid_weight_sum_is_rejected(isolated_env):
    _write_minimal_processed_dataset(isolated_env)
    _write_subset(isolated_env, None, "u1")

    service = _build_service(isolated_env)

    with pytest.raises(ValueError, match="sum to 1.00"):
        service.build_hybrid_svd_agentic_reranker(
            subset_size=1,
            svd_weight=0.7,
            agentic_weight=0.2,
            diversity_weight=0.05,
        )


def test_input_artifact_prefix_controls_only_frozen_input_and_artifact_prefix_only_output_names(isolated_env):
    _write_minimal_processed_dataset(isolated_env)
    _write_subset(isolated_env, None, "default-user")
    _write_subset(isolated_env, "frozen-source", "prefixed-user")

    service = _build_service(isolated_env)
    service.build_hybrid_svd_agentic_reranker(
        subset_size=1,
        input_artifact_prefix="frozen-source",
        artifact_prefix="fresh-output",
        allow_overwrite=False,
    )

    output_path = isolated_env.hybrid_svd_agentic_recommendations_top10_json_path(1, artifact_prefix="fresh-output")
    validation_path = isolated_env.hybrid_svd_agentic_validation_report_top10_path(1, artifact_prefix="fresh-output")

    assert output_path.exists()
    assert validation_path.exists()
    assert not isolated_env.hybrid_svd_agentic_recommendations_top10_json_path(1).exists()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))

    assert payload[0]["customer_id"] == "prefixed-user"
    assert validation["input_artifact_prefix"] == "frozen-source"
    assert validation["artifact_prefix"] == "fresh-output"


def test_hybrid_formal_path_does_not_call_openai(isolated_env, monkeypatch: pytest.MonkeyPatch):
    _write_minimal_processed_dataset(isolated_env)
    _write_subset(isolated_env, None, "u1")

    service = _build_service(isolated_env)
    monkeypatch.setattr(
        service.agentic_service,
        "_request_structured_completion",
        lambda *args, **kwargs: pytest.fail("OpenAI-backed method should not be called"),
    )

    report = service.build_hybrid_svd_agentic_reranker(subset_size=1)

    assert report["users_evaluated"] == 1


def test_original_weights_reproduce_saved_original_hybrid_metrics_from_same_frozen_input(isolated_env):
    shutil.copyfile(
        REPO_PROCESSED_DIR / "processed_interactions_with_articles.csv",
        isolated_env.processed_interactions_with_articles_csv_path,
    )
    source_prefix = "reproduction_guard"
    shutil.copyfile(
        REPO_PROCESSED_DIR / "evaluation_base_table_svd_top10_1000.json",
        isolated_env.evaluation_base_table_svd_top10_json_path(1000, artifact_prefix=source_prefix),
    )

    saved_metrics = json.loads(
        (REPO_PROCESSED_DIR / "metric_summary_top10_1000_three_methods.json").read_text(encoding="utf-8")
    )
    expected_hr = float(saved_metrics["hybrid"]["hit_rate_at_10"])
    expected_ndcg = float(saved_metrics["hybrid"]["ndcg_at_10"])

    service = _build_service(isolated_env)
    service.build_hybrid_svd_agentic_reranker(
        subset_size=1000,
        random_state=42,
        input_artifact_prefix=source_prefix,
        artifact_prefix="reproduced",
        svd_weight=0.70,
        agentic_weight=0.25,
        diversity_weight=0.05,
        allow_overwrite=False,
    )

    evaluation_rows = json.loads(
        isolated_env.evaluation_base_table_svd_top10_json_path(1000, artifact_prefix=source_prefix).read_text(
            encoding="utf-8"
        )
    )
    recommendation_rows = json.loads(
        isolated_env.hybrid_svd_agentic_recommendations_top10_json_path(1000, artifact_prefix="reproduced").read_text(
            encoding="utf-8"
        )
    )
    actual_hr, actual_ndcg = _compute_hr_and_ndcg(evaluation_rows, recommendation_rows)

    assert expected_hr == EXPECTED_ORIGINAL_HR
    assert expected_ndcg == EXPECTED_ORIGINAL_NDCG
    assert actual_hr == pytest.approx(expected_hr, abs=1e-6)
    assert actual_ndcg == pytest.approx(expected_ndcg, abs=1e-6)

