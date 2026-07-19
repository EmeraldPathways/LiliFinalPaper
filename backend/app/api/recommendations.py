from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Query

from app.api.experiment import get_experiment_service
from app.config import Settings
from app.models.schemas import RecommendationComparisonResponse
from app.services.experiment_service import ExperimentService

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


def _map_formal_svd_item(item: dict[str, object]) -> dict[str, object]:
    product_type = str(item.get("product_type_name") or "Unknown")
    return {
        "article_id": str(item.get("article_id") or ""),
        "product_name": product_type,
        "product_type": product_type,
        "product_group": str(item.get("product_group_name") or "Unknown"),
        "colour": str(item.get("colour_group_name") or "Unknown"),
        "appearance": str(item.get("graphical_appearance_name") or "Unknown"),
        "score": float(item.get("score") or 0.0),
        "model": "svd_matrix_factorization",
    }


def _map_formal_agentic_item(item: dict[str, object]) -> dict[str, object]:
    product_type = str(item.get("product_type_name") or "Unknown")
    return {
        "article_id": str(item.get("article_id") or ""),
        "product_name": product_type,
        "product_type": product_type,
        "product_group": str(item.get("product_group_name") or "Unknown"),
        "colour": str(item.get("colour_group_name") or "Unknown"),
        "appearance": str(item.get("graphical_appearance_name") or "Unknown"),
        "score": float(item.get("score") or 0.0),
        "model": "agentic_ai_framework",
        "reason": str(item.get("recommendation_reason") or ""),
        "intent_match": 0.0,
        "preference_alignment": 0.0,
        "product_relevance": 0.0,
        "diversity": 0.0,
        "behavioural_signal": 0.0,
    }


@router.get("/compare/{user_id}", response_model=RecommendationComparisonResponse)
def compare_recommendations(
    user_id: str,
    mode: str = Query(default=Settings.LEGACY_DEBUG_EXPERIMENT_MODE),
    service: ExperimentService = Depends(get_experiment_service),
) -> dict[str, object]:
    cf_output_path = service.settings.experiment_artifact_path(mode, "cf_output")
    agentic_output_path = service.settings.experiment_artifact_path(mode, "agentic_output")
    agentic_trace_path = service.settings.experiment_artifact_path(mode, "agentic_trace")

    if mode != service.settings.LEGACY_DEBUG_EXPERIMENT_MODE and (
        not cf_output_path.exists() or not agentic_output_path.exists()
    ):
        if (
            service.settings.svd_recommendations_top10_100_json_path.exists()
            and service.settings.agentic_recommendations_top10_100_json_path.exists()
        ):
            svd_rows = json.loads(
                service.settings.svd_recommendations_top10_100_json_path.read_text(encoding="utf-8")
            )
            agentic_rows = json.loads(
                service.settings.agentic_recommendations_top10_100_json_path.read_text(encoding="utf-8")
            )
            svd_lookup = {str(row.get("customer_id")): row for row in svd_rows}
            agentic_lookup = {str(row.get("customer_id")): row for row in agentic_rows}
            svd_row = svd_lookup.get(user_id, {})
            agentic_row = agentic_lookup.get(user_id, {})
            return {
                "user_id": user_id,
                "cf_recommendations": [
                    _map_formal_svd_item(item) for item in svd_row.get("top_10_recommendations", [])
                ],
                "agentic_recommendations": [
                    _map_formal_agentic_item(item)
                    for item in agentic_row.get("top_10_recommendations", [])
                ],
                "agentic_process": [],
            }

    if not cf_output_path.exists() or not agentic_output_path.exists():
        raise FileNotFoundError("Run the experiment first to generate recommendation outputs.")

    cf_payload = json.loads(cf_output_path.read_text(encoding="utf-8"))
    agentic_payload = json.loads(agentic_output_path.read_text(encoding="utf-8"))
    agentic_trace = (
        json.loads(agentic_trace_path.read_text(encoding="utf-8")) if agentic_trace_path.exists() else {}
    )

    if mode != service.settings.LEGACY_DEBUG_EXPERIMENT_MODE:
        return {
            "user_id": user_id,
            "cf_recommendations": cf_payload.get(user_id, []),
            "agentic_recommendations": agentic_payload.get(user_id, []),
            "agentic_process": agentic_trace.get(user_id, [])[:3],
        }

    train_df = service.data_service.load_train()

    if user_id not in cf_payload:
        matrix = service.cf_service._build_user_item_matrix(train_df)
        metadata = service.cf_service._article_metadata(train_df)
        user_history = train_df.groupby("customer_id")["article_id"].agg(set).to_dict()
        cf_payload[user_id] = service.cf_service.recommend_for_user(user_id, matrix, metadata, user_history)
        cf_output_path.write_text(json.dumps(cf_payload, indent=2), encoding="utf-8")

    if user_id not in agentic_payload or user_id not in agentic_trace:
        recommendations, trace = service.agentic_service.generate_for_user(user_id, train_df)
        agentic_payload[user_id] = recommendations
        agentic_trace[user_id] = trace
        agentic_output_path.write_text(
            json.dumps(agentic_payload, indent=2),
            encoding="utf-8",
        )
        agentic_trace_path.write_text(
            json.dumps(agentic_trace, indent=2),
            encoding="utf-8",
        )

    return {
        "user_id": user_id,
        "cf_recommendations": cf_payload.get(user_id, []),
        "agentic_recommendations": agentic_payload.get(user_id, []),
        "agentic_process": agentic_trace.get(user_id, []),
    }
