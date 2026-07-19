from __future__ import annotations

import json

import pandas as pd
from fastapi import APIRouter, Depends
from fastapi import Query

from app.api.experiment import get_experiment_service
from app.config import Settings
from app.models.schemas import FeedbackRequest, FeedbackResponse, UserIntentResponse
from app.services.agentic_service import AgenticServiceError
from app.services.data_service import DataValidationError
from app.services.experiment_service import ExperimentService

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/{user_id}/intent", response_model=UserIntentResponse)
def get_user_intent(
    user_id: str,
    mode: str = Query(default=Settings.LEGACY_DEBUG_EXPERIMENT_MODE),
    service: ExperimentService = Depends(get_experiment_service),
) -> dict[str, object]:
    try:
        if mode == service.settings.SVD_TOP10_EXPERIMENT_MODE:
            formal_rows_path = service.settings.agentic_recommendations_top10_100_json_path
            if formal_rows_path.exists():
                rows = json.loads(formal_rows_path.read_text(encoding="utf-8"))
                for row in rows:
                    if str(row.get("customer_id")) == user_id and isinstance(row.get("preference_profile"), dict):
                        return row["preference_profile"]

            formal_train_path = service.settings.experiment_artifact_path(mode, "train")
            if formal_train_path.exists():
                train_df = pd.read_csv(
                    formal_train_path,
                    dtype={"article_id": "string", "customer_id": "string"},
                )
                train_df = service.data_service._normalize_interaction_ids(train_df)
                return service.agentic_service.infer_user_intent(user_id, train_df)

        train_df = service.data_service.load_train()
        return service.agentic_service.infer_user_intent(user_id, train_df)
    except FileNotFoundError as exc:
        raise DataValidationError("Run the experiment first to generate training data.") from exc


@router.post("/{user_id}/feedback", response_model=FeedbackResponse)
def submit_feedback(
    user_id: str,
    payload: FeedbackRequest,
    service: ExperimentService = Depends(get_experiment_service),
) -> dict[str, object]:
    updated_weights = service.agentic_service.adapt_from_feedback(
        user_id=user_id,
        article_id=payload.article_id,
        feedback_type=payload.feedback_type,
    )
    return {
        "status": "updated",
        "message": f"User preference weights updated based on {payload.feedback_type} feedback.",
        "user_id": user_id,
        "article_id": payload.article_id,
        "feedback_type": payload.feedback_type,
        "updated_weights": updated_weights,
    }
