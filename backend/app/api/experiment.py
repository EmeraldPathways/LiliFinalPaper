from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.config import Settings, get_settings
from app.models.schemas import ExperimentSetupResponse, RunExperimentResponse
from app.services.agentic_service import AgenticRecommendationService
from app.services.cf_service import CollaborativeFilteringService
from app.services.data_service import DataService
from app.services.evaluation_service import EvaluationService
from app.services.experiment_service import ExperimentService

router = APIRouter(prefix="/experiment", tags=["experiment"])


def get_experiment_service(settings: Settings = Depends(get_settings)) -> ExperimentService:
    data_service = DataService(settings)
    return ExperimentService(
        settings=settings,
        data_service=data_service,
        cf_service=CollaborativeFilteringService(settings),
        agentic_service=AgenticRecommendationService(settings),
        evaluation_service=EvaluationService(settings),
    )


@router.post("/run", response_model=RunExperimentResponse)
def run_experiment(
    mode: str = Query(default=Settings.LEGACY_DEBUG_EXPERIMENT_MODE),
    service: ExperimentService = Depends(get_experiment_service),
) -> dict[str, object]:
    payload = service.run(mode)
    return {
        "dataset": payload["dataset"],
        "status": payload["status"],
        "experiment_mode": payload.get("experiment_mode"),
        "sample_size": payload["sample_size"],
        "train_size": payload["train_size"],
        "test_size": payload["test_size"],
        "models": payload["models"],
        "metrics_ready": payload["metrics_ready"],
        "evaluated_users": payload["evaluated_users"],
    }


@router.get("/setup", response_model=ExperimentSetupResponse)
def get_setup(
    mode: str = Query(default=Settings.LEGACY_DEBUG_EXPERIMENT_MODE),
    settings: Settings = Depends(get_settings),
    service: ExperimentService = Depends(get_experiment_service),
) -> dict[str, object]:
    state = service.load_state(mode)
    sample_size = int(state["sample_size"]) if state else settings.sample_size
    if mode == settings.SVD_TOP10_EXPERIMENT_MODE:
        summary_path = settings.experiment_artifact_path(mode, "summary")
        return {
            "dataset": settings.dataset_name,
            "sample_size": sample_size,
            "experiment_mode": mode,
            "split_method": "Leave-one-out per user",
            "benchmark": "SVD Matrix Factorisation",
            "proposed_framework": "Agentic AI Recommendation Framework",
            "evaluation_metrics": ["Hit Rate@10", "NDCG@10", "Intra-list Diversity@10"],
            "summary": DataService(settings).load_summary(summary_path),
        }
    return {
        "dataset": settings.dataset_name,
        "sample_size": sample_size,
        "experiment_mode": mode,
        "split_method": "80% historical behaviour / 20% future behaviour",
        "benchmark": "Collaborative Filtering",
        "proposed_framework": "Agentic AI Recommendation Framework",
        "evaluation_metrics": ["Hit Rate@10", "Preference Alignment", "Diversity"],
        "summary": DataService(settings).load_summary(),
    }
