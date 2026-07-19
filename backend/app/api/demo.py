from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.experiment import get_experiment_service
from app.models.schemas import ArtifactDemoWorkflowCasesResponse
from app.services.artifact_demo_service import ArtifactDemoService
from app.services.experiment_service import ExperimentService

router = APIRouter(prefix="/demo", tags=["demo"])


@router.get("/workflow-cases", response_model=ArtifactDemoWorkflowCasesResponse)
def get_workflow_cases(
    artifact_prefix: str = Query(default="seed99_robustness"),
    explainability_prefix: str = Query(default="seed99_full_retry"),
    service: ExperimentService = Depends(get_experiment_service),
) -> dict[str, object]:
    artifact_demo_service = ArtifactDemoService(service.settings)
    return artifact_demo_service.load_workflow_cases(
        artifact_prefix=artifact_prefix,
        explainability_prefix=explainability_prefix,
    )
