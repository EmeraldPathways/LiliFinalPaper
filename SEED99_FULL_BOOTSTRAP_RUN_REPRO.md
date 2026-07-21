# Reproducing the Full Seed99 Bootstrap Run

## Important Note

The repository does not expose the full three-method Seed99 pipeline as a single packaged CLI command. The exact reproducible run is a sequence of service calls. The command block below reproduces the full saved `seed99_robustness_*` formal experiment artefacts from the current codebase, including the bootstrap confidence interval outputs.

## Exact Full Seed99 Run Command

Run this from the repository root:

```powershell
cd backend
@'
from app.config import get_settings
from app.services.data_service import DataService
from app.services.cf_service import CollaborativeFilteringService
from app.services.agentic_service import AgenticRecommendationService
from app.services.hybrid_service import HybridRecommendationService
from app.services.evaluation_service import EvaluationService

PREFIX = "seed99_robustness"

settings = get_settings()
data_service = DataService(settings)
cf_service = CollaborativeFilteringService(settings)
agentic_service = AgenticRecommendationService(settings)
hybrid_service = HybridRecommendationService(settings, cf_service, agentic_service)
evaluation_service = EvaluationService(settings)

print("starting")
data_service.build_processed_interactions_with_articles()
print("processed done")
data_service.build_leave_one_out_evaluation_base()
print("loo done")
data_service.build_svd_top10_subset(
    sample_size=1000,
    random_seed=99,
    candidate_pool_size=100,
    artifact_prefix=PREFIX,
)
print("subset done")
cf_service.build_svd_top10_baseline(
    subset_size=1000,
    random_state=99,
    artifact_prefix=PREFIX,
)
print("svd done")
agentic_service._build_top10_formal_experiment(
    subset_size=1000,
    artifact_prefix=PREFIX,
)
print("agentic done")
hybrid_service.build_hybrid_svd_agentic_reranker(
    subset_size=1000,
    random_state=99,
    artifact_prefix=PREFIX,
)
print("hybrid done")
evaluation_service.compute_three_method_top10_metrics_from_saved_artifacts(
    subset_size=1000,
    bootstrap_samples=1000,
    random_seed=42,
    artifact_prefix=PREFIX,
)
print("evaluation done")
'@ | .\.venv\Scripts\python.exe -
```

## What This Command Produces

Formal run writes saved artefacts to `backend/app/data/processed/`. Expected key outputs include:

- `seed99_robustness_evaluation_base_table_top10_1000.json`
- `seed99_robustness_svd_recommendations_top10_1000.json`
- `seed99_robustness_agentic_recommendations_top10_1000.json`
- `seed99_robustness_hybrid_svd_agentic_recommendations_top10_1000.json`
- `seed99_robustness_per_user_metrics_top10_1000_three_methods.json`
- `seed99_robustness_metric_summary_top10_1000_three_methods.json`
- `seed99_robustness_bootstrap_ci_report_top10_1000_three_methods.json`
- `seed99_robustness_validation_report_top10_1000_three_methods.json`
- `seed99_robustness_experiment_report_top10_1000.md`

A full list of Seed99 artefacts can be found at:

`SECOND_1000_USER_FILES.md`
