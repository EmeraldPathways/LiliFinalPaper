**Reproducing the First Formal 1000-User Experiment**

**Important Note**
**The repository does not expose the full three-method first 1000-user pipeline as a single packaged CLI command. The exact reproducible run is a sequence of service calls. The command block below reproduces the saved original 1000-user formal experiment artefacts from the current codebase.**

**Exact First 1000-User Run Command**
**Run this from the repository root:**
**cd backend**
**@'**
**from app.config import get_settings**
**from app.services.data_service import DataService**
**from app.services.cf_service import CollaborativeFilteringService**
**from app.services.agentic_service import AgenticRecommendationService**
**from app.services.hybrid_service import HybridRecommendationService**
**from app.services.evaluation_service import EvaluationService**

**settings = get_settings()**
**data_service = DataService(settings)**
**cf_service = CollaborativeFilteringService(settings)**
**agentic_service = AgenticRecommendationService(settings)**
**hybrid_service = HybridRecommendationService(settings, cf_service, agentic_service)**
**evaluation_service = EvaluationService(settings)**

**data_service.build_processed_interactions_with_articles()**
**data_service.build_leave_one_out_evaluation_base()**
**data_service.build_svd_top10_subset(**
    **sample_size=1000,**
    **random_seed=42,**
    **candidate_pool_size=100,**
**)**
**cf_service.build_svd_top10_baseline(**
    **subset_size=1000,**
    **random_state=42,**
**)**
**agentic_service._build_top10_formal_experiment(**
    **subset_size=1000,**
**)**
**hybrid_service.build_hybrid_svd_agentic_reranker(**
    **subset_size=1000,**
    **random_state=42,**
**)**
**evaluation_service.compute_three_method_top10_metrics_from_saved_artifacts(**
    **subset_size=1000,**
    **bootstrap_samples=1000,**
    **random_seed=42,**
**)**
**'@ | .\\.venv\\Scripts\\python.exe -**




**What This Command Produces**
**Formal run writes saved artefacts to the backend/app/data/processed/. Expected key outputs include:**
**evaluation_base_table_svd_top10_1000.json**
**svd_recommendations_top10_1000.json**
**agentic_recommendations_top10_1000.json**
**hybrid_svd_agentic_recommendations_top10_1000.json**
**per_user_metrics_top10_1000_three_methods.json**
**metric_summary_top10_1000_three_methods.json**
**bootstrap_ci_report_top10_1000_three_methods.json**
**hybrid_svd_agentic_audit_report_top10_1000.md**
