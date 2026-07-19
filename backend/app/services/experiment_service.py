from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from app.config import Settings
from app.models.experiment_state import ExperimentState
from app.services.agentic_service import AgenticRecommendationService
from app.services.cf_service import CollaborativeFilteringService
from app.services.data_service import DataService
from app.services.evaluation_service import EvaluationService


@dataclass
class ExperimentService:
    settings: Settings
    data_service: DataService
    cf_service: CollaborativeFilteringService
    agentic_service: AgenticRecommendationService
    evaluation_service: EvaluationService

    def run(self, experiment_mode: str = Settings.LEGACY_DEBUG_EXPERIMENT_MODE) -> dict[str, object]:
        if experiment_mode == self.settings.SVD_TOP10_EXPERIMENT_MODE:
            return self._run_svd_top10_experiment()
        return self._run_legacy_debug_experiment()

    def _run_legacy_debug_experiment(self) -> dict[str, object]:
        summary = self.data_service.preprocess()
        train_df = self.data_service.load_train()
        test_df = self.data_service.load_test()
        user_ids = self._select_evaluable_users(train_df, test_df)
        summary["evaluated_user_ids"] = user_ids
        summary["evaluated_users"] = len(user_ids)
        self.settings.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        cf_recommendations = self.cf_service.generate_all(train_df, user_ids)
        agentic_recommendations = self.agentic_service.generate_all(train_df, user_ids)
        metrics = self.evaluation_service.evaluate(
            train_df, test_df, cf_recommendations, agentic_recommendations
        )

        state = ExperimentState(
            status="completed",
            dataset=self.settings.dataset_name,
            sample_size=int(summary["sample_size"]),
            train_size=int(summary["train_size"]),
            test_size=int(summary["test_size"]),
            evaluated_users=len(user_ids),
            models=["collaborative_filtering", "agentic_ai_framework"],
            metrics_ready=True,
            split_boundary_date=str(summary["split_boundary_date"]),
            artifacts={
                "interactions": str(self.settings.interactions_path),
                "train": str(self.settings.train_path),
                "test": str(self.settings.test_path),
                "cf_recommendations": str(self.settings.cf_output_path),
                "agentic_recommendations": str(self.settings.agentic_output_path),
                "metrics": str(self.settings.metrics_path),
            },
            updated_at=datetime.now(timezone.utc),
        )
        self.settings.experiment_state_path.write_text(
            state.model_dump_json(indent=2),
            encoding="utf-8",
        )

        return {
            "dataset": self.settings.dataset_name,
            "status": "completed",
            "sample_size": int(summary["sample_size"]),
            "train_size": int(summary["train_size"]),
            "test_size": int(summary["test_size"]),
            "models": ["collaborative_filtering", "agentic_ai_framework"],
            "metrics_ready": True,
            "evaluated_users": len(user_ids),
            "metrics": metrics,
        }

    def _run_svd_top10_experiment(self) -> dict[str, object]:
        interactions_path = self.settings.experiment_artifact_path("svd_top10_experiment", "interactions")
        summary_path = self.settings.experiment_artifact_path("svd_top10_experiment", "summary")
        train_path = self.settings.experiment_artifact_path("svd_top10_experiment", "train")
        test_path = self.settings.experiment_artifact_path("svd_top10_experiment", "test")
        summary = self.data_service.preprocess(
            interactions_path=interactions_path,
            summary_path=summary_path,
            train_path=train_path,
            test_path=test_path,
            split_method="leave_one_out",
        )
        train_df = pd.read_csv(train_path, dtype={"article_id": "string", "customer_id": "string"})
        test_df = pd.read_csv(test_path, dtype={"article_id": "string", "customer_id": "string"})
        user_ids = self._select_evaluable_users(train_df, test_df)
        summary["evaluated_user_ids"] = user_ids
        summary["evaluated_users"] = len(user_ids)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        candidate_pools = self.agentic_service.build_candidate_pools(train_df, user_ids)
        svd_output_path = self.settings.experiment_artifact_path("svd_top10_experiment", "cf_output")
        agentic_output_path = self.settings.experiment_artifact_path("svd_top10_experiment", "agentic_output")
        agentic_trace_path = self.settings.experiment_artifact_path("svd_top10_experiment", "agentic_trace")
        metrics_path = self.settings.experiment_artifact_path("svd_top10_experiment", "metrics")
        state_path = self.settings.experiment_artifact_path("svd_top10_experiment", "experiment_state")

        svd_recommendations = self.cf_service.generate_all_svd(
            train_df,
            user_ids,
            candidate_pools,
            output_path=svd_output_path,
        )
        agentic_recommendations = self.agentic_service.generate_all(
            train_df,
            user_ids,
            output_path=agentic_output_path,
            trace_path=agentic_trace_path,
        )
        metrics = self.evaluation_service.evaluate_top10_experiment(
            train_df,
            test_df,
            svd_recommendations,
            agentic_recommendations,
            output_path=metrics_path,
        )

        state = ExperimentState(
            status="completed",
            dataset=self.settings.dataset_name,
            sample_size=int(summary["sample_size"]),
            train_size=int(summary["train_size"]),
            test_size=int(summary["test_size"]),
            evaluated_users=len(user_ids),
            models=["svd_matrix_factorization", "agentic_ai_framework"],
            metrics_ready=True,
            split_boundary_date=str(summary["split_boundary_date"]),
            artifacts={
                "interactions": str(interactions_path),
                "train": str(train_path),
                "test": str(test_path),
                "svd_recommendations": str(svd_output_path),
                "agentic_recommendations": str(agentic_output_path),
                "agentic_trace": str(agentic_trace_path),
                "metrics": str(metrics_path),
            },
            updated_at=datetime.now(timezone.utc),
        )
        state_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")

        return {
            "dataset": self.settings.dataset_name,
            "status": "completed",
            "experiment_mode": "svd_top10_experiment",
            "sample_size": int(summary["sample_size"]),
            "train_size": int(summary["train_size"]),
            "test_size": int(summary["test_size"]),
            "models": ["svd_matrix_factorization", "agentic_ai_framework"],
            "metrics_ready": True,
            "evaluated_users": len(user_ids),
            "metrics": metrics,
        }

    def load_state(self, experiment_mode: str = Settings.LEGACY_DEBUG_EXPERIMENT_MODE) -> dict[str, object] | None:
        state_path = self.settings.experiment_state_path
        if experiment_mode != self.settings.LEGACY_DEBUG_EXPERIMENT_MODE:
            state_path = self.settings.experiment_artifact_path(experiment_mode, "experiment_state")
        if not state_path.exists():
            return None
        return json.loads(state_path.read_text(encoding="utf-8"))

    def _select_evaluable_users(self, train_df, test_df) -> list[str]:
        train_counts = train_df.groupby("customer_id").size()
        test_counts = test_df.groupby("customer_id").size()
        common_users = set(train_counts.index.astype(str)) & set(test_counts.index.astype(str))

        ranked_users = sorted(
            common_users,
            key=lambda user_id: (
                int(test_counts.get(user_id, 0)),
                int(train_counts.get(user_id, 0)),
                user_id,
            ),
            reverse=True,
        )
        return ranked_users[: self.settings.max_eval_users]
