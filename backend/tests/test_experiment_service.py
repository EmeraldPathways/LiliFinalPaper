from __future__ import annotations

import pandas as pd

from app.services.experiment_service import ExperimentService


class StubDataService:
    def __init__(self, settings, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        self.settings = settings
        self._train_df = train_df
        self._test_df = test_df

    def preprocess(self, **kwargs) -> dict[str, object]:
        interactions_path = kwargs.get("interactions_path", self.settings.interactions_path)
        summary_path = kwargs.get("summary_path", self.settings.summary_path)
        train_path = kwargs.get("train_path", self.settings.train_path)
        test_path = kwargs.get("test_path", self.settings.test_path)
        combined = pd.concat([self._train_df, self._test_df], ignore_index=True)
        combined.to_csv(interactions_path, index=False)
        self._train_df.to_csv(train_path, index=False)
        self._test_df.to_csv(test_path, index=False)
        return {
            "dataset": "fixture",
            "sample_size": len(self._train_df) + len(self._test_df),
            "train_size": len(self._train_df),
            "test_size": len(self._test_df),
            "split_boundary_date": "2024-01-03",
        }

    def load_train(self) -> pd.DataFrame:
        return self._train_df.copy()

    def load_test(self) -> pd.DataFrame:
        return self._test_df.copy()


class StubCollaborativeFilteringService:
    def __init__(self) -> None:
        self.called = False
        self.called_mode: str | None = None

    def generate_all(self, train_df: pd.DataFrame, user_ids: list[str]) -> dict[str, list[dict[str, object]]]:
        self.called = True
        self.called_mode = "legacy"
        return {user_id: [] for user_id in user_ids}

    def generate_all_svd(
        self,
        train_df: pd.DataFrame,
        user_ids: list[str],
        candidate_pools: dict[str, list[str]],
        **kwargs,
    ) -> dict[str, list[dict[str, object]]]:
        self.called = True
        self.called_mode = "svd"
        return {user_id: [{"article_id": candidate_pools[user_id][0], "model": "svd_matrix_factorization"}] for user_id in user_ids}


class StubAgenticRecommendationService:
    def build_candidate_pools(
        self, train_df: pd.DataFrame, user_ids: list[str]
    ) -> dict[str, list[str]]:
        return {user_id: ["a4", "a5"] for user_id in user_ids}

    def generate_all(self, train_df: pd.DataFrame, user_ids: list[str], **kwargs) -> dict[str, list[dict[str, object]]]:
        return {user_id: [{"article_id": "a5", "model": "agentic_ai_framework"}] for user_id in user_ids}


class StubEvaluationService:
    def __init__(self) -> None:
        self.called_mode: str | None = None

    def evaluate(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        cf_recommendations: dict[str, list[dict[str, object]]],
        agentic_recommendations: dict[str, list[dict[str, object]]],
    ) -> dict[str, object]:
        self.called_mode = "legacy"
        return {"collaborative_filtering": {}, "agentic_ai_framework": {}}

    def evaluate_top10_experiment(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        svd_recommendations: dict[str, list[dict[str, object]]],
        agentic_recommendations: dict[str, list[dict[str, object]]],
        **kwargs,
    ) -> dict[str, object]:
        self.called_mode = "svd_top10_experiment"
        return {"svd_matrix_factorization": {}, "agentic_ai_framework": {}}


def test_experiment_service_routes_svd_top10_mode_without_touching_legacy_defaults(
    isolated_env,
):
    train_df = pd.DataFrame(
        [
            ["u1", "a1", "Shirt", "Upper", "Blue", "Solid", "Blue Shirt", "2024-01-01"],
            ["u1", "a2", "Top", "Upper", "White", "Plain", "White Top", "2024-01-02"],
            ["u2", "a3", "Dress", "Full", "Red", "Patterned", "Red Dress", "2024-01-03"],
        ],
        columns=[
            "customer_id",
            "article_id",
            "product_type",
            "product_group",
            "colour",
            "appearance",
            "product_name",
            "transaction_date",
        ],
    )
    test_df = pd.DataFrame(
        [["u1", "a4", "Shirt", "Upper", "Blue", "Solid", "Oxford Shirt", "2024-01-04"]],
        columns=train_df.columns,
    )
    cf_service = StubCollaborativeFilteringService()
    evaluation_service = StubEvaluationService()

    service = ExperimentService(
        settings=isolated_env,
        data_service=StubDataService(isolated_env, train_df, test_df),
        cf_service=cf_service,
        agentic_service=StubAgenticRecommendationService(),
        evaluation_service=evaluation_service,
    )

    result = service.run("svd_top10_experiment")

    assert result["experiment_mode"] == "svd_top10_experiment"
    assert result["models"] == ["svd_matrix_factorization", "agentic_ai_framework"]
    assert cf_service.called_mode == "svd"
    assert evaluation_service.called_mode == "svd_top10_experiment"
