from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.config import Settings
from app.services.agentic_service import AgenticRecommendationService
from app.services.cf_service import CollaborativeFilteringService
from app.utils.normalization import normalize_article_id, normalize_customer_id


@dataclass
class HybridRecommendationService:
    settings: Settings
    cf_service: CollaborativeFilteringService
    agentic_service: AgenticRecommendationService

    def build_hybrid_svd_agentic_reranker(
        self,
        subset_size: int = 1000,
        random_state: int = 42,
        artifact_prefix: str | None = None,
        allow_overwrite: bool | None = None,
    ) -> dict[str, object]:
        processed = self.cf_service._load_processed_interactions_with_articles()
        subset_rows = json.loads(
            self.settings.evaluation_base_table_svd_top10_json_path(
                subset_size,
                artifact_prefix=artifact_prefix,
            ).read_text(encoding="utf-8")
        )
        metadata = self.cf_service._formal_article_metadata(processed)
        catalogue = self.agentic_service._build_formal_catalogue(processed)
        train_df = self.cf_service._build_svd_training_interactions(processed, subset_rows)
        relevant_item_ids = self.cf_service._subset_relevant_item_ids(subset_rows)
        train_df = train_df[train_df["article_id"].isin(relevant_item_ids)].copy()
        sparse_matrix, user_index, item_index = self.cf_service._build_sparse_user_item_matrix(train_df)
        user_factors, item_factors, _ = self.cf_service._sparse_svd_factors(
            sparse_matrix,
            n_components=50,
            random_state=random_state,
        )

        results: list[dict[str, object]] = []
        users_with_hybrid_recommendations = 0
        users_with_full_top_10 = 0
        users_where_top_10_all_inside_candidate_pool = 0
        users_where_top_10_contains_training_items = 0
        users_with_equal_svd_scores = 0
        users_with_equal_agentic_scores = 0
        missing_metadata_for_diversity_count = 0
        example_hybrid_recommendation_users: list[str] = []
        example_hybrid_failure_users: list[str] = []

        for row in subset_rows:
            result = self._build_hybrid_result_for_user(
                row=row,
                metadata=metadata,
                catalogue=catalogue,
                user_index=user_index,
                item_index=item_index,
                user_factors=user_factors,
                item_factors=item_factors,
            )
            results.append(result)
            if int(result["recommendation_count"]) > 0:
                users_with_hybrid_recommendations += 1
                if len(example_hybrid_recommendation_users) < 5:
                    example_hybrid_recommendation_users.append(result["customer_id"])
            else:
                if len(example_hybrid_failure_users) < 5:
                    example_hybrid_failure_users.append(result["customer_id"])
            if int(result["recommendation_count"]) == self.settings.top_n:
                users_with_full_top_10 += 1
            if bool(result["top_10_all_inside_candidate_pool"]):
                users_where_top_10_all_inside_candidate_pool += 1
            if bool(result["top_10_contains_training_items"]):
                users_where_top_10_contains_training_items += 1
            if bool(result["equal_svd_scores"]):
                users_with_equal_svd_scores += 1
            if bool(result["equal_agentic_scores"]):
                users_with_equal_agentic_scores += 1
            missing_metadata_for_diversity_count += int(result["missing_metadata_for_diversity_count"])

        output_json_path = self.settings.ensure_output_path(
            self.settings.hybrid_svd_agentic_recommendations_top10_json_path(
                subset_size,
                artifact_prefix=artifact_prefix,
            ),
            artifact_prefix=artifact_prefix,
            allow_overwrite=allow_overwrite,
        )
        output_csv_path = self.settings.ensure_output_path(
            self.settings.hybrid_svd_agentic_recommendations_top10_csv_path(
                subset_size,
                artifact_prefix=artifact_prefix,
            ),
            artifact_prefix=artifact_prefix,
            allow_overwrite=allow_overwrite,
        )
        validation_path = self.settings.ensure_output_path(
            self.settings.hybrid_svd_agentic_validation_report_top10_path(
                subset_size,
                artifact_prefix=artifact_prefix,
            ),
            artifact_prefix=artifact_prefix,
            allow_overwrite=allow_overwrite,
        )
        output_json_path.write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )
        output_csv_path.write_text(
            pd.DataFrame(
                [
                    {
                        **result,
                        "top_10_recommendations": json.dumps(result["top_10_recommendations"]),
                    }
                    for result in results
                ]
            ).to_csv(index=False),
            encoding="utf-8",
        )

        report = {
            "users_evaluated": len(subset_rows),
            "users_with_hybrid_recommendations": users_with_hybrid_recommendations,
            "users_with_full_top_10": users_with_full_top_10,
            "users_where_top_10_all_inside_candidate_pool": users_where_top_10_all_inside_candidate_pool,
            "users_where_top_10_contains_training_items": users_where_top_10_contains_training_items,
            "users_with_equal_svd_scores": users_with_equal_svd_scores,
            "users_with_equal_agentic_scores": users_with_equal_agentic_scores,
            "missing_metadata_for_diversity_count": missing_metadata_for_diversity_count,
            "average_top_10_length": round(
                sum(int(result["recommendation_count"]) for result in results) / len(results), 2
            )
            if results
            else 0.0,
            "example_hybrid_recommendation_users": example_hybrid_recommendation_users,
            "example_hybrid_failure_users": example_hybrid_failure_users,
        }
        validation_path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        return report

    def _build_hybrid_result_for_user(
        self,
        *,
        row: dict[str, object],
        metadata: dict[str, dict[str, str | None]],
        catalogue: pd.DataFrame,
        user_index: dict[str, int],
        item_index: dict[str, int],
        user_factors: np.ndarray,
        item_factors: np.ndarray,
    ) -> dict[str, object]:
        customer_id = normalize_customer_id(row["customer_id"])
        ground_truth_article_id = normalize_article_id(row["ground_truth_article_id"])
        train_article_ids = [normalize_article_id(article_id) for article_id in row.get("train_article_ids", [])]
        train_article_id_set = {article_id for article_id in train_article_ids if article_id}
        candidate_pool_article_ids = [
            normalize_article_id(article_id) for article_id in row.get("candidate_pool_article_ids", [])
        ]
        candidate_pool_set = set(candidate_pool_article_ids)
        eligible_candidate_ids = [
            article_id for article_id in candidate_pool_article_ids if article_id not in train_article_id_set
        ]

        train_history = self.agentic_service._build_formal_history_frame(customer_id, train_article_ids, catalogue)
        preference_profile = self.agentic_service._infer_formal_user_profile(customer_id, train_history)
        candidate_frame = catalogue[catalogue["article_id"].isin(eligible_candidate_ids)].copy()
        candidate_frame = candidate_frame.sort_values("transaction_date", ascending=False).copy()

        svd_scores = self._score_svd_candidates(
            customer_id=customer_id,
            candidate_article_ids=eligible_candidate_ids,
            user_index=user_index,
            item_index=item_index,
            user_factors=user_factors,
            item_factors=item_factors,
        )
        agentic_scored = self.agentic_service.score_candidates(
            preference_profile,
            candidate_frame,
            train_history,
            limit=len(candidate_frame),
        )
        agentic_scores = {normalize_article_id(item["article_id"]): float(item["score"]) for item in agentic_scored}

        normalized_svd_scores, equal_svd_scores = self._normalize_score_map(
            {article_id: svd_scores.get(article_id, 0.0) for article_id in eligible_candidate_ids}
        )
        normalized_agentic_scores, equal_agentic_scores = self._normalize_score_map(
            {article_id: agentic_scores.get(article_id, 0.0) for article_id in eligible_candidate_ids}
        )

        candidate_rows = {
            normalize_article_id(article_id): candidate_frame.loc[candidate_frame["article_id"] == article_id].iloc[0]
            for article_id in candidate_frame["article_id"].tolist()
        }
        selected_article_ids: list[str] = []
        recommendation_rows: list[dict[str, object]] = []
        missing_metadata_for_diversity_count = 0

        while len(selected_article_ids) < self.settings.top_n:
            remaining_ids = [article_id for article_id in eligible_candidate_ids if article_id not in selected_article_ids]
            if not remaining_ids:
                break
            scored_candidates: list[dict[str, object]] = []
            for article_id in remaining_ids:
                article_row = candidate_rows.get(article_id)
                diversity_bonus, missing_count = self._compute_diversity_bonus(
                    selected_article_ids=selected_article_ids,
                    candidate_article_id=article_id,
                    candidate_rows=candidate_rows,
                )
                missing_metadata_for_diversity_count += missing_count
                hybrid_score = (
                    0.70 * normalized_svd_scores.get(article_id, 0.0)
                    + 0.25 * normalized_agentic_scores.get(article_id, 0.0)
                    + 0.05 * diversity_bonus
                )
                scored_candidates.append(
                    {
                        "article_id": article_id,
                        "hybrid_score": hybrid_score,
                        "normalized_svd_score": normalized_svd_scores.get(article_id, 0.0),
                        "normalized_agentic_score": normalized_agentic_scores.get(article_id, 0.0),
                        "diversity_bonus": diversity_bonus,
                        "article_row": article_row,
                    }
                )

            scored_candidates.sort(
                key=lambda item: (
                    float(item["hybrid_score"]),
                    float(item["normalized_svd_score"]),
                    float(item["normalized_agentic_score"]),
                    str(item["article_id"]),
                ),
                reverse=True,
            )
            chosen = scored_candidates[0]
            selected_article_ids.append(chosen["article_id"])
            article_row = chosen["article_row"]
            recommendation_rows.append(
                {
                    "rank": len(selected_article_ids),
                    "article_id": chosen["article_id"],
                    "hybrid_score": round(float(chosen["hybrid_score"]), 6),
                    "normalized_svd_score": round(float(chosen["normalized_svd_score"]), 6),
                    "normalized_agentic_score": round(float(chosen["normalized_agentic_score"]), 6),
                    "diversity_bonus": round(float(chosen["diversity_bonus"]), 6),
                    "product_type_name": metadata.get(chosen["article_id"], {}).get("product_type_name"),
                    "product_group_name": metadata.get(chosen["article_id"], {}).get("product_group_name"),
                    "colour_group_name": metadata.get(chosen["article_id"], {}).get("colour_group_name"),
                    "graphical_appearance_name": metadata.get(chosen["article_id"], {}).get("graphical_appearance_name"),
                    "garment_group_name": metadata.get(chosen["article_id"], {}).get("garment_group_name"),
                    "recommendation_reason": (
                        "Recommended because it combines SVD behavioural relevance with agentic metadata evidence "
                        "and a small diversity adjustment."
                    ),
                }
            )

        return {
            "customer_id": customer_id,
            "method": "hybrid_svd_agentic_reranker",
            "ground_truth_article_id": ground_truth_article_id,
            "candidate_pool_size": int(row.get("candidate_pool_size", len(candidate_pool_article_ids))),
            "top_10_recommendations": recommendation_rows,
            "recommendation_count": len(recommendation_rows),
            "top_10_all_inside_candidate_pool": all(article_id in candidate_pool_set for article_id in selected_article_ids),
            "top_10_contains_training_items": any(article_id in train_article_id_set for article_id in selected_article_ids),
            "equal_svd_scores": equal_svd_scores,
            "equal_agentic_scores": equal_agentic_scores,
            "missing_metadata_for_diversity_count": missing_metadata_for_diversity_count,
        }

    @staticmethod
    def _score_svd_candidates(
        *,
        customer_id: str,
        candidate_article_ids: list[str],
        user_index: dict[str, int],
        item_index: dict[str, int],
        user_factors: np.ndarray,
        item_factors: np.ndarray,
    ) -> dict[str, float]:
        if customer_id not in user_index:
            return {article_id: 0.0 for article_id in candidate_article_ids}
        user_vector = user_factors[user_index[customer_id]]
        scores: dict[str, float] = {}
        for article_id in candidate_article_ids:
            if article_id not in item_index:
                scores[article_id] = 0.0
                continue
            scores[article_id] = float(np.dot(user_vector, item_factors[item_index[article_id]]))
        return scores

    @staticmethod
    def _normalize_score_map(score_map: dict[str, float]) -> tuple[dict[str, float], bool]:
        if not score_map:
            return {}, False
        values = list(score_map.values())
        min_value = min(values)
        max_value = max(values)
        if abs(max_value - min_value) < 1e-12:
            return ({key: 0.5 for key in score_map}, True)
        return (
            {
                key: (float(value) - min_value) / (max_value - min_value)
                for key, value in score_map.items()
            },
            False,
        )

    @staticmethod
    def _compute_diversity_bonus(
        *,
        selected_article_ids: list[str],
        candidate_article_id: str,
        candidate_rows: dict[str, pd.Series],
    ) -> tuple[float, int]:
        if not selected_article_ids:
            return 0.0, 0
        fields = (
            "product_group_name",
            "colour_group_name",
            "graphical_appearance_name",
        )
        selected_sets: dict[str, set[str]] = {field: set() for field in fields}
        missing_metadata_count = 0
        for article_id in selected_article_ids:
            row = candidate_rows.get(article_id)
            if row is None:
                continue
            for field in fields:
                value = row.get(field)
                if value is None or str(value).strip() == "":
                    missing_metadata_count += 1
                    continue
                selected_sets[field].add(str(value))
        row = candidate_rows.get(candidate_article_id)
        if row is None:
            return 0.0, missing_metadata_count
        bonus = 0.0
        for field in fields:
            value = row.get(field)
            if value is None or str(value).strip() == "":
                missing_metadata_count += 1
                continue
            if str(value) not in selected_sets[field]:
                bonus += 1.0
        return bonus / 3.0, missing_metadata_count
