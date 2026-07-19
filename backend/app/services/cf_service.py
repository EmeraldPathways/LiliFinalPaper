from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from app.config import Settings
from app.utils.normalization import normalize_article_id, normalize_customer_id


@dataclass
class CollaborativeFilteringService:
    settings: Settings

    # The previous user-based cosine CF baseline is retained only for debugging.
    # The formal baseline follows the supervisor feedback and uses SVD Matrix Factorisation.

    def generate_all(self, train_df: pd.DataFrame, user_ids: list[str]) -> dict[str, list[dict[str, object]]]:
        train_df = self._normalize_ids(train_df)
        user_ids = [normalize_customer_id(user_id) for user_id in user_ids]
        matrix = self._build_user_item_matrix(train_df)
        metadata = self._article_metadata(train_df)
        user_history = train_df.groupby("customer_id")["article_id"].agg(set).to_dict()

        recommendations: dict[str, list[dict[str, object]]] = {}
        for user_id in user_ids:
            recommendations[user_id] = self.recommend_for_user(user_id, matrix, metadata, user_history)

        self.settings.cf_output_path.write_text(json.dumps(recommendations, indent=2), encoding="utf-8")
        return recommendations

    def generate_all_svd(
        self,
        train_df: pd.DataFrame,
        user_ids: list[str],
        candidate_pools: dict[str, list[str]],
        output_path=None,
    ) -> dict[str, list[dict[str, object]]]:
        train_df = self._normalize_ids(train_df)
        user_ids = [normalize_customer_id(user_id) for user_id in user_ids]
        candidate_pools = {
            normalize_customer_id(user_id): [normalize_article_id(article_id) for article_id in article_ids]
            for user_id, article_ids in candidate_pools.items()
        }
        matrix = self._build_user_item_matrix(train_df)
        metadata = self._article_metadata(train_df)
        user_history = train_df.groupby("customer_id")["article_id"].agg(set).to_dict()
        user_factors, item_factors = self._svd_factors(matrix)

        recommendations: dict[str, list[dict[str, object]]] = {}
        for user_id in user_ids:
            recommendations[user_id] = self.recommend_for_user_svd(
                user_id=user_id,
                matrix=matrix,
                metadata=metadata,
                user_history=user_history,
                user_factors=user_factors,
                item_factors=item_factors,
                candidate_article_ids=candidate_pools.get(user_id, []),
            )

        target_path = output_path or self.settings.cf_output_path
        target_path.write_text(json.dumps(recommendations, indent=2), encoding="utf-8")
        return recommendations

    def build_svd_top10_baseline(
        self,
        subset_size: int = 100,
        n_components: int = 50,
        random_state: int = 42,
        artifact_prefix: str | None = None,
        allow_overwrite: bool | None = None,
    ) -> dict[str, object]:
        processed = self._load_processed_interactions_with_articles()
        subset_rows = json.loads(
            self.settings.evaluation_base_table_svd_top10_json_path(
                subset_size,
                artifact_prefix=artifact_prefix,
            ).read_text(encoding="utf-8")
        )
        metadata = self._formal_article_metadata(processed)
        train_df = self._build_svd_training_interactions(processed, subset_rows)
        relevant_item_ids = self._subset_relevant_item_ids(subset_rows)
        train_df = train_df[train_df["article_id"].isin(relevant_item_ids)].copy()
        sparse_matrix, user_index, item_index = self._build_sparse_user_item_matrix(train_df)
        user_factors, item_factors, n_components_used = self._sparse_svd_factors(
            sparse_matrix,
            n_components=n_components,
            random_state=random_state,
        )

        results: list[dict[str, object]] = []
        users_with_svd_recommendations = 0
        users_without_svd_recommendations = 0
        users_with_full_top_10 = 0
        users_where_top_10_all_inside_candidate_pool = 0
        users_where_top_10_contains_training_items = 0
        users_with_unscoreable_candidates = 0
        scoreable_candidate_counts: list[int] = []
        example_svd_recommendation_users: list[str] = []
        example_svd_failure_users: list[str] = []

        for row in subset_rows:
            result = self._build_svd_result_for_user(
                row=row,
                metadata=metadata,
                user_index=user_index,
                item_index=item_index,
                user_factors=user_factors,
                item_factors=item_factors,
                top_k=self.settings.top_n,
            )
            results.append(result)
            top_10 = result["top_10_recommendations"]
            if top_10:
                users_with_svd_recommendations += 1
                if len(example_svd_recommendation_users) < 5:
                    example_svd_recommendation_users.append(result["customer_id"])
            else:
                users_without_svd_recommendations += 1
                if len(example_svd_failure_users) < 5:
                    example_svd_failure_users.append(result["customer_id"])
            if int(result["recommendation_count"]) == self.settings.top_n:
                users_with_full_top_10 += 1
            if result["top_10_all_inside_candidate_pool"]:
                users_where_top_10_all_inside_candidate_pool += 1
            if result["top_10_contains_training_items"]:
                users_where_top_10_contains_training_items += 1
            if int(result["svd_unscoreable_candidate_count"]) > 0:
                users_with_unscoreable_candidates += 1
            scoreable_candidate_counts.append(int(result["svd_scoreable_candidate_count"]))

        output_json_path = self.settings.ensure_output_path(
            self.settings.svd_recommendations_top10_json_path(
                subset_size,
                artifact_prefix=artifact_prefix,
            ),
            artifact_prefix=artifact_prefix,
            allow_overwrite=allow_overwrite,
        )
        output_csv_path = self.settings.ensure_output_path(
            self.settings.svd_recommendations_top10_csv_path(
                subset_size,
                artifact_prefix=artifact_prefix,
            ),
            artifact_prefix=artifact_prefix,
            allow_overwrite=allow_overwrite,
        )
        validation_path = self.settings.ensure_output_path(
            self.settings.svd_baseline_validation_report_top10_path(
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
            "evaluated_users_requested": len(subset_rows),
            "users_with_svd_recommendations": users_with_svd_recommendations,
            "users_without_svd_recommendations": users_without_svd_recommendations,
            "average_top_10_length": round(
                sum(int(result["recommendation_count"]) for result in results) / len(results), 2
            )
            if results
            else 0.0,
            "users_with_full_top_10": users_with_full_top_10,
            "users_where_top_10_all_inside_candidate_pool": users_where_top_10_all_inside_candidate_pool,
            "users_where_top_10_contains_training_items": users_where_top_10_contains_training_items,
            "users_with_unscoreable_candidates": users_with_unscoreable_candidates,
            "average_scoreable_candidate_count": round(sum(scoreable_candidate_counts) / len(scoreable_candidate_counts), 2)
            if scoreable_candidate_counts
            else 0.0,
            "min_scoreable_candidate_count": min(scoreable_candidate_counts) if scoreable_candidate_counts else 0,
            "max_scoreable_candidate_count": max(scoreable_candidate_counts) if scoreable_candidate_counts else 0,
            "svd_matrix_shape": [int(sparse_matrix.shape[0]), int(sparse_matrix.shape[1])],
            "svd_n_components_used": n_components_used,
            "random_state": random_state,
            "example_svd_recommendation_users": example_svd_recommendation_users,
            "example_svd_failure_users": example_svd_failure_users,
        }
        validation_path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        print(
            "SVD baseline summary: "
            f"matrix_shape={tuple(report['svd_matrix_shape'])}, "
            f"n_components_used={report['svd_n_components_used']}, "
            f"users_evaluated={report['evaluated_users_requested']}, "
            f"users_with_valid_top_10={report['users_with_full_top_10']}, "
            f"average_top_10_length={report['average_top_10_length']}, "
            f"top_10_all_inside_candidate_pool_count={report['users_where_top_10_all_inside_candidate_pool']}, "
            f"top_10_contains_training_items_count={report['users_where_top_10_contains_training_items']}"
        )
        return report

    def build_svd_top10_debug_baseline(
        self,
        n_components: int = 50,
        random_state: int = 42,
    ) -> dict[str, object]:
        return self.build_svd_top10_baseline(
            subset_size=100,
            n_components=n_components,
            random_state=random_state,
        )

    def recommend_for_user(
        self,
        user_id: str,
        matrix: pd.DataFrame,
        metadata: dict[str, dict[str, str]],
        user_history: dict[str, set[str]],
    ) -> list[dict[str, object]]:
        user_id = normalize_customer_id(user_id)
        if user_id not in matrix.index:
            return []

        target = matrix.loc[user_id].to_numpy(dtype=float)
        norms = np.linalg.norm(matrix.to_numpy(dtype=float), axis=1)
        target_norm = np.linalg.norm(target)
        similarities: dict[str, float] = {}
        for idx, other_user in enumerate(matrix.index):
            if other_user == user_id or norms[idx] == 0 or target_norm == 0:
                continue
            sim = float(np.dot(target, matrix.iloc[idx].to_numpy(dtype=float)) / (target_norm * norms[idx]))
            if sim > 0:
                similarities[str(other_user)] = sim

        scores: defaultdict[str, float] = defaultdict(float)
        purchased = user_history.get(user_id, set())
        for neighbor_id, similarity in similarities.items():
            for article_id in user_history.get(neighbor_id, set()):
                if article_id in purchased:
                    continue
                scores[normalize_article_id(article_id)] += similarity

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[: self.settings.top_n]
        result = []
        for article_id, score in ranked:
            details = metadata.get(article_id)
            if not details:
                continue
            result.append(
                {
                    "article_id": article_id,
                    "product_name": details["product_name"],
                    "product_type": details["product_type"],
                    "product_group": details["product_group"],
                    "colour": details["colour"],
                    "appearance": details["appearance"],
                    "score": round(float(score), 4),
                    "model": "collaborative_filtering",
                }
            )
        return result

    def recommend_for_user_svd(
        self,
        user_id: str,
        matrix: pd.DataFrame,
        metadata: dict[str, dict[str, str]],
        user_history: dict[str, set[str]],
        user_factors: pd.DataFrame,
        item_factors: pd.DataFrame,
        candidate_article_ids: list[str],
    ) -> list[dict[str, object]]:
        user_id = normalize_customer_id(user_id)
        if user_id not in user_factors.index:
            return []

        purchased = user_history.get(user_id, set())
        scored_candidates: list[tuple[str, float]] = []
        for article_id in [normalize_article_id(article_id) for article_id in candidate_article_ids]:
            if article_id in purchased or article_id not in item_factors.index:
                continue
            score = float(np.dot(user_factors.loc[user_id], item_factors.loc[article_id]))
            scored_candidates.append((article_id, score))

        ranked = sorted(scored_candidates, key=lambda item: item[1], reverse=True)[: self.settings.top_n]
        results = []
        for article_id, score in ranked:
            details = metadata.get(article_id)
            if not details:
                continue
            results.append(
                {
                    "article_id": article_id,
                    "product_name": details["product_name"],
                    "product_type": details["product_type"],
                    "product_group": details["product_group"],
                    "colour": details["colour"],
                    "appearance": details["appearance"],
                    "score": round(score, 4),
                    "model": "svd_matrix_factorization",
                }
            )
        return results

    @staticmethod
    def _build_user_item_matrix(train_df: pd.DataFrame) -> pd.DataFrame:
        interactions = train_df.assign(interaction=1)
        return interactions.pivot_table(
            index="customer_id",
            columns="article_id",
            values="interaction",
            aggfunc="max",
            fill_value=0,
        )

    @staticmethod
    def _build_sparse_user_item_matrix(
        train_df: pd.DataFrame,
    ) -> tuple[csr_matrix, dict[str, int], dict[str, int]]:
        deduped = train_df.drop_duplicates(["customer_id", "article_id"]).copy()
        user_ids = sorted(deduped["customer_id"].astype(str).unique().tolist())
        item_ids = sorted(deduped["article_id"].astype(str).unique().tolist())
        user_index = {user_id: index for index, user_id in enumerate(user_ids)}
        item_index = {item_id: index for index, item_id in enumerate(item_ids)}
        row_indices = deduped["customer_id"].map(user_index).to_numpy(dtype=int)
        col_indices = deduped["article_id"].map(item_index).to_numpy(dtype=int)
        data = np.ones(len(deduped), dtype=np.float32)
        matrix = csr_matrix((data, (row_indices, col_indices)), shape=(len(user_ids), len(item_ids)))
        return matrix, user_index, item_index

    @staticmethod
    def _article_metadata(train_df: pd.DataFrame) -> dict[str, dict[str, str]]:
        deduped = train_df.drop_duplicates("article_id")
        return {
            normalize_article_id(row["article_id"]): {
                "product_name": str(row["product_name"]),
                "product_type": str(row["product_type"]),
                "product_group": str(row["product_group"]),
                "colour": str(row["colour"]),
                "appearance": str(row["appearance"]),
            }
            for _, row in deduped.iterrows()
        }

    @staticmethod
    def _normalize_ids(frame: pd.DataFrame) -> pd.DataFrame:
        normalized = frame.copy()
        normalized["article_id"] = normalized["article_id"].map(normalize_article_id)
        normalized["customer_id"] = normalized["customer_id"].map(normalize_customer_id)
        return normalized

    @staticmethod
    def _svd_factors(
        matrix: pd.DataFrame,
        n_components: int = 50,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame, int]:
        if matrix.empty:
            return pd.DataFrame(index=matrix.index), pd.DataFrame(index=matrix.columns), 0

        values = matrix.to_numpy(dtype=float)
        rng = np.random.default_rng(random_state)
        user_means = values.mean(axis=1, keepdims=True)
        centered = values - user_means
        centered = centered + rng.normal(0.0, 1e-9, centered.shape)
        u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        max_rank = min(matrix.shape[0], matrix.shape[1])
        rank = max(1, min(n_components, len(singular_values), max_rank))
        sigma = np.diag(np.sqrt(singular_values[:rank]))
        user_latent = pd.DataFrame(
            u[:, :rank] @ sigma,
            index=matrix.index,
        )
        item_latent = pd.DataFrame(
            (sigma @ vt[:rank, :]).T,
            index=matrix.columns,
        )
        return user_latent, item_latent, rank

    @staticmethod
    def _sparse_svd_factors(
        matrix: csr_matrix,
        n_components: int = 50,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            return np.empty((matrix.shape[0], 0)), np.empty((matrix.shape[1], 0)), 0
        max_rank = min(matrix.shape[0] - 1, matrix.shape[1] - 1)
        if max_rank <= 0:
            return matrix.toarray().astype(float), np.eye(matrix.shape[1], dtype=float), 1
        rank = max(1, min(n_components, max_rank))
        svd = TruncatedSVD(n_components=rank, random_state=random_state)
        user_factors = svd.fit_transform(matrix)
        item_factors = svd.components_.T
        return user_factors, item_factors, rank

    def _load_processed_interactions_with_articles(self) -> pd.DataFrame:
        processed = pd.read_csv(
            self.settings.processed_interactions_with_articles_csv_path,
            dtype={"article_id": "string", "customer_id": "string"},
            parse_dates=["t_dat"],
        )
        processed["article_id"] = processed["article_id"].map(normalize_article_id)
        processed["customer_id"] = processed["customer_id"].map(normalize_customer_id)
        return processed

    @staticmethod
    def _formal_article_metadata(processed: pd.DataFrame) -> dict[str, dict[str, str | None]]:
        deduped = processed.drop_duplicates("article_id")
        return {
            normalize_article_id(row["article_id"]): {
                "product_type_name": None if pd.isna(row["product_type_name"]) else str(row["product_type_name"]),
                "product_group_name": None if pd.isna(row["product_group_name"]) else str(row["product_group_name"]),
                "colour_group_name": None if pd.isna(row["colour_group_name"]) else str(row["colour_group_name"]),
                "graphical_appearance_name": None
                if pd.isna(row["graphical_appearance_name"])
                else str(row["graphical_appearance_name"]),
                "garment_group_name": None if pd.isna(row["garment_group_name"]) else str(row["garment_group_name"]),
            }
            for _, row in deduped.iterrows()
        }

    def _build_svd_training_interactions(
        self,
        processed: pd.DataFrame,
        subset_rows: list[dict[str, object]],
    ) -> pd.DataFrame:
        subset_customer_ids = {normalize_customer_id(row["customer_id"]) for row in subset_rows}
        non_subset = processed[~processed["customer_id"].isin(subset_customer_ids)].copy()
        subset_training_rows = []
        metadata_lookup = processed.drop_duplicates("article_id").set_index("article_id")
        for row in subset_rows:
            customer_id = normalize_customer_id(row["customer_id"])
            for article_id in row.get("train_article_ids", []):
                normalized_article_id = normalize_article_id(article_id)
                if normalized_article_id not in metadata_lookup.index:
                    continue
                article_row = metadata_lookup.loc[normalized_article_id]
                subset_training_rows.append(
                    {
                        "customer_id": customer_id,
                        "article_id": normalized_article_id,
                        "product_type_name": article_row["product_type_name"],
                        "product_group_name": article_row["product_group_name"],
                        "graphical_appearance_name": article_row["graphical_appearance_name"],
                        "colour_group_name": article_row["colour_group_name"],
                        "garment_group_name": article_row["garment_group_name"],
                        "department_name": article_row["department_name"],
                        "section_name": article_row["section_name"],
                        "index_name": article_row["index_name"],
                        "detail_desc": article_row["detail_desc"],
                    }
                )
        subset_training_df = pd.DataFrame(subset_training_rows)
        training_df = pd.concat(
            [
                non_subset[
                    [
                        "customer_id",
                        "article_id",
                        "product_type_name",
                        "product_group_name",
                        "graphical_appearance_name",
                        "colour_group_name",
                        "garment_group_name",
                        "department_name",
                        "section_name",
                        "index_name",
                        "detail_desc",
                    ]
                ],
                subset_training_df,
            ],
            ignore_index=True,
        )
        return training_df.drop_duplicates(["customer_id", "article_id"]).reset_index(drop=True)

    @staticmethod
    def _subset_relevant_item_ids(subset_rows: list[dict[str, object]]) -> set[str]:
        relevant_item_ids: set[str] = set()
        for row in subset_rows:
            relevant_item_ids.update(
                normalize_article_id(article_id) for article_id in row.get("train_article_ids", [])
            )
            relevant_item_ids.update(
                normalize_article_id(article_id) for article_id in row.get("candidate_pool_article_ids", [])
            )
            relevant_item_ids.add(normalize_article_id(row.get("ground_truth_article_id")))
        return {article_id for article_id in relevant_item_ids if article_id}

    def _build_svd_result_for_user(
        self,
        *,
        row: dict[str, object],
        metadata: dict[str, dict[str, str | None]],
        user_index: dict[str, int],
        item_index: dict[str, int],
        user_factors: np.ndarray,
        item_factors: np.ndarray,
        top_k: int,
    ) -> dict[str, object]:
        customer_id = normalize_customer_id(row["customer_id"])
        train_article_ids = {normalize_article_id(article_id) for article_id in row.get("train_article_ids", [])}
        candidate_pool_article_ids = [normalize_article_id(article_id) for article_id in row.get("candidate_pool_article_ids", [])]
        scoreable_candidates = [
            article_id
            for article_id in candidate_pool_article_ids
            if article_id not in train_article_ids and article_id in item_index
        ]
        unscoreable_candidates = [
            article_id
            for article_id in candidate_pool_article_ids
            if article_id not in train_article_ids and article_id not in item_index
        ]
        recommendations: list[dict[str, object]] = []
        if customer_id in user_index:
            user_vector = user_factors[user_index[customer_id]]
            ranked = sorted(
                (
                    (article_id, float(np.dot(user_vector, item_factors[item_index[article_id]])))
                    for article_id in scoreable_candidates
                ),
                key=lambda item: item[1],
                reverse=True,
            )[:top_k]
            for rank, (article_id, score) in enumerate(ranked, start=1):
                details = metadata.get(article_id, {})
                recommendations.append(
                    {
                        "rank": rank,
                        "article_id": article_id,
                        "score": round(score, 6),
                        "product_type_name": details.get("product_type_name"),
                        "product_group_name": details.get("product_group_name"),
                        "colour_group_name": details.get("colour_group_name"),
                        "graphical_appearance_name": details.get("graphical_appearance_name"),
                        "garment_group_name": details.get("garment_group_name"),
                    }
                )

        top_10_article_ids = [item["article_id"] for item in recommendations]
        candidate_pool_set = set(candidate_pool_article_ids)
        return {
            "customer_id": customer_id,
            "method": "svd_matrix_factorisation",
            "ground_truth_article_id": normalize_article_id(row["ground_truth_article_id"]),
            "candidate_pool_size": int(row.get("candidate_pool_size", len(candidate_pool_article_ids))),
            "top_10_recommendations": recommendations,
            "recommendation_count": len(recommendations),
            "top_10_all_inside_candidate_pool": all(article_id in candidate_pool_set for article_id in top_10_article_ids),
            "top_10_contains_training_items": any(article_id in train_article_ids for article_id in top_10_article_ids),
            "svd_scoreable_candidate_count": len(scoreable_candidates),
            "svd_unscoreable_candidate_count": len(unscoreable_candidates),
        }
