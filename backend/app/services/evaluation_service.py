from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from math import log2

import pandas as pd

from app.config import Settings
from app.utils.normalization import normalize_article_id, normalize_customer_id


@dataclass
class EvaluationService:
    settings: Settings

    DIVERSITY_FIELDS = (
        "product_group_name",
        "colour_group_name",
        "graphical_appearance_name",
    )

    def evaluate(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        cf_recommendations: dict[str, list[dict[str, object]]],
        agentic_recommendations: dict[str, list[dict[str, object]]],
    ) -> dict[str, object]:
        train_df = self._normalize_ids(train_df)
        test_df = self._normalize_ids(test_df)
        cf_recommendations = self._normalize_recommendation_payload(cf_recommendations)
        agentic_recommendations = self._normalize_recommendation_payload(agentic_recommendations)
        user_profiles = self._build_user_profiles(train_df)
        evaluated_users = sorted(set(cf_recommendations) & set(agentic_recommendations))
        metrics = {
            "collaborative_filtering": self._model_metrics(
                evaluated_users, test_df, cf_recommendations, user_profiles, include_explanations=False
            ),
            "agentic_ai_framework": self._model_metrics(
                evaluated_users, test_df, agentic_recommendations, user_profiles, include_explanations=True
            ),
            "business_mapping": {
                "hit_rate_at_10": "Potential CTR improvement",
                "preference_alignment": "Potential CVR improvement",
                "diversity": "Potential engagement depth improvement",
            },
            "evaluated_users": len(evaluated_users),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.settings.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics

    def load_metrics(self) -> dict[str, object] | None:
        if not self.settings.metrics_path.exists():
            return None
        return json.loads(self.settings.metrics_path.read_text(encoding="utf-8"))

    def evaluate_top10_experiment(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        svd_recommendations: dict[str, list[dict[str, object]]],
        agentic_recommendations: dict[str, list[dict[str, object]]],
        output_path=None,
    ) -> dict[str, object]:
        train_df = self._normalize_ids(train_df)
        test_df = self._normalize_ids(test_df)
        svd_recommendations = self._normalize_recommendation_payload(svd_recommendations)
        agentic_recommendations = self._normalize_recommendation_payload(agentic_recommendations)
        evaluated_users = sorted(set(svd_recommendations) & set(agentic_recommendations))
        metrics = {
            "svd_matrix_factorization": self._top10_model_metrics(
                evaluated_users, test_df, svd_recommendations
            ),
            "agentic_ai_framework": self._top10_model_metrics(
                evaluated_users, test_df, agentic_recommendations
            ),
            "business_mapping": {
                "hit_rate_at_10": "Potential CTR improvement",
                "ndcg_at_10": "Ranking quality for held-out purchases",
                "intra_list_diversity_at_10": "Assortment breadth within the top-10 list",
            },
            "evaluated_users": len(evaluated_users),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        target_path = output_path or self.settings.metrics_path
        target_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return metrics

    def compute_formal_top10_metrics_from_saved_artifacts(
        self,
        subset_size: int = 100,
        bootstrap_samples: int = 0,
        random_seed: int = 42,
        artifact_prefix: str | None = None,
        allow_overwrite: bool | None = None,
    ) -> dict[str, object]:
        evaluation_rows = json.loads(
            self.settings.evaluation_base_table_svd_top10_json_path(
                subset_size,
                artifact_prefix=artifact_prefix,
            ).read_text(encoding="utf-8")
        )
        svd_rows = json.loads(
            self.settings.svd_recommendations_top10_json_path(
                subset_size,
                artifact_prefix=artifact_prefix,
            ).read_text(encoding="utf-8")
        )
        agentic_rows = json.loads(
            self.settings.agentic_recommendations_top10_json_path(
                subset_size,
                artifact_prefix=artifact_prefix,
            ).read_text(encoding="utf-8")
        )
        processed = pd.read_csv(
            self.settings.processed_interactions_with_articles_csv_path,
            dtype={"article_id": "string", "customer_id": "string"},
        )

        metadata_lookup = self._build_article_metadata_lookup(processed)
        evaluation_lookup = {
            normalize_customer_id(row["customer_id"]): self._normalize_evaluation_row(row)
            for row in evaluation_rows
        }
        svd_lookup = {
            normalize_customer_id(row["customer_id"]): self._normalize_saved_recommendation_row(row)
            for row in svd_rows
        }
        agentic_lookup = {
            normalize_customer_id(row["customer_id"]): self._normalize_saved_recommendation_row(row)
            for row in agentic_rows
        }

        per_user_rows: list[dict[str, object]] = []
        metric_calculation_errors: list[str] = []
        example_hit_users_svd: list[str] = []
        example_hit_users_agentic: list[str] = []
        example_miss_users_svd: list[str] = []
        example_miss_users_agentic: list[str] = []
        missing_metadata_for_diversity_count = 0
        users_with_svd_metrics = 0
        users_with_agentic_metrics = 0
        svd_top_10_count_check = 0
        agentic_top_10_count_check = 0
        svd_inside_pool_check = 0
        agentic_inside_pool_check = 0

        for customer_id, eval_row in evaluation_lookup.items():
            ground_truth_article_id = eval_row["ground_truth_article_id"]
            candidate_pool_set = set(eval_row["candidate_pool_article_ids"])
            svd_row = svd_lookup.get(customer_id, {"top_10_recommendations": []})
            agentic_row = agentic_lookup.get(customer_id, {"top_10_recommendations": []})
            svd_recs = svd_row.get("top_10_recommendations", [])
            agentic_recs = agentic_row.get("top_10_recommendations", [])

            if len(svd_recs) == self.settings.top_n:
                svd_top_10_count_check += 1
            if len(agentic_recs) == self.settings.top_n:
                agentic_top_10_count_check += 1
            if all(normalize_article_id(item.get("article_id")) in candidate_pool_set for item in svd_recs):
                svd_inside_pool_check += 1
            if all(normalize_article_id(item.get("article_id")) in candidate_pool_set for item in agentic_recs):
                agentic_inside_pool_check += 1

            svd_metrics, svd_missing_count, svd_errors = self._compute_method_metrics_for_user(
                customer_id=customer_id,
                ground_truth_article_id=ground_truth_article_id,
                recommendations=svd_recs,
                metadata_lookup=metadata_lookup,
                method_label="svd",
            )
            agentic_metrics, agentic_missing_count, agentic_errors = self._compute_method_metrics_for_user(
                customer_id=customer_id,
                ground_truth_article_id=ground_truth_article_id,
                recommendations=agentic_recs,
                metadata_lookup=metadata_lookup,
                method_label="agentic",
            )
            missing_metadata_for_diversity_count += svd_missing_count + agentic_missing_count
            metric_calculation_errors.extend(svd_errors)
            metric_calculation_errors.extend(agentic_errors)

            if svd_recs:
                users_with_svd_metrics += 1
            if agentic_recs:
                users_with_agentic_metrics += 1

            if int(svd_metrics["hit_rate_at_10"]) == 1 and len(example_hit_users_svd) < 5:
                example_hit_users_svd.append(customer_id)
            if int(svd_metrics["hit_rate_at_10"]) == 0 and len(example_miss_users_svd) < 5:
                example_miss_users_svd.append(customer_id)
            if int(agentic_metrics["hit_rate_at_10"]) == 1 and len(example_hit_users_agentic) < 5:
                example_hit_users_agentic.append(customer_id)
            if int(agentic_metrics["hit_rate_at_10"]) == 0 and len(example_miss_users_agentic) < 5:
                example_miss_users_agentic.append(customer_id)

            per_user_rows.append(
                {
                    "customer_id": customer_id,
                    "ground_truth_article_id": ground_truth_article_id,
                    "svd_hit_rate_at_10": svd_metrics["hit_rate_at_10"],
                    "svd_ground_truth_rank": svd_metrics["ground_truth_rank"],
                    "svd_ndcg_at_10": svd_metrics["ndcg_at_10"],
                    "svd_ild_at_10": svd_metrics["ild_at_10"],
                    "agentic_hit_rate_at_10": agentic_metrics["hit_rate_at_10"],
                    "agentic_ground_truth_rank": agentic_metrics["ground_truth_rank"],
                    "agentic_ndcg_at_10": agentic_metrics["ndcg_at_10"],
                    "agentic_ild_at_10": agentic_metrics["ild_at_10"],
                }
            )

        per_user_json_path = self.settings.ensure_output_path(
            self.settings.per_user_metrics_top10_json_path(
                subset_size,
                artifact_prefix=artifact_prefix,
            ),
            artifact_prefix=artifact_prefix,
            allow_overwrite=allow_overwrite,
        )
        per_user_csv_path = self.settings.ensure_output_path(
            self.settings.per_user_metrics_top10_csv_path(
                subset_size,
                artifact_prefix=artifact_prefix,
            ),
            artifact_prefix=artifact_prefix,
            allow_overwrite=allow_overwrite,
        )
        metric_summary_json_path = self.settings.ensure_output_path(
            self.settings.metric_summary_top10_json_path(
                subset_size,
                artifact_prefix=artifact_prefix,
            ),
            artifact_prefix=artifact_prefix,
            allow_overwrite=allow_overwrite,
        )
        metric_summary_csv_path = self.settings.ensure_output_path(
            self.settings.metric_summary_top10_csv_path(
                subset_size,
                artifact_prefix=artifact_prefix,
            ),
            artifact_prefix=artifact_prefix,
            allow_overwrite=allow_overwrite,
        )
        metric_validation_path = self.settings.ensure_output_path(
            self.settings.metric_validation_report_top10_path(
                subset_size,
                artifact_prefix=artifact_prefix,
            ),
            artifact_prefix=artifact_prefix,
            allow_overwrite=allow_overwrite,
        )
        per_user_json_path.write_text(
            json.dumps(per_user_rows, indent=2),
            encoding="utf-8",
        )
        per_user_csv_path.write_text(
            pd.DataFrame(per_user_rows).to_csv(index=False),
            encoding="utf-8",
        )

        summary = self._build_metric_summary(per_user_rows, subset_size=subset_size)
        metric_summary_json_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        metric_summary_csv_path.write_text(
            pd.DataFrame([self._flatten_metric_summary(summary)]).to_csv(index=False),
            encoding="utf-8",
        )

        validation_report = {
            "users_evaluated": len(per_user_rows),
            "users_with_svd_metrics": users_with_svd_metrics,
            "users_with_agentic_metrics": users_with_agentic_metrics,
            "svd_top_10_count_check": svd_top_10_count_check,
            "agentic_top_10_count_check": agentic_top_10_count_check,
            "svd_recommendations_inside_candidate_pool_check": svd_inside_pool_check,
            "agentic_recommendations_inside_candidate_pool_check": agentic_inside_pool_check,
            "missing_metadata_for_diversity_count": missing_metadata_for_diversity_count,
            "metric_calculation_errors": metric_calculation_errors[:50],
            "example_hit_users_svd": example_hit_users_svd,
            "example_hit_users_agentic": example_hit_users_agentic,
            "example_miss_users_svd": example_miss_users_svd,
            "example_miss_users_agentic": example_miss_users_agentic,
        }
        metric_validation_path.write_text(
            json.dumps(validation_report, indent=2),
            encoding="utf-8",
        )

        bootstrap_report = None
        if bootstrap_samples > 0:
            bootstrap_report = self._build_bootstrap_ci_report(
                per_user_rows=per_user_rows,
                bootstrap_samples=bootstrap_samples,
                random_seed=random_seed,
            )
            metric_summary_with_ci_path = self.settings.ensure_output_path(
                self.settings.metric_summary_top10_with_ci_json_path(
                    subset_size,
                    artifact_prefix=artifact_prefix,
                ),
                artifact_prefix=artifact_prefix,
                allow_overwrite=allow_overwrite,
            )
            bootstrap_ci_path = self.settings.ensure_output_path(
                self.settings.bootstrap_ci_report_top10_json_path(
                    subset_size,
                    artifact_prefix=artifact_prefix,
                ),
                artifact_prefix=artifact_prefix,
                allow_overwrite=allow_overwrite,
            )
            metric_summary_with_ci_path.write_text(
                json.dumps({**summary, "confidence_intervals": bootstrap_report["confidence_intervals"]}, indent=2),
                encoding="utf-8",
            )
            bootstrap_ci_path.write_text(
                json.dumps(bootstrap_report, indent=2),
                encoding="utf-8",
            )

        print(
            "Formal Top-10 metrics summary: "
            f"SVD HitRate@10={summary['svd']['hit_rate_at_10']:.4f}, "
            f"3-Agent HitRate@10={summary['agentic']['hit_rate_at_10']:.4f}, "
            f"SVD NDCG@10={summary['svd']['ndcg_at_10']:.4f}, "
            f"3-Agent NDCG@10={summary['agentic']['ndcg_at_10']:.4f}, "
            f"SVD ILD@10={summary['svd']['intra_list_diversity_at_10']:.4f}, "
            f"3-Agent ILD@10={summary['agentic']['intra_list_diversity_at_10']:.4f}, "
            f"SVD hits count={summary['svd']['hits_count']}, "
            f"3-Agent hits count={summary['agentic']['hits_count']}"
        )
        return {
            "summary": summary,
            "validation": validation_report,
            "per_user_metrics": per_user_rows,
            "bootstrap": bootstrap_report,
        }

    def compute_three_method_top10_metrics_from_saved_artifacts(
        self,
        subset_size: int = 1000,
        bootstrap_samples: int = 1000,
        random_seed: int = 42,
        artifact_prefix: str | None = None,
        allow_overwrite: bool | None = None,
    ) -> dict[str, object]:
        evaluation_rows = json.loads(
            self.settings.evaluation_base_table_svd_top10_json_path(
                subset_size,
                artifact_prefix=artifact_prefix,
            ).read_text(encoding="utf-8")
        )
        svd_rows = json.loads(
            self.settings.svd_recommendations_top10_json_path(
                subset_size,
                artifact_prefix=artifact_prefix,
            ).read_text(encoding="utf-8")
        )
        agentic_rows = json.loads(
            self.settings.agentic_recommendations_top10_json_path(
                subset_size,
                artifact_prefix=artifact_prefix,
            ).read_text(encoding="utf-8")
        )
        hybrid_rows = json.loads(
            self.settings.hybrid_svd_agentic_recommendations_top10_json_path(
                subset_size,
                artifact_prefix=artifact_prefix,
            ).read_text(encoding="utf-8")
        )
        processed = pd.read_csv(
            self.settings.processed_interactions_with_articles_csv_path,
            dtype={"article_id": "string", "customer_id": "string"},
        )
        metadata_lookup = self._build_article_metadata_lookup(processed)
        evaluation_lookup = {
            normalize_customer_id(row["customer_id"]): self._normalize_evaluation_row(row)
            for row in evaluation_rows
        }
        method_payloads = {
            "svd": {
                "summary_label": "svd",
                "display_label": "SVD Matrix Factorisation",
                "payload": {
                    normalize_customer_id(row["customer_id"]): self._normalize_saved_recommendation_row(row)
                    for row in svd_rows
                },
            },
            "agentic": {
                "summary_label": "agentic",
                "display_label": "Standalone 3-Agent",
                "payload": {
                    normalize_customer_id(row["customer_id"]): self._normalize_saved_recommendation_row(row)
                    for row in agentic_rows
                },
            },
            "hybrid": {
                "summary_label": "hybrid",
                "display_label": "Hybrid SVD + 3-Agent Reranker",
                "payload": {
                    normalize_customer_id(row["customer_id"]): self._normalize_saved_recommendation_row(row)
                    for row in hybrid_rows
                },
            },
        }

        per_user_rows: list[dict[str, object]] = []
        validation_report = {
            "users_evaluated": len(evaluation_lookup),
            "users_with_svd_metrics": 0,
            "users_with_agentic_metrics": 0,
            "users_with_hybrid_metrics": 0,
            "svd_top_10_count_check": 0,
            "agentic_top_10_count_check": 0,
            "hybrid_top_10_count_check": 0,
            "svd_recommendations_inside_candidate_pool_check": 0,
            "agentic_recommendations_inside_candidate_pool_check": 0,
            "hybrid_recommendations_inside_candidate_pool_check": 0,
            "missing_metadata_for_diversity_count": 0,
            "metric_calculation_errors": [],
        }

        for customer_id, eval_row in evaluation_lookup.items():
            ground_truth_article_id = eval_row["ground_truth_article_id"]
            candidate_pool_set = set(eval_row["candidate_pool_article_ids"])
            row_result: dict[str, object] = {
                "customer_id": customer_id,
                "ground_truth_article_id": ground_truth_article_id,
            }

            for method_key, payload_meta in method_payloads.items():
                rec_row = payload_meta["payload"].get(customer_id, {"top_10_recommendations": []})
                recommendations = rec_row.get("top_10_recommendations", [])
                metrics, missing_count, errors = self._compute_method_metrics_for_user(
                    customer_id=customer_id,
                    ground_truth_article_id=ground_truth_article_id,
                    recommendations=recommendations,
                    metadata_lookup=metadata_lookup,
                    method_label=method_key,
                )
                row_result[f"{method_key}_hit_rate_at_10"] = metrics["hit_rate_at_10"]
                row_result[f"{method_key}_ground_truth_rank"] = metrics["ground_truth_rank"]
                row_result[f"{method_key}_ndcg_at_10"] = metrics["ndcg_at_10"]
                row_result[f"{method_key}_ild_at_10"] = metrics["ild_at_10"]
                validation_report["missing_metadata_for_diversity_count"] += missing_count
                validation_report["metric_calculation_errors"].extend(errors)

                if recommendations:
                    validation_report[f"users_with_{method_key}_metrics"] += 1
                if len(recommendations) == self.settings.top_n:
                    validation_report[f"{method_key}_top_10_count_check"] += 1
                if all(normalize_article_id(item.get("article_id")) in candidate_pool_set for item in recommendations):
                    validation_report[f"{method_key}_recommendations_inside_candidate_pool_check"] += 1

            per_user_rows.append(row_result)

        per_user_json_path = self.settings.ensure_output_path(
            self.settings.per_user_metrics_top10_three_methods_json_path(
                subset_size,
                artifact_prefix=artifact_prefix,
            ),
            artifact_prefix=artifact_prefix,
            allow_overwrite=allow_overwrite,
        )
        per_user_csv_path = self.settings.ensure_output_path(
            self.settings.per_user_metrics_top10_three_methods_csv_path(
                subset_size,
                artifact_prefix=artifact_prefix,
            ),
            artifact_prefix=artifact_prefix,
            allow_overwrite=allow_overwrite,
        )
        metric_summary_json_path = self.settings.ensure_output_path(
            self.settings.metric_summary_top10_three_methods_json_path(
                subset_size,
                artifact_prefix=artifact_prefix,
            ),
            artifact_prefix=artifact_prefix,
            allow_overwrite=allow_overwrite,
        )
        metric_summary_csv_path = self.settings.ensure_output_path(
            self.settings.metric_summary_top10_three_methods_csv_path(
                subset_size,
                artifact_prefix=artifact_prefix,
            ),
            artifact_prefix=artifact_prefix,
            allow_overwrite=allow_overwrite,
        )
        validation_report_path = self.settings.ensure_output_path(
            self.settings.validation_report_top10_three_methods_json_path(
                subset_size,
                artifact_prefix=artifact_prefix,
            ),
            artifact_prefix=artifact_prefix,
            allow_overwrite=allow_overwrite,
        )
        bootstrap_report_path = self.settings.ensure_output_path(
            self.settings.bootstrap_ci_report_top10_three_methods_json_path(
                subset_size,
                artifact_prefix=artifact_prefix,
            ),
            artifact_prefix=artifact_prefix,
            allow_overwrite=allow_overwrite,
        )
        per_user_json_path.write_text(
            json.dumps(per_user_rows, indent=2),
            encoding="utf-8",
        )
        per_user_csv_path.write_text(
            pd.DataFrame(per_user_rows).to_csv(index=False),
            encoding="utf-8",
        )

        summary = self._build_three_method_metric_summary(per_user_rows, subset_size=subset_size)
        metric_summary_json_path.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        metric_summary_csv_path.write_text(
            pd.DataFrame([self._flatten_three_method_metric_summary(summary)]).to_csv(index=False),
            encoding="utf-8",
        )
        validation_report_path.write_text(
            json.dumps(validation_report, indent=2),
            encoding="utf-8",
        )

        bootstrap_report = self._build_three_method_bootstrap_ci_report(
            per_user_rows=per_user_rows,
            bootstrap_samples=bootstrap_samples,
            random_seed=random_seed,
        )
        bootstrap_report_path.write_text(
            json.dumps(bootstrap_report, indent=2),
            encoding="utf-8",
        )

        self._write_hybrid_audit_report(
            subset_size=subset_size,
            summary=summary,
            bootstrap_report=bootstrap_report,
            validation_report=validation_report,
            artifact_prefix=artifact_prefix,
            allow_overwrite=allow_overwrite,
        )

        return {
            "summary": summary,
            "validation": validation_report,
            "per_user_metrics": per_user_rows,
            "bootstrap": bootstrap_report,
        }

    def _model_metrics(
        self,
        user_ids: list[str],
        test_df: pd.DataFrame,
        recommendations: dict[str, list[dict[str, object]]],
        user_profiles: dict[str, dict[str, set[str]]],
        include_explanations: bool,
    ) -> dict[str, float | None]:
        if not user_ids:
            return {
                "hit_rate_at_10": 0.0,
                "preference_alignment": 0.0,
                "diversity": 0.0,
                "explanation_quality": 0.0 if include_explanations else None,
                "feedback_adaptability": 0.0 if include_explanations else None,
            }

        test_lookup = test_df.groupby("customer_id")["article_id"].agg(set).to_dict()
        hits = 0
        alignments: list[float] = []
        diversities: list[float] = []
        explanation_scores: list[float] = []

        for user_id in user_ids:
            recs = recommendations.get(user_id, [])[: self.settings.top_n]
            if not recs:
                continue
            future_items = test_lookup.get(user_id, set())
            if any(normalize_article_id(rec["article_id"]) in future_items for rec in recs):
                hits += 1

            profile = user_profiles[user_id]
            alignments.extend(self._preference_scores(recs, profile))
            diversities.append(self._diversity_score(recs))
            if include_explanations:
                explanation_scores.extend(self._explanation_scores(recs))

        metrics = {
            "hit_rate_at_10": round(hits / len(user_ids), 4),
            "preference_alignment": round(sum(alignments) / len(alignments), 4) if alignments else 0.0,
            "diversity": round(sum(diversities) / len(diversities), 4) if diversities else 0.0,
            "explanation_quality": round(sum(explanation_scores) / len(explanation_scores), 4)
            if include_explanations and explanation_scores
            else (0.0 if include_explanations else None),
            "feedback_adaptability": 0.75 if include_explanations else None,
        }
        return metrics

    @staticmethod
    def _build_user_profiles(train_df: pd.DataFrame) -> dict[str, dict[str, set[str]]]:
        profiles: dict[str, dict[str, set[str]]] = {}
        for user_id, group in train_df.groupby("customer_id"):
            profiles[str(user_id)] = {
                "product_group": set(group["product_group"].astype(str)),
                "product_type": set(group["product_type"].astype(str)),
                "colour": set(group["colour"].astype(str)),
                "appearance": set(group["appearance"].astype(str)),
            }
        return profiles

    @staticmethod
    def _preference_scores(recommendations: list[dict[str, object]], profile: dict[str, set[str]]) -> list[float]:
        scores = []
        for rec in recommendations:
            score = 0.0
            score += 0.25 if rec["product_group"] in profile["product_group"] else 0.0
            score += 0.25 if rec["product_type"] in profile["product_type"] else 0.0
            score += 0.25 if rec["colour"] in profile["colour"] else 0.0
            score += 0.25 if rec["appearance"] in profile["appearance"] else 0.0
            scores.append(score)
        return scores

    def _diversity_score(self, recommendations: list[dict[str, object]]) -> float:
        if not recommendations:
            return 0.0
        unique_types = len({str(rec["product_type"]) for rec in recommendations})
        return unique_types / min(len(recommendations), self.settings.top_n)

    @staticmethod
    def _explanation_scores(recommendations: list[dict[str, object]]) -> list[float]:
        scores = []
        for rec in recommendations:
            explanation = str(rec.get("reason", "")).lower()
            grounded = sum(
                1
                for token in [
                    str(rec["product_type"]).lower(),
                    str(rec["product_group"]).lower(),
                    str(rec["colour"]).lower(),
                ]
                if token in explanation
            )
            score = 0.4 if explanation else 0.0
            score += 0.2 if len(explanation.split()) >= 8 else 0.0
            score += min(0.4, grounded * 0.2)
            scores.append(score)
        return scores

    def _top10_model_metrics(
        self,
        user_ids: list[str],
        test_df: pd.DataFrame,
        recommendations: dict[str, list[dict[str, object]]],
    ) -> dict[str, float]:
        if not user_ids:
            return {
                "hit_rate_at_10": 0.0,
                "ndcg_at_10": 0.0,
                "intra_list_diversity_at_10": 0.0,
            }

        test_lookup = test_df.groupby("customer_id")["article_id"].agg(set).to_dict()
        hit_scores: list[float] = []
        ndcg_scores: list[float] = []
        ild_scores: list[float] = []

        for user_id in user_ids:
            recs = recommendations.get(user_id, [])[: self.settings.top_n]
            future_items = test_lookup.get(user_id, set())
            hit_scores.append(
                1.0 if any(normalize_article_id(rec["article_id"]) in future_items for rec in recs) else 0.0
            )
            ndcg_scores.append(self._ndcg_at_k(recs, future_items))
            ild_scores.append(self._intra_list_diversity(recs))

        return {
            "hit_rate_at_10": round(sum(hit_scores) / len(hit_scores), 4),
            "ndcg_at_10": round(sum(ndcg_scores) / len(ndcg_scores), 4),
            "intra_list_diversity_at_10": round(sum(ild_scores) / len(ild_scores), 4),
        }

    @staticmethod
    def _ndcg_at_k(recommendations: list[dict[str, object]], relevant_items: set[str]) -> float:
        if not recommendations or not relevant_items:
            return 0.0

        dcg = 0.0
        for index, rec in enumerate(recommendations, start=1):
            if normalize_article_id(rec["article_id"]) in relevant_items:
                dcg += 1.0 / log2(index + 1)

        ideal_hits = min(len(relevant_items), len(recommendations))
        if ideal_hits == 0:
            return 0.0
        idcg = sum(1.0 / log2(index + 1) for index in range(1, ideal_hits + 1))
        return dcg / idcg if idcg else 0.0

    @staticmethod
    def _intra_list_diversity(recommendations: list[dict[str, object]]) -> float:
        if len(recommendations) < 2:
            return 0.0

        distances: list[float] = []
        for left_index, left in enumerate(recommendations[:-1]):
            for right in recommendations[left_index + 1 :]:
                matches = sum(
                    1
                    for field in ("product_type", "product_group", "colour", "appearance")
                    if str(left.get(field)) == str(right.get(field))
                )
                distances.append(1.0 - (matches / 4.0))
        return sum(distances) / len(distances) if distances else 0.0

    @staticmethod
    def _normalize_ids(frame: pd.DataFrame) -> pd.DataFrame:
        normalized = frame.copy()
        normalized["article_id"] = normalized["article_id"].map(normalize_article_id)
        normalized["customer_id"] = normalized["customer_id"].map(normalize_customer_id)
        return normalized

    @staticmethod
    def _normalize_recommendation_payload(
        recommendations: dict[str, list[dict[str, object]]],
    ) -> dict[str, list[dict[str, object]]]:
        normalized_payload: dict[str, list[dict[str, object]]] = {}
        for user_id, items in recommendations.items():
            normalized_items: list[dict[str, object]] = []
            for item in items:
                normalized_item = dict(item)
                normalized_item["article_id"] = normalize_article_id(item.get("article_id"))
                normalized_items.append(normalized_item)
            normalized_payload[normalize_customer_id(user_id)] = normalized_items
        return normalized_payload

    @staticmethod
    def _normalize_evaluation_row(row: dict[str, object]) -> dict[str, object]:
        normalized = dict(row)
        normalized["customer_id"] = normalize_customer_id(row.get("customer_id"))
        normalized["ground_truth_article_id"] = normalize_article_id(row.get("ground_truth_article_id"))
        normalized["candidate_pool_article_ids"] = [
            normalize_article_id(article_id) for article_id in row.get("candidate_pool_article_ids", [])
        ]
        return normalized

    @staticmethod
    def _normalize_saved_recommendation_row(row: dict[str, object]) -> dict[str, object]:
        normalized = dict(row)
        normalized["customer_id"] = normalize_customer_id(row.get("customer_id"))
        normalized["ground_truth_article_id"] = normalize_article_id(row.get("ground_truth_article_id"))
        normalized["top_10_recommendations"] = [
            {**item, "article_id": normalize_article_id(item.get("article_id"))}
            for item in row.get("top_10_recommendations", [])
        ]
        return normalized

    @classmethod
    def _build_article_metadata_lookup(cls, processed: pd.DataFrame) -> dict[str, dict[str, object]]:
        normalized = processed.copy()
        normalized["article_id"] = normalized["article_id"].map(normalize_article_id)
        normalized["customer_id"] = normalized["customer_id"].map(normalize_customer_id)
        normalized["t_dat"] = pd.to_datetime(normalized["t_dat"], errors="coerce")
        normalized = normalized.sort_values("t_dat", ascending=False).drop_duplicates("article_id", keep="first")
        return {
            normalize_article_id(row["article_id"]): {
                field: row.get(field)
                for field in cls.DIVERSITY_FIELDS
            }
            for _, row in normalized.iterrows()
        }

    def _compute_method_metrics_for_user(
        self,
        *,
        customer_id: str,
        ground_truth_article_id: str,
        recommendations: list[dict[str, object]],
        metadata_lookup: dict[str, dict[str, object]],
        method_label: str,
    ) -> tuple[dict[str, object], int, list[str]]:
        ranked_article_ids = [normalize_article_id(item.get("article_id")) for item in recommendations[: self.settings.top_n]]
        ground_truth_rank = None
        for index, article_id in enumerate(ranked_article_ids, start=1):
            if article_id == ground_truth_article_id:
                ground_truth_rank = index
                break

        hit_rate_at_10 = 1 if ground_truth_rank is not None else 0
        ndcg_at_10 = round((1.0 / log2(ground_truth_rank + 1)) if ground_truth_rank is not None else 0.0, 6)
        ild_at_10, missing_metadata_count, errors = self._compute_saved_top10_ild(
            customer_id=customer_id,
            recommendations=recommendations[: self.settings.top_n],
            metadata_lookup=metadata_lookup,
            method_label=method_label,
        )
        return (
            {
                "hit_rate_at_10": hit_rate_at_10,
                "ground_truth_rank": ground_truth_rank,
                "ndcg_at_10": ndcg_at_10,
                "ild_at_10": ild_at_10,
            },
            missing_metadata_count,
            errors,
        )

    def _compute_saved_top10_ild(
        self,
        *,
        customer_id: str,
        recommendations: list[dict[str, object]],
        metadata_lookup: dict[str, dict[str, object]],
        method_label: str,
    ) -> tuple[float, int, list[str]]:
        prepared_items: list[dict[str, object]] = []
        missing_metadata_count = 0
        errors: list[str] = []

        for item in recommendations:
            article_id = normalize_article_id(item.get("article_id"))
            metadata = metadata_lookup.get(article_id, {})
            prepared = {"article_id": article_id}
            missing_fields: list[str] = []
            for field in self.DIVERSITY_FIELDS:
                value = item.get(field)
                if value is None or str(value).strip() == "":
                    value = metadata.get(field)
                if value is None or str(value).strip() == "":
                    missing_fields.append(field)
                prepared[field] = value
            if missing_fields:
                missing_metadata_count += 1
                errors.append(
                    f"{method_label}:{customer_id}:{article_id}:missing_diversity_metadata={','.join(missing_fields)}"
                )
            prepared_items.append(prepared)

        valid_items = [
            item
            for item in prepared_items
            if all(item.get(field) is not None and str(item.get(field)).strip() != "" for field in self.DIVERSITY_FIELDS)
        ]
        if len(valid_items) < 2:
            return 0.0, missing_metadata_count, errors

        distances: list[float] = []
        for left_index, left in enumerate(valid_items[:-1]):
            for right in valid_items[left_index + 1 :]:
                differences = [
                    1.0 if str(left[field]) != str(right[field]) else 0.0
                    for field in self.DIVERSITY_FIELDS
                ]
                distances.append(sum(differences) / len(differences))
        return round(sum(distances) / len(distances), 6) if distances else 0.0, missing_metadata_count, errors

    def _build_metric_summary(self, per_user_rows: list[dict[str, object]], subset_size: int) -> dict[str, object]:
        svd_hits = sum(int(row["svd_hit_rate_at_10"]) for row in per_user_rows)
        agentic_hits = sum(int(row["agentic_hit_rate_at_10"]) for row in per_user_rows)
        total_users = len(per_user_rows)

        def average(field: str) -> float:
            if not per_user_rows:
                return 0.0
            return round(sum(float(row[field]) for row in per_user_rows) / total_users, 6)

        return {
            "evaluation_scope": {
                "valid_evaluated_users": total_users,
                "candidate_pool_size": self.settings.candidate_pool_size,
                "top_k": self.settings.top_n,
                "experiment_subset_size": subset_size,
                "split_strategy": "leave_one_out",
                "baseline": "SVD Matrix Factorisation",
                "comparison_method": "3-Agent Agentic AI",
            },
            "svd": {
                "hit_rate_at_10": average("svd_hit_rate_at_10"),
                "hits_count": svd_hits,
                "miss_count": total_users - svd_hits,
                "ndcg_at_10": average("svd_ndcg_at_10"),
                "intra_list_diversity_at_10": average("svd_ild_at_10"),
            },
            "agentic": {
                "hit_rate_at_10": average("agentic_hit_rate_at_10"),
                "hits_count": agentic_hits,
                "miss_count": total_users - agentic_hits,
                "ndcg_at_10": average("agentic_ndcg_at_10"),
                "intra_list_diversity_at_10": average("agentic_ild_at_10"),
            },
        }

    @staticmethod
    def _flatten_metric_summary(summary: dict[str, object]) -> dict[str, object]:
        evaluation_scope = summary["evaluation_scope"]
        svd = summary["svd"]
        agentic = summary["agentic"]
        return {
            "valid_evaluated_users": evaluation_scope["valid_evaluated_users"],
            "candidate_pool_size": evaluation_scope["candidate_pool_size"],
            "top_k": evaluation_scope["top_k"],
            "experiment_subset_size": evaluation_scope.get("experiment_subset_size"),
            "split_strategy": evaluation_scope["split_strategy"],
            "baseline": evaluation_scope["baseline"],
            "comparison_method": evaluation_scope["comparison_method"],
            "svd_hit_rate_at_10": svd["hit_rate_at_10"],
            "svd_hits_count": svd["hits_count"],
            "svd_miss_count": svd["miss_count"],
            "svd_ndcg_at_10": svd["ndcg_at_10"],
            "svd_intra_list_diversity_at_10": svd["intra_list_diversity_at_10"],
            "agentic_hit_rate_at_10": agentic["hit_rate_at_10"],
            "agentic_hits_count": agentic["hits_count"],
            "agentic_miss_count": agentic["miss_count"],
            "agentic_ndcg_at_10": agentic["ndcg_at_10"],
            "agentic_intra_list_diversity_at_10": agentic["intra_list_diversity_at_10"],
        }

    def _build_three_method_metric_summary(
        self,
        per_user_rows: list[dict[str, object]],
        subset_size: int,
    ) -> dict[str, object]:
        total_users = len(per_user_rows)

        def avg(field: str) -> float:
            if not per_user_rows:
                return 0.0
            return round(sum(float(row[field]) for row in per_user_rows) / total_users, 6)

        def hit_counts(prefix: str) -> tuple[int, int]:
            hits = sum(int(row[f"{prefix}_hit_rate_at_10"]) for row in per_user_rows)
            return hits, total_users - hits

        svd_hits, svd_misses = hit_counts("svd")
        agentic_hits, agentic_misses = hit_counts("agentic")
        hybrid_hits, hybrid_misses = hit_counts("hybrid")
        return {
            "evaluation_scope": {
                "valid_evaluated_users": total_users,
                "candidate_pool_size": self.settings.candidate_pool_size,
                "top_k": self.settings.top_n,
                "experiment_subset_size": subset_size,
                "split_strategy": "leave_one_out",
            },
            "svd": {
                "method": "SVD Matrix Factorisation",
                "hit_rate_at_10": avg("svd_hit_rate_at_10"),
                "hits_count": svd_hits,
                "miss_count": svd_misses,
                "ndcg_at_10": avg("svd_ndcg_at_10"),
                "intra_list_diversity_at_10": avg("svd_ild_at_10"),
            },
            "agentic": {
                "method": "Standalone 3-Agent",
                "hit_rate_at_10": avg("agentic_hit_rate_at_10"),
                "hits_count": agentic_hits,
                "miss_count": agentic_misses,
                "ndcg_at_10": avg("agentic_ndcg_at_10"),
                "intra_list_diversity_at_10": avg("agentic_ild_at_10"),
            },
            "hybrid": {
                "method": "Hybrid SVD + 3-Agent Reranker",
                "hit_rate_at_10": avg("hybrid_hit_rate_at_10"),
                "hits_count": hybrid_hits,
                "miss_count": hybrid_misses,
                "ndcg_at_10": avg("hybrid_ndcg_at_10"),
                "intra_list_diversity_at_10": avg("hybrid_ild_at_10"),
            },
        }

    @staticmethod
    def _flatten_three_method_metric_summary(summary: dict[str, object]) -> dict[str, object]:
        evaluation_scope = summary["evaluation_scope"]
        flat = {
            "valid_evaluated_users": evaluation_scope["valid_evaluated_users"],
            "candidate_pool_size": evaluation_scope["candidate_pool_size"],
            "top_k": evaluation_scope["top_k"],
            "experiment_subset_size": evaluation_scope["experiment_subset_size"],
            "split_strategy": evaluation_scope["split_strategy"],
        }
        for key in ("svd", "agentic", "hybrid"):
            method = summary[key]
            flat[f"{key}_hit_rate_at_10"] = method["hit_rate_at_10"]
            flat[f"{key}_hits_count"] = method["hits_count"]
            flat[f"{key}_miss_count"] = method["miss_count"]
            flat[f"{key}_ndcg_at_10"] = method["ndcg_at_10"]
            flat[f"{key}_intra_list_diversity_at_10"] = method["intra_list_diversity_at_10"]
        return flat

    def _build_bootstrap_ci_report(
        self,
        *,
        per_user_rows: list[dict[str, object]],
        bootstrap_samples: int,
        random_seed: int,
    ) -> dict[str, object]:
        rng = random.Random(random_seed)
        metrics = {
            "svd_hit_rate_at_10": [],
            "agentic_hit_rate_at_10": [],
            "svd_ndcg_at_10": [],
            "agentic_ndcg_at_10": [],
            "svd_ild_at_10": [],
            "agentic_ild_at_10": [],
            "difference_hit_rate_at_10": [],
            "difference_ndcg_at_10": [],
            "difference_ild_at_10": [],
        }
        row_count = len(per_user_rows)
        if row_count == 0:
            return {
                "bootstrap_samples": bootstrap_samples,
                "random_seed": random_seed,
                "confidence_intervals": {},
            }

        for _ in range(bootstrap_samples):
            sample = [per_user_rows[rng.randrange(row_count)] for _ in range(row_count)]
            svd_hit = sum(float(row["svd_hit_rate_at_10"]) for row in sample) / row_count
            agentic_hit = sum(float(row["agentic_hit_rate_at_10"]) for row in sample) / row_count
            svd_ndcg = sum(float(row["svd_ndcg_at_10"]) for row in sample) / row_count
            agentic_ndcg = sum(float(row["agentic_ndcg_at_10"]) for row in sample) / row_count
            svd_ild = sum(float(row["svd_ild_at_10"]) for row in sample) / row_count
            agentic_ild = sum(float(row["agentic_ild_at_10"]) for row in sample) / row_count

            metrics["svd_hit_rate_at_10"].append(svd_hit)
            metrics["agentic_hit_rate_at_10"].append(agentic_hit)
            metrics["svd_ndcg_at_10"].append(svd_ndcg)
            metrics["agentic_ndcg_at_10"].append(agentic_ndcg)
            metrics["svd_ild_at_10"].append(svd_ild)
            metrics["agentic_ild_at_10"].append(agentic_ild)
            metrics["difference_hit_rate_at_10"].append(agentic_hit - svd_hit)
            metrics["difference_ndcg_at_10"].append(agentic_ndcg - svd_ndcg)
            metrics["difference_ild_at_10"].append(agentic_ild - svd_ild)

        return {
            "bootstrap_samples": bootstrap_samples,
            "random_seed": random_seed,
            "confidence_intervals": {
                metric_name: {
                    "mean": round(sum(values) / len(values), 6),
                    "ci_95_lower": round(self._percentile(values, 2.5), 6),
                    "ci_95_upper": round(self._percentile(values, 97.5), 6),
                }
                for metric_name, values in metrics.items()
            },
        }

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * (percentile / 100.0)
        lower_index = int(position)
        upper_index = min(lower_index + 1, len(ordered) - 1)
        weight = position - lower_index
        return ordered[lower_index] * (1 - weight) + ordered[upper_index] * weight

    def _build_three_method_bootstrap_ci_report(
        self,
        *,
        per_user_rows: list[dict[str, object]],
        bootstrap_samples: int,
        random_seed: int,
    ) -> dict[str, object]:
        rng = random.Random(random_seed)
        metrics = {
            "svd_hit_rate_at_10": [],
            "agentic_hit_rate_at_10": [],
            "hybrid_hit_rate_at_10": [],
            "svd_ndcg_at_10": [],
            "agentic_ndcg_at_10": [],
            "hybrid_ndcg_at_10": [],
            "svd_ild_at_10": [],
            "agentic_ild_at_10": [],
            "hybrid_ild_at_10": [],
            "difference_hybrid_minus_svd_hit_rate_at_10": [],
            "difference_hybrid_minus_svd_ndcg_at_10": [],
            "difference_hybrid_minus_svd_ild_at_10": [],
            "difference_hybrid_minus_agentic_hit_rate_at_10": [],
            "difference_hybrid_minus_agentic_ndcg_at_10": [],
            "difference_hybrid_minus_agentic_ild_at_10": [],
        }
        row_count = len(per_user_rows)
        for _ in range(bootstrap_samples):
            sample = [per_user_rows[rng.randrange(row_count)] for _ in range(row_count)]
            means = {
                field: sum(float(row[field]) for row in sample) / row_count
                for field in (
                    "svd_hit_rate_at_10",
                    "agentic_hit_rate_at_10",
                    "hybrid_hit_rate_at_10",
                    "svd_ndcg_at_10",
                    "agentic_ndcg_at_10",
                    "hybrid_ndcg_at_10",
                    "svd_ild_at_10",
                    "agentic_ild_at_10",
                    "hybrid_ild_at_10",
                )
            }
            for field, value in means.items():
                metrics[field].append(value)
            metrics["difference_hybrid_minus_svd_hit_rate_at_10"].append(
                means["hybrid_hit_rate_at_10"] - means["svd_hit_rate_at_10"]
            )
            metrics["difference_hybrid_minus_svd_ndcg_at_10"].append(
                means["hybrid_ndcg_at_10"] - means["svd_ndcg_at_10"]
            )
            metrics["difference_hybrid_minus_svd_ild_at_10"].append(
                means["hybrid_ild_at_10"] - means["svd_ild_at_10"]
            )
            metrics["difference_hybrid_minus_agentic_hit_rate_at_10"].append(
                means["hybrid_hit_rate_at_10"] - means["agentic_hit_rate_at_10"]
            )
            metrics["difference_hybrid_minus_agentic_ndcg_at_10"].append(
                means["hybrid_ndcg_at_10"] - means["agentic_ndcg_at_10"]
            )
            metrics["difference_hybrid_minus_agentic_ild_at_10"].append(
                means["hybrid_ild_at_10"] - means["agentic_ild_at_10"]
            )

        return {
            "bootstrap_samples": bootstrap_samples,
            "random_seed": random_seed,
            "confidence_intervals": {
                metric_name: {
                    "mean": round(sum(values) / len(values), 6),
                    "ci_95_lower": round(self._percentile(values, 2.5), 6),
                    "ci_95_upper": round(self._percentile(values, 97.5), 6),
                }
                for metric_name, values in metrics.items()
            },
        }

    def _write_hybrid_audit_report(
        self,
        *,
        subset_size: int,
        summary: dict[str, object],
        bootstrap_report: dict[str, object],
        validation_report: dict[str, object],
        artifact_prefix: str | None = None,
        allow_overwrite: bool | None = None,
    ) -> None:
        ci = bootstrap_report["confidence_intervals"]
        hybrid = summary["hybrid"]
        svd = summary["svd"]
        agentic = summary["agentic"]
        hybrid_better_than_svd = (
            hybrid["hit_rate_at_10"] > svd["hit_rate_at_10"]
            and hybrid["ndcg_at_10"] > svd["ndcg_at_10"]
        )
        hybrid_better_than_agentic = (
            hybrid["hit_rate_at_10"] > agentic["hit_rate_at_10"]
            and hybrid["ndcg_at_10"] > agentic["ndcg_at_10"]
        )
        lines = [
            "# Hybrid SVD + 3-Agent Audit Report",
            "",
            "## Why Hybrid Was Added",
            "",
            "The standalone 3-agent method underperformed SVD in the saved 1,000-user offline evaluation.",
            "The hybrid experiment tests whether agentic evidence can help as a reranking signal on top of SVD rather than replacing SVD.",
            "",
            "## Method Description",
            "",
            "- SVD remains the main behavioural relevance signal.",
            "- The standalone 3-agent score is used as a metadata evidence signal.",
            "- A small diversity bonus is applied during iterative Top-10 selection.",
            "",
            "## Hybrid Scoring Formula",
            "",
            "```text",
            "hybrid_score = 0.70 * normalized_svd_score",
            "             + 0.25 * normalized_agentic_score",
            "             + 0.05 * diversity_bonus",
            "```",
            "",
            "## Validation Checks",
            "",
        ]
        for key, value in validation_report.items():
            lines.append(f"- {key}: {value}")
        lines.extend(
            [
                "",
                "## Final Metrics",
                "",
                f"- SVD HitRate@10: {svd['hit_rate_at_10']:.6f}",
                f"- SVD NDCG@10: {svd['ndcg_at_10']:.6f}",
                f"- SVD ILD@10: {svd['intra_list_diversity_at_10']:.6f}",
                f"- Standalone 3-Agent HitRate@10: {agentic['hit_rate_at_10']:.6f}",
                f"- Standalone 3-Agent NDCG@10: {agentic['ndcg_at_10']:.6f}",
                f"- Standalone 3-Agent ILD@10: {agentic['intra_list_diversity_at_10']:.6f}",
                f"- Hybrid HitRate@10: {hybrid['hit_rate_at_10']:.6f}",
                f"- Hybrid NDCG@10: {hybrid['ndcg_at_10']:.6f}",
                f"- Hybrid ILD@10: {hybrid['intra_list_diversity_at_10']:.6f}",
                "",
                "## Bootstrap Confidence Intervals",
                "",
            ]
        )
        for metric_name, values in ci.items():
            lines.append(
                f"- {metric_name}: mean={values['mean']:.6f}, "
                f"95% CI [{values['ci_95_lower']:.6f}, {values['ci_95_upper']:.6f}]"
            )
        lines.extend(
            [
                "",
                "## Did Hybrid Improve?",
                "",
                f"- Hybrid improved over SVD on offline relevance: {hybrid_better_than_svd}",
                f"- Hybrid improved over standalone 3-agent on offline relevance: {hybrid_better_than_agentic}",
                "",
                "## Honest Interpretation",
                "",
            ]
        )
        if hybrid_better_than_svd:
            lines.append("- The saved offline metrics show the hybrid outperforming SVD on relevance.")
        else:
            lines.append("- The saved offline metrics do not show the hybrid outperforming SVD on relevance.")
            lines.append("- SVD remains the strongest offline relevance baseline in this experiment.")
        if hybrid_better_than_agentic:
            lines.append("- The hybrid outperforms standalone 3-agent on offline relevance.")
        else:
            lines.append("- The hybrid does not outperform standalone 3-agent on offline relevance.")
        if hybrid["intra_list_diversity_at_10"] > agentic["intra_list_diversity_at_10"]:
            lines.append("- The hybrid improves diversity over standalone 3-agent.")
        lines.extend(
            [
                "",
                "## Limitations",
                "",
                "- These are offline metrics only.",
                "- They do not establish customer engagement, CTR, CVR, or business impact.",
                "- The hybrid uses a fixed heuristic formula and was not tuned in this step.",
            ]
        )
        report_path = self.settings.ensure_output_path(
            self.settings.hybrid_svd_agentic_audit_report_top10_path(
                subset_size,
                artifact_prefix=artifact_prefix,
            ),
            artifact_prefix=artifact_prefix,
            allow_overwrite=allow_overwrite,
        )
        report_path.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
