from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from app.config import Settings
from app.utils.normalization import normalize_article_id, normalize_customer_id


@dataclass
class ArtifactDemoService:
    settings: Settings

    def load_workflow_cases(
        self,
        *,
        artifact_prefix: str = "seed99_robustness",
        explainability_prefix: str = "seed99_full_retry",
        sample_size: int = 1000,
        max_cases: int = 5,
    ) -> dict[str, object]:
        paths = {
            "evaluation": self.settings.evaluation_base_table_svd_top10_json_path(
                sample_size, artifact_prefix=artifact_prefix
            ),
            "svd": self.settings.svd_recommendations_top10_json_path(
                sample_size, artifact_prefix=artifact_prefix
            ),
            "agentic": self.settings.agentic_recommendations_top10_json_path(
                sample_size, artifact_prefix=artifact_prefix
            ),
            "hybrid": self.settings.hybrid_svd_agentic_recommendations_top10_json_path(
                sample_size, artifact_prefix=artifact_prefix
            ),
            "per_user": self.settings.per_user_metrics_top10_three_methods_json_path(
                sample_size, artifact_prefix=artifact_prefix
            ),
            "examples": self.settings.explainability_examples_csv_path(explainability_prefix),
        }
        missing = [str(path) for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing required walkthrough artifacts: " + "; ".join(missing))

        evaluation_rows = json.loads(paths["evaluation"].read_text(encoding="utf-8"))
        svd_rows = json.loads(paths["svd"].read_text(encoding="utf-8"))
        agentic_rows = json.loads(paths["agentic"].read_text(encoding="utf-8"))
        hybrid_rows = json.loads(paths["hybrid"].read_text(encoding="utf-8"))
        per_user_rows = json.loads(paths["per_user"].read_text(encoding="utf-8"))
        explainability_examples = pd.read_csv(
            paths["examples"],
            dtype={"customer_id": "string", "article_id": "string"},
            keep_default_na=False,
        ).fillna("")

        evaluation_lookup = {normalize_customer_id(row.get("customer_id")): row for row in evaluation_rows}
        svd_lookup = {normalize_customer_id(row.get("customer_id")): row for row in svd_rows}
        agentic_lookup = {normalize_customer_id(row.get("customer_id")): row for row in agentic_rows}
        hybrid_lookup = {normalize_customer_id(row.get("customer_id")): row for row in hybrid_rows}
        per_user_lookup = {normalize_customer_id(row.get("customer_id")): row for row in per_user_rows}
        explainability_lookup = self._group_examples_by_customer(explainability_examples)

        complete_users = sorted(
            set(evaluation_lookup)
            & set(svd_lookup)
            & set(agentic_lookup)
            & set(hybrid_lookup)
            & set(per_user_lookup)
        )
        prioritized_users = sorted(
            complete_users,
            key=lambda customer_id: (
                0 if explainability_lookup.get(customer_id) else 1,
                customer_id,
            ),
        )[:max_cases]
        cases = [
            self._build_case(
                label=f"Demo User {index}",
                customer_id=customer_id,
                evaluation_row=evaluation_lookup[customer_id],
                svd_row=svd_lookup[customer_id],
                agentic_row=agentic_lookup[customer_id],
                hybrid_row=hybrid_lookup[customer_id],
                per_user_row=per_user_lookup[customer_id],
                explainability_rows=explainability_lookup.get(customer_id, []),
                history_rows=[],
            )
            for index, customer_id in enumerate(prioritized_users, start=1)
        ]

        return {
            "artifact_prefix": artifact_prefix,
            "explainability_prefix": explainability_prefix,
            "cases": cases,
        }

    @staticmethod
    def _group_examples_by_customer(examples: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
        lookup: dict[str, list[dict[str, object]]] = {}
        for row in examples.to_dict(orient="records"):
            customer_id = normalize_customer_id(row.get("customer_id"))
            lookup.setdefault(customer_id, []).append(row)
        return lookup

    def _build_case(
        self,
        *,
        label: str,
        customer_id: str,
        evaluation_row: dict[str, object],
        svd_row: dict[str, object],
        agentic_row: dict[str, object],
        hybrid_row: dict[str, object],
        per_user_row: dict[str, object],
        explainability_rows: list[dict[str, object]],
        history_rows: list[dict[str, object]],
    ) -> dict[str, object]:
        ground_truth_article_id = normalize_article_id(evaluation_row.get("ground_truth_article_id"))
        preference_profile = agentic_row.get("preference_profile") or {}
        explanation = self._select_explanation_row(
            explainability_rows=explainability_rows,
            ground_truth_article_id=ground_truth_article_id,
        )
        history_summary = self._build_history_summary(
            history_rows=history_rows,
            train_article_ids=evaluation_row.get("train_article_ids", []),
        )

        svd_items = self._map_method_items(
            svd_row.get("top_10_recommendations", []),
            ground_truth_article_id=ground_truth_article_id,
            score_field="score",
        )
        agentic_items = self._map_method_items(
            agentic_row.get("top_10_recommendations", []),
            ground_truth_article_id=ground_truth_article_id,
            score_field="score",
            include_reason=True,
            include_evidence=True,
        )
        hybrid_items = self._map_method_items(
            hybrid_row.get("top_10_recommendations", []),
            ground_truth_article_id=ground_truth_article_id,
            score_field="hybrid_score",
            include_reason=True,
            include_hybrid_components=True,
        )

        candidate_pool_article_ids = [
            normalize_article_id(article_id) for article_id in evaluation_row.get("candidate_pool_article_ids", [])
        ]
        training_history_count = self._coerce_int(evaluation_row.get("train_count"))
        if training_history_count is None:
            training_history_count = len(evaluation_row.get("train_article_ids", []))

        candidate_pool_size = self._coerce_int(evaluation_row.get("candidate_pool_size"))
        if candidate_pool_size is None:
            candidate_pool_size = len(candidate_pool_article_ids)

        ground_truth_in_candidate_pool = evaluation_row.get("ground_truth_in_candidate_pool")
        if ground_truth_in_candidate_pool in ("", None):
            ground_truth_in_candidate_pool = ground_truth_article_id in candidate_pool_article_ids
        else:
            ground_truth_in_candidate_pool = self._is_truthy(ground_truth_in_candidate_pool)

        evidence_item = agentic_row.get("top_10_recommendations", [{}])[0] if agentic_row.get("top_10_recommendations") else {}
        decision_item = agentic_row.get("top_10_recommendations", [{}])[0] if agentic_row.get("top_10_recommendations") else {}
        explainability_metadata = {
            "article_id": explanation.get("article_id"),
            "product_type_name": explanation.get("product_type_name"),
            "product_group_name": explanation.get("product_group_name"),
            "colour_group_name": explanation.get("colour_group_name"),
            "graphical_appearance_name": explanation.get("graphical_appearance_name"),
            "garment_group_name": explanation.get("garment_group_name"),
            "prod_name": explanation.get("prod_name"),
        }
        score_breakdown = {
            "normalized_svd_score": self._coerce_float(explanation.get("normalized_svd_score")),
            "normalized_agentic_score": self._coerce_float(explanation.get("normalized_agentic_score")),
            "diversity_bonus": self._coerce_float(explanation.get("diversity_bonus")),
            "hybrid_score": self._coerce_float(explanation.get("hybrid_score")),
        }
        groundedness_status = None
        if explanation:
            groundedness_status = "Grounded" if self._coerce_int(explanation.get("ungrounded_claim_count")) == 0 else "Check warnings"

        rank_shift = self._coerce_int(explanation.get("rank_shift"))
        if rank_shift is None:
            hybrid_rank = self._coerce_int(per_user_row.get("hybrid_ground_truth_rank"))
            svd_rank = self._coerce_int(per_user_row.get("svd_ground_truth_rank"))
            if hybrid_rank is not None and svd_rank is not None:
                rank_shift = svd_rank - hybrid_rank

        has_preference_summary = any(
            [
                preference_profile.get("inferred_intent"),
                preference_profile.get("preferred_categories"),
                preference_profile.get("preferred_product_types"),
                preference_profile.get("preferred_colours"),
                preference_profile.get("preferred_appearance"),
            ]
        )
        has_history_summary = any(history_summary.values())
        has_item_metadata = any(
            value for key, value in explainability_metadata.items() if key not in {"article_id", "prod_name"}
        ) or any(
            evidence_item.get(field)
            for field in (
                "product_type_name",
                "product_group_name",
                "colour_group_name",
                "graphical_appearance_name",
                "garment_group_name",
            )
        )
        has_score_breakdown = all(value is not None for value in score_breakdown.values())

        return {
            "label": label,
            "demo_user": {
                "label": label,
                "customer_id": customer_id,
                "customer_id_short": self._short_customer_id(customer_id),
            },
            "leave_one_out": {
                "training_history_count": training_history_count,
                "ground_truth_article_id": ground_truth_article_id,
                "candidate_pool_size": candidate_pool_size,
                "ground_truth_in_candidate_pool": ground_truth_in_candidate_pool,
            },
            "ground_truth": {
                "article_id": ground_truth_article_id,
                "product_type_name": evaluation_row.get("ground_truth_product_type_name") or None,
                "product_group_name": evaluation_row.get("ground_truth_product_group_name") or None,
                "colour_group_name": evaluation_row.get("ground_truth_colour_group_name") or None,
                "graphical_appearance_name": evaluation_row.get("ground_truth_graphical_appearance_name") or None,
                "garment_group_name": evaluation_row.get("ground_truth_garment_group_name") or None,
            },
            "method_hits": {
                "svd": any(item["is_ground_truth"] for item in svd_items),
                "agentic": any(item["is_ground_truth"] for item in agentic_items),
                "hybrid": any(item["is_ground_truth"] for item in hybrid_items),
            },
            "preference_agent": {
                "inferred_intent": preference_profile.get("inferred_intent") or None,
                "preferred_categories": preference_profile.get("preferred_categories", []),
                "preferred_product_types": preference_profile.get("preferred_product_types", []),
                "preferred_colours": preference_profile.get("preferred_colours", []),
                "preferred_appearance": preference_profile.get("preferred_appearance", []),
                "frequent_product_types": history_summary["frequent_product_types"],
                "frequent_product_groups": history_summary["frequent_product_groups"],
                "frequent_colours": history_summary["frequent_colours"],
                "frequent_graphical_appearances": history_summary["frequent_graphical_appearances"],
                "availability": {
                    "has_preference_summary": has_preference_summary,
                    "has_history_summary": has_history_summary,
                },
            },
            "evidence_agent": {
                "headline": evidence_item.get("recommendation_reason") or None,
                "matched_evidence": evidence_item.get("matched_evidence", []) or [],
                "item_metadata": {
                    "article_id": normalize_article_id(evidence_item.get("article_id")),
                    "product_type_name": evidence_item.get("product_type_name") or None,
                    "product_group_name": evidence_item.get("product_group_name") or None,
                    "colour_group_name": evidence_item.get("colour_group_name") or None,
                    "graphical_appearance_name": evidence_item.get("graphical_appearance_name") or None,
                    "garment_group_name": evidence_item.get("garment_group_name") or None,
                    "prod_name": None,
                },
                "availability": {
                    "has_item_metadata": has_item_metadata,
                    "has_matched_evidence": bool(evidence_item.get("matched_evidence")),
                },
            },
            "decision_agent": {
                "headline": decision_item.get("recommendation_reason") or None,
                "selected_article_id": normalize_article_id(decision_item.get("article_id")),
                "selected_rank": self._coerce_int(decision_item.get("rank")),
                "selected_score": self._coerce_float(decision_item.get("score")),
            },
            "svd_top10": svd_items,
            "agentic_top10": agentic_items,
            "hybrid_top10": hybrid_items,
            "hybrid_explainability": {
                "article_id": explanation.get("article_id"),
                "hybrid_rank": self._coerce_int(explanation.get("hybrid_rank")),
                "svd_rank": self._coerce_int(explanation.get("svd_rank")),
                "rank_shift": rank_shift,
                "is_ground_truth": self._is_truthy(explanation.get("is_ground_truth")),
                "explanation_text": explanation.get("explanation_text") or None,
                "matched_preference_fields": explanation.get("matched_preference_fields", []),
                "score_components": score_breakdown,
                "item_metadata": explainability_metadata,
                "groundedness_status": groundedness_status,
                "warnings": self._build_explanation_warnings(explanation),
                "availability": {
                    "has_explanation_text": bool(explanation.get("explanation_text")),
                    "has_score_breakdown": has_score_breakdown,
                    "has_rank_shift": rank_shift is not None,
                    "has_item_metadata": has_item_metadata,
                    "prod_name_available": bool(explanation.get("prod_name")),
                },
            },
        }

    def _select_explanation_row(
        self,
        *,
        explainability_rows: list[dict[str, object]],
        ground_truth_article_id: str,
    ) -> dict[str, object]:
        if not explainability_rows:
            return {}

        sorted_rows = sorted(
            explainability_rows,
            key=lambda row: (
                not self._is_truthy(row.get("is_ground_truth")),
                normalize_article_id(row.get("article_id")) != ground_truth_article_id,
                self._coerce_int(row.get("hybrid_rank")) or 999,
            ),
        )
        row = sorted_rows[0]
        matched_fields: list[dict[str, object]] = []
        raw_fields = row.get("matched_preference_fields_json")
        if isinstance(raw_fields, str) and raw_fields.strip():
            try:
                parsed = json.loads(raw_fields)
                if isinstance(parsed, list):
                    matched_fields = [item for item in parsed if isinstance(item, dict)]
            except json.JSONDecodeError:
                matched_fields = []

        return {
            "article_id": normalize_article_id(row.get("article_id")),
            "hybrid_rank": self._coerce_int(row.get("hybrid_rank")),
            "svd_rank": self._coerce_int(row.get("svd_rank")),
            "rank_shift": self._coerce_int(row.get("rank_shift")),
            "is_ground_truth": self._is_truthy(row.get("is_ground_truth")),
            "product_type_name": row.get("product_type_name") or None,
            "product_group_name": row.get("product_group_name") or None,
            "colour_group_name": row.get("colour_group_name") or None,
            "graphical_appearance_name": row.get("graphical_appearance_name") or None,
            "garment_group_name": row.get("garment_group_name") or None,
            "prod_name": row.get("prod_name") or None,
            "explanation_text": row.get("explanation_text") or "",
            "matched_preference_fields": matched_fields,
            "normalized_svd_score": self._coerce_float(row.get("normalized_svd_score")),
            "normalized_agentic_score": self._coerce_float(row.get("normalized_agentic_score")),
            "diversity_bonus": self._coerce_float(row.get("diversity_bonus")),
            "hybrid_score": self._coerce_float(row.get("hybrid_score")),
            "ungrounded_claim_count": self._coerce_int(row.get("ungrounded_claim_count")),
        }

    def _build_history_summary(
        self,
        *,
        history_rows: list[dict[str, object]],
        train_article_ids: list[object],
    ) -> dict[str, list[dict[str, object]]]:
        normalized_train_ids = {normalize_article_id(article_id) for article_id in train_article_ids}
        filtered_rows = [row for row in history_rows if row.get("article_id") in normalized_train_ids] or history_rows

        return {
            "frequent_product_types": self._count_values(filtered_rows, "product_type_name"),
            "frequent_product_groups": self._count_values(filtered_rows, "product_group_name"),
            "frequent_colours": self._count_values(filtered_rows, "colour_group_name"),
            "frequent_graphical_appearances": self._count_values(filtered_rows, "graphical_appearance_name"),
        }

    @staticmethod
    def _count_values(rows: list[dict[str, object]], field: str) -> list[dict[str, object]]:
        counts: dict[str, int] = {}
        for row in rows:
            value = str(row.get(field) or "").strip()
            if not value:
                continue
            counts[value] = counts.get(value, 0) + 1
        return [
            {"value": value, "count": count}
            for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:4]
        ]

    def _map_method_items(
        self,
        items: list[dict[str, object]],
        *,
        ground_truth_article_id: str,
        score_field: str,
        include_reason: bool = False,
        include_evidence: bool = False,
        include_hybrid_components: bool = False,
    ) -> list[dict[str, object]]:
        mapped: list[dict[str, object]] = []
        for item in items[:10]:
            mapped.append(
                {
                    "article_id": normalize_article_id(item.get("article_id")),
                    "rank": self._coerce_int(item.get("rank")),
                    "score": self._coerce_float(item.get(score_field)),
                    "hybrid_score": self._coerce_float(item.get("hybrid_score")) if include_hybrid_components else None,
                    "normalized_svd_score": self._coerce_float(item.get("normalized_svd_score")) if include_hybrid_components else None,
                    "normalized_agentic_score": self._coerce_float(item.get("normalized_agentic_score")) if include_hybrid_components else None,
                    "diversity_bonus": self._coerce_float(item.get("diversity_bonus")) if include_hybrid_components else None,
                    "reason": item.get("recommendation_reason") if include_reason else None,
                    "matched_evidence": list(item.get("matched_evidence", []) or []) if include_evidence else [],
                    "is_ground_truth": normalize_article_id(item.get("article_id")) == ground_truth_article_id,
                    "product_type_name": item.get("product_type_name") or None,
                    "product_group_name": item.get("product_group_name") or None,
                    "colour_group_name": item.get("colour_group_name") or None,
                    "graphical_appearance_name": item.get("graphical_appearance_name") or None,
                    "garment_group_name": item.get("garment_group_name") or None,
                }
            )
        return mapped

    @staticmethod
    def _build_explanation_warnings(explanation: dict[str, object]) -> list[str]:
        warnings: list[str] = []
        if not explanation:
            warnings.append("Not available")
            return warnings
        if not explanation.get("prod_name"):
            warnings.append("Product name metadata not available in saved artifacts.")
        if not explanation.get("explanation_text"):
            warnings.append("Saved explanation text not available.")
        return warnings

    @staticmethod
    def _short_customer_id(customer_id: str) -> str:
        if len(customer_id) <= 12:
            return customer_id
        return f"{customer_id[:6]}...{customer_id[-4:]}"

    @staticmethod
    def _coerce_float(value: object) -> float | None:
        if value in ("", None):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_int(value: object) -> int | None:
        if value in ("", None, " "):
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _is_truthy(value: object) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"true", "1", "yes"}
