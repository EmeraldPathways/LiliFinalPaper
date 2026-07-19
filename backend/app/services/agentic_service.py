from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

import httpx
import pandas as pd

from app.config import Settings
from app.models.schemas import FeedbackType
from app.utils.normalization import normalize_article_id, normalize_customer_id


class AgenticServiceError(Exception):
    pass


@dataclass
class AgenticRecommendationService:
    settings: Settings

    BASE_WEIGHTS = {
        "category_weight": 1.0,
        "colour_weight": 1.0,
        "product_type_weight": 1.0,
        "appearance_weight": 1.0,
        "diversity_penalty": 1.0,
    }

    def infer_user_intent(self, user_id: str, train_df: pd.DataFrame) -> dict[str, Any]:
        user_id = normalize_customer_id(user_id)
        train_df = self._normalize_ids(train_df)
        history = train_df[train_df["customer_id"] == user_id].copy()
        if history.empty:
            raise AgenticServiceError(f"User {user_id} has no training history.")

        dominant_groups = self._top_values(history["product_group"])
        dominant_types = self._top_values(history["product_type"])
        dominant_colours = self._top_values(history["colour"])
        dominant_appearance = self._top_values(history["appearance"])

        prompt_payload = {
            "user_id": user_id,
            "dominant_product_groups": dominant_groups,
            "dominant_product_types": dominant_types,
            "dominant_colours": dominant_colours,
            "dominant_appearance": dominant_appearance,
        }
        llm_result = self._request_structured_completion(
            system_prompt=(
                "You are an e-commerce recommendation analyst. "
                "Return JSON only with keys: inferred_intent, preferred_categories, "
                "preferred_product_types, preferred_colours, preferred_appearance, shopping_context."
            ),
            user_payload=prompt_payload,
        )

        required_keys = {
            "inferred_intent",
            "preferred_categories",
            "preferred_product_types",
            "preferred_colours",
            "preferred_appearance",
            "shopping_context",
        }
        if not required_keys.issubset(llm_result):
            raise AgenticServiceError("LLM response was missing one or more required user-intent fields.")

        return {
            "user_id": user_id,
            "inferred_intent": str(llm_result["inferred_intent"]),
            "preferred_categories": [str(item) for item in llm_result["preferred_categories"]],
            "preferred_product_types": [str(item) for item in llm_result["preferred_product_types"]],
            "preferred_colours": [str(item) for item in llm_result["preferred_colours"]],
            "preferred_appearance": [str(item) for item in llm_result["preferred_appearance"]],
            "shopping_context": str(llm_result["shopping_context"]),
        }

    def retrieve_candidate_products(
        self,
        user_profile: dict[str, Any],
        train_df: pd.DataFrame,
    ) -> pd.DataFrame:
        train_df = self._normalize_ids(train_df)
        user_history = set(
            train_df.loc[train_df["customer_id"] == normalize_customer_id(user_profile["user_id"]), "article_id"]
        )
        catalogue = train_df.drop_duplicates("article_id").copy()
        catalogue = catalogue[~catalogue["article_id"].isin(user_history)].copy()

        preferred_groups = set(user_profile["preferred_categories"])
        preferred_types = set(user_profile["preferred_product_types"])
        preferred_colours = set(user_profile["preferred_colours"])
        preferred_appearance = set(user_profile["preferred_appearance"])
        intent_terms = set(str(user_profile["inferred_intent"]).lower().split())

        def retrieval_score(row: pd.Series) -> float:
            name_terms = set(str(row["product_name"]).lower().split())
            score = 0.0
            score += 2.0 if row["product_group"] in preferred_groups else 0.0
            score += 2.5 if row["product_type"] in preferred_types else 0.0
            score += 1.5 if row["colour"] in preferred_colours else 0.0
            score += 1.5 if row["appearance"] in preferred_appearance else 0.0
            score += min(1.0, len(intent_terms & name_terms) * 0.3)
            return score

        catalogue["retrieval_score"] = catalogue.apply(retrieval_score, axis=1)
        ranked = catalogue.sort_values(
            ["retrieval_score", "transaction_date"], ascending=[False, False]
        ).head(self.settings.candidate_pool_size)
        return ranked

    def score_candidates(
        self,
        user_profile: dict[str, Any],
        candidates: pd.DataFrame,
        train_df: pd.DataFrame,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        train_df = self._normalize_ids(train_df)
        if candidates.empty:
            return []

        preferences = self._load_feedback_weights(user_profile["user_id"])
        profile_history = train_df[train_df["customer_id"] == user_profile["user_id"]]
        type_counts = Counter(profile_history["product_type"])
        group_counts = Counter(profile_history["product_group"])
        colour_counts = Counter(profile_history["colour"])
        appearance_counts = Counter(profile_history["appearance"])

        seen_types: set[str] = set()
        scored_items: list[dict[str, Any]] = []
        intent_terms = set(user_profile["inferred_intent"].lower().split())

        for _, row in candidates.iterrows():
            intent_match = self._intent_match(row, user_profile, intent_terms)
            preference_alignment = self._preference_alignment(row, user_profile, preferences)
            product_relevance = self._product_relevance(
                row, group_counts, type_counts, colour_counts, appearance_counts
            )
            diversity = 1.0 if row["product_type"] not in seen_types else 0.4 / preferences["diversity_penalty"]
            behavioural_signal = self._behavioural_signal(row, group_counts, type_counts)
            final_score = (
                0.30 * intent_match
                + 0.25 * preference_alignment
                + 0.20 * product_relevance
                + 0.15 * diversity
                + 0.10 * behavioural_signal
            )
            seen_types.add(str(row["product_type"]))
            scored_items.append(
                {
                    "article_id": normalize_article_id(row["article_id"]),
                    "product_name": str(row["product_name"]),
                    "product_type": str(row["product_type"]),
                    "product_group": str(row["product_group"]),
                    "colour": str(row["colour"]),
                    "appearance": str(row["appearance"]),
                    "score": round(float(final_score), 4),
                    "model": "agentic_ai_framework",
                    "intent_match": round(float(intent_match), 4),
                    "preference_alignment": round(float(preference_alignment), 4),
                    "product_relevance": round(float(product_relevance), 4),
                    "diversity": round(float(diversity), 4),
                    "behavioural_signal": round(float(behavioural_signal), 4),
                }
            )

        scored_items.sort(key=lambda item: item["score"], reverse=True)
        top_limit = self.settings.top_n if limit is None else limit
        if top_limit <= 0:
            return []
        return scored_items[:top_limit]

    def generate_explanation(self, user_profile: dict[str, Any], scored_item: dict[str, Any]) -> str:
        llm_result = self._request_structured_completion(
            system_prompt=(
                "You explain why an e-commerce product was recommended. "
                "Return JSON only with a single key called explanation. "
                "Keep the explanation to one sentence and ground it in the provided profile and scores."
            ),
            user_payload={
                "user_profile": user_profile,
                "recommendation": scored_item,
            },
        )
        explanation = llm_result.get("explanation")
        if not explanation:
            raise AgenticServiceError("LLM response did not include an explanation field.")
        return str(explanation)

    def adapt_from_feedback(
        self,
        user_id: str,
        article_id: str,
        feedback_type: FeedbackType,
    ) -> dict[str, float]:
        user_id = normalize_customer_id(user_id)
        article_id = normalize_article_id(article_id)
        weights = self._load_feedback_weights(user_id)
        if feedback_type == "click":
            weights["category_weight"] += 0.05
            weights["colour_weight"] += 0.05
        elif feedback_type == "add_to_cart":
            weights["product_type_weight"] += 0.12
            weights["appearance_weight"] += 0.12
        elif feedback_type == "ignore":
            weights["diversity_penalty"] += 0.08
        elif feedback_type == "purchase":
            weights["product_type_weight"] += 0.2
            weights["category_weight"] += 0.15
        else:
            raise AgenticServiceError(f"Unsupported feedback type: {feedback_type}")

        state = self._read_json_file(self.settings.feedback_state_path, default={})
        state[user_id] = {"weights": weights, "last_article_id": article_id, "last_feedback": feedback_type}
        self.settings.feedback_state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        return weights

    def generate_all(
        self,
        train_df: pd.DataFrame,
        user_ids: list[str],
        output_path=None,
        trace_path=None,
    ) -> dict[str, list[dict[str, Any]]]:
        train_df = self._normalize_ids(train_df)
        user_ids = [normalize_customer_id(user_id) for user_id in user_ids]
        outputs: dict[str, list[dict[str, Any]]] = {}
        traces: dict[str, list[dict[str, Any]]] = {}
        for user_id in user_ids:
            enriched, trace = self.generate_for_user(user_id, train_df)
            outputs[user_id] = enriched
            traces[user_id] = trace

        target_output_path = output_path or self.settings.agentic_output_path
        target_trace_path = trace_path or self.settings.agentic_trace_path
        target_output_path.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
        target_trace_path.write_text(json.dumps(traces, indent=2), encoding="utf-8")
        return outputs

    def build_candidate_pools(self, train_df: pd.DataFrame, user_ids: list[str]) -> dict[str, list[str]]:
        train_df = self._normalize_ids(train_df)
        candidate_pools: dict[str, list[str]] = {}
        for user_id in user_ids:
            profile = self.infer_user_intent(user_id, train_df)
            candidates = self.retrieve_candidate_products(profile, train_df)
            candidate_pools[normalize_customer_id(user_id)] = [
                normalize_article_id(article_id) for article_id in candidates["article_id"]
            ]
        return candidate_pools

    def generate_for_user(
        self,
        user_id: str,
        train_df: pd.DataFrame,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        user_id = normalize_customer_id(user_id)
        train_df = self._normalize_ids(train_df)
        profile = self.infer_user_intent(user_id, train_df)
        candidates = self.retrieve_candidate_products(profile, train_df)
        scored = self.score_candidates(profile, candidates, train_df)
        enriched = []
        for item in scored:
            enriched_item = dict(item)
            enriched_item["reason"] = self.generate_explanation(profile, item)
            enriched.append(enriched_item)
        trace = self._build_agent_trace(user_id, profile, candidates, scored, enriched)
        return enriched, trace

    def build_top10_formal_experiment(self, subset_size: int = 100) -> dict[str, object]:
        return self._build_top10_formal_experiment(subset_size=subset_size)

    def _build_top10_formal_experiment(
        self,
        *,
        subset_size: int = 100,
        artifact_prefix: str | None = None,
        allow_overwrite: bool | None = None,
    ) -> dict[str, object]:
        processed = pd.read_csv(
            self.settings.processed_interactions_with_articles_csv_path,
            dtype={"article_id": "string", "customer_id": "string"},
        )
        processed["article_id"] = processed["article_id"].map(normalize_article_id)
        processed["customer_id"] = processed["customer_id"].map(normalize_customer_id)
        processed["t_dat"] = pd.to_datetime(processed["t_dat"], errors="coerce")

        evaluation_rows = json.loads(
            self.settings.evaluation_base_table_svd_top10_json_path(
                subset_size,
                artifact_prefix=artifact_prefix,
            ).read_text(encoding="utf-8")
        )
        catalogue = self._build_formal_catalogue(processed)

        results: list[dict[str, Any]] = []
        users_with_agentic_recommendations = 0
        users_without_agentic_recommendations = 0
        users_with_full_top_10 = 0
        users_where_top_10_all_inside_candidate_pool = 0
        users_where_top_10_contains_training_items = 0
        users_where_detail_desc_missing_but_handled = 0
        example_agentic_recommendation_users: list[str] = []
        example_agentic_failure_users: list[str] = []

        for row in evaluation_rows:
            result = self._build_formal_agentic_result_for_user(row=row, catalogue=catalogue)
            results.append(result)

            if int(result["recommendation_count"]) > 0:
                users_with_agentic_recommendations += 1
                if len(example_agentic_recommendation_users) < 5:
                    example_agentic_recommendation_users.append(result["customer_id"])
            else:
                users_without_agentic_recommendations += 1
                if len(example_agentic_failure_users) < 5:
                    example_agentic_failure_users.append(result["customer_id"])

            if int(result["recommendation_count"]) == self.settings.top_n:
                users_with_full_top_10 += 1
            if bool(result["top_10_all_inside_candidate_pool"]):
                users_where_top_10_all_inside_candidate_pool += 1
            if bool(result["top_10_contains_training_items"]):
                users_where_top_10_contains_training_items += 1
            if bool(result["detail_desc_missing_but_handled"]):
                users_where_detail_desc_missing_but_handled += 1

        output_json_path = self.settings.ensure_output_path(
            self.settings.agentic_recommendations_top10_json_path(
                subset_size,
                artifact_prefix=artifact_prefix,
            ),
            artifact_prefix=artifact_prefix,
            allow_overwrite=allow_overwrite,
        )
        output_csv_path = self.settings.ensure_output_path(
            self.settings.agentic_recommendations_top10_csv_path(
                subset_size,
                artifact_prefix=artifact_prefix,
            ),
            artifact_prefix=artifact_prefix,
            allow_overwrite=allow_overwrite,
        )
        validation_path = self.settings.ensure_output_path(
            self.settings.agentic_top10_validation_report_path(
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
                        "preference_profile": json.dumps(result["preference_profile"]),
                        "top_10_recommendations": json.dumps(result["top_10_recommendations"]),
                    }
                    for result in results
                ]
            ).to_csv(index=False),
            encoding="utf-8",
        )

        report = {
            "evaluated_users_requested": len(evaluation_rows),
            "users_with_agentic_recommendations": users_with_agentic_recommendations,
            "users_without_agentic_recommendations": users_without_agentic_recommendations,
            "average_top_10_length": round(
                sum(int(result["recommendation_count"]) for result in results) / len(results), 2
            )
            if results
            else 0.0,
            "users_with_full_top_10": users_with_full_top_10,
            "users_where_top_10_all_inside_candidate_pool": users_where_top_10_all_inside_candidate_pool,
            "users_where_top_10_contains_training_items": users_where_top_10_contains_training_items,
            "users_where_detail_desc_missing_but_handled": users_where_detail_desc_missing_but_handled,
            "example_agentic_recommendation_users": example_agentic_recommendation_users,
            "example_agentic_failure_users": example_agentic_failure_users,
        }
        validation_path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        print(
            "3-agent Top-10 summary: "
            f"users_evaluated={report['evaluated_users_requested']}, "
            f"users_with_valid_top_10={report['users_with_full_top_10']}, "
            f"average_top_10_length={report['average_top_10_length']}, "
            f"top_10_all_inside_candidate_pool_count={report['users_where_top_10_all_inside_candidate_pool']}, "
            f"top_10_contains_training_items_count={report['users_where_top_10_contains_training_items']}, "
            f"detail_desc_missing_handled_count={report['users_where_detail_desc_missing_but_handled']}"
        )
        return report

    def load_traces(self) -> dict[str, list[dict[str, Any]]]:
        return self._read_json_file(self.settings.agentic_trace_path, default={})

    def load_outputs(self) -> dict[str, list[dict[str, Any]]]:
        return self._read_json_file(self.settings.agentic_output_path, default={})

    def _load_feedback_weights(self, user_id: str) -> dict[str, float]:
        user_id = normalize_customer_id(user_id)
        state = self._read_json_file(self.settings.feedback_state_path, default={})
        weights = state.get(user_id, {}).get("weights", {})
        return {key: float(weights.get(key, value)) for key, value in self.BASE_WEIGHTS.items()}

    @staticmethod
    def _normalize_ids(frame: pd.DataFrame) -> pd.DataFrame:
        normalized = frame.copy()
        normalized["article_id"] = normalized["article_id"].map(normalize_article_id)
        normalized["customer_id"] = normalized["customer_id"].map(normalize_customer_id)
        return normalized

    def _request_structured_completion(self, system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
        if not self.settings.openai_api_key:
            raise AgenticServiceError(
                "OPENAI_API_KEY is required for the agentic intention and explanation services."
            )

        request_body = {
            "model": self.settings.openai_model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
        }

        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        try:
            with httpx.Client(timeout=self.settings.llm_timeout_seconds) as client:
                response = client.post(
                    "https://api.openai.com/v1/chat/completions",
                    json=request_body,
                    headers=headers,
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise AgenticServiceError(f"OpenAI request failed: {exc}") from exc

        payload = response.json()
        try:
            content = payload["choices"][0]["message"]["content"]
            return json.loads(content)
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise AgenticServiceError("Unable to parse structured JSON from OpenAI response.") from exc

    @staticmethod
    def _top_values(series: pd.Series, limit: int = 3) -> list[str]:
        return [str(value) for value in series.value_counts().head(limit).index.tolist()]

    @staticmethod
    def _intent_match(row: pd.Series, user_profile: dict[str, Any], intent_terms: set[str]) -> float:
        score = 0.0
        if row["product_group"] in user_profile["preferred_categories"]:
            score += 0.35
        if row["product_type"] in user_profile["preferred_product_types"]:
            score += 0.35
        if row["colour"] in user_profile["preferred_colours"]:
            score += 0.15
        if row["appearance"] in user_profile["preferred_appearance"]:
            score += 0.1
        score += min(0.05, len(intent_terms & set(str(row["product_name"]).lower().split())) * 0.02)
        return min(score, 1.0)

    @staticmethod
    def _preference_alignment(
        row: pd.Series,
        user_profile: dict[str, Any],
        preferences: dict[str, float],
    ) -> float:
        score = 0.0
        score += 0.25 * preferences["category_weight"] if row["product_group"] in user_profile["preferred_categories"] else 0.0
        score += 0.25 * preferences["product_type_weight"] if row["product_type"] in user_profile["preferred_product_types"] else 0.0
        score += 0.25 * preferences["colour_weight"] if row["colour"] in user_profile["preferred_colours"] else 0.0
        score += 0.25 * preferences["appearance_weight"] if row["appearance"] in user_profile["preferred_appearance"] else 0.0
        return min(score / max(preferences["product_type_weight"], 1.0), 1.0)

    @staticmethod
    def _product_relevance(
        row: pd.Series,
        group_counts: Counter[str],
        type_counts: Counter[str],
        colour_counts: Counter[str],
        appearance_counts: Counter[str],
    ) -> float:
        def normalized(counter: Counter[str], key: str) -> float:
            if not counter:
                return 0.0
            return counter.get(str(key), 0) / max(counter.values())

        return min(
            1.0,
            (
                normalized(group_counts, row["product_group"])
                + normalized(type_counts, row["product_type"])
                + normalized(colour_counts, row["colour"])
                + normalized(appearance_counts, row["appearance"])
            )
            / 4,
        )

    @staticmethod
    def _behavioural_signal(row: pd.Series, group_counts: Counter[str], type_counts: Counter[str]) -> float:
        total = sum(group_counts.values()) + sum(type_counts.values())
        if total == 0:
            return 0.0
        return min(
            1.0,
            (group_counts.get(str(row["product_group"]), 0) + type_counts.get(str(row["product_type"]), 0))
            / total
            * 2,
        )

    @staticmethod
    def _read_json_file(path, default: dict[str, Any]) -> dict[str, Any]:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def _build_formal_agentic_result_for_user(
        self,
        *,
        row: dict[str, Any],
        catalogue: pd.DataFrame,
    ) -> dict[str, Any]:
        customer_id = normalize_customer_id(row["customer_id"])
        train_article_ids = [normalize_article_id(article_id) for article_id in row.get("train_article_ids", [])]
        train_article_id_set = {article_id for article_id in train_article_ids if article_id}
        candidate_pool_article_ids = [
            normalize_article_id(article_id) for article_id in row.get("candidate_pool_article_ids", [])
        ]
        candidate_pool_set = set(candidate_pool_article_ids)

        train_history = self._build_formal_history_frame(customer_id, train_article_ids, catalogue)
        preference_profile = self._infer_formal_user_profile(customer_id, train_history)

        candidate_frame = catalogue[catalogue["article_id"].isin(candidate_pool_article_ids)].copy()
        candidate_frame = candidate_frame[~candidate_frame["article_id"].isin(train_article_id_set)].copy()
        candidate_frame = candidate_frame.sort_values("transaction_date", ascending=False)

        missing_detail_desc_handled = bool(
            candidate_frame["detail_desc"].isna().any()
            or candidate_frame["detail_desc"].astype("string").str.strip().eq("").any()
        )
        scored = self.score_candidates(preference_profile, candidate_frame, train_history)

        recommendations: list[dict[str, Any]] = []
        for rank, item in enumerate(scored[: self.settings.top_n], start=1):
            article_row = candidate_frame.loc[candidate_frame["article_id"] == item["article_id"]].iloc[0]
            recommendations.append(
                {
                    "rank": rank,
                    "article_id": item["article_id"],
                    "score": item["score"],
                    "product_type_name": str(article_row["product_type_name"]),
                    "product_group_name": str(article_row["product_group_name"]),
                    "colour_group_name": str(article_row["colour_group_name"]),
                    "graphical_appearance_name": str(article_row["graphical_appearance_name"]),
                    "garment_group_name": str(article_row["garment_group_name"]),
                    "recommendation_reason": self._build_formal_recommendation_reason(preference_profile, article_row, item),
                    "matched_evidence": self._build_matched_evidence(preference_profile, article_row),
                }
            )

        top_10_article_ids = [item["article_id"] for item in recommendations]
        return {
            "customer_id": customer_id,
            "method": "3_agent",
            "ground_truth_article_id": normalize_article_id(row["ground_truth_article_id"]),
            "candidate_pool_size": int(row.get("candidate_pool_size", len(candidate_pool_article_ids))),
            "preference_profile": preference_profile,
            "top_10_recommendations": recommendations,
            "recommendation_count": len(recommendations),
            "top_10_all_inside_candidate_pool": all(article_id in candidate_pool_set for article_id in top_10_article_ids),
            "top_10_contains_training_items": any(article_id in train_article_id_set for article_id in top_10_article_ids),
            "detail_desc_missing_but_handled": missing_detail_desc_handled,
        }

    @staticmethod
    def _build_formal_catalogue(processed: pd.DataFrame) -> pd.DataFrame:
        catalogue = processed.copy()
        catalogue["product_type"] = catalogue["product_type_name"].fillna("Unknown").astype(str)
        catalogue["product_group"] = catalogue["product_group_name"].fillna("Unknown").astype(str)
        catalogue["colour"] = catalogue["colour_group_name"].fillna("Unknown").astype(str)
        catalogue["appearance"] = catalogue["graphical_appearance_name"].fillna("Unknown").astype(str)
        catalogue["product_name"] = catalogue["product_type_name"].fillna("Unknown").astype(str)
        catalogue["transaction_date"] = pd.to_datetime(catalogue["t_dat"], errors="coerce")
        catalogue = catalogue.sort_values("transaction_date", ascending=False)
        return catalogue.drop_duplicates("article_id", keep="first").reset_index(drop=True)

    @staticmethod
    def _build_formal_history_frame(
        customer_id: str,
        train_article_ids: list[str],
        catalogue: pd.DataFrame,
    ) -> pd.DataFrame:
        catalogue_index = catalogue.set_index("article_id", drop=False)
        history_rows: list[dict[str, Any]] = []
        for article_id in train_article_ids:
            if article_id not in catalogue_index.index:
                continue
            article_row = catalogue_index.loc[article_id]
            history_rows.append(
                {
                    "customer_id": customer_id,
                    "article_id": article_id,
                    "product_type": str(article_row["product_type"]),
                    "product_group": str(article_row["product_group"]),
                    "colour": str(article_row["colour"]),
                    "appearance": str(article_row["appearance"]),
                    "product_name": str(article_row["product_name"]),
                    "transaction_date": article_row["transaction_date"],
                }
            )
        return pd.DataFrame(history_rows)

    def _infer_formal_user_profile(self, customer_id: str, train_history: pd.DataFrame) -> dict[str, Any]:
        if train_history.empty:
            raise AgenticServiceError(f"User {customer_id} has no formal training history.")

        preferred_categories = self._top_values(train_history["product_group"])
        preferred_product_types = self._top_values(train_history["product_type"])
        preferred_colours = self._top_values(train_history["colour"])
        preferred_appearance = self._top_values(train_history["appearance"])
        inferred_intent = " ".join(preferred_product_types + preferred_categories).lower()

        return {
            "user_id": customer_id,
            "inferred_intent": inferred_intent,
            "preferred_categories": preferred_categories,
            "preferred_product_types": preferred_product_types,
            "preferred_colours": preferred_colours,
            "preferred_appearance": preferred_appearance,
            "shopping_context": "offline historical preference evaluation",
        }

    @staticmethod
    def _build_formal_recommendation_reason(
        user_profile: dict[str, Any],
        article_row: pd.Series,
        scored_item: dict[str, Any],
    ) -> str:
        evidence = AgenticRecommendationService._build_matched_evidence(user_profile, article_row)
        if evidence:
            return (
                f"Matches historical preferences on {', '.join(evidence[:3])} "
                f"with final score {float(scored_item['score']):.4f}."
            )
        return f"Selected by the existing 3-agent scoring logic with final score {float(scored_item['score']):.4f}."

    @staticmethod
    def _build_matched_evidence(user_profile: dict[str, Any], article_row: pd.Series) -> list[str]:
        evidence: list[str] = []
        if str(article_row["product_group"]) in user_profile["preferred_categories"]:
            evidence.append(f"product_group_name={article_row['product_group_name']}")
        if str(article_row["product_type"]) in user_profile["preferred_product_types"]:
            evidence.append(f"product_type_name={article_row['product_type_name']}")
        if str(article_row["colour"]) in user_profile["preferred_colours"]:
            evidence.append(f"colour_group_name={article_row['colour_group_name']}")
        if str(article_row["appearance"]) in user_profile["preferred_appearance"]:
            evidence.append(f"graphical_appearance_name={article_row['graphical_appearance_name']}")
        return evidence

    def _build_agent_trace(
        self,
        user_id: str,
        user_profile: dict[str, Any],
        candidates: pd.DataFrame,
        scored_items: list[dict[str, Any]],
        enriched_items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        candidate_preview = [
            {
                "article_id": normalize_article_id(row["article_id"]),
                "product_name": str(row["product_name"]),
                "product_type": str(row["product_type"]),
                "colour": str(row["colour"]),
                "retrieval_score": round(float(row["retrieval_score"]), 4),
            }
            for _, row in candidates.head(5).iterrows()
        ]
        reasoning_preview = [
            {
                "article_id": item["article_id"],
                "product_name": item["product_name"],
                "final_score": item["score"],
                "intent_match": item["intent_match"],
                "preference_alignment": item["preference_alignment"],
                "product_relevance": item["product_relevance"],
                "diversity": item["diversity"],
                "behavioural_signal": item["behavioural_signal"],
            }
            for item in scored_items[:5]
        ]
        explanation_preview = [
            {
                "article_id": item["article_id"],
                "product_name": item["product_name"],
                "reason": item["reason"],
            }
            for item in enriched_items[:3]
        ]
        feedback_state = self._read_json_file(self.settings.feedback_state_path, default={}).get(user_id, {})
        weights = self._load_feedback_weights(user_id)

        return [
            {
                "agent": "Agent 1",
                "title": "User Shopping Intention Understanding",
                "summary": "The framework converts observed category, colour, type, and appearance patterns into a structured shopping-intent profile.",
                "payload": {
                    "user_profile": user_profile,
                },
            },
            {
                "agent": "Agent 2",
                "title": "Product Retrieval",
                "summary": f"{len(candidates)} candidate products were retrieved from the filtered catalogue before ranking.",
                "payload": {
                    "candidate_count": len(candidates),
                    "candidate_preview": candidate_preview,
                },
            },
            {
                "agent": "Agent 3",
                "title": "Recommendation Reasoning",
                "summary": "Candidates are ranked with an explicit weighted score covering intent match, alignment, relevance, diversity, and behavioural signal.",
                "payload": {
                    "scoring_formula": "0.30*Intent Match + 0.25*Preference Alignment + 0.20*Product Relevance + 0.15*Diversity + 0.10*Behavioural Signal",
                    "top_scored_items": reasoning_preview,
                },
            },
            {
                "agent": "Agent 4",
                "title": "Recommendation Explanation",
                "summary": "The LLM generates one grounded explanation per top recommendation without changing ranking scores.",
                "payload": {
                    "explanation_preview": explanation_preview,
                },
            },
            {
                "agent": "Agent 5",
                "title": "Feedback Adaptation",
                "summary": "Per-user feedback weights are stored separately so the demo can show how future recommendations would adapt.",
                "payload": {
                    "current_weights": weights,
                    "last_feedback": feedback_state.get("last_feedback"),
                    "last_article_id": feedback_state.get("last_article_id"),
                    "adaptation_rules": {
                        "click": "Slightly increase category and colour weight.",
                        "add_to_cart": "Strongly increase product type and appearance weight.",
                        "ignore": "Increase diversity penalty to reduce similar-item priority.",
                        "purchase": "Treat as a strong category and product-type preference signal.",
                    },
                },
            },
        ]
