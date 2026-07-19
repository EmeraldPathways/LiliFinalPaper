from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import Settings
from app.utils.normalization import normalize_article_id, normalize_customer_id


@dataclass
class ExplainabilityService:
    settings: Settings

    CORE_METADATA_FIELDS = (
        "product_type_name",
        "product_group_name",
        "colour_group_name",
        "graphical_appearance_name",
    )
    METADATA_FIELDS = (
        "product_type_name",
        "product_group_name",
        "graphical_appearance_name",
        "colour_group_name",
        "garment_group_name",
        "department_name",
        "section_name",
        "index_name",
        "prod_name",
        "detail_desc",
    )
    SCORE_COMPONENT_FIELDS = (
        "normalized_svd_score",
        "normalized_agentic_score",
        "diversity_bonus",
        "hybrid_score",
    )
    HISTORY_FIELD_MAP = {
        "product_group_name": "frequent_product_groups",
        "product_type_name": "frequent_product_types",
        "colour_group_name": "frequent_colours",
        "graphical_appearance_name": "frequent_appearances",
        "garment_group_name": "frequent_garment_groups",
    }
    RANK_SHIFT_DEFINITION = (
        "Positive rank_shift means Hybrid promoted the item upward. Negative means Hybrid demoted the item. "
        "Null means rank shift is not observable from saved artifacts."
    )
    LIMITATION_TEXT = (
        "This explainability audit is offline and does not prove CTR, CVR, customer engagement, conversion, add-to-cart, dwell time, or live feedback adaptation improvement."
    )
    TRADEOFF_TEXT = (
        "Hybrid improves explainability and remains competitive on ranking quality, but it reduces intra-list diversity compared with SVD."
    )
    SAFE_CLAIM_TEXT = (
        "The Hybrid method exposes source-grounded evidence from user history, item metadata, score components and reranking movement, supporting its role as an explainability-oriented augmentation layer over SVD."
    )

    def generate_explainability_artifacts(
        self,
        *,
        artifact_prefix: str = "seed99_robustness",
        output_prefix: str = "seed99",
        sample_size: int = 100,
        allow_overwrite: bool | None = None,
    ) -> dict[str, object]:
        discovered = self._discover_source_artifacts(artifact_prefix)
        evaluation_rows = self._load_json_rows(discovered["evaluation"])
        svd_rows = self._load_json_rows(discovered["svd"])
        hybrid_rows = self._load_json_rows(discovered["hybrid"])
        evaluation_lookup = {
            normalize_customer_id(row.get("customer_id")): row
            for row in evaluation_rows
        }
        generated_at = self._deterministic_generated_at(
            [
                discovered["evaluation"],
                discovered["svd"],
                discovered["hybrid"],
                self.settings.processed_interactions_with_articles_csv_path,
            ]
        )
        users_available = len(hybrid_rows)
        selected_hybrid_rows = sorted(
            hybrid_rows,
            key=lambda row: normalize_customer_id(row.get("customer_id")),
        )[:sample_size]
        selected_customer_ids = {
            normalize_customer_id(row.get("customer_id"))
            for row in selected_hybrid_rows
        }
        selected_train_article_ids_by_customer = {
            customer_id: evaluation_lookup.get(customer_id, {}).get("train_article_ids", [])
            for customer_id in selected_customer_ids
        }
        needed_article_ids = {
            normalize_article_id(article_id)
            for train_article_ids in selected_train_article_ids_by_customer.values()
            for article_id in train_article_ids
        }
        needed_article_ids.update(
            normalize_article_id(item.get("article_id"))
            for row in selected_hybrid_rows
            for item in row.get("top_10_recommendations", [])
        )
        processed = self._load_filtered_processed_rows(
            selected_customer_ids=selected_customer_ids,
            needed_article_ids=needed_article_ids,
        )
        metadata_lookup = self._build_metadata_lookup(processed)
        user_history_summaries = self._build_user_history_summaries(
            processed=processed,
            train_article_ids_by_customer=selected_train_article_ids_by_customer,
        )
        svd_lookup = {
            normalize_customer_id(row.get("customer_id")): row
            for row in svd_rows
        }

        missing_optional_fields = sorted(
            field for field in self.METADATA_FIELDS if field not in processed.columns
        )
        warnings: list[str] = []
        if missing_optional_fields:
            warnings.append(
                "Optional metadata fields unavailable in processed source: " + ", ".join(missing_optional_fields)
            )

        records: list[dict[str, object]] = []
        concentration_rows: list[dict[str, object]] = []
        for hybrid_row in selected_hybrid_rows:
            customer_id = normalize_customer_id(hybrid_row.get("customer_id"))
            evaluation_row = evaluation_lookup.get(customer_id)
            if evaluation_row is None:
                continue
            svd_row = svd_lookup.get(customer_id, {})
            user_history_summary = user_history_summaries.get(
                customer_id,
                self._empty_user_history_summary(),
            )
            svd_rank_lookup = {
                normalize_article_id(item.get("article_id")): item
                for item in svd_row.get("top_10_recommendations", [])
            }
            concentration_rows.append(
                self._build_diversity_concentration_row(
                    customer_id=customer_id,
                    svd_recommendations=svd_row.get("top_10_recommendations", []),
                    hybrid_recommendations=hybrid_row.get("top_10_recommendations", []),
                )
            )
            for item in sorted(hybrid_row.get("top_10_recommendations", []), key=lambda rec: int(rec.get("rank", 0))):
                records.append(
                    self._build_explanation_record(
                        customer_id=customer_id,
                        ground_truth_article_id=normalize_article_id(evaluation_row.get("ground_truth_article_id")),
                        item=item,
                        svd_rank_lookup=svd_rank_lookup,
                        metadata_lookup=metadata_lookup,
                        user_history_summary=user_history_summary,
                    )
                )

        records.sort(key=lambda row: (str(row["customer_id"]), int(row["hybrid_rank"])))
        rank_shift_rows = [self._build_rank_shift_row(record) for record in records]
        audit = self._build_audit_payload(
            records=records,
            warnings=warnings,
            missing_optional_fields=missing_optional_fields,
            concentration_rows=concentration_rows,
            artifact_prefix=artifact_prefix,
            output_prefix=output_prefix,
            sample_size=sample_size,
            users_available=users_available,
        )
        summary = self._build_summary_payload(
            audit=audit,
            artifact_prefix=artifact_prefix,
            output_prefix=output_prefix,
            sample_size=sample_size,
            users_available=users_available,
            users_included=len(selected_hybrid_rows),
            recommendations_explained=len(records),
            generated_at=generated_at,
        )
        case_studies = self._select_case_studies(records=records, concentration_rows=concentration_rows)
        audit["case_study_preview"] = case_studies[0] if case_studies else None
        self._write_outputs(
            summary=summary,
            audit=audit,
            records=records,
            rank_shift_rows=rank_shift_rows,
            case_studies=case_studies,
            output_prefix=output_prefix,
            allow_overwrite=allow_overwrite,
        )
        return {
            "summary": summary,
            "audit": audit,
            "records": records,
            "case_studies": case_studies,
        }

    def load_explainability_page(
        self,
        *,
        output_prefix: str = "seed99",
    ) -> dict[str, object]:
        summary_path = self.settings.explainability_summary_path(output_prefix)
        audit_path = self.settings.explainability_audit_path(output_prefix)
        examples_path = self.settings.explainability_examples_csv_path(output_prefix)
        if not summary_path.exists() or not audit_path.exists() or not examples_path.exists():
            raise FileNotFoundError(
                "Explainability artifacts not found. Run python -m backend.scripts.run_explainability_audit "
                "--artifact-prefix seed99_robustness --sample-size 100 --output-prefix seed99 first."
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        examples = pd.read_csv(examples_path, dtype={"customer_id": "string", "article_id": "string"}).fillna("")
        return {
            "summary": summary,
            "examples": examples.to_dict(orient="records"),
            "rank_shift_highlights": {
                "examples_where_hybrid_promoted_ground_truth": audit["examples_where_hybrid_promoted_ground_truth"],
                "examples_where_hybrid_demoted_items": audit["examples_where_hybrid_demoted_items"],
                "diversity_concentration_examples": audit["diversity_concentration_summary"]["examples"],
            },
            "case_study": audit.get("case_study_preview"),
            "warnings": summary["warnings"],
            "limitations": [
                summary["interpretation"]["limitation"],
                summary["interpretation"]["diversity_tradeoff"],
            ],
        }

    def _discover_source_artifacts(self, artifact_prefix: str) -> dict[str, Path]:
        patterns = {
            "evaluation": f"{artifact_prefix}_evaluation_base_table_top10_*.json",
            "svd": f"{artifact_prefix}_svd_recommendations_top10_*.json",
            "hybrid": f"{artifact_prefix}_hybrid_svd_agentic_recommendations_top10_*.json",
        }
        discovered: dict[str, Path] = {}
        missing: list[str] = []
        for key, pattern in patterns.items():
            matches = list(self.settings.processed_data_dir.glob(pattern))
            if not matches:
                missing.append(str(self.settings.processed_data_dir / pattern))
                continue
            discovered[key] = max(matches, key=self._extract_size_from_name)
        if not self.settings.processed_interactions_with_articles_csv_path.exists():
            missing.append(str(self.settings.processed_interactions_with_articles_csv_path))
        if missing:
            raise FileNotFoundError(
                "Missing required explainability source artifacts for "
                f"{artifact_prefix}. Searched: " + "; ".join(missing)
            )
        return discovered

    @staticmethod
    def _extract_size_from_name(path: Path) -> int:
        stem = path.stem
        try:
            return int(stem.split("_")[-1])
        except ValueError:
            return 0

    @staticmethod
    def _load_json_rows(path: Path) -> list[dict[str, object]]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _normalize_processed(self, processed: pd.DataFrame) -> pd.DataFrame:
        normalized = processed.copy()
        normalized["customer_id"] = normalized["customer_id"].map(normalize_customer_id)
        normalized["article_id"] = normalized["article_id"].map(normalize_article_id)
        return normalized

    def _load_filtered_processed_rows(
        self,
        *,
        selected_customer_ids: set[str],
        needed_article_ids: set[str],
    ) -> pd.DataFrame:
        available_columns = pd.read_csv(
            self.settings.processed_interactions_with_articles_csv_path,
            nrows=0,
        ).columns.tolist()
        usecols = [
            column
            for column in ("customer_id", "article_id", *self.METADATA_FIELDS)
            if column in available_columns
        ]
        chunks: list[pd.DataFrame] = []
        for chunk in pd.read_csv(
            self.settings.processed_interactions_with_articles_csv_path,
            usecols=usecols,
            dtype={"customer_id": "string", "article_id": "string"},
            chunksize=250_000,
        ):
            normalized_chunk = self._normalize_processed(chunk)
            filtered_chunk = normalized_chunk[
                normalized_chunk["customer_id"].isin(selected_customer_ids)
                | normalized_chunk["article_id"].isin(needed_article_ids)
            ].copy()
            if not filtered_chunk.empty:
                chunks.append(filtered_chunk)
        if not chunks:
            return pd.DataFrame(columns=usecols)
        return pd.concat(chunks, ignore_index=True)

    def _build_metadata_lookup(self, processed: pd.DataFrame) -> dict[str, dict[str, object]]:
        fields = [field for field in self.METADATA_FIELDS if field in processed.columns]
        lookup: dict[str, dict[str, object]] = {}
        for article_id, frame in processed.groupby("article_id", sort=False):
            row = frame.iloc[-1]
            lookup[str(article_id)] = {field: self._clean_value(row.get(field)) for field in fields}
        return lookup

    def _build_user_history_summaries(
        self,
        *,
        processed: pd.DataFrame,
        train_article_ids_by_customer: dict[str, list[object]],
    ) -> dict[str, dict[str, list[dict[str, object]]]]:
        summaries: dict[str, dict[str, list[dict[str, object]]]] = {}
        processed_by_customer = {
            customer_id: frame.copy()
            for customer_id, frame in processed.groupby("customer_id", sort=False)
        }
        for customer_id, train_article_ids in train_article_ids_by_customer.items():
            summaries[customer_id] = self._build_user_history_summary(
                processed=processed,
                customer_history=processed_by_customer.get(customer_id),
                train_article_ids=train_article_ids,
            )
        return summaries

    def _build_user_history_summary(
        self,
        *,
        processed: pd.DataFrame,
        customer_history: pd.DataFrame | None,
        train_article_ids: list[object],
    ) -> dict[str, list[dict[str, object]]]:
        normalized_train_ids = [normalize_article_id(article_id) for article_id in train_article_ids]
        if customer_history is not None and not customer_history.empty:
            history = customer_history[customer_history["article_id"].isin(normalized_train_ids)].copy()
        else:
            history = pd.DataFrame(columns=processed.columns)
        if history.empty:
            history = processed[processed["article_id"].isin(normalized_train_ids)].copy()
        summary: dict[str, list[dict[str, object]]] = {}
        for metadata_field, summary_key in self.HISTORY_FIELD_MAP.items():
            if metadata_field not in history.columns:
                summary[summary_key] = []
                continue
            counts = Counter(
                str(value)
                for value in history[metadata_field].tolist()
                if self._clean_value(value) is not None
            )
            summary[summary_key] = [
                {"value": value, "count": count}
                for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            ]
        return summary

    def _empty_user_history_summary(self) -> dict[str, list[dict[str, object]]]:
        return {
            summary_key: []
            for summary_key in self.HISTORY_FIELD_MAP.values()
        }

    def _build_explanation_record(
        self,
        *,
        customer_id: str,
        ground_truth_article_id: str,
        item: dict[str, object],
        svd_rank_lookup: dict[str, dict[str, object]],
        metadata_lookup: dict[str, dict[str, object]],
        user_history_summary: dict[str, list[dict[str, object]]],
    ) -> dict[str, object]:
        article_id = normalize_article_id(item.get("article_id"))
        metadata = metadata_lookup.get(article_id, {})
        merged_metadata = {
            field: self._clean_value(item.get(field))
            if self._clean_value(item.get(field)) is not None
            else self._clean_value(metadata.get(field))
            for field in self.METADATA_FIELDS
        }
        svd_item = svd_rank_lookup.get(article_id)
        hybrid_rank = int(item.get("rank") or 0)
        svd_rank = int(svd_item.get("rank")) if svd_item and svd_item.get("rank") is not None else None
        rank_shift = svd_rank - hybrid_rank if svd_rank is not None else None
        matched_preference_fields = self._build_matched_preference_fields(merged_metadata, user_history_summary)
        explanation_claims = self._build_explanation_claims(
            metadata=merged_metadata,
            matched_preference_fields=matched_preference_fields,
            score_components={field: item.get(field) for field in self.SCORE_COMPONENT_FIELDS},
            hybrid_rank=hybrid_rank,
            svd_rank=svd_rank,
            rank_shift=rank_shift,
        )
        grounded_claim_count = sum(1 for claim in explanation_claims if claim["is_grounded"])
        ungrounded_claim_count = sum(1 for claim in explanation_claims if not claim["is_grounded"])
        record = {
            "customer_id": customer_id,
            "article_id": article_id,
            "hybrid_rank": hybrid_rank,
            "svd_rank": svd_rank,
            "rank_shift": rank_shift,
            "rank_shift_definition": self.RANK_SHIFT_DEFINITION,
            "is_ground_truth": article_id == ground_truth_article_id,
            **{field: self._round_float(item.get(field)) for field in self.SCORE_COMPONENT_FIELDS},
            **merged_metadata,
            **user_history_summary,
            "matched_preference_fields": matched_preference_fields,
            "explanation_claims": explanation_claims,
            "grounded_claim_count": grounded_claim_count,
            "ungrounded_claim_count": ungrounded_claim_count,
        }
        record["explanation_text"] = self._build_explanation_text(record)
        return record

    def _build_matched_preference_fields(
        self,
        metadata: dict[str, object],
        user_history_summary: dict[str, list[dict[str, object]]],
    ) -> list[dict[str, object]]:
        matches: list[dict[str, object]] = []
        for metadata_field, summary_key in self.HISTORY_FIELD_MAP.items():
            value = self._clean_value(metadata.get(metadata_field))
            if value is None:
                continue
            for entry in user_history_summary.get(summary_key, []):
                if entry["value"] == value:
                    matches.append(
                        {
                            "field": metadata_field,
                            "item_value": value,
                            "matched_user_history_count": int(entry["count"]),
                        }
                    )
                    break
        return matches

    def _build_explanation_claims(
        self,
        *,
        metadata: dict[str, object],
        matched_preference_fields: list[dict[str, object]],
        score_components: dict[str, object],
        hybrid_rank: int,
        svd_rank: int | None,
        rank_shift: int | None,
    ) -> list[dict[str, object]]:
        claims: list[dict[str, object]] = []
        product_group = self._clean_value(metadata.get("product_group_name"))
        colour = self._clean_value(metadata.get("colour_group_name"))
        if product_group is not None:
            claims.append(
                {
                    "claim_type": "item_metadata",
                    "claim": f"The item belongs to product group '{product_group}'.",
                    "source_type": "item_metadata",
                    "source_fields": ["product_group_name"],
                    "source_values": {"product_group_name": product_group},
                    "is_grounded": True,
                }
            )
        if colour is not None:
            claims.append(
                {
                    "claim_type": "item_metadata",
                    "claim": f"The item colour is '{colour}'.",
                    "source_type": "item_metadata",
                    "source_fields": ["colour_group_name"],
                    "source_values": {"colour_group_name": colour},
                    "is_grounded": True,
                }
            )
        for match in matched_preference_fields:
            claims.append(
                {
                    "claim_type": "metadata_preference_match",
                    "claim": f"The item {match['field']} matches a frequent user-history attribute.",
                    "source_type": "user_history + item_metadata",
                    "source_fields": [match["field"]],
                    "source_values": {
                        "item_value": match["item_value"],
                        "user_history_count": match["matched_user_history_count"],
                    },
                    "is_grounded": True,
                }
            )
        if all(score_components.get(field) is not None for field in self.SCORE_COMPONENT_FIELDS):
            claims.append(
                {
                    "claim_type": "score_components",
                    "claim": "All Hybrid score components are available for this recommendation.",
                    "source_type": "saved_hybrid_scores",
                    "source_fields": list(self.SCORE_COMPONENT_FIELDS),
                    "source_values": {
                        field: self._round_float(score_components.get(field))
                        for field in self.SCORE_COMPONENT_FIELDS
                    },
                    "is_grounded": True,
                }
            )
        if svd_rank is not None and rank_shift is not None:
            claims.append(
                {
                    "claim_type": "rank_shift",
                    "claim": "The Hybrid reranker changed the item's position relative to saved SVD Top-10 output.",
                    "source_type": "saved_rank_positions",
                    "source_fields": ["svd_rank", "hybrid_rank", "rank_shift"],
                    "source_values": {
                        "svd_rank": svd_rank,
                        "hybrid_rank": hybrid_rank,
                        "rank_shift": rank_shift,
                    },
                    "is_grounded": True,
                }
            )
        return claims

    def _build_explanation_text(self, record: dict[str, object]) -> str:
        clauses = [
            "This item was recommended by the Hybrid reranker."
        ]
        metadata_bits = []
        product_group = record.get("product_group_name")
        colour = record.get("colour_group_name")
        if product_group:
            metadata_bits.append(f"product group '{product_group}'")
        if colour:
            metadata_bits.append(f"colour '{colour}'")
        if metadata_bits:
            clauses.append("The item metadata shows " + " and ".join(metadata_bits) + ".")
        for match in record["matched_preference_fields"]:
            clauses.append(
                f"The item {match['field']} value '{match['item_value']}' appears {match['matched_user_history_count']} times in the user's training history."
            )
        if all(record.get(field) is not None for field in self.SCORE_COMPONENT_FIELDS):
            clauses.append(
                "Saved Hybrid score components are available with "
                f"SVD score {record['normalized_svd_score']:.3f}, "
                f"agentic score {record['normalized_agentic_score']:.3f}, "
                f"diversity bonus {record['diversity_bonus']:.3f}, "
                f"and hybrid score {record['hybrid_score']:.3f}."
            )
        if record.get("rank_shift") is not None:
            shift = int(record["rank_shift"])
            direction = "promoted" if shift > 0 else "demoted" if shift < 0 else "kept"
            if shift == 0:
                clauses.append(
                    f"The item stayed at the same position in saved SVD and Hybrid Top-10 outputs at rank {record['hybrid_rank']}."
                )
            else:
                clauses.append(
                    f"The Hybrid rank is {record['hybrid_rank']}, compared with SVD rank {record['svd_rank']}, so the reranking layer {direction} this item by {abs(shift)} positions."
                )
        return " ".join(clauses)

    def _build_diversity_concentration_row(
        self,
        *,
        customer_id: str,
        svd_recommendations: list[dict[str, object]],
        hybrid_recommendations: list[dict[str, object]],
    ) -> dict[str, object]:
        row: dict[str, object] = {"customer_id": customer_id}
        flags = []
        for field, label in (
            ("product_group_name", "product_groups"),
            ("colour_group_name", "colours"),
            ("graphical_appearance_name", "appearances"),
        ):
            svd_distinct = self._distinct_count(svd_recommendations, field)
            hybrid_distinct = self._distinct_count(hybrid_recommendations, field)
            row[f"svd_distinct_{label}"] = svd_distinct
            row[f"hybrid_distinct_{label}"] = hybrid_distinct
            row[f"concentrated_{label}"] = hybrid_distinct < svd_distinct
            if hybrid_distinct < svd_distinct:
                flags.append(field)
        row["concentration_fields"] = flags
        row["is_concentrated"] = bool(flags)
        return row

    @staticmethod
    def _distinct_count(recommendations: list[dict[str, object]], field: str) -> int:
        return len(
            {
                str(item.get(field))
                for item in recommendations
                if item.get(field) is not None and str(item.get(field)).strip() != ""
            }
        )

    def _build_rank_shift_row(self, record: dict[str, object]) -> dict[str, object]:
        return {
            "customer_id": record["customer_id"],
            "article_id": record["article_id"],
            "is_ground_truth": record["is_ground_truth"],
            "svd_rank": record["svd_rank"],
            "hybrid_rank": record["hybrid_rank"],
            "rank_shift": record["rank_shift"],
            "promoted_by_hybrid": bool(record["rank_shift"] is not None and int(record["rank_shift"]) > 0),
            "demoted_by_hybrid": bool(record["rank_shift"] is not None and int(record["rank_shift"]) < 0),
            "product_group_name": record.get("product_group_name"),
            "colour_group_name": record.get("colour_group_name"),
            "graphical_appearance_name": record.get("graphical_appearance_name"),
            "matched_preference_fields_json": json.dumps(record["matched_preference_fields"], sort_keys=True),
            "score_component_summary_json": json.dumps(
                {field: record.get(field) for field in self.SCORE_COMPONENT_FIELDS},
                sort_keys=True,
            ),
        }

    def _build_audit_payload(
        self,
        *,
        records: list[dict[str, object]],
        warnings: list[str],
        missing_optional_fields: list[str],
        concentration_rows: list[dict[str, object]],
        artifact_prefix: str,
        output_prefix: str,
        sample_size: int,
        users_available: int,
    ) -> dict[str, object]:
        total_records = len(records)
        total_claims = sum(len(record["explanation_claims"]) for record in records)
        grounded_claims = sum(record["grounded_claim_count"] for record in records)
        rank_shift_records = [record for record in records if record["rank_shift"] is not None]
        ground_truth_hits = [
            record
            for record in records
            if record["is_ground_truth"] and record["rank_shift"] is not None
        ]
        promoted_ground_truth = [
            self._example_row(record)
            for record in records
            if record["is_ground_truth"] and record["rank_shift"] is not None and int(record["rank_shift"]) > 0
        ]
        demoted_items = [
            self._example_row(record)
            for record in records
            if record["rank_shift"] is not None and int(record["rank_shift"]) < 0
        ]
        concentrated = [row for row in concentration_rows if row["is_concentrated"]]
        if total_claims == 0:
            warnings.append("No structured explanation claims were generated; groundedness_rate is reported as 0.")
        if not ground_truth_hits:
            warnings.append("No Hybrid ground-truth hits with observable rank shift were available.")
        return {
            "artifact_prefix": artifact_prefix,
            "output_prefix": output_prefix,
            "sample_size_requested": sample_size,
            "users_available": users_available,
            "users_included": len({record["customer_id"] for record in records}),
            "recommendation_count": total_records,
            "missing_required_artifacts": [],
            "missing_optional_fields": missing_optional_fields,
            "score_component_availability": {
                field: {
                    "present_count": sum(1 for record in records if record.get(field) is not None),
                    "missing_count": sum(1 for record in records if record.get(field) is None),
                }
                for field in self.SCORE_COMPONENT_FIELDS
            },
            "groundedness_validation": {
                "total_claims": total_claims,
                "grounded_claims": grounded_claims,
                "ungrounded_claims": sum(record["ungrounded_claim_count"] for record in records),
            },
            "rank_shift_coverage": {
                "observable_count": len(rank_shift_records),
                "missing_count": total_records - len(rank_shift_records),
            },
            "examples_where_hybrid_promoted_ground_truth": promoted_ground_truth[:5],
            "examples_where_hybrid_demoted_items": demoted_items[:5],
            "diversity_concentration_summary": {
                "user_count_with_concentration": len(concentrated),
                "examples": concentrated[:5],
            },
            "summary_metrics": {
                "evidence_coverage_rate": self._rate(
                    sum(
                        1
                        for record in records
                        if any(record.get(field) is not None for field in self.CORE_METADATA_FIELDS)
                    ),
                    total_records,
                ),
                "preference_trace_rate": self._rate(
                    sum(1 for record in records if record["matched_preference_fields"]),
                    total_records,
                ),
                "score_component_coverage_rate": self._rate(
                    sum(
                        1
                        for record in records
                        if all(record.get(field) is not None for field in self.SCORE_COMPONENT_FIELDS)
                    ),
                    total_records,
                ),
                "groundedness_rate": self._rate(grounded_claims, total_claims),
                "rank_shift_coverage_rate": self._rate(len(rank_shift_records), total_records),
                "ungrounded_claim_count": sum(record["ungrounded_claim_count"] for record in records),
                "average_rank_shift_for_ground_truth_hits": (
                    round(
                        sum(int(record["rank_shift"]) for record in ground_truth_hits) / len(ground_truth_hits),
                        6,
                    )
                    if ground_truth_hits
                    else None
                ),
            },
            "validation_warnings": warnings,
            "errors": [],
        }

    def _build_summary_payload(
        self,
        *,
        audit: dict[str, object],
        artifact_prefix: str,
        output_prefix: str,
        sample_size: int,
        users_available: int,
        users_included: int,
        recommendations_explained: int,
        generated_at: str,
    ) -> dict[str, object]:
        return {
            "run_context": {
                "artifact_prefix": artifact_prefix,
                "output_prefix": output_prefix,
                "sample_size_requested": sample_size,
                "users_available": users_available,
                "users_included": users_included,
                "recommendations_explained": recommendations_explained,
                "generated_at": generated_at,
            },
            "summary_metrics": audit["summary_metrics"],
            "interpretation": {
                "safe_claim": self.SAFE_CLAIM_TEXT,
                "limitation": self.LIMITATION_TEXT,
                "diversity_tradeoff": self.TRADEOFF_TEXT,
            },
            "warnings": audit["validation_warnings"],
        }

    def _select_case_studies(
        self,
        *,
        records: list[dict[str, object]],
        concentration_rows: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        concentrated_customers = {
            row["customer_id"]
            for row in concentration_rows
            if row["is_concentrated"]
        }

        def priority(record: dict[str, object]) -> tuple[int, int, int, str, int]:
            if record["is_ground_truth"] and record["rank_shift"] is not None and int(record["rank_shift"]) > 0:
                bucket = 0
            elif record["matched_preference_fields"] and record["rank_shift"] is not None and int(record["rank_shift"]) > 0:
                bucket = 1
            elif record["customer_id"] in concentrated_customers:
                bucket = 2
            elif all(record.get(field) is not None for field in self.SCORE_COMPONENT_FIELDS):
                bucket = 3
            else:
                bucket = 4
            rank_shift = int(record["rank_shift"]) if record["rank_shift"] is not None else -999
            return (bucket, -rank_shift, -int(record["grounded_claim_count"]), str(record["customer_id"]), int(record["hybrid_rank"]))

        selected = sorted(records, key=priority)[:10]
        return [
            {
                "customer_id": record["customer_id"],
                "ground_truth_article_id": record["article_id"] if record["is_ground_truth"] else None,
                "recommended_article_id": record["article_id"],
                "is_ground_truth": record["is_ground_truth"],
                "user_history_summary": {
                    key: record[key]
                    for key in (
                        "frequent_product_groups",
                        "frequent_product_types",
                        "frequent_colours",
                        "frequent_appearances",
                        "frequent_garment_groups",
                    )
                },
                "item_metadata": {
                    field: record.get(field)
                    for field in self.METADATA_FIELDS
                },
                "svd_rank": record["svd_rank"],
                "hybrid_rank": record["hybrid_rank"],
                "rank_shift": record["rank_shift"],
                "score_components": {
                    field: record.get(field)
                    for field in self.SCORE_COMPONENT_FIELDS
                },
                "matched_preference_fields": record["matched_preference_fields"],
                "explanation_text": record["explanation_text"],
                "limitation_note": self._case_study_limitation(record),
            }
            for record in selected
        ]

    def _case_study_limitation(self, record: dict[str, object]) -> str | None:
        notes = []
        if record.get("rank_shift") is None:
            notes.append("Rank shift is unavailable because the item is not observable in saved SVD Top-10 output.")
        if not all(record.get(field) is not None for field in self.SCORE_COMPONENT_FIELDS):
            notes.append("Some saved Hybrid score components are unavailable for this recommendation.")
        return " ".join(notes) if notes else None

    def _write_outputs(
        self,
        *,
        summary: dict[str, object],
        audit: dict[str, object],
        records: list[dict[str, object]],
        rank_shift_rows: list[dict[str, object]],
        case_studies: list[dict[str, object]],
        output_prefix: str,
        allow_overwrite: bool | None,
    ) -> None:
        summary_path = self.settings.ensure_output_path(
            self.settings.explainability_summary_path(output_prefix),
            allow_overwrite=allow_overwrite,
        )
        audit_path = self.settings.ensure_output_path(
            self.settings.explainability_audit_path(output_prefix),
            allow_overwrite=allow_overwrite,
        )
        examples_path = self.settings.ensure_output_path(
            self.settings.explainability_examples_csv_path(output_prefix),
            allow_overwrite=allow_overwrite,
        )
        rank_shift_path = self.settings.ensure_output_path(
            self.settings.rank_shift_analysis_csv_path(output_prefix),
            allow_overwrite=allow_overwrite,
        )
        case_studies_path = self.settings.ensure_output_path(
            self.settings.explainability_case_studies_path(output_prefix),
            allow_overwrite=allow_overwrite,
        )
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        pd.DataFrame([self._csv_record(record) for record in records]).to_csv(examples_path, index=False)
        pd.DataFrame(rank_shift_rows).to_csv(rank_shift_path, index=False)
        case_studies_path.write_text(self._render_case_studies(case_studies), encoding="utf-8")

    def _csv_record(self, record: dict[str, object]) -> dict[str, object]:
        return {
            "customer_id": record["customer_id"],
            "article_id": record["article_id"],
            "hybrid_rank": record["hybrid_rank"],
            "svd_rank": record["svd_rank"],
            "rank_shift": record["rank_shift"],
            "is_ground_truth": record["is_ground_truth"],
            **{field: record.get(field) for field in self.SCORE_COMPONENT_FIELDS},
            **{field: record.get(field) for field in self.METADATA_FIELDS},
            "matched_preference_fields_json": json.dumps(record["matched_preference_fields"], sort_keys=True),
            "explanation_text": record["explanation_text"],
            "grounded_claim_count": record["grounded_claim_count"],
            "ungrounded_claim_count": record["ungrounded_claim_count"],
        }

    @staticmethod
    def _render_case_studies(case_studies: list[dict[str, object]]) -> str:
        lines = ["# Explainability Case Studies", ""]
        for index, case_study in enumerate(case_studies, start=1):
            lines.extend(
                [
                    f"## Case Study {index}",
                    f"- customer_id: {case_study['customer_id']}",
                    f"- ground_truth_article_id: {case_study['ground_truth_article_id']}",
                    f"- recommended_article_id: {case_study['recommended_article_id']}",
                    f"- is_ground_truth: {case_study['is_ground_truth']}",
                    f"- svd_rank: {case_study['svd_rank']}",
                    f"- hybrid_rank: {case_study['hybrid_rank']}",
                    f"- rank_shift: {case_study['rank_shift']}",
                    f"- explanation_text: {case_study['explanation_text']}",
                    f"- limitation_note: {case_study['limitation_note']}",
                    "",
                    "### User History Summary",
                    json.dumps(case_study["user_history_summary"], indent=2, sort_keys=True),
                    "",
                    "### Item Metadata",
                    json.dumps(case_study["item_metadata"], indent=2, sort_keys=True),
                    "",
                    "### Score Components",
                    json.dumps(case_study["score_components"], indent=2, sort_keys=True),
                    "",
                    "### Matched Preference Fields",
                    json.dumps(case_study["matched_preference_fields"], indent=2, sort_keys=True),
                    "",
                ]
            )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _deterministic_generated_at(paths: list[Path]) -> str:
        latest_mtime = max(path.stat().st_mtime for path in paths)
        return datetime.fromtimestamp(latest_mtime, tz=timezone.utc).isoformat()

    @staticmethod
    def _example_row(record: dict[str, object]) -> dict[str, object]:
        return {
            "customer_id": record["customer_id"],
            "article_id": record["article_id"],
            "hybrid_rank": record["hybrid_rank"],
            "svd_rank": record["svd_rank"],
            "rank_shift": record["rank_shift"],
            "is_ground_truth": record["is_ground_truth"],
            "product_group_name": record.get("product_group_name"),
            "colour_group_name": record.get("colour_group_name"),
        }

    @staticmethod
    def _clean_value(value: object) -> str | None:
        if value is None or pd.isna(value):
            return None
        text = str(value).strip()
        return text if text else None

    @staticmethod
    def _round_float(value: object) -> float | None:
        if value is None or pd.isna(value):
            return None
        return round(float(value), 6)

    @staticmethod
    def _rate(numerator: int, denominator: int) -> float:
        if denominator == 0:
            return 0.0
        return round(numerator / denominator, 6)
