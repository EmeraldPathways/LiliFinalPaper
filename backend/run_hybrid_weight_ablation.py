from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

# This experiment must not require an OpenAI key.
os.environ.pop("OPENAI_API_KEY", None)

from app.config import get_settings
from app.services.agentic_service import AgenticRecommendationService
from app.services.cf_service import CollaborativeFilteringService
from app.services.hybrid_service import HybridRecommendationService


# ---------------------------------------------------------------------
# Frozen experiment configuration
# ---------------------------------------------------------------------

SUBSET_SIZE = 1000
TOP_K = 10
RANDOM_STATE = 42

# This must identify the already-saved frozen input from the original run.
# Do not call DataService.build_* methods in this experiment.
SOURCE_PREFIX: str | None = None

# Every new file created by this experiment must begin with this prefix.
RUN_PREFIX = "hybrid_weight_ablation_first1000_v1"

WEIGHT_GRID = [
    {
        "id": "w1_svd090_agent005_div005",
        "svd_weight": 0.90,
        "agentic_weight": 0.05,
        "diversity_weight": 0.05,
        "is_original": False,
    },
    {
        "id": "w2_svd080_agent015_div005",
        "svd_weight": 0.80,
        "agentic_weight": 0.15,
        "diversity_weight": 0.05,
        "is_original": False,
    },
    {
        "id": "w3_svd070_agent025_div005",
        "svd_weight": 0.70,
        "agentic_weight": 0.25,
        "diversity_weight": 0.05,
        "is_original": True,
    },
    {
        "id": "w4_svd060_agent035_div005",
        "svd_weight": 0.60,
        "agentic_weight": 0.35,
        "diversity_weight": 0.05,
        "is_original": False,
    },
    {
        "id": "w5_svd050_agent045_div005",
        "svd_weight": 0.50,
        "agentic_weight": 0.45,
        "diversity_weight": 0.05,
        "is_original": False,
    },
]

# Expected result for the original first-1,000-user configuration.
# This is a reproducibility guard, not a target to force.
EXPECTED_ORIGINAL_HR = 0.498000
EXPECTED_ORIGINAL_NDCG = 0.304552
REPRODUCTION_TOLERANCE = 1e-6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def git_value(*arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def validate_weight_grid() -> None:
    if len(WEIGHT_GRID) != 5:
        raise RuntimeError("Exactly five weight configurations are required.")

    original_count = 0

    for configuration in WEIGHT_GRID:
        svd_weight = float(configuration["svd_weight"])
        agentic_weight = float(configuration["agentic_weight"])
        diversity_weight = float(configuration["diversity_weight"])

        if min(svd_weight, agentic_weight, diversity_weight) < 0:
            raise RuntimeError(
                f"Negative weight found in {configuration['id']}."
            )

        if not math.isclose(
            svd_weight + agentic_weight + diversity_weight,
            1.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"Weights do not sum to 1.00 in {configuration['id']}."
            )

        if not math.isclose(diversity_weight, 0.05, abs_tol=1e-12):
            raise RuntimeError(
                f"Diversity weight changed in {configuration['id']}."
            )

        if not math.isclose(
            svd_weight + agentic_weight,
            0.95,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                f"SVD and Agentic weights do not sum to 0.95 "
                f"in {configuration['id']}."
            )

        if bool(configuration["is_original"]):
            original_count += 1

    if original_count != 1:
        raise RuntimeError(
            "Exactly one configuration must be marked as the original."
        )


def validate_frozen_input(rows: list[dict[str, Any]]) -> None:
    if len(rows) != SUBSET_SIZE:
        raise RuntimeError(
            f"Frozen input contains {len(rows)} users; "
            f"expected {SUBSET_SIZE}."
        )

    customer_ids: list[str] = []

    for index, row in enumerate(rows):
        customer_id = row.get("customer_id")
        ground_truth = row.get("ground_truth_article_id")
        train_ids = row.get("train_article_ids", [])
        candidate_ids = row.get("candidate_pool_article_ids", [])

        if not isinstance(customer_id, str):
            raise RuntimeError(
                f"customer_id is not a string at row {index}."
            )

        if not isinstance(ground_truth, str):
            raise RuntimeError(
                f"ground_truth_article_id is not a string at row {index}."
            )

        if not isinstance(train_ids, list) or not all(
            isinstance(item, str) for item in train_ids
        ):
            raise RuntimeError(
                f"Training history contains non-string IDs at row {index}."
            )

        if not isinstance(candidate_ids, list) or not all(
            isinstance(item, str) for item in candidate_ids
        ):
            raise RuntimeError(
                f"Candidate pool contains non-string IDs at row {index}."
            )

        if len(candidate_ids) != 100:
            raise RuntimeError(
                f"Candidate pool size is {len(candidate_ids)} "
                f"at row {index}; expected 100."
            )

        if len(set(candidate_ids)) != 100:
            raise RuntimeError(
                f"Duplicate candidate IDs found at row {index}."
            )

        if ground_truth not in candidate_ids:
            raise RuntimeError(
                f"Ground truth is missing from the candidate pool "
                f"at row {index}."
            )

        customer_ids.append(customer_id)

    if len(set(customer_ids)) != SUBSET_SIZE:
        raise RuntimeError("Frozen input does not contain 1,000 unique users.")


def compute_hr_and_ndcg(
    evaluation_rows: list[dict[str, Any]],
    recommendation_rows: list[dict[str, Any]],
) -> tuple[float, float]:
    evaluation_lookup = {
        str(row["customer_id"]): row for row in evaluation_rows
    }
    recommendation_lookup = {
        str(row["customer_id"]): row for row in recommendation_rows
    }

    if set(evaluation_lookup) != set(recommendation_lookup):
        missing = set(evaluation_lookup) - set(recommendation_lookup)
        extra = set(recommendation_lookup) - set(evaluation_lookup)

        raise RuntimeError(
            "Recommendation users do not match the frozen input. "
            f"Missing={len(missing)}, extra={len(extra)}."
        )

    hits: list[float] = []
    ndcg_values: list[float] = []

    for customer_id, evaluation_row in evaluation_lookup.items():
        ground_truth = str(evaluation_row["ground_truth_article_id"])
        candidate_pool = set(
            str(item)
            for item in evaluation_row["candidate_pool_article_ids"]
        )
        training_items = set(
            str(item)
            for item in evaluation_row.get("train_article_ids", [])
        )

        recommendation_row = recommendation_lookup[customer_id]
        recommendations = recommendation_row.get(
            "top_10_recommendations",
            [],
        )

        if len(recommendations) != TOP_K:
            raise RuntimeError(
                f"User {customer_id} has {len(recommendations)} "
                f"recommendations; expected {TOP_K}."
            )

        recommendation_ids = [
            str(item["article_id"]) for item in recommendations
        ]

        if len(set(recommendation_ids)) != TOP_K:
            raise RuntimeError(
                f"Duplicate Top-10 recommendations for {customer_id}."
            )

        if not set(recommendation_ids).issubset(candidate_pool):
            raise RuntimeError(
                f"Recommendation outside candidate pool for {customer_id}."
            )

        leaked_items = set(recommendation_ids).intersection(training_items)
        if leaked_items:
            raise RuntimeError(
                f"Training-item leakage for {customer_id}: "
                f"{sorted(leaked_items)}"
            )

        rank = next(
            (
                position
                for position, article_id in enumerate(
                    recommendation_ids,
                    start=1,
                )
                if article_id == ground_truth
            ),
            None,
        )

        if rank is None:
            hits.append(0.0)
            ndcg_values.append(0.0)
        else:
            hits.append(1.0)
            ndcg_values.append(1.0 / math.log2(rank + 1))

    hit_rate = sum(hits) / len(hits)
    ndcg = sum(ndcg_values) / len(ndcg_values)

    return hit_rate, ndcg


def ensure_new_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing experiment output: {path}"
        )


def main() -> None:
    validate_weight_grid()

    settings = get_settings()

    frozen_input_path = (
        settings.evaluation_base_table_svd_top10_json_path(
            SUBSET_SIZE,
            artifact_prefix=SOURCE_PREFIX,
        )
    )

    if not frozen_input_path.exists():
        raise FileNotFoundError(
            "Frozen input was not found. Expected:\n"
            f"{frozen_input_path}\n\n"
            "Do not rebuild it. Verify SOURCE_PREFIX instead."
        )

    frozen_input_bytes_before = frozen_input_path.read_bytes()
    frozen_input_hash = sha256_file(frozen_input_path)

    evaluation_rows = json.loads(
        frozen_input_path.read_text(encoding="utf-8")
    )
    validate_frozen_input(evaluation_rows)

    summary_json_path = (
        settings.processed_data_dir
        / f"{RUN_PREFIX}_metric_summary.json"
    )
    summary_csv_path = (
        settings.processed_data_dir
        / f"{RUN_PREFIX}_metric_summary.csv"
    )
    manifest_path = (
        settings.processed_data_dir
        / f"{RUN_PREFIX}_manifest.json"
    )

    for path in (
        summary_json_path,
        summary_csv_path,
        manifest_path,
    ):
        ensure_new_output(path)

    cf_service = CollaborativeFilteringService(settings)
    agentic_service = AgenticRecommendationService(settings)
    hybrid_service = HybridRecommendationService(
        settings,
        cf_service,
        agentic_service,
    )

    metric_rows: list[dict[str, Any]] = []

    print("Hybrid weight ablation starting")
    print(f"Frozen input: {frozen_input_path}")
    print(f"Frozen input SHA-256: {frozen_input_hash}")
    print(f"Users: {len(evaluation_rows)}")
    print()

    for configuration in WEIGHT_GRID:
        configuration_id = str(configuration["id"])
        output_prefix = f"{RUN_PREFIX}_{configuration_id}"

        print(
            f"Running {configuration_id}: "
            f"SVD={configuration['svd_weight']:.2f}, "
            f"Agentic={configuration['agentic_weight']:.2f}, "
            f"Diversity={configuration['diversity_weight']:.2f}"
        )

        hybrid_service.build_hybrid_svd_agentic_reranker(
            subset_size=SUBSET_SIZE,
            random_state=RANDOM_STATE,
            input_artifact_prefix=SOURCE_PREFIX,
            artifact_prefix=output_prefix,
            svd_weight=float(configuration["svd_weight"]),
            agentic_weight=float(configuration["agentic_weight"]),
            diversity_weight=float(configuration["diversity_weight"]),
            allow_overwrite=False,
        )

        recommendation_path = (
            settings.hybrid_svd_agentic_recommendations_top10_json_path(
                SUBSET_SIZE,
                artifact_prefix=output_prefix,
            )
        )

        recommendation_rows = json.loads(
            recommendation_path.read_text(encoding="utf-8")
        )

        hit_rate, ndcg = compute_hr_and_ndcg(
            evaluation_rows,
            recommendation_rows,
        )

        metric_row = {
            "configuration": configuration_id,
            "is_original_configuration": bool(
                configuration["is_original"]
            ),
            "svd_weight": float(configuration["svd_weight"]),
            "agentic_weight": float(
                configuration["agentic_weight"]
            ),
            "diversity_weight": float(
                configuration["diversity_weight"]
            ),
            "users_evaluated": len(evaluation_rows),
            "hit_rate_at_10": round(hit_rate, 6),
            "ndcg_at_10": round(ndcg, 6),
            "recommendation_file": str(recommendation_path),
        }
        metric_rows.append(metric_row)

        print(
            f"Completed {configuration_id}: "
            f"HR@10={hit_rate:.6f}, "
            f"NDCG@10={ndcg:.6f}"
        )
        print()

    original_row = next(
        row
        for row in metric_rows
        if row["is_original_configuration"]
    )

    original_hr = float(original_row["hit_rate_at_10"])
    original_ndcg = float(original_row["ndcg_at_10"])

    if not math.isclose(
        original_hr,
        EXPECTED_ORIGINAL_HR,
        abs_tol=REPRODUCTION_TOLERANCE,
    ):
        raise RuntimeError(
            "Original W3 HitRate@10 was not reproduced. "
            f"Expected {EXPECTED_ORIGINAL_HR:.6f}, "
            f"received {original_hr:.6f}. "
            "Stop and investigate; do not report the sweep."
        )

    if not math.isclose(
        original_ndcg,
        EXPECTED_ORIGINAL_NDCG,
        abs_tol=REPRODUCTION_TOLERANCE,
    ):
        raise RuntimeError(
            "Original W3 NDCG@10 was not reproduced. "
            f"Expected {EXPECTED_ORIGINAL_NDCG:.6f}, "
            f"received {original_ndcg:.6f}. "
            "Stop and investigate; do not report the sweep."
        )

    for row in metric_rows:
        row["delta_hr_vs_original"] = round(
            float(row["hit_rate_at_10"]) - original_hr,
            6,
        )
        row["delta_ndcg_vs_original"] = round(
            float(row["ndcg_at_10"]) - original_ndcg,
            6,
        )

    hr_order = sorted(
        metric_rows,
        key=lambda row: float(row["hit_rate_at_10"]),
        reverse=True,
    )
    ndcg_order = sorted(
        metric_rows,
        key=lambda row: float(row["ndcg_at_10"]),
        reverse=True,
    )

    for rank, row in enumerate(hr_order, start=1):
        row["hr_rank"] = rank

    for rank, row in enumerate(ndcg_order, start=1):
        row["ndcg_rank"] = rank

    manifest = {
        "experiment": "Hybrid weight-sensitivity ablation",
        "source_prefix": SOURCE_PREFIX,
        "run_prefix": RUN_PREFIX,
        "subset_size": SUBSET_SIZE,
        "top_k": TOP_K,
        "random_state": RANDOM_STATE,
        "frozen_input_path": str(frozen_input_path),
        "frozen_input_size_bytes": frozen_input_path.stat().st_size,
        "frozen_input_sha256": frozen_input_hash,
        "git_branch": git_value("branch", "--show-current"),
        "git_commit": git_value("rev-parse", "HEAD"),
        "openai_used": False,
        "controlled_variables": {
            "same_users": True,
            "same_user_order": True,
            "same_training_histories": True,
            "same_ground_truths": True,
            "same_candidate_pools": True,
            "same_candidate_order": True,
            "same_svd_configuration": True,
            "same_deterministic_agentic_logic": True,
            "same_diversity_calculation": True,
            "same_top_k": True,
            "diversity_weight_fixed": 0.05,
        },
        "weight_grid": WEIGHT_GRID,
    }

    summary_payload = {
        "manifest": manifest,
        "metrics": metric_rows,
        "original_configuration_reproduced": True,
        "expected_original_metrics": {
            "hit_rate_at_10": EXPECTED_ORIGINAL_HR,
            "ndcg_at_10": EXPECTED_ORIGINAL_NDCG,
        },
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    summary_json_path.write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )

    with summary_csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file_handle:
        writer = csv.DictWriter(
            file_handle,
            fieldnames=[
                "configuration",
                "is_original_configuration",
                "svd_weight",
                "agentic_weight",
                "diversity_weight",
                "users_evaluated",
                "hit_rate_at_10",
                "ndcg_at_10",
                "delta_hr_vs_original",
                "delta_ndcg_vs_original",
                "hr_rank",
                "ndcg_rank",
                "recommendation_file",
            ],
        )
        writer.writeheader()
        writer.writerows(metric_rows)

    frozen_input_bytes_after = frozen_input_path.read_bytes()

    if frozen_input_bytes_before != frozen_input_bytes_after:
        raise RuntimeError(
            "Frozen input changed during execution."
        )

    if sha256_file(frozen_input_path) != frozen_input_hash:
        raise RuntimeError(
            "Frozen-input SHA-256 changed during execution."
        )

    print("Final comparison")
    print("-" * 104)
    print(
        f"{'Configuration':36}"
        f"{'SVD':>8}"
        f"{'Agent':>8}"
        f"{'Div':>8}"
        f"{'HR@10':>12}"
        f"{'NDCG@10':>12}"
        f"{'HR Rank':>10}"
        f"{'NDCG Rank':>12}"
    )
    print("-" * 104)

    for row in metric_rows:
        print(
            f"{row['configuration']:36}"
            f"{row['svd_weight']:>8.2f}"
            f"{row['agentic_weight']:>8.2f}"
            f"{row['diversity_weight']:>8.2f}"
            f"{row['hit_rate_at_10']:>12.6f}"
            f"{row['ndcg_at_10']:>12.6f}"
            f"{row['hr_rank']:>10}"
            f"{row['ndcg_rank']:>12}"
        )

    print("-" * 104)
    print(f"Summary JSON: {summary_json_path}")
    print(f"Summary CSV:  {summary_csv_path}")
    print(f"Manifest:     {manifest_path}")
    print("ABLATION_RUN_VALIDATED")


if __name__ == "__main__":
    main()
