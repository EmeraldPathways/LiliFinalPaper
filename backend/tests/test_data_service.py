from __future__ import annotations

import json

import pandas as pd

from app.services.data_service import DataService
from app.utils.normalization import normalize_article_id, normalize_customer_id


def test_time_based_split_preserves_order(isolated_env, sample_interactions: pd.DataFrame):
    service = DataService(isolated_env)
    train_df, test_df, boundary_date = service.create_time_based_split(sample_interactions)

    assert len(train_df) == 9
    assert len(test_df) == 3
    assert boundary_date == "2024-01-10"
    assert train_df.iloc[0]["transaction_date"] == "2024-01-01"
    assert test_df.iloc[0]["transaction_date"] == "2024-01-11"


def test_dense_user_sampling_prefers_repeated_histories(isolated_env):
    service = DataService(isolated_env)
    interactions = pd.DataFrame(
        [
            ["u_dense_1", "a1", "Top", "Garment Upper body", "Black", "Solid", "Top 1", "2024-01-01"],
            ["u_dense_1", "a2", "Top", "Garment Upper body", "Black", "Solid", "Top 2", "2024-01-02"],
            ["u_dense_1", "a3", "Top", "Garment Upper body", "Black", "Solid", "Top 3", "2024-01-03"],
            ["u_dense_1", "a4", "Top", "Garment Upper body", "Black", "Solid", "Top 4", "2024-01-04"],
            ["u_dense_2", "b1", "Dress", "Garment Full body", "Blue", "Solid", "Dress 1", "2024-01-05"],
            ["u_dense_2", "b2", "Dress", "Garment Full body", "Blue", "Solid", "Dress 2", "2024-01-06"],
            ["u_dense_2", "b3", "Dress", "Garment Full body", "Blue", "Solid", "Dress 3", "2024-01-07"],
            ["u_sparse_1", "c1", "Skirt", "Garment Lower body", "White", "Stripe", "Skirt 1", "2024-01-08"],
            ["u_sparse_2", "d1", "Skirt", "Garment Lower body", "White", "Stripe", "Skirt 2", "2024-01-09"],
            ["u_sparse_3", "e1", "Skirt", "Garment Lower body", "White", "Stripe", "Skirt 3", "2024-01-10"],
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
    isolated_env.sample_size = 7

    sampled = service._sample_dense_user_histories(interactions)

    assert len(sampled) == 7
    assert set(sampled["customer_id"]) == {"u_dense_1", "u_dense_2"}
    assert sampled["customer_id"].value_counts().to_dict() == {"u_dense_1": 4, "u_dense_2": 3}


def test_dense_user_sampling_caps_single_user_domination(isolated_env):
    service = DataService(isolated_env)
    rows = []
    for index in range(12):
        rows.append(
            [
                "u_heavy",
                f"a{index}",
                "Top",
                "Garment Upper body",
                "Black",
                "Solid",
                f"Heavy {index}",
                f"2024-01-{index + 1:02d}",
            ]
        )
    for user_number in range(1, 5):
        for interaction_number in range(3):
            rows.append(
                [
                    f"u_repeat_{user_number}",
                    f"r{user_number}{interaction_number}",
                    "Dress",
                    "Garment Full body",
                    "Blue",
                    "Solid",
                    f"Repeat {user_number}-{interaction_number}",
                    f"2024-02-{len(rows) + 1:02d}",
                ]
            )
        columns = [
            "customer_id",
            "article_id",
            "product_type",
            "product_group",
            "colour",
            "appearance",
            "product_name",
            "transaction_date",
        ]
    interactions = pd.DataFrame(rows, columns=columns)
    isolated_env.sample_size = 12
    isolated_env.max_eval_users = 2

    sampled = service._sample_dense_user_histories(interactions)

    assert len(sampled) == 12
    assert sampled["customer_id"].nunique() > 1
    assert sampled["customer_id"].value_counts()["u_heavy"] < 12


def test_build_processed_interactions_with_articles_outputs_join_and_validation_report(isolated_env):
    service = DataService(isolated_env)
    transactions = pd.DataFrame(
        [
            ["2024-01-01", "000abc", 12345, 19.99, 1],
            ["2024-01-02", "000def", "00067890", 29.99, 2],
            ["2024-01-03", "000ghi", "99999999", 9.99, 1],
        ],
        columns=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"],
    )
    articles = pd.DataFrame(
        [
            [normalize_article_id(12345), "Top", "Garment Upper body", "Solid", "Black", "Jersey Basic", "Tops", "Women", "Ladieswear", "Cotton top"],
            [normalize_article_id("00067890"), "Dress", "Garment Full body", None, "Red", "Dresses", "Dresses", "Women", "Ladieswear", None],
        ],
        columns=[
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
        ],
    )
    customers = pd.DataFrame(
        [["000abc"], ["000def"], ["000ghi"]],
        columns=["customer_id"],
    )
    transactions.to_csv(isolated_env.transactions_path, index=False)
    articles.to_csv(isolated_env.articles_path, index=False)
    customers.to_csv(isolated_env.customers_path, index=False)

    report = service.build_processed_interactions_with_articles()

    processed_csv = isolated_env.processed_data_dir / "processed_interactions_with_articles.csv"
    processed_json = isolated_env.processed_data_dir / "processed_interactions_with_articles.json"
    validation_json = isolated_env.processed_data_dir / "processed_data_validation_report.json"

    assert processed_csv.exists()
    assert processed_json.exists()
    assert validation_json.exists()

    processed = pd.read_csv(
        processed_csv,
        dtype={"article_id": "string", "customer_id": "string"},
    )
    assert list(processed.columns) == [
        "customer_id",
        "article_id",
        "t_dat",
        "price",
        "sales_channel_id",
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
    assert processed["article_id"].tolist() == [normalize_article_id(12345), normalize_article_id("00067890")]
    assert processed["customer_id"].tolist() == [
        normalize_customer_id("000abc"),
        normalize_customer_id("000def"),
    ]
    assert report["raw_transaction_row_count"] == 3
    assert report["joined_row_count"] == 2
    assert report["dropped_transaction_rows_due_to_missing_article_match"] == 1
    assert report["article_id_type_check"] == "string"
    assert report["customer_id_type_check"] == "string"
    assert report["missing_metadata_counts"]["graphical_appearance_name"] == 1
    assert report["missing_metadata_counts"]["detail_desc"] == 1

    json_payload = json.loads(processed_json.read_text(encoding="utf-8"))
    assert json_payload[0]["article_id"] == normalize_article_id(12345)
    assert json_payload[1]["article_id"] == normalize_article_id("00067890")


def test_normalize_helpers_preserve_string_ids():
    assert normalize_article_id(12345) == "12345"
    assert normalize_article_id("0012345") == "0012345"
    assert normalize_customer_id("000abc") == "000abc"


def test_build_leave_one_out_evaluation_base_outputs_validity_flags_and_subset_function(isolated_env):
    service = DataService(isolated_env)
    processed = pd.DataFrame(
        [
            ["u1", "0001", "2024-01-01", 10.0, 1, "Top", "Upper", "Solid", "Black", "Jersey", "Dept1", "Sec1", "Idx1", "desc1"],
            ["u1", "0002", "2024-01-02", 10.0, 1, "Top", "Upper", "Solid", "Black", "Jersey", "Dept1", "Sec1", "Idx1", "desc2"],
            ["u1", "0003", "2024-01-03", 10.0, 1, "Top", "Upper", "Solid", "Black", "Jersey", "Dept1", "Sec1", "Idx1", "desc3"],
            ["u1", "0004", "2024-01-04", 10.0, 1, "Top", "Upper", "Solid", "Black", "Jersey", "Dept1", "Sec1", "Idx1", "desc4"],
            ["u1", "0005", "2024-01-05", 10.0, 1, "Top", "Upper", "Solid", "Black", "Jersey", "Dept1", "Sec1", "Idx1", "desc5"],
            ["u1", "0006", "2024-01-06", 10.0, 1, "Top", "Upper", "Solid", "Black", "Jersey", "Dept1", "Sec1", "Idx1", "desc6"],
            ["u1", "0007", "2024-01-07", 10.0, 1, "Top", "Upper", "Solid", "Black", "Jersey", "Dept1", "Sec1", "Idx1", "desc7"],
            ["u1", "0008", "2024-01-08", 10.0, 1, "Top", "Upper", "Solid", "Black", "Jersey", "Dept1", "Sec1", "Idx1", "desc8"],
            ["u1", "0009", "2024-01-09", 10.0, 1, "Top", "Upper", "Solid", "Black", "Jersey", "Dept1", "Sec1", "Idx1", "desc9"],
            ["u1", "0010", "2024-01-10", 10.0, 1, "Top", "Upper", "Solid", "Black", "Jersey", "Dept1", "Sec1", "Idx1", None],
            ["u2", "0101", "2024-01-01", 12.0, 1, "Dress", "Full", "Patterned", "Red", "Dresses", "Dept2", "Sec2", "Idx2", "d1"],
            ["u2", "0102", "2024-01-02", 12.0, 1, "Dress", "Full", "Patterned", "Red", "Dresses", "Dept2", "Sec2", "Idx2", "d2"],
            ["u2", "0103", "2024-01-03", 12.0, 1, "Dress", "Full", "Patterned", "Red", "Dresses", "Dept2", "Sec2", "Idx2", "d3"],
            ["u2", "0104", "2024-01-04", 12.0, 1, "Dress", "Full", "Patterned", "Red", "Dresses", "Dept2", "Sec2", "Idx2", "d4"],
            ["u2", "0105", "2024-01-05", 12.0, 1, "Dress", "Full", "Patterned", "Red", "Dresses", "Dept2", "Sec2", "Idx2", "d5"],
            ["u2", "0106", "2024-01-06", 12.0, 1, "Dress", "Full", "Patterned", "Red", "Dresses", "Dept2", "Sec2", "Idx2", "d6"],
            ["u2", "0107", "2024-01-07", 12.0, 1, "Dress", "Full", "Patterned", "Red", "Dresses", "Dept2", "Sec2", "Idx2", "d7"],
            ["u2", "0108", "2024-01-08", 12.0, 1, "Dress", "Full", "Patterned", "Red", "Dresses", "Dept2", "Sec2", "Idx2", "d8"],
            ["u2", "0109", "2024-01-09", 12.0, 1, "Dress", "Full", "Patterned", "Red", "Dresses", "Dept2", "Sec2", "Idx2", "d9"],
        ],
        columns=[
            "customer_id",
            "article_id",
            "t_dat",
            "price",
            "sales_channel_id",
            "product_type_name",
            "product_group_name",
            "graphical_appearance_name",
            "colour_group_name",
            "garment_group_name",
            "department_name",
            "section_name",
            "index_name",
            "detail_desc",
        ],
    )
    processed.to_csv(isolated_env.processed_interactions_with_articles_csv_path, index=False)

    report = service.build_leave_one_out_evaluation_base()

    output_csv = isolated_env.processed_data_dir / "evaluation_base_table_svd_top10_all_valid.csv"
    output_json = isolated_env.processed_data_dir / "evaluation_base_table_svd_top10_all_valid.json"
    report_json = isolated_env.processed_data_dir / "evaluation_base_validation_report_svd_top10_all_valid.json"

    assert output_csv.exists()
    assert output_json.exists()
    assert report_json.exists()
    assert report["total_customers_before_filtering"] == 2
    assert report["customers_with_at_least_10_transactions"] == 1
    assert report["valid_final_comparison_base_users"] == 1
    assert report["invalid_user_count"] == 1
    assert report["all_base_users_ground_truth_in_catalog_count"] == 2
    assert report["valid_users_ground_truth_in_catalog_count"] == 1
    assert report["valid_users_ground_truth_in_catalog_all_true"] is True
    assert report["valid_users_train_items_in_catalog_all_true"] is True
    assert report["valid_users_core_metadata_complete_all_true"] is True

    base_table = json.loads(output_json.read_text(encoding="utf-8"))
    assert len(base_table) == 2
    valid_user = next(item for item in base_table if item["customer_id"] == "u1")
    invalid_user = next(item for item in base_table if item["customer_id"] == "u2")
    assert valid_user["train_count"] == 9
    assert valid_user["ground_truth_article_id"] == "0010"
    assert valid_user["ground_truth_detail_desc_missing"] is True
    assert valid_user["is_valid_for_final_comparison_base"] is True
    assert invalid_user["is_valid_for_final_comparison_base"] is False
    assert "total_transaction_count_below_10" in invalid_user["invalid_reason"]

    subset = service.create_experiment_subset(sample_size=1, random_seed=7)
    assert len(subset) == 1
    assert subset[0]["customer_id"] == "u1"

    rebuilt_report = service.rebuild_evaluation_base_validation_report()
    assert rebuilt_report["all_base_users_ground_truth_in_catalog_count"] == 2
    assert rebuilt_report["valid_users_ground_truth_in_catalog_count"] == 1


def test_build_svd_top10_debug_subset_outputs_candidate_pool_validation(isolated_env):
    service = DataService(isolated_env)
    base_rows = []
    for user_number in range(1, 4):
        train_article_ids = [f"{user_number:02d}{index:02d}" for index in range(1, 10)]
        base_rows.append(
                {
                    "customer_id": f"u{user_number}",
                    "total_transaction_count": 10,
                    "train_count": 9,
                    "train_article_ids": train_article_ids,
                    "train_transaction_dates": [f"2024-01-{index:02d}" for index in range(1, 10)],
                    "ground_truth_article_id": "0201" if user_number == 1 else ("0101" if user_number == 2 else "9999"),
                "ground_truth_transaction_date": "2024-01-10",
                "ground_truth_product_type_name": "Top",
                "ground_truth_product_group_name": "Upper",
                "ground_truth_colour_group_name": "Black",
                "ground_truth_graphical_appearance_name": "Solid",
                "ground_truth_garment_group_name": "Jersey",
                "ground_truth_department_name": "Dept",
                "ground_truth_section_name": "Sec",
                "ground_truth_index_name": "Idx",
                "ground_truth_detail_desc": None,
                "ground_truth_detail_desc_missing": True,
                "train_items_in_catalog": True,
                "ground_truth_in_catalog": True,
                "article_id_format_check": "passed",
                "customer_id_format_check": "passed",
                "core_metadata_complete": True,
                "is_valid_for_svd_evaluation_base": True,
                "is_valid_for_agentic_evaluation_base": True,
                "is_valid_for_final_comparison_base": True,
                "invalid_reason": "",
            }
        )
    isolated_env.evaluation_base_table_svd_top10_all_valid_json_path.write_text(
        json.dumps(base_rows, indent=2),
        encoding="utf-8",
    )
    processed_rows = []
    for row in base_rows:
        for article_id in row["train_article_ids"] + [row["ground_truth_article_id"]]:
            processed_rows.append(
                [
                    row["customer_id"],
                    article_id,
                    "2024-01-01",
                    10.0,
                    1,
                    "Top",
                    "Upper",
                    "Solid",
                    "Black",
                    "Jersey",
                    "Dept",
                    "Sec",
                    "Idx",
                    None,
                ]
            )
    for article_id in ["9801", "9802", "9803", "9804", "9805"]:
        processed_rows.append(
            ["catalog", article_id, "2024-02-01", 11.0, 1, "Top", "Upper", "Solid", "Black", "Jersey", "Dept", "Sec", "Idx", None]
        )
    pd.DataFrame(
        processed_rows,
        columns=[
            "customer_id",
            "article_id",
            "t_dat",
            "price",
            "sales_channel_id",
            "product_type_name",
            "product_group_name",
            "graphical_appearance_name",
            "colour_group_name",
            "garment_group_name",
            "department_name",
            "section_name",
            "index_name",
            "detail_desc",
        ],
    ).to_csv(isolated_env.processed_interactions_with_articles_csv_path, index=False)
    isolated_env.candidate_pool_size = 4

    report = service.build_svd_top10_debug_subset(sample_size=2, random_seed=42)

    assert report["full_valid_user_pool_size"] == 3
    assert report["svd_scoreable_valid_users_count"] == 2
    assert report["actual_selected_subset_size"] == 2
    assert report["users_with_valid_candidate_pool"] == 2
    assert report["users_with_invalid_candidate_pool"] == 0
    assert report["ground_truth_in_candidate_pool_count"] == 2
    assert report["candidate_pool_all_items_in_catalog_count"] == 2
    assert report["candidate_pool_all_items_in_global_training_vocab_count"] == 2
    assert report["duplicate_candidate_pool_count"] == 0

    subset_payload = json.loads(
        isolated_env.evaluation_base_table_svd_top10_json_path(2).read_text(encoding="utf-8")
    )
    assert len(subset_payload) == 2
    assert all(row["candidate_pool_valid"] is True for row in subset_payload)
    assert all(row["ground_truth_in_candidate_pool"] is True for row in subset_payload)
    assert all(row["candidate_pool_has_duplicates"] is False for row in subset_payload)
