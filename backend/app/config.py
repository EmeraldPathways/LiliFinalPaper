from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import ClassVar

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Agentic AI Recommendation Demo API"
    dataset_name: str = "H&M Personalized Fashion Recommendations"
    frontend_origin: str = Field(
        default="http://localhost:3009,http://127.0.0.1:3009",
        alias="FRONTEND_ORIGIN",
    )
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")
    backend_data_dir: Path | None = Field(default=None, alias="BACKEND_DATA_DIR")
    sample_size: int = Field(default=20_000, alias="SAMPLE_SIZE")
    top_n: int = Field(default=10, alias="TOP_N")
    candidate_pool_size: int = Field(default=100, alias="CANDIDATE_POOL_SIZE")
    min_user_interactions: int = Field(default=3, alias="MIN_USER_INTERACTIONS")
    min_product_interactions: int = Field(default=2, alias="MIN_PRODUCT_INTERACTIONS")
    max_eval_users: int = Field(default=50, alias="MAX_EVAL_USERS")
    llm_timeout_seconds: float = Field(default=40.0, alias="LLM_TIMEOUT_SECONDS")

    LEGACY_DEBUG_EXPERIMENT_MODE: ClassVar[str] = "legacy_debug"
    SVD_TOP10_EXPERIMENT_MODE: ClassVar[str] = "svd_top10_experiment"

    @staticmethod
    def _apply_artifact_prefix(filename: str, artifact_prefix: str | None = None) -> str:
        normalized_prefix = (artifact_prefix or "").strip()
        if not normalized_prefix:
            return filename
        return f"{normalized_prefix}_{filename}"

    def _processed_artifact_path(self, filename: str, artifact_prefix: str | None = None) -> Path:
        return self.processed_data_dir / self._apply_artifact_prefix(filename, artifact_prefix)

    def ensure_output_path(
        self,
        path: Path,
        *,
        artifact_prefix: str | None = None,
        allow_overwrite: bool | None = None,
    ) -> Path:
        overwrite_allowed = artifact_prefix is None if allow_overwrite is None else allow_overwrite
        if path.exists() and not overwrite_allowed:
            raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
        return path

    @property
    def raw_data_dir(self) -> Path:
        if self.backend_data_dir:
            return Path(self.backend_data_dir) / "raw"
        return RAW_DATA_DIR

    @property
    def processed_data_dir(self) -> Path:
        if self.backend_data_dir:
            return Path(self.backend_data_dir) / "processed"
        return PROCESSED_DATA_DIR

    @property
    def explainability_data_dir(self) -> Path:
        return self.processed_data_dir / "explainability"

    @property
    def transactions_path(self) -> Path:
        return self.raw_data_dir / "transactions_train.csv"

    @property
    def articles_path(self) -> Path:
        return self.raw_data_dir / "articles.csv"

    @property
    def customers_path(self) -> Path:
        return self.raw_data_dir / "customers.csv"

    @property
    def frontend_origins(self) -> list[str]:
        return [origin.strip() for origin in self.frontend_origin.split(",") if origin.strip()]

    @property
    def interactions_path(self) -> Path:
        return self.processed_data_dir / "interactions_sample.csv"

    @property
    def train_path(self) -> Path:
        return self.processed_data_dir / "train.csv"

    @property
    def test_path(self) -> Path:
        return self.processed_data_dir / "test.csv"

    @property
    def summary_path(self) -> Path:
        return self.processed_data_dir / "experiment_summary.json"

    @property
    def cf_output_path(self) -> Path:
        return self.processed_data_dir / "cf_recommendations.json"

    @property
    def agentic_output_path(self) -> Path:
        return self.processed_data_dir / "agentic_recommendations.json"

    @property
    def agentic_trace_path(self) -> Path:
        return self.processed_data_dir / "agentic_trace.json"

    @property
    def metrics_path(self) -> Path:
        return self.processed_data_dir / "metrics.json"

    @property
    def experiment_state_path(self) -> Path:
        return self.processed_data_dir / "experiment_state.json"

    @property
    def feedback_state_path(self) -> Path:
        return self.processed_data_dir / "feedback_state.json"

    @property
    def processed_interactions_with_articles_csv_path(self) -> Path:
        return self.processed_data_dir / "processed_interactions_with_articles.csv"

    @property
    def processed_interactions_with_articles_json_path(self) -> Path:
        return self.processed_data_dir / "processed_interactions_with_articles.json"

    @property
    def processed_data_validation_report_path(self) -> Path:
        return self.processed_data_dir / "processed_data_validation_report.json"

    @property
    def evaluation_base_table_svd_top10_all_valid_json_path(self) -> Path:
        return self.processed_data_dir / "evaluation_base_table_svd_top10_all_valid.json"

    @property
    def evaluation_base_table_svd_top10_all_valid_csv_path(self) -> Path:
        return self.processed_data_dir / "evaluation_base_table_svd_top10_all_valid.csv"

    @property
    def evaluation_base_validation_report_svd_top10_all_valid_path(self) -> Path:
        return self.processed_data_dir / "evaluation_base_validation_report_svd_top10_all_valid.json"

    @property
    def evaluation_base_table_svd_top10_100_json_path(self) -> Path:
        return self.processed_data_dir / "evaluation_base_table_svd_top10_100.json"

    @property
    def evaluation_base_table_svd_top10_100_csv_path(self) -> Path:
        return self.processed_data_dir / "evaluation_base_table_svd_top10_100.csv"

    @property
    def candidate_pool_validation_report_svd_top10_100_path(self) -> Path:
        return self.processed_data_dir / "candidate_pool_validation_report_svd_top10_100.json"

    @property
    def svd_recommendations_top10_100_json_path(self) -> Path:
        return self.processed_data_dir / "svd_recommendations_top10_100.json"

    @property
    def svd_recommendations_top10_100_csv_path(self) -> Path:
        return self.processed_data_dir / "svd_recommendations_top10_100.csv"

    @property
    def svd_baseline_validation_report_top10_100_path(self) -> Path:
        return self.processed_data_dir / "svd_baseline_validation_report_top10_100.json"

    @property
    def agentic_recommendations_top10_100_json_path(self) -> Path:
        return self.processed_data_dir / "agentic_recommendations_top10_100.json"

    @property
    def agentic_recommendations_top10_100_csv_path(self) -> Path:
        return self.processed_data_dir / "agentic_recommendations_top10_100.csv"

    @property
    def agentic_top10_validation_report_100_path(self) -> Path:
        return self.processed_data_dir / "agentic_top10_validation_report_100.json"

    @property
    def per_user_metrics_top10_100_json_path(self) -> Path:
        return self.processed_data_dir / "per_user_metrics_top10_100.json"

    @property
    def per_user_metrics_top10_100_csv_path(self) -> Path:
        return self.processed_data_dir / "per_user_metrics_top10_100.csv"

    @property
    def metric_summary_top10_100_json_path(self) -> Path:
        return self.processed_data_dir / "metric_summary_top10_100.json"

    @property
    def metric_summary_top10_100_csv_path(self) -> Path:
        return self.processed_data_dir / "metric_summary_top10_100.csv"

    @property
    def metric_validation_report_top10_100_path(self) -> Path:
        return self.processed_data_dir / "metric_validation_report_top10_100.json"

    def evaluation_base_table_svd_top10_json_path(
        self,
        sample_size: int,
        artifact_prefix: str | None = None,
    ) -> Path:
        filename = (
            f"evaluation_base_table_svd_top10_{sample_size}.json"
            if artifact_prefix is None
            else f"evaluation_base_table_top10_{sample_size}.json"
        )
        return self._processed_artifact_path(filename, artifact_prefix)

    def evaluation_base_table_svd_top10_csv_path(
        self,
        sample_size: int,
        artifact_prefix: str | None = None,
    ) -> Path:
        filename = (
            f"evaluation_base_table_svd_top10_{sample_size}.csv"
            if artifact_prefix is None
            else f"evaluation_base_table_top10_{sample_size}.csv"
        )
        return self._processed_artifact_path(filename, artifact_prefix)

    def candidate_pool_validation_report_svd_top10_path(
        self,
        sample_size: int,
        artifact_prefix: str | None = None,
    ) -> Path:
        filename = (
            f"candidate_pool_validation_report_svd_top10_{sample_size}.json"
            if artifact_prefix is None
            else f"candidate_pools_top10_{sample_size}.json"
        )
        return self._processed_artifact_path(filename, artifact_prefix)

    def svd_recommendations_top10_json_path(
        self,
        sample_size: int,
        artifact_prefix: str | None = None,
    ) -> Path:
        return self._processed_artifact_path(
            f"svd_recommendations_top10_{sample_size}.json",
            artifact_prefix,
        )

    def svd_recommendations_top10_csv_path(
        self,
        sample_size: int,
        artifact_prefix: str | None = None,
    ) -> Path:
        return self._processed_artifact_path(
            f"svd_recommendations_top10_{sample_size}.csv",
            artifact_prefix,
        )

    def svd_baseline_validation_report_top10_path(
        self,
        sample_size: int,
        artifact_prefix: str | None = None,
    ) -> Path:
        return self._processed_artifact_path(
            f"svd_baseline_validation_report_top10_{sample_size}.json",
            artifact_prefix,
        )

    def agentic_recommendations_top10_json_path(
        self,
        sample_size: int,
        artifact_prefix: str | None = None,
    ) -> Path:
        return self._processed_artifact_path(
            f"agentic_recommendations_top10_{sample_size}.json",
            artifact_prefix,
        )

    def agentic_recommendations_top10_csv_path(
        self,
        sample_size: int,
        artifact_prefix: str | None = None,
    ) -> Path:
        return self._processed_artifact_path(
            f"agentic_recommendations_top10_{sample_size}.csv",
            artifact_prefix,
        )

    def agentic_top10_validation_report_path(
        self,
        sample_size: int,
        artifact_prefix: str | None = None,
    ) -> Path:
        return self._processed_artifact_path(
            f"agentic_top10_validation_report_{sample_size}.json",
            artifact_prefix,
        )

    def per_user_metrics_top10_json_path(
        self,
        sample_size: int,
        artifact_prefix: str | None = None,
    ) -> Path:
        return self._processed_artifact_path(
            f"per_user_metrics_top10_{sample_size}.json",
            artifact_prefix,
        )

    def per_user_metrics_top10_csv_path(
        self,
        sample_size: int,
        artifact_prefix: str | None = None,
    ) -> Path:
        return self._processed_artifact_path(
            f"per_user_metrics_top10_{sample_size}.csv",
            artifact_prefix,
        )

    def metric_summary_top10_json_path(
        self,
        sample_size: int,
        artifact_prefix: str | None = None,
    ) -> Path:
        return self._processed_artifact_path(
            f"metric_summary_top10_{sample_size}.json",
            artifact_prefix,
        )

    def metric_summary_top10_csv_path(
        self,
        sample_size: int,
        artifact_prefix: str | None = None,
    ) -> Path:
        return self._processed_artifact_path(
            f"metric_summary_top10_{sample_size}.csv",
            artifact_prefix,
        )

    def metric_validation_report_top10_path(
        self,
        sample_size: int,
        artifact_prefix: str | None = None,
    ) -> Path:
        return self._processed_artifact_path(
            f"metric_validation_report_top10_{sample_size}.json",
            artifact_prefix,
        )

    def metric_summary_top10_with_ci_json_path(
        self,
        sample_size: int,
        artifact_prefix: str | None = None,
    ) -> Path:
        return self._processed_artifact_path(
            f"metric_summary_top10_{sample_size}_with_ci.json",
            artifact_prefix,
        )

    def bootstrap_ci_report_top10_json_path(
        self,
        sample_size: int,
        artifact_prefix: str | None = None,
    ) -> Path:
        return self._processed_artifact_path(
            f"bootstrap_ci_report_top10_{sample_size}.json",
            artifact_prefix,
        )

    def hybrid_svd_agentic_recommendations_top10_json_path(
        self,
        sample_size: int,
        artifact_prefix: str | None = None,
    ) -> Path:
        return self._processed_artifact_path(
            f"hybrid_svd_agentic_recommendations_top10_{sample_size}.json",
            artifact_prefix,
        )

    def hybrid_svd_agentic_recommendations_top10_csv_path(
        self,
        sample_size: int,
        artifact_prefix: str | None = None,
    ) -> Path:
        return self._processed_artifact_path(
            f"hybrid_svd_agentic_recommendations_top10_{sample_size}.csv",
            artifact_prefix,
        )

    def hybrid_svd_agentic_validation_report_top10_path(
        self,
        sample_size: int,
        artifact_prefix: str | None = None,
    ) -> Path:
        return self._processed_artifact_path(
            f"hybrid_svd_agentic_validation_report_top10_{sample_size}.json",
            artifact_prefix,
        )

    def per_user_metrics_top10_three_methods_json_path(
        self,
        sample_size: int,
        artifact_prefix: str | None = None,
    ) -> Path:
        return self._processed_artifact_path(
            f"per_user_metrics_top10_{sample_size}_three_methods.json",
            artifact_prefix,
        )

    def per_user_metrics_top10_three_methods_csv_path(
        self,
        sample_size: int,
        artifact_prefix: str | None = None,
    ) -> Path:
        return self._processed_artifact_path(
            f"per_user_metrics_top10_{sample_size}_three_methods.csv",
            artifact_prefix,
        )

    def metric_summary_top10_three_methods_json_path(
        self,
        sample_size: int,
        artifact_prefix: str | None = None,
    ) -> Path:
        return self._processed_artifact_path(
            f"metric_summary_top10_{sample_size}_three_methods.json",
            artifact_prefix,
        )

    def metric_summary_top10_three_methods_csv_path(
        self,
        sample_size: int,
        artifact_prefix: str | None = None,
    ) -> Path:
        return self._processed_artifact_path(
            f"metric_summary_top10_{sample_size}_three_methods.csv",
            artifact_prefix,
        )

    def bootstrap_ci_report_top10_three_methods_json_path(
        self,
        sample_size: int,
        artifact_prefix: str | None = None,
    ) -> Path:
        return self._processed_artifact_path(
            f"bootstrap_ci_report_top10_{sample_size}_three_methods.json",
            artifact_prefix,
        )

    def validation_report_top10_three_methods_json_path(
        self,
        sample_size: int,
        artifact_prefix: str | None = None,
    ) -> Path:
        return self._processed_artifact_path(
            f"validation_report_top10_{sample_size}_three_methods.json",
            artifact_prefix,
        )

    def hybrid_svd_agentic_audit_report_top10_path(
        self,
        sample_size: int,
        artifact_prefix: str | None = None,
    ) -> Path:
        filename = (
            f"hybrid_svd_agentic_audit_report_top10_{sample_size}.md"
            if artifact_prefix is None
            else f"experiment_report_top10_{sample_size}.md"
        )
        return self._processed_artifact_path(filename, artifact_prefix)

    def seed99_output_naming_readiness_report_path(self) -> Path:
        return self.processed_data_dir / "seed99_output_naming_readiness_report.md"

    def experiment_artifact_path(self, experiment_mode: str, artifact_name: str) -> Path:
        if experiment_mode == self.LEGACY_DEBUG_EXPERIMENT_MODE:
            return getattr(self, f"{artifact_name}_path")
        return self.processed_data_dir / f"{experiment_mode}_{artifact_name}.json"

    def explainability_summary_path(self, output_prefix: str) -> Path:
        return self.explainability_data_dir / f"{output_prefix}_explainability_summary.json"

    def explainability_audit_path(self, output_prefix: str) -> Path:
        return self.explainability_data_dir / f"{output_prefix}_explainability_audit.json"

    def explainability_examples_csv_path(self, output_prefix: str) -> Path:
        return self.explainability_data_dir / f"{output_prefix}_explainability_examples.csv"

    def rank_shift_analysis_csv_path(self, output_prefix: str) -> Path:
        return self.explainability_data_dir / f"{output_prefix}_rank_shift_analysis.csv"

    def explainability_case_studies_path(self, output_prefix: str) -> Path:
        return self.explainability_data_dir / f"{output_prefix}_case_studies.md"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    env_file = os.getenv("BACKEND_ENV_FILE")
    if env_file:
        settings = Settings(_env_file=env_file)
    else:
        settings = Settings()
    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
    settings.explainability_data_dir.mkdir(parents=True, exist_ok=True)
    return settings
