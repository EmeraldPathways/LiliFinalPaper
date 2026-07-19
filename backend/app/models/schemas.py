from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


FeedbackType = Literal["click", "add_to_cart", "ignore", "purchase"]


class RecommendationItem(BaseModel):
    article_id: str
    product_name: str
    product_type: str
    product_group: str
    colour: str
    appearance: str
    score: float
    model: str


class AgenticRecommendationItem(RecommendationItem):
    reason: str
    intent_match: float
    preference_alignment: float
    product_relevance: float
    diversity: float
    behavioural_signal: float


class AgentProcessStage(BaseModel):
    agent: str
    title: str
    summary: str
    payload: dict[str, object]


class UserIntentResponse(BaseModel):
    user_id: str
    inferred_intent: str
    preferred_categories: list[str]
    preferred_product_types: list[str]
    preferred_colours: list[str]
    preferred_appearance: list[str]
    shopping_context: str


class RecommendationComparisonResponse(BaseModel):
    user_id: str
    cf_recommendations: list[RecommendationItem]
    agentic_recommendations: list[AgenticRecommendationItem]
    agentic_process: list[AgentProcessStage]


class ModelMetrics(BaseModel):
    hit_rate_at_10: float
    preference_alignment: float | None = None
    diversity: float | None = None
    ndcg_at_10: float | None = None
    intra_list_diversity_at_10: float | None = None
    hits_count: int | None = None
    miss_count: int | None = None
    explanation_quality: float | None = None
    feedback_adaptability: float | None = None


class MetricsResponse(BaseModel):
    collaborative_filtering: ModelMetrics | None = None
    svd_matrix_factorization: ModelMetrics | None = None
    agentic_ai_framework: ModelMetrics
    business_mapping: dict[str, str]
    evaluated_users: int
    generated_at: datetime | None = None


class RunExperimentResponse(BaseModel):
    dataset: str
    status: str
    experiment_mode: str | None = None
    sample_size: int
    train_size: int
    test_size: int
    models: list[str]
    metrics_ready: bool
    evaluated_users: int


class ExperimentSetupResponse(BaseModel):
    dataset: str
    sample_size: int
    experiment_mode: str | None = None
    split_method: str
    benchmark: str
    proposed_framework: str
    evaluation_metrics: list[str]
    summary: dict[str, object] | None = None


class FeedbackRequest(BaseModel):
    article_id: str
    feedback_type: FeedbackType


class FeedbackResponse(BaseModel):
    status: str
    message: str
    user_id: str
    article_id: str
    feedback_type: FeedbackType
    updated_weights: dict[str, float]


class ProcessingSummaryResponse(BaseModel):
    dataset: str
    sample_size: int
    distinct_users: int
    distinct_products: int
    top_product_groups: list[dict[str, object]]
    top_colours: list[dict[str, object]]
    top_appearances: list[dict[str, object]]
    train_size: int
    test_size: int
    split_boundary_date: str


class ExplainabilitySummaryMetrics(BaseModel):
    evidence_coverage_rate: float
    preference_trace_rate: float
    score_component_coverage_rate: float
    groundedness_rate: float
    rank_shift_coverage_rate: float
    ungrounded_claim_count: int
    average_rank_shift_for_ground_truth_hits: float | None = None


class ExplainabilityExampleRow(BaseModel):
    customer_id: str
    article_id: str
    hybrid_rank: int
    svd_rank: int | str | None = None
    rank_shift: int | str | None = None
    is_ground_truth: bool | str
    normalized_svd_score: float | str | None = None
    normalized_agentic_score: float | str | None = None
    diversity_bonus: float | str | None = None
    hybrid_score: float | str | None = None
    product_type_name: str | None = None
    product_group_name: str | None = None
    graphical_appearance_name: str | None = None
    colour_group_name: str | None = None
    garment_group_name: str | None = None
    department_name: str | None = None
    section_name: str | None = None
    index_name: str | None = None
    prod_name: str | None = None
    detail_desc: str | None = None
    matched_preference_fields_json: str
    explanation_text: str
    grounded_claim_count: int
    ungrounded_claim_count: int


class ExplainabilityCaseStudy(BaseModel):
    customer_id: str
    ground_truth_article_id: str | None = None
    recommended_article_id: str
    is_ground_truth: bool
    user_history_summary: dict[str, object]
    item_metadata: dict[str, object]
    svd_rank: int | None = None
    hybrid_rank: int
    rank_shift: int | None = None
    score_components: dict[str, object]
    matched_preference_fields: list[dict[str, object]]
    explanation_text: str
    limitation_note: str | None = None


class ExplainabilityPageResponse(BaseModel):
    summary: dict[str, object]
    examples: list[ExplainabilityExampleRow]
    rank_shift_highlights: dict[str, object]
    case_study: ExplainabilityCaseStudy | None = None
    warnings: list[str]
    limitations: list[str]


class ArtifactDemoMethodItem(BaseModel):
    article_id: str
    rank: int | None = None
    score: float | None = None
    hybrid_score: float | None = None
    normalized_svd_score: float | None = None
    normalized_agentic_score: float | None = None
    diversity_bonus: float | None = None
    reason: str | None = None
    matched_evidence: list[str] = Field(default_factory=list)
    is_ground_truth: bool = False
    product_type_name: str | None = None
    product_group_name: str | None = None
    colour_group_name: str | None = None
    graphical_appearance_name: str | None = None
    garment_group_name: str | None = None


class ArtifactDemoValueCount(BaseModel):
    value: str
    count: int


class ArtifactDemoDemoUser(BaseModel):
    label: str
    customer_id: str
    customer_id_short: str


class ArtifactDemoLeaveOneOutContext(BaseModel):
    training_history_count: int
    ground_truth_article_id: str | None = None
    candidate_pool_size: int
    ground_truth_in_candidate_pool: bool | None = None


class ArtifactDemoGroundTruth(BaseModel):
    article_id: str | None = None
    product_type_name: str | None = None
    product_group_name: str | None = None
    colour_group_name: str | None = None
    graphical_appearance_name: str | None = None
    garment_group_name: str | None = None


class ArtifactDemoPreferenceAvailability(BaseModel):
    has_preference_summary: bool
    has_history_summary: bool


class ArtifactDemoPreferenceSummary(BaseModel):
    inferred_intent: str | None = None
    preferred_categories: list[str] = Field(default_factory=list)
    preferred_product_types: list[str] = Field(default_factory=list)
    preferred_colours: list[str] = Field(default_factory=list)
    preferred_appearance: list[str] = Field(default_factory=list)
    frequent_product_types: list[ArtifactDemoValueCount] = Field(default_factory=list)
    frequent_product_groups: list[ArtifactDemoValueCount] = Field(default_factory=list)
    frequent_colours: list[ArtifactDemoValueCount] = Field(default_factory=list)
    frequent_graphical_appearances: list[ArtifactDemoValueCount] = Field(default_factory=list)
    availability: ArtifactDemoPreferenceAvailability


class ArtifactDemoEvidenceAvailability(BaseModel):
    has_item_metadata: bool
    has_matched_evidence: bool


class ArtifactDemoItemMetadata(BaseModel):
    article_id: str | None = None
    product_type_name: str | None = None
    product_group_name: str | None = None
    colour_group_name: str | None = None
    graphical_appearance_name: str | None = None
    garment_group_name: str | None = None
    prod_name: str | None = None


class ArtifactDemoEvidencePanel(BaseModel):
    headline: str | None = None
    matched_evidence: list[str] = Field(default_factory=list)
    item_metadata: ArtifactDemoItemMetadata
    availability: ArtifactDemoEvidenceAvailability


class ArtifactDemoDecisionPanel(BaseModel):
    headline: str | None = None
    selected_article_id: str | None = None
    selected_rank: int | None = None
    selected_score: float | None = None


class ArtifactDemoMethodHits(BaseModel):
    svd: bool
    agentic: bool
    hybrid: bool


class ArtifactDemoScoreBreakdown(BaseModel):
    normalized_svd_score: float | None = None
    normalized_agentic_score: float | None = None
    diversity_bonus: float | None = None
    hybrid_score: float | None = None


class ArtifactDemoExplanationAvailability(BaseModel):
    has_explanation_text: bool
    has_score_breakdown: bool
    has_rank_shift: bool
    has_item_metadata: bool
    prod_name_available: bool


class ArtifactDemoExplanationPanel(BaseModel):
    article_id: str | None = None
    hybrid_rank: int | None = None
    svd_rank: int | None = None
    rank_shift: int | None = None
    is_ground_truth: bool = False
    explanation_text: str | None = None
    matched_preference_fields: list[dict[str, object]] = Field(default_factory=list)
    score_components: ArtifactDemoScoreBreakdown
    item_metadata: ArtifactDemoItemMetadata
    groundedness_status: str | None = None
    warnings: list[str] = Field(default_factory=list)
    availability: ArtifactDemoExplanationAvailability


class ArtifactDemoWorkflowCase(BaseModel):
    demo_user: ArtifactDemoDemoUser
    leave_one_out: ArtifactDemoLeaveOneOutContext
    ground_truth: ArtifactDemoGroundTruth
    method_hits: ArtifactDemoMethodHits
    preference_agent: ArtifactDemoPreferenceSummary
    evidence_agent: ArtifactDemoEvidencePanel
    decision_agent: ArtifactDemoDecisionPanel
    label: str
    svd_top10: list[ArtifactDemoMethodItem]
    agentic_top10: list[ArtifactDemoMethodItem]
    hybrid_top10: list[ArtifactDemoMethodItem]
    hybrid_explainability: ArtifactDemoExplanationPanel


class ArtifactDemoWorkflowCasesResponse(BaseModel):
    artifact_prefix: str
    explainability_prefix: str
    cases: list[ArtifactDemoWorkflowCase]
