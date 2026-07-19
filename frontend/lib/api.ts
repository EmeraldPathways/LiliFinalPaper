export type SummaryCount = {
  label: string;
  value: number;
};

export type ExperimentSummary = {
  dataset: string;
  sample_size: number;
  distinct_users: number;
  distinct_products: number;
  repeat_user_ratio: number;
  average_interactions_per_user: number;
  average_interactions_per_product: number;
  top_product_groups: SummaryCount[];
  top_product_types: SummaryCount[];
  top_colours: SummaryCount[];
  top_appearances: SummaryCount[];
  train_size: number;
  test_size: number;
  split_boundary_date: string;
  sample_user_ids?: string[];
  evaluated_user_ids?: string[];
  evaluated_users?: number;
};

export type ExperimentSetup = {
  dataset: string;
  sample_size: number;
  experiment_mode?: string | null;
  split_method: string;
  benchmark: string;
  proposed_framework: string;
  evaluation_metrics: string[];
  summary: ExperimentSummary | null;
};

export type UserIntent = {
  user_id: string;
  inferred_intent: string;
  preferred_categories: string[];
  preferred_product_types: string[];
  preferred_colours: string[];
  preferred_appearance: string[];
  shopping_context: string;
};

export type RecommendationItem = {
  article_id: string;
  product_name: string;
  product_type: string;
  product_group: string;
  colour: string;
  appearance: string;
  score: number;
  model: string;
};

export type AgenticRecommendationItem = RecommendationItem & {
  reason: string;
  intent_match: number;
  preference_alignment: number;
  product_relevance: number;
  diversity: number;
  behavioural_signal: number;
};

export type RecommendationComparison = {
  user_id: string;
  cf_recommendations: RecommendationItem[];
  agentic_recommendations: AgenticRecommendationItem[];
  agentic_process: AgentProcessStage[];
};

export type AgentProcessStage = {
  agent: string;
  title: string;
  summary: string;
  payload: Record<string, unknown>;
};

export type ModelMetrics = {
  hit_rate_at_10: number;
  preference_alignment?: number | null;
  diversity?: number | null;
  ndcg_at_10?: number | null;
  intra_list_diversity_at_10?: number | null;
  hits_count?: number | null;
  miss_count?: number | null;
  explanation_quality?: number | null;
  feedback_adaptability?: number | null;
};

export type MetricsResponse = {
  collaborative_filtering?: ModelMetrics | null;
  svd_matrix_factorization?: ModelMetrics | null;
  agentic_ai_framework: ModelMetrics;
  business_mapping: Record<string, string>;
  evaluated_users: number;
  generated_at?: string | null;
};

export type ExplainabilitySummaryMetrics = {
  evidence_coverage_rate: number;
  preference_trace_rate: number;
  score_component_coverage_rate: number;
  groundedness_rate: number;
  rank_shift_coverage_rate: number;
  ungrounded_claim_count: number;
  average_rank_shift_for_ground_truth_hits?: number | null;
};

export type ExplainabilitySummary = {
  run_context: {
    artifact_prefix: string;
    output_prefix: string;
    sample_size_requested: number;
    users_available: number;
    users_included: number;
    recommendations_explained: number;
    generated_at: string;
  };
  summary_metrics: ExplainabilitySummaryMetrics;
  interpretation: {
    safe_claim: string;
    limitation: string;
    diversity_tradeoff: string;
  };
  warnings: string[];
};

export type ExplainabilityExampleRow = {
  customer_id: string;
  article_id: string;
  hybrid_rank: number;
  svd_rank?: number | string | null;
  rank_shift?: number | string | null;
  is_ground_truth: boolean | string;
  normalized_svd_score?: number | string | null;
  normalized_agentic_score?: number | string | null;
  diversity_bonus?: number | string | null;
  hybrid_score?: number | string | null;
  product_type_name?: string | null;
  product_group_name?: string | null;
  graphical_appearance_name?: string | null;
  colour_group_name?: string | null;
  garment_group_name?: string | null;
  department_name?: string | null;
  section_name?: string | null;
  index_name?: string | null;
  prod_name?: string | null;
  detail_desc?: string | null;
  matched_preference_fields_json: string;
  explanation_text: string;
  grounded_claim_count: number;
  ungrounded_claim_count: number;
};

export type ExplainabilityCaseStudy = {
  customer_id: string;
  ground_truth_article_id?: string | null;
  recommended_article_id: string;
  is_ground_truth: boolean;
  user_history_summary: Record<string, unknown>;
  item_metadata: Record<string, unknown>;
  svd_rank?: number | null;
  hybrid_rank: number;
  rank_shift?: number | null;
  score_components: Record<string, unknown>;
  matched_preference_fields: Array<Record<string, unknown>>;
  explanation_text: string;
  limitation_note?: string | null;
};

export type ExplainabilityPageResponse = {
  summary: ExplainabilitySummary;
  examples: ExplainabilityExampleRow[];
  rank_shift_highlights: Record<string, unknown>;
  case_study?: ExplainabilityCaseStudy | null;
  warnings: string[];
  limitations: string[];
};

export type ArtifactDemoMethodItem = {
  article_id: string;
  rank?: number | null;
  score?: number | null;
  hybrid_score?: number | null;
  normalized_svd_score?: number | null;
  normalized_agentic_score?: number | null;
  diversity_bonus?: number | null;
  reason?: string | null;
  matched_evidence: string[];
  is_ground_truth: boolean;
  product_type_name?: string | null;
  product_group_name?: string | null;
  colour_group_name?: string | null;
  graphical_appearance_name?: string | null;
  garment_group_name?: string | null;
};

export type ArtifactDemoValueCount = {
  value: string;
  count: number;
};

export type ArtifactDemoDemoUser = {
  label: string;
  customer_id: string;
  customer_id_short: string;
};

export type ArtifactDemoLeaveOneOutContext = {
  training_history_count: number;
  ground_truth_article_id?: string | null;
  candidate_pool_size: number;
  ground_truth_in_candidate_pool?: boolean | null;
};

export type ArtifactDemoGroundTruth = {
  article_id?: string | null;
  product_type_name?: string | null;
  product_group_name?: string | null;
  colour_group_name?: string | null;
  graphical_appearance_name?: string | null;
  garment_group_name?: string | null;
};

export type ArtifactDemoMethodHits = {
  svd: boolean;
  agentic: boolean;
  hybrid: boolean;
};

export type ArtifactDemoPreferenceSummary = {
  inferred_intent?: string | null;
  preferred_categories: string[];
  preferred_product_types: string[];
  preferred_colours: string[];
  preferred_appearance: string[];
  frequent_product_types: ArtifactDemoValueCount[];
  frequent_product_groups: ArtifactDemoValueCount[];
  frequent_colours: ArtifactDemoValueCount[];
  frequent_graphical_appearances: ArtifactDemoValueCount[];
  availability: {
    has_preference_summary: boolean;
    has_history_summary: boolean;
  };
};

export type ArtifactDemoItemMetadata = {
  article_id?: string | null;
  product_type_name?: string | null;
  product_group_name?: string | null;
  colour_group_name?: string | null;
  graphical_appearance_name?: string | null;
  garment_group_name?: string | null;
  prod_name?: string | null;
};

export type ArtifactDemoEvidencePanel = {
  headline?: string | null;
  matched_evidence: string[];
  item_metadata: ArtifactDemoItemMetadata;
  availability: {
    has_item_metadata: boolean;
    has_matched_evidence: boolean;
  };
};

export type ArtifactDemoDecisionPanel = {
  headline?: string | null;
  selected_article_id?: string | null;
  selected_rank?: number | null;
  selected_score?: number | null;
};

export type ArtifactDemoExplanationPanel = {
  article_id?: string | null;
  hybrid_rank?: number | null;
  svd_rank?: number | null;
  rank_shift?: number | null;
  is_ground_truth: boolean;
  explanation_text?: string | null;
  matched_preference_fields: Array<Record<string, unknown>>;
  score_components: {
    normalized_svd_score?: number | null;
    normalized_agentic_score?: number | null;
    diversity_bonus?: number | null;
    hybrid_score?: number | null;
  };
  item_metadata: ArtifactDemoItemMetadata;
  groundedness_status?: string | null;
  warnings: string[];
  availability: {
    has_explanation_text: boolean;
    has_score_breakdown: boolean;
    has_rank_shift: boolean;
    has_item_metadata: boolean;
    prod_name_available: boolean;
  };
};

export type ArtifactDemoWorkflowCase = {
  demo_user: ArtifactDemoDemoUser;
  leave_one_out: ArtifactDemoLeaveOneOutContext;
  ground_truth: ArtifactDemoGroundTruth;
  method_hits: ArtifactDemoMethodHits;
  preference_agent: ArtifactDemoPreferenceSummary;
  evidence_agent: ArtifactDemoEvidencePanel;
  decision_agent: ArtifactDemoDecisionPanel;
  label: string;
  svd_top10: ArtifactDemoMethodItem[];
  agentic_top10: ArtifactDemoMethodItem[];
  hybrid_top10: ArtifactDemoMethodItem[];
  hybrid_explainability: ArtifactDemoExplanationPanel;
};

export type ArtifactDemoWorkflowCasesResponse = {
  artifact_prefix: string;
  explainability_prefix: string;
  cases: ArtifactDemoWorkflowCase[];
};

const CLIENT_API_BASE_URL = "http://127.0.0.1:8009";
const SERVER_API_BASE_URL = "http://127.0.0.1:8009";
const API_BASE_URL =
  typeof window === "undefined" ? SERVER_API_BASE_URL : CLIENT_API_BASE_URL;
const FORMAL_EXPERIMENT_MODE = "svd_top10_experiment";

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  if (init?.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers,
    cache: "no-store",
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(payload.detail ?? `Request failed with ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export async function getSetup(): Promise<ExperimentSetup | null> {
  try {
    return await apiFetch<ExperimentSetup>(`/experiment/setup?mode=${FORMAL_EXPERIMENT_MODE}`);
  } catch {
    return null;
  }
}

export async function getMetrics(): Promise<MetricsResponse | null> {
  try {
    return await apiFetch<MetricsResponse>(`/metrics?mode=${FORMAL_EXPERIMENT_MODE}`);
  } catch {
    return null;
  }
}

export async function getUserIntent(userId: string): Promise<UserIntent> {
  return apiFetch<UserIntent>(`/users/${userId}/intent?mode=${FORMAL_EXPERIMENT_MODE}`);
}

export async function getRecommendationComparison(userId: string): Promise<RecommendationComparison> {
  return apiFetch<RecommendationComparison>(
    `/recommendations/compare/${userId}?mode=${FORMAL_EXPERIMENT_MODE}`,
  );
}

export async function getExplainabilityEvidence(options?: {
  artifactPrefix?: string;
  outputPrefix?: string;
}): Promise<ExplainabilityPageResponse | null> {
  try {
    const artifactPrefix = options?.artifactPrefix ?? "seed99_robustness";
    const outputPrefix = options?.outputPrefix ?? "seed99";
    return await apiFetch<ExplainabilityPageResponse>(
      `/metrics/explainability?mode=svd_top10_experiment&artifact_prefix=${artifactPrefix}&output_prefix=${outputPrefix}`,
    );
  } catch {
    return null;
  }
}

export async function getWorkflowCases(): Promise<ArtifactDemoWorkflowCasesResponse | null> {
  try {
    return await apiFetch<ArtifactDemoWorkflowCasesResponse>(
      "/demo/workflow-cases?artifact_prefix=seed99_robustness&explainability_prefix=seed99_full_retry",
    );
  } catch {
    return null;
  }
}

export async function sendFeedback(
  userId: string,
  articleId: string,
  feedbackType: "click" | "add_to_cart" | "ignore" | "purchase",
): Promise<{ message: string; updated_weights: Record<string, number> }> {
  return apiFetch(`/users/${userId}/feedback`, {
    method: "POST",
    body: JSON.stringify({
      article_id: articleId,
      feedback_type: feedbackType,
    }),
  });
}
