# Project

## Overview

This project is an offline recommendation-system demo and evaluation environment built around the H&M Personalized Fashion Recommendations dataset.

It contains two parallel tracks:

1. A legacy demo track used for the original presentation and debugging.
2. A formal experiment track used for the supervisor-requested comparison:
   - SVD Matrix Factorisation baseline
   - existing 3-agent recommendation framework
   - same users
   - same leave-one-out split
   - same candidate pools
   - Top-10 evaluation surface

The formal experiment mode is named:

- `svd_top10_experiment`

The previous user-based cosine collaborative filtering baseline is still present in the codebase, but it is retained only for debugging and legacy/demo compatibility.

## Objectives

The repository supports three related goals:

1. Demonstrate a recommendation workflow through a FastAPI backend and a frontend dashboard.
2. Build repeatable offline experiment artifacts from raw H&M data.
3. Compare a formal collaborative filtering baseline against the existing agentic framework under controlled conditions.

The current supervisor-facing artifact now uses the final three-method dissertation framing:

- SVD Matrix Factorisation baseline
- Standalone 3-Agent comparison / ablation method
- Hybrid SVD + 3-Agent reranker as the final artifact method

## Current Comparison Design

### Legacy debug path

The original path remains available for compatibility with the earlier demo:

- time-based train/test split
- user-based cosine collaborative filtering
- agentic framework
- JSON artifact outputs used by the presentation flow

This path is not the formal baseline for the paper comparison.

### Formal experiment path

The formal path follows supervisor feedback:

- leave-one-out per customer
- formal joined H&M interaction table
- customer-level validation base
- fixed 100-user debug subset for controlled evaluation
- shared 100-item candidate pool per user
- SVD Matrix Factorisation baseline
- unchanged 3-agent ranking logic
- metrics at `K=10`:
  - `HitRate@10`
  - `NDCG@10`
  - `Intra-list Diversity@10`

## Repository Layout

```text
.
|- backend/
|  |- app/
|  |  |- api/
|  |  |- data/
|  |  |  |- raw/
|  |  |  |- processed/
|  |  |- models/
|  |  |- services/
|  |  |- utils/
|  |  |- config.py
|  |  `- main.py
|  |- tests/
|  |- requirements.txt
|  `- README.md
|- frontend/
|- README.md
`- PROJECT.md
```

## Backend Architecture

### Entry point

- `backend/app/main.py`

Creates the FastAPI app, configures CORS, and mounts the API routers.

### Configuration

- `backend/app/config.py`

Centralizes:

- data paths
- experiment artifact paths
- environment variables
- experiment mode constants
- model and API settings

Important constants:

- `LEGACY_DEBUG_EXPERIMENT_MODE = "legacy_debug"`
- `SVD_TOP10_EXPERIMENT_MODE = "svd_top10_experiment"`

### Services

#### `DataService`

- `backend/app/services/data_service.py`

Responsible for:

- validating raw H&M files
- loading and normalizing transactions/articles
- building the formal processed joined table
- constructing the leave-one-out evaluation base
- constructing the 100-user subset
- constructing shared candidate pools
- building legacy sampled train/test artifacts

Key formal methods:

- `build_processed_interactions_with_articles()`
- `build_leave_one_out_evaluation_base()`
- `rebuild_evaluation_base_validation_report()`
- `build_svd_top10_debug_subset()`

Key legacy/general method:

- `preprocess(...)`

#### `CollaborativeFilteringService`

- `backend/app/services/cf_service.py`

Contains two CF paths:

1. Legacy/debug:
   - user-based cosine CF
   - retained for demo/debug purposes

2. Formal baseline:
   - SVD Matrix Factorisation
   - sparse matrix + `TruncatedSVD`
   - candidate-pool-restricted Top-10 generation

Key methods:

- `generate_all(...)`
- `generate_all_svd(...)`
- `build_svd_top10_debug_baseline()`

The code contains the formal comment requested for the baseline transition.

#### `AgenticRecommendationService`

- `backend/app/services/agentic_service.py`

Implements the existing multi-stage recommendation logic. The internal scoring logic was preserved during the formal experiment work.

Core responsibilities:

- infer preference profile from user history
- build candidate evidence
- score and rank products
- generate explanations for the main demo path
- support feedback-based weight adaptation
- run the formal 100-user Top-10 constrained experiment

Key methods:

- `infer_user_intent(...)`
- `retrieve_candidate_products(...)`
- `score_candidates(...)`
- `generate_for_user(...)`
- `generate_all(...)`
- `build_top10_formal_experiment()`

#### `EvaluationService`

- `backend/app/services/evaluation_service.py`

Provides:

- legacy evaluation outputs
- formal Top-10 evaluation outputs

Key methods:

- `evaluate(...)`
- `evaluate_top10_experiment(...)`

Formal metrics:

- `hit_rate_at_10`
- `ndcg_at_10`
- `intra_list_diversity_at_10`

#### `ExperimentService`

- `backend/app/services/experiment_service.py`

Coordinates end-to-end runs by experiment mode:

- legacy debug run
- `svd_top10_experiment`

#### `ArtifactDemoService`

- `backend/app/services/artifact_demo_service.py`

Provides a read-only adapter for the final artifact demo UI. It does not generate recommendations or write artifacts. Instead, it assembles saved formal experiment outputs and saved explainability rows into deterministic walkthrough cases for the supervisor-facing demo route.

## API Surface

### Health

- `GET /health`

### Experiment

- `POST /experiment/run`
- `GET /experiment/setup`

`/experiment/run` accepts the `mode` query parameter.

Examples:

- `legacy_debug`
- `svd_top10_experiment`

### User and recommendations

- `GET /users/{user_id}/intent`
- `POST /users/{user_id}/feedback`
- `GET /recommendations/compare/{user_id}`

### Demo

- `GET /demo/workflow-cases`

This endpoint is read-only and returns saved walkthrough cases for the final `/artifact-demo` route using existing offline artifacts only.

### Metrics

- `GET /metrics`

## Frontend

The frontend is a Next.js application intended to consume backend artifacts and API responses for the research/demo UI.

The primary supervisor-facing route is now:

- `/artifact-demo`

The current artifact page is a one-page dark card layout with exactly three visible method tabs:

1. `SVD`
2. `3-Agent`
3. `Hybrid`

The route uses the final dissertation framing:

- SVD Matrix Factorisation as the formal behavioural baseline
- Standalone 3-Agent as the comparison / ablation method
- Hybrid SVD + 3-Agent as the final reranking and explainability-oriented artifact

The UI is now card-based rather than table-heavy:

- header card with dissertation framing
- shared real demo-user selector
- SVD recommendation cards
- 3-Agent preference profile, evidence set, and final recommendation cards
- Hybrid workflow cards, formula card, Top-10 cards, and structured explainability audit cards

The live supervisor-facing app is served from this repository's FastAPI + Next.js stack, with `/artifact-demo` as the dissertation artefact route.

This document focuses on the backend experiment system because that is where the formal SVD vs 3-agent comparison is implemented.

## Raw Data Requirements

Place the following Kaggle H&M files in:

- `backend/app/data/raw/`

Required files:

- `transactions_train.csv`
- `articles.csv`
- `customers.csv`

For the formal experiment build, recommendation logic does not use customer demographic attributes. `customers.csv` is loaded only as part of the raw-data validation and compatibility path.

## Data Normalization Rules

Normalization is centralized in:

- `backend/app/utils/normalization.py`

Rules:

- `article_id` is always handled as `string`
- `customer_id` is always handled as `string`
- leading zeros must be preserved

The article normalization helper is applied across:

- transactions
- articles
- train/test splits
- evaluation base tables
- candidate pools
- SVD recommendations
- 3-agent recommendations
- metric calculation

## Formal Data Pipeline

### Step 1: Build processed interactions with article metadata

Inputs:

- `transactions_train.csv`
- `articles.csv`

Fields loaded from transactions:

- `t_dat`
- `customer_id`
- `article_id`
- `price`
- `sales_channel_id`

Fields loaded from articles:

- `article_id`
- `product_type_name`
- `product_group_name`
- `graphical_appearance_name`
- `colour_group_name`
- `garment_group_name`
- `department_name`
- `section_name`
- `index_name`
- `detail_desc`

Join rule:

- inner join on `article_id`

Outputs:

- `backend/app/data/processed/processed_interactions_with_articles.csv`
- `backend/app/data/processed/processed_interactions_with_articles.json`
- `backend/app/data/processed/processed_data_validation_report.json`

### Step 2: Build leave-one-out evaluation base

Source:

- `processed_interactions_with_articles.csv`

Per-customer logic:

- sort the customer’s transactions
- keep the last transaction as ground truth
- use earlier transactions as train history
- require validity checks for formal comparison eligibility

Primary output:

- `backend/app/data/processed/evaluation_base_table_svd_top10_all_valid.json`
- `backend/app/data/processed/evaluation_base_table_svd_top10_all_valid.csv`

Validation output:

- `backend/app/data/processed/evaluation_base_validation_report_svd_top10_all_valid.json`

### Step 3: Build the 100-user subset and shared candidate pools

Source:

- `evaluation_base_table_svd_top10_all_valid.json`

Subset design:

- customer-level sample
- shared candidate pool size target: 100
- held-out ground truth included
- candidate items constrained to catalog/global training vocabulary

Outputs:

- `backend/app/data/processed/evaluation_base_table_svd_top10_100.json`
- `backend/app/data/processed/evaluation_base_table_svd_top10_100.csv`
- `backend/app/data/processed/candidate_pool_validation_report_svd_top10_100.json`

### Step 4: Generate SVD baseline recommendations

Source:

- `processed_interactions_with_articles.csv`
- `evaluation_base_table_svd_top10_100.json`

Rules:

- use training interactions only
- do not use held-out ground truth items as that user’s train history
- binary implicit feedback
- recommend only from `candidate_pool_article_ids`
- exclude `train_article_ids`
- return Top 10

Outputs:

- `backend/app/data/processed/svd_recommendations_top10_100.json`
- `backend/app/data/processed/svd_recommendations_top10_100.csv`
- `backend/app/data/processed/svd_baseline_validation_report_top10_100.json`

### Step 5: Generate 3-agent recommendations for the same users and pools

Source:

- `processed_interactions_with_articles.csv`
- `evaluation_base_table_svd_top10_100.json`

Rules:

- same 100 users
- same candidate pools
- same train histories
- same held-out items
- unchanged 3-agent ranking logic
- Top 10 output
- no recommendation of training items
- no recommendation outside the shared candidate pool

Outputs:

- `backend/app/data/processed/agentic_recommendations_top10_100.json`
- `backend/app/data/processed/agentic_recommendations_top10_100.csv`
- `backend/app/data/processed/agentic_top10_validation_report_100.json`

### Step 6: Formal Top-10 metrics

The metric service supports the final comparison once both formal recommendation sets exist.

Formal metric output is written through:

- `EvaluationService.evaluate_top10_experiment(...)`

Metrics:

- `HitRate@10`
- `NDCG@10`
- `Intra-list Diversity@10`

## Important Process Constraints

The current formal experiment implementation was built under these constraints:

- do not delete the old debug files
- do not modify the original 10-user presentation demo flow
- do not change the 3-agent ranking logic during the formal restructuring
- do not tune 3-agent weights for the formal run
- do not force recommendation hits
- keep candidate pools shared between the two methods

## Key Processed Artifacts

### Formal joined data

- `processed_interactions_with_articles.csv`
- `processed_interactions_with_articles.json`
- `processed_data_validation_report.json`

### Formal evaluation base

- `evaluation_base_table_svd_top10_all_valid.json`
- `evaluation_base_table_svd_top10_all_valid.csv`
- `evaluation_base_validation_report_svd_top10_all_valid.json`

### 100-user experiment base

- `evaluation_base_table_svd_top10_100.json`
- `evaluation_base_table_svd_top10_100.csv`
- `candidate_pool_validation_report_svd_top10_100.json`

### Formal baseline outputs

- `svd_recommendations_top10_100.json`
- `svd_recommendations_top10_100.csv`
- `svd_baseline_validation_report_top10_100.json`

### Formal 3-agent outputs

- `agentic_recommendations_top10_100.json`
- `agentic_recommendations_top10_100.csv`
- `agentic_top10_validation_report_100.json`

### Generic experiment/demo outputs

- `experiment_summary.json`
- `experiment_state.json`
- `cf_recommendations.json`
- `agentic_recommendations.json`
- `agentic_trace.json`
- `metrics.json`

## Validation Reports

The repository now contains multiple validation layers.

### Processed data validation

Confirms:

- raw transaction row count
- raw article row count
- joined row count
- dropped transaction rows after article join
- unique customer count
- unique article count
- ID type checks
- missing metadata counts

### Evaluation base validation

Confirms:

- number of valid final comparison users
- ID format checks
- ground truth catalog membership
- train-item catalog membership
- metadata completeness
- train count statistics
- invalid-reason breakdown

It also includes scoped fields such as:

- `all_base_users_ground_truth_in_catalog_count`
- `valid_users_ground_truth_in_catalog_count`
- `valid_users_ground_truth_in_catalog_all_true`
- `valid_users_train_items_in_catalog_all_true`
- `valid_users_core_metadata_complete_all_true`

### Candidate pool validation

Confirms:

- actual selected subset size
- number of users with valid candidate pools
- ground truth presence in each candidate pool
- all candidate items in catalog
- all candidate items in global training vocabulary
- duplicate candidate count

### SVD validation

Confirms:

- users with recommendations
- full Top-10 coverage
- no train-item leakage
- recommendation containment within candidate pools
- SVD matrix shape
- chosen number of components
- scoreable candidate statistics

### 3-agent formal validation

Confirms:

- users with recommendations
- full Top-10 coverage
- no train-item leakage
- recommendation containment within candidate pools
- handling of missing `detail_desc`

## Recommendation Logic Summary

### Legacy collaborative filtering

The old collaborative filtering implementation is user-based cosine similarity over a dense user-item matrix.

Status:

- retained
- legacy/debug only
- not the formal baseline

### SVD baseline

The formal baseline uses:

- binary implicit feedback
- sparse user-item matrix
- `TruncatedSVD`
- score ranking only inside the user’s shared candidate pool

### 3-agent framework

The existing framework follows a staged structure:

1. preference inference from historical purchases
2. candidate evidence construction
3. decision/ranking
4. explanation generation for the main demo path

For the formal 100-user comparison:

- the same ranking logic is preserved
- recommendations are constrained to the saved candidate pools
- output format is standardized to Top 10

## Environment Variables

Main backend environment variables:

- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `FRONTEND_ORIGIN`
- `BACKEND_DATA_DIR`
- `SAMPLE_SIZE`
- `TOP_N`
- `CANDIDATE_POOL_SIZE`
- `MIN_USER_INTERACTIONS`
- `MIN_PRODUCT_INTERACTIONS`
- `MAX_EVAL_USERS`
- `LLM_TIMEOUT_SECONDS`

Notes:

- `TOP_N` defaults to `10`
- `CANDIDATE_POOL_SIZE` defaults to `100`
- the formal data-building pipeline does not require customer demographics in the recommendation algorithms

## Installation

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Frontend

```bash
cd frontend
npm.cmd install
```

## Running the API

```bash
cd backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8009
```

Health check:

```bash
curl http://127.0.0.1:8009/health
```

Notes:

- The working backend port for the current artifact demo is `8009`.
- Run from `backend/` so `app.main:app` resolves correctly.
- Use the project venv command above instead of relying on a global `uvicorn`.

## Running the Frontend

Install once:

```bash
cd frontend
npm.cmd install
```

Development mode:

```bash
cd frontend
$env:PORT='3009'
$env:NEXT_PUBLIC_API_BASE_URL='http://127.0.0.1:8009'
npm.cmd run dev
```

Production-style local run used for the artifact demo:

```bash
cd frontend
$env:NEXT_PUBLIC_API_BASE_URL='http://127.0.0.1:8009'
npm.cmd run build
npm.cmd run start -- --port 3009
```

Notes:

- The working frontend port for the current artifact demo is `3009`.
- Always set `NEXT_PUBLIC_API_BASE_URL` to `http://127.0.0.1:8009` before starting the frontend.
- The live supervisor-facing page is `http://127.0.0.1:3009/artifact-demo`.
- If `npx.cmd tsc --noEmit` complains about missing `.next/types/...`, run `npm.cmd run build` first, then rerun typecheck.
- If another Next.js process for this repo is already running, stop it before starting `3009` to avoid shared `.next` state conflicts.
- If the page shows an old layout after restart, do one hard refresh in the browser.

## Standard Startup Sequence

From a clean terminal, use this order every time:

1. Start the backend:

```bash
cd backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8009
```

2. In a second terminal, start the frontend:

```bash
cd frontend
$env:NEXT_PUBLIC_API_BASE_URL='http://127.0.0.1:8009'
npm.cmd run build
npm.cmd run start -- --port 3009
```

3. Open the artifact page:

```text
http://127.0.0.1:3009/artifact-demo
```

4. Optional backend health check:

```text
http://127.0.0.1:8009/health
```

Current known-good local process ports:

- backend: `8009`
- frontend: `3009`

## Running Tests

From the repository root:

```bash
python -m pytest backend/tests
```

Targeted backend verification examples:

```bash
python -m pytest backend/tests/test_data_service.py
python -m pytest backend/tests/test_cf_service.py
python -m pytest backend/tests/test_agentic_service.py
python -m pytest backend/tests/test_formal_evaluation_service.py
```

## Typical Formal Experiment Workflow

This is the intended order for the formal comparison artifacts.

1. Build the joined processed interaction layer.
2. Build the leave-one-out evaluation base.
3. Build the 100-user experiment subset and candidate pools.
4. Build SVD Top-10 recommendations.
5. Build 3-agent Top-10 recommendations for the same users and pools.
6. Compute formal Top-10 metrics.

## Suggested Python Entry Snippets

The repository currently exposes most of the experiment logic through service methods. A typical script pattern is:

```python
from app.config import get_settings
from app.services.data_service import DataService
from app.services.cf_service import CollaborativeFilteringService
from app.services.agentic_service import AgenticRecommendationService

get_settings.cache_clear()
settings = get_settings()

data_service = DataService(settings)
cf_service = CollaborativeFilteringService(settings)
agentic_service = AgenticRecommendationService(settings)

data_service.build_processed_interactions_with_articles()
data_service.build_leave_one_out_evaluation_base()
data_service.build_svd_top10_debug_subset(sample_size=100, random_seed=42)
cf_service.build_svd_top10_debug_baseline()
agentic_service.build_top10_formal_experiment()
```

## Dependencies

Backend dependencies in `backend/requirements.txt` include:

- `fastapi`
- `uvicorn`
- `pandas`
- `numpy`
- `pydantic`
- `pydantic-settings`
- `httpx`
- `openai`
- `pytest`
- `scipy`
- `scikit-learn`

## Testing Coverage

The backend test suite includes coverage for:

- API behavior
- data preprocessing
- collaborative filtering
- agentic scoring
- experiment routing
- formal Top-10 evaluation metrics

Key test files:

- `backend/tests/test_data_service.py`
- `backend/tests/test_cf_service.py`
- `backend/tests/test_agentic_service.py`
- `backend/tests/test_experiment_service.py`
- `backend/tests/test_formal_evaluation_service.py`

## Known Boundaries

1. Raw dataset files are expected locally and are not committed.
2. The main demo path and the formal experiment path coexist, which means some outputs are intentionally duplicated.
3. The legacy collaborative filtering path is still callable and still writes its old artifacts.
4. The formal 100-user path is currently artifact-driven rather than exposed as a dedicated CLI command.
5. The agentic demo flow uses LLM-backed intent/explanation generation, while the constrained formal 100-user path is structured for reproducible offline evaluation.

## Recommended Next Steps

1. Add a dedicated CLI entry point for each formal pipeline stage.
2. Add a single orchestration command for the full formal comparison run.
3. Persist the final metric report for the formal 100-user SVD vs 3-agent comparison in a dedicated artifact path.
4. Document frontend pages against the current backend artifacts if the UI will be part of the final paper demo.

## Explainability Evidence Layer

The repository now includes a read-only explainability evidence layer for the saved Hybrid SVD + 3-Agent reranker artifacts.

Key additions:

- backend service: `backend/app/services/explainability_service.py`
- backend walkthrough adapter: `backend/app/services/artifact_demo_service.py`
- CLI runner: `python -m backend.scripts.run_explainability_audit --artifact-prefix seed99_robustness --sample-size 100 --output-prefix seed99`
- API endpoint: `GET /metrics/explainability`
- walkthrough API endpoint: `GET /demo/workflow-cases`
- primary frontend page: `/artifact-demo`
- frontend page: `/explainability-evidence`

The explainability evidence layer does not change the recommendation algorithms or formal ranking metrics. Instead, it audits whether the Hybrid SVD + 3-Agent reranker exposes source-grounded evidence from user history, item metadata, score components and reranking movement. This supports the interpretation of Hybrid as an explainability-oriented augmentation layer over SVD, while preserving the limitation that the study is offline and cannot establish live customer engagement or conversion gains.

The current final frontend artifact page now keeps the walkthrough focused on three method tabs only:

- `SVD` shows the behavioural baseline as Top-10 recommendation cards
- `3-Agent` shows preference, evidence, and decision workflow sections
- `Hybrid` shows reranking, score components, and read-only explainability audit sections

Global experiment summaries, old comparison panels, and old five-tab artifact structure are no longer the primary `/artifact-demo` presentation.

Explainability outputs are written to:

- `backend/app/data/processed/explainability/`

Explainability metrics include:

- `evidence_coverage_rate`
- `preference_trace_rate`
- `score_component_coverage_rate`
- `groundedness_rate`
- `rank_shift_coverage_rate`
- `ungrounded_claim_count`
- `average_rank_shift_for_ground_truth_hits`

### Explainability performance note

The explainability audit originally timed out at the full 1,000-user scale because it repeatedly filtered the full `processed_interactions_with_articles.csv` file inside the per-user loop. The current implementation avoids repeated full-dataframe scans by:

- selecting users from the saved Hybrid artifact first
- collecting only the needed `customer_id` and `article_id` values
- reading only required columns from the processed CSV
- filtering relevant rows in chunks
- precomputing metadata and user-history summaries once before explanation generation

This is a read-only explainability-path optimisation only. It does not change recommendation logic, ranking metrics, candidate pools, or saved formal experiment artifacts.

### Explainability audit results

Two explainability audit scales are now available for the `seed99_robustness` artifacts:

- 100-user audit:
  - output prefix: `seed99`
  - results path: `backend/app/data/processed/explainability/seed99_explainability_summary.json`
- full 1,000-user audit:
  - output prefix: `seed99_full_retry`
  - results path: `backend/app/data/processed/explainability/seed99_full_retry_explainability_summary.json`

Key full 1,000-user explainability metrics:

- `users_included = 1000`
- `recommendations_explained = 10000`
- `evidence_coverage_rate = 1.0000`
- `preference_trace_rate = 0.9954`
- `score_component_coverage_rate = 1.0000`
- `groundedness_rate = 1.0000`
- `rank_shift_coverage_rate = 0.6974`
- `ungrounded_claim_count = 0`
- `average_rank_shift_for_ground_truth_hits = 0.274725`

Current explainability warning:

- `prod_name` is not available in the processed source and is correctly reported as missing rather than fabricated.

Interpretation limits remain strict:

- this is not a live user study
- it does not prove CTR, CVR, customer engagement, conversion, add-to-cart, dwell time, or live feedback adaptation
- Hybrid improves explainability and remains competitive on ranking quality, but it reduces intra-list diversity compared with SVD

## Summary

This codebase now supports both:

- the original demo/debug recommendation workflow
- a structurally separate formal experiment workflow for SVD vs 3-agent evaluation

The important architectural decision is that the old user-based cosine collaborative filtering baseline remains in place only for legacy/debug use, while the formal comparison baseline is SVD Matrix Factorisation under a controlled leave-one-out, shared-candidate-pool, Top-10 evaluation design.

## Current Working State

Current branch during this update:

- `artifacts`

Latest artifact-demo implementation status:

- frontend route: `/artifact-demo`
- backend read-only walkthrough endpoint: `GET /demo/workflow-cases`
- selected users come from saved formal seed99 artifacts only
- the page uses a one-page dark card layout for the dissertation artefact
- the dropdown is backed by a client-side recovery fetch if the initial server render misses backend data
- no product image placeholders are used
- no Top-5 or Hit@5 wording is used

Latest verification commands that passed:

```bash
cd frontend
npx.cmd tsc --noEmit
npm.cmd run build
```

Backend tests were previously verified with:

```bash
.\backend\.venv\Scripts\python.exe -m pytest backend/tests -q -p no:cacheprovider
```
