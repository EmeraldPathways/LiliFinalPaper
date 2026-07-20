# 1 Introduction

This configuration manual provides the technical information required to configure, run and inspect the H&M Hybrid SVD + 3-Agent Recommendation Artefact. Its purpose is to support reproducibility by explaining the required software, project structure, dataset preparation, backend and frontend configuration, experiment execution, validation procedures and artefact demonstration process.

# 2 Project Overview

This project develops an offline recommendation artefact using the H&M Personalized Fashion Recommendations dataset. It compares three methods: SVD Matrix Factorisation, a structured Standalone 3-Agent workflow, and a Hybrid SVD + 3-Agent reranker, generating Top-10 recommendations.

The Hybrid method combines SVD behavioural scores, agentic evidence scores and a small diversity adjustment. The backend is implemented with Python and FastAPI, while the frontend uses Next.js, React and TypeScript. The artefact presents saved experiment and explainability outputs through a read-only demonstration interface. It is designed for offline evaluation and reproducibility rather than real-time recommendation or live customer-engagement measurement.

# 3 System Requirements

## Hardware Specifications

This project was developed and tested on the following hardware:

| Component | Specification |
| --- | --- |
| Processor | Intel Core i7-10700KF CPU @ 3.80 GHz |
| Installed RAM | 64.0 GB (63.9 GB usable) |
| Storage | SSD 850 EVO 500 GB, 1.82 TB HDD SAMSUNG HD204UI |
| Graphics Card | NVIDIA GeForce RTX 3060 (12 GB) |
| System Type | 64-bit operating system, x64-based processor |

Note: these hardware values are reproduced from the supplied document. They are machine-specific rather than repo-defined.

## Software Specifications

The following tools and libraries were used in this project for implementation, testing and analysis:

1. Programming Language: Python 3.13.2
2. Development Environments: Visual Studio Code (VSC)
3. Version Control: GitHub

### Libraries and Frameworks

- fastapi
- uvicorn
- pandas
- numpy
- pytest
- pydantic
- pydantic-settings
- httpx
- openai
- scipy
- scikit-learn

## Objectives

- Demonstrate a recommendation workflow through a FastAPI backend and a frontend dashboard.
- Build repeatable offline experiment artefacts from raw H&M data.
- Compare a formal collaborative filtering baseline against the existing structured 3-Agent workflow, which is a deterministic, role-specialised software architecture. Preference extraction, candidate metadata matching, scoring, ranking and recommendation reasons are produced through fixed Python rules and source-grounded templates. The formal SVD, Standalone 3-Agent, Hybrid and explainability outputs do not require or call the OpenAI API.

## Project Layout

The repository is split into two top-level directories:

- `backend/` - FastAPI API, experiment pipeline, tests, and local data folders
- `frontend/` - Next.js dashboard including the `/artifact-demo` dissertation artefact page

## Technology Stack

- Backend: FastAPI + pandas
- Frontend: Next.js App Router + TypeScript
- Storage: local CSV / JSON artefacts

## Repository Root

In the commands below, `<PROJECT_ROOT>` refers to the main project folder containing the `backend` and `frontend` directories.

## Runtime Versions

| Runtime / Package | Version |
| --- | --- |
| Python | 3.13.2 |
| Node.js | 24.14.1 |
| npm | 11.11.0 |
| Next.js | 15.5.18 |
| React | 19.0.0 |
| React DOM | 19.0.0 |
| TypeScript | 5.8.3 |
| fastapi | 0.115.12 |
| uvicorn[standard] | 0.34.2 |
| pandas | 2.2.3 |
| numpy | 2.2.5 |
| openai | 1.78.1 |
| pytest | 8.3.5 |
| scipy | 1.17.1 |
| scikit-learn | 1.9.0 |

## Dataset Source

The project is based on the H&M Personalized Fashion Recommendations dataset. The expected download source is the Kaggle H&M competition page:

`https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations`

After downloading and extracting the dataset, place the raw CSV files inside:

`backend/app/data/raw/`

## Required Raw Files

- `backend/app/data/raw/transactions_train.csv`
- `backend/app/data/raw/articles.csv`
- `backend/app/data/raw/customers.csv`

# 4 Run the Project

## Backend Setup

From the repository root:

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you prefer not to activate the environment, use the explicit interpreter path in all backend commands:

```powershell
.\backend\.venv\Scripts\python.exe
```

## Frontend Setup

From the repository root:

```powershell
cd frontend
npm.cmd install
```

## Environment Configuration

The backend settings load from a `.env` file. Create `backend/.env` using this template:

```env
FRONTEND_ORIGIN=http://localhost:3009,http://127.0.0.1:3009

# Optional overrides
# OPENAI_API_KEY=sk-REDACTED
# OPENAI_MODEL=gpt-4.1-mini
# BACKEND_DATA_DIR=D:\path\to\alternate\data\folder
# SAMPLE_SIZE=20000
# TOP_N=10
# CANDIDATE_POOL_SIZE=100
# MIN_USER_INTERACTIONS=3
# MIN_PRODUCT_INTERACTIONS=2
# MAX_EVAL_USERS=50
# LLM_TIMEOUT_SECONDS=40
```

Do not place the real API key in the dissertation document. Use a placeholder such as `OPENAI_API_KEY=sk-REDACTED`.

## Reproducing the Seed99 Formal Experiment

### Important Note

The repository does not expose the full three-method Seed99 pipeline as a single packaged CLI command. The exact reproducible run is a sequence of service calls. The command block below reproduces the saved `seed99_robustness_*` formal experiment artefacts from the current codebase.

### Exact Seed99 Run Command

Run this from the repository root:

```powershell
@'
from pathlib import Path
import sys

backend_root = Path("backend").resolve()
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from app.config import get_settings
from app.services.data_service import DataService
from app.services.cf_service import CollaborativeFilteringService
from app.services.agentic_service import AgenticRecommendationService
from app.services.hybrid_service import HybridRecommendationService
from app.services.evaluation_service import EvaluationService

settings = get_settings()
data_service = DataService(settings)
cf_service = CollaborativeFilteringService(settings)
agentic_service = AgenticRecommendationService(settings)
hybrid_service = HybridRecommendationService(settings, cf_service, agentic_service)
evaluation_service = EvaluationService(settings)

data_service.build_processed_interactions_with_articles()
data_service.build_leave_one_out_evaluation_base()
data_service.build_svd_top10_subset(
    sample_size=1000,
    random_seed=99,
    candidate_pool_size=100,
    artifact_prefix="seed99_robustness",
)
cf_service.build_svd_top10_baseline(
    subset_size=1000,
    random_state=99,
    artifact_prefix="seed99_robustness",
)
agentic_service._build_top10_formal_experiment(
    subset_size=1000,
    artifact_prefix="seed99_robustness",
)
hybrid_service.build_hybrid_svd_agentic_reranker(
    subset_size=1000,
    random_state=99,
    artifact_prefix="seed99_robustness",
)
evaluation_service.compute_three_method_top10_metrics_from_saved_artifacts(
    subset_size=1000,
    bootstrap_samples=1000,
    random_seed=42,
    artifact_prefix="seed99_robustness",
)
'@ | .\backend\.venv\Scripts\python.exe -
```

### What This Command Produces

Formal run writes saved artefacts to `backend/app/data/processed/`. Expected key outputs include:

- `seed99_robustness_evaluation_base_table_top10_1000.json`
- `seed99_robustness_svd_recommendations_top10_1000.json`
- `seed99_robustness_agentic_recommendations_top10_1000.json`
- `seed99_robustness_hybrid_svd_agentic_recommendations_top10_1000.json`
- `seed99_robustness_per_user_metrics_top10_1000_three_methods.json`
- `seed99_robustness_metric_summary_top10_1000_three_methods.json`
- `seed99_robustness_bootstrap_ci_report_top10_1000_three_methods.json`
- `seed99_robustness_validation_report_top10_1000_three_methods.json`
- `seed99_robustness_experiment_report_top10_1000.md`

### Bootstrap Confidence Interval Procedure

The bootstrap confidence interval logic is in `backend/app/services/evaluation_service.py`. For the three-method Seed99 comparison, the procedure is:

1. Read the saved per-user SVD, standalone 3-Agent, and Hybrid recommendation results.
2. Compute per-user values for: HitRate@10, NDCG@10, ILD@10.
3. Draw 1,000 bootstrap resamples with replacement from the full set of per-user rows.
4. For each resample, compute method-level mean metrics for SVD, standalone 3-Agent, and Hybrid; and pairwise differences (Hybrid minus SVD; Hybrid minus standalone 3-Agent).
5. Estimate the 95% confidence interval using the empirical 2.5th and 97.5th percentiles.

The Seed99 run uses `bootstrap_samples=1000` and `random_seed=42`. The resulting report is saved as:

`backend/app/data/processed/seed99_robustness_bootstrap_ci_report_top10_1000_three_methods.json`

## Explainability Replay Artefacts for `/artifact-demo`

After the Seed99 formal experiment is complete, run:

```powershell
.\backend\.venv\Scripts\python.exe -m backend.scripts.run_explainability_audit --artifact-prefix seed99_robustness --sample-size 1000 --output-prefix seed99_full_retry
```

This writes explainability artefacts to `backend/app/data/processed/explainability/`. Key outputs include:

- `seed99_full_retry_explainability_summary.json`
- `seed99_full_retry_explainability_audit.json`
- `seed99_full_retry_explainability_examples.csv`
- `seed99_full_retry_rank_shift_analysis.csv`
- `seed99_full_retry_case_studies.md`

## Backend Launch Command

```powershell
cd backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8009
```

Repository-root helper: `start-backend.cmd`

Health check: `http://127.0.0.1:8009/health`

## Frontend Launch Commands

### Development Mode

```powershell
cd frontend
npm.cmd run dev -- --hostname 127.0.0.1 --port 3009
```

### Production-Style Local Run (Artefact)

```powershell
cd frontend
npm.cmd run build
npm.cmd run start -- --hostname 127.0.0.1 --port 3009
```

Repository-root helper: `start-frontend.cmd`

## How to Access the Demo

Once the backend and frontend are both running, open:

`http://127.0.0.1:3009/artifact-demo`

The backend walkthrough route that feeds this page is:

`GET /demo/workflow-cases?artifact_prefix=seed99_robustness&explainability_prefix=seed99_full_retry`

## Read-Only Replay Confirmation

The `/artifact-demo` page is a read-only replay of saved offline outputs. It:

- Reads saved experiment artefacts from `backend/app/data/processed/`
- Reads saved explainability artefacts from `backend/app/data/processed/explainability/`
- Renders a deterministic walkthrough for selected saved users

The `/artifact-demo` page does not:

- Rerun the formal experiment when the page loads
- Regenerate recommendations live
- Retrain SVD
- Regenerate the Hybrid ranking live
- Run a live user study

## Pytest Commands

Full backend test suite from the repository root:

```powershell
python -m pytest backend/tests
```

Windows-safe low-cache variant used when cache locking causes issues:

```powershell
.\backend\.venv\Scripts\python.exe -m pytest backend/tests -q -p no:cacheprovider
```

Targeted tests:

```powershell
python -m pytest backend/tests/test_data_service.py
python -m pytest backend/tests/test_cf_service.py
python -m pytest backend/tests/test_agentic_service.py
python -m pytest backend/tests/test_formal_evaluation_service.py
```

## Saved Artefact Locations

| Location | Path |
| --- | --- |
| Raw Data | `backend/app/data/raw/` |
| Processed Experiment Outputs | `backend/app/data/processed/` |
| Explainability Outputs | `backend/app/data/processed/explainability/` |

# References

References should be formatted using APA or Harvard style as detailed in the NCI Library Referencing Guide available at `https://libguides.ncirl.ie/referencing`. You can use a reference management system such as Zotero or Mendeley to cite in MS Word.

