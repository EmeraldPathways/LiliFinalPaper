# H&M Hybrid SVD + 3-Agent Recommendation Artefact

Offline dissertation artefact comparing SVD Matrix Factorisation, a structured 3-Agent recommendation workflow, and a Hybrid SVD + 3-Agent reranker on saved H&M experiment artefacts.

## Stack

- Backend: FastAPI + pandas
- Frontend: Next.js App Router + TypeScript
- Storage: local CSV/JSON artifacts
- LLM integration: optional for legacy/demo routes; /artifact-demo is read-only and uses saved offline experiment and explainability artefacts.

## Project Layout

- `backend/` FastAPI API, experiment pipeline, tests, and local data folders
- `frontend/` Next.js dashboard including the /artifact-demo dissertation artefact page

## Artifact Demo Provenance

`/artifact-demo` is a runnable offline dissertation artefact demo with three method views: SVD, 3-Agent, and Hybrid. It is pinned to selected saved formal experiment artefacts for reproducible demonstration, so it does not rerun experiments or generate new live recommendations. The route reads saved offline outputs for evaluation, SVD, Standalone 3-Agent, Hybrid, and explainability. H&M fields such as `product_type_name`, `product_group_name`, `colour_group_name`, and `graphical_appearance_name` are real article metadata fields, while recommendation scores, match scores, match counts, rank shifts, and explanation text are project-generated saved artefact outputs or derived presentation fields rather than raw H&M dataset columns. Static UI labels and fallback text are presentation-layer copy, not additional experiment evidence. The artefact does not claim CTR, CVR, conversion, live customer engagement, add-to-cart, dwell time, or live feedback adaptation.

## Dataset Placement

Drop these files into `backend/app/data/raw/` before running the experiment:

- `transactions_train.csv`
- `articles.csv`
- `customers.csv`

The raw dataset is not committed to git.

## Backend Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Frontend Setup

```bash
cd frontend
npm.cmd install
npm.cmd run dev
```

## Environment

Backend expects:

- `OPENAI_API_KEY`
- Optional `OPENAI_MODEL` defaulting to `gpt-4.1-mini`
- Optional `FRONTEND_ORIGIN` defaulting to `http://localhost:3000`

## Main Endpoints

- `POST /experiment/run`
- `GET /experiment/setup`
- `GET /users/{user_id}/intent`
- `GET /recommendations/compare/{user_id}`
- `GET /demo/workflow-cases`
- `GET /metrics`
- `POST /users/{user_id}/feedback`

## Artifact Demo Flow

1. Ensure saved processed artefacts are present under `backend/app/data/processed/`
2. Start the FastAPI backend
3. Start the Next.js frontend
4. Open `/artifact-demo`
5. Review the SVD, 3-Agent, and Hybrid method views

To rerun the full experiment pipeline, place the raw H&M CSV files in `backend/app/data/raw/` and use `POST /experiment/run`. This is not required for viewing `/artifact-demo`.
