# Reproducing the Seed99 Explainability Audit

## Important Note

The repository does expose the Seed99 explainability audit as a dedicated script entrypoint. The command block below reproduces the saved `seed99_full_retry_*` explainability artefacts from the current codebase using the formal `seed99_robustness` experiment outputs as input.

## Exact Seed99 Explainability Audit Run Command

Run this from the repository root after the Seed99 formal experiment artefacts already exist:

```powershell
.\backend\.venv\Scripts\python.exe -m backend.scripts.run_explainability_audit --artifact-prefix seed99_robustness --sample-size 1000 --output-prefix seed99_full_retry
```

## What This Command Produces

The script writes explainability artefacts to `backend/app/data/processed/explainability/`. Expected key outputs include:

- `seed99_full_retry_explainability_summary.json`
- `seed99_full_retry_explainability_audit.json`
- `seed99_full_retry_explainability_examples.csv`
- `seed99_full_retry_rank_shift_analysis.csv`
- `seed99_full_retry_case_studies.md`

A full list of Seed99 artefacts can be found at:

`SECOND_1000_USER_FILES.md`

## Explainability Audit Procedure

The explainability audit logic is implemented in `backend/app/services/explainability_service.py`. For the Seed99 explainability replay, the procedure is:

1. Read the saved `seed99_robustness` evaluation, SVD, and Hybrid recommendation artefacts.
2. Select the first 1,000 Hybrid user rows when `--sample-size 1000` is used.
3. Load the required rows from `processed_interactions_with_articles.csv` for those selected users and the referenced article IDs.
4. Build user-history summaries and item metadata lookups from the processed interactions source.
5. Generate per-recommendation explanation records containing source-grounded metadata, score components, and rank-shift evidence.
6. Write the explainability summary, audit JSON, examples CSV, rank-shift analysis CSV, and case studies markdown outputs.

This Seed99 explainability replay uses:

- `artifact_prefix=seed99_robustness`
- `sample_size=1000`
- `output_prefix=seed99_full_retry`

The resulting summary report is saved as:

`backend/app/data/processed/explainability/seed99_full_retry_explainability_summary.json`
