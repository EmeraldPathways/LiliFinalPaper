# Hybrid Weight Ablation First 1000 Report

This note summarizes the completed Hybrid weight-sensitivity ablation run over the saved first 1,000-user frozen evaluation input.

## Run Identity

- Run prefix: `hybrid_weight_ablation_first1000_v1`
- Frozen input: `backend/app/data/processed/evaluation_base_table_svd_top10_1000.json`
- Frozen input SHA-256: `427DA0D55124107551602B12DAD82EE05B0EAE1E296EE9F612AEA8326080D08E`
- Users evaluated: `1000`
- Random state: `42`
- Diversity weight: fixed at `0.05`
- OpenAI used: `False`

## Results

| Configuration | SVD | Agentic | Diversity | HR@10 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `w1_svd090_agent005_div005` | 0.90 | 0.05 | 0.05 | 0.504000 | 0.298478 |
| `w2_svd080_agent015_div005` | 0.80 | 0.15 | 0.05 | 0.507000 | 0.303812 |
| `w3_svd070_agent025_div005` | 0.70 | 0.25 | 0.05 | 0.498000 | 0.304552 |
| `w4_svd060_agent035_div005` | 0.60 | 0.35 | 0.05 | 0.473000 | 0.296664 |
| `w5_svd050_agent045_div005` | 0.50 | 0.45 | 0.05 | 0.434000 | 0.282121 |

## Reproduction Check

The original weight setting was:

- `svd_weight = 0.70`
- `agentic_weight = 0.25`
- `diversity_weight = 0.05`

The ablation reproduced the saved original Hybrid metrics exactly for `w3_svd070_agent025_div005`:

- `HR@10 = 0.498000`
- `NDCG@10 = 0.304552`

## Controlled Variables

The run manifest confirms:

- same users
- same user order
- same training histories
- same ground truths
- same candidate pools
- same candidate order
- same SVD configuration
- same deterministic agentic logic
- same diversity calculation
- same top-k

## Output Files

A full list of ablation artefacts can be found at:

- `HYBRID_WEIGHT_ABLATION_FIRST1000_FILES.md`
