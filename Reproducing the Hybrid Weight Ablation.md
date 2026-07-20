**Reproducing the Hybrid Weight Ablation First 1000-User Experiment**

**Important Note**
**The repository now exposes the Hybrid weight-sensitivity ablation as a dedicated runner script. This command reuses the saved unprefixed first 1000-user frozen evaluation input from the current codebase and writes a new isolated `hybrid_weight_ablation_first1000_v1_` output set.**

**Exact Hybrid Weight Ablation Run Command**
**Run this from the repository root:**
**cd backend**
**.\\.venv\\Scripts\\python.exe run_hybrid_weight_ablation.py**


**What This Command Uses**
**The runner reads the saved frozen first 1000-user evaluation input from:**
**backend/app/data/processed/evaluation_base_table_svd_top10_1000.json**

**The runner keeps these settings fixed:**
**users = same frozen 1000 users**
**random_state = 42**
**diversity_weight = 0.05**
**same histories, ground truths, candidate pools and candidate order**
**same deterministic agentic scoring logic**
**same SVD configuration**

**The runner evaluates exactly these five Hybrid weight configurations:**
**w1_svd090_agent005_div005**
**w2_svd080_agent015_div005**
**w3_svd070_agent025_div005**
**w4_svd060_agent035_div005**
**w5_svd050_agent045_div005**


**What This Command Produces**
**Formal ablation run writes saved artefacts to the backend/app/data/processed/. Expected key outputs include:**
**hybrid_weight_ablation_first1000_v1_manifest.json**
**hybrid_weight_ablation_first1000_v1_metric_summary.json**
**hybrid_weight_ablation_first1000_v1_metric_summary.csv**
**hybrid_weight_ablation_first1000_v1_w1_svd090_agent005_div005_hybrid_svd_agentic_recommendations_top10_1000.json**
**hybrid_weight_ablation_first1000_v1_w2_svd080_agent015_div005_hybrid_svd_agentic_recommendations_top10_1000.json**
**hybrid_weight_ablation_first1000_v1_w3_svd070_agent025_div005_hybrid_svd_agentic_recommendations_top10_1000.json**
**hybrid_weight_ablation_first1000_v1_w4_svd060_agent035_div005_hybrid_svd_agentic_recommendations_top10_1000.json**
**hybrid_weight_ablation_first1000_v1_w5_svd050_agent045_div005_hybrid_svd_agentic_recommendations_top10_1000.json**

**A full list of Hybrid weight ablation artefacts can be found at:**
**HYBRID_WEIGHT_ABLATION_FIRST1000_FILES.md**
