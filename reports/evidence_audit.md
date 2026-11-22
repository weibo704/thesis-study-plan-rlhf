# Evidence audit

`python -m pipeline.evidence_audit` · every headline number in the draft, recomputed from its source and compared to what the draft says.

**45 of 45 claims verified.**

## Quantities two reports disagree on

Both files are correct for what they computed. The risk is a reader opening both and finding a contradiction nobody flagged, so each is reconciled here and the thesis cites one consistently.

| quantity | statistical_analysis.md | experiment report | why they differ | thesis uses |
|---|---|---|---|---|
| learning win rate | 95% | 80% | paired per learner (n=500) vs per learner-seed (n=2,500); means agree, dispersion does not | 80% (per learner-seed) |
| planning win rate | 60% | 53% | paired per learner (n=500) vs per learner-seed (n=2,500); means agree, dispersion does not | 53% (per learner-seed) |

## Verified claims and the file that supports each

| § | claim | source |
|---|---|---|
| 3.1/3.2 | `1,899` | `data/agent/arm_features.parquet` |
| 3.3 | `188` | `data/catalog/skills.parquet` |
| 3.3 | `254` | `data/world/skill_prereqs.parquet` |
| 3.3 | `122` | `data/world/study_plans.parquet` |
| 3.4 | `500` | `data/world/users.parquet` |
| 4.4 | `0.5985` | `data/agent/bestofn_selection_poc-v6-stratified.parquet` |
| 4.4 | `0.4045` | `data/agent/bestofn_selection_poc-v6-stratified.parquet` |
| 4.4 | `0.7121` | `data/agent/bestofn_selection_poc-v6-stratified.parquet` |
| 4.4 | `0.6819` | `data/agent/bestofn_selection_poc-v6-stratified.parquet` |
| 4.4 | `0.8438` | `data/agent/bestofn_selection_poc-v6-stratified.parquet` |
| 4.4 | `+0.1136` | `reports/bestofn_poc-v6-stratified.md` |
| 4.4 | `±0.0745` | `reports/bestofn_poc-v6-stratified.md` |
| 4.4 | `+0.3075` | `reports/bestofn_poc-v6-stratified.md` |
| 4.4 | `±0.0995` | `reports/bestofn_poc-v6-stratified.md` |
| 4.4 | `-0.1940` | `reports/bestofn_poc-v6-stratified.md` |
| 4.4 | `±0.1001` | `reports/bestofn_poc-v6-stratified.md` |
| 4.4 | `+0.1317` | `reports/bestofn_poc-v6-stratified.md` |
| 4.4 | `±0.0465` | `reports/bestofn_poc-v6-stratified.md` |
| 4.4 | `24%` | `data/agent/bestofn_selection parquet` |
| 4.4 | `67%` | `data/agent/bestofn_selection parquet` |
| 4.4 | `+0.1489` | `data/agent/bestofn_selection parquet` |
| 4.5 | `+0.313` | `data/agent/reward_model_poc-v6-stratified.json` |
| 4.5 | `-0.838` | `data/agent/reward_model_poc-v6-stratified.json` |
| 4.5 | `-0.184` | `data/agent/reward_model_poc-v6-stratified.json` |
| 4.5 | `+0.366` | `data/agent/reward_model_poc-v6-stratified.json` |
| 4.5 | `+0.182` | `data/agent/reward_model_poc-v6-stratified.json` |
| 4.5 | `+0.368` | `data/agent/reward_model_poc-v6-stratified.json` |
| 4.5 | `+0.000` | `data/agent/reward_model_poc-v6-stratified.json` |
| 4.5 | `4.6` | `ratio of the two difficulty weights` |
| 4.3 | `0.602` | `data/agent/reward_model_poc-v6-stratified.json` |
| 3.8 | `64%` | `data/agent/reward_model_poc-v6-stratified.json` |
| 4.7 | `+0.2042` | `reports/experiment_poc-v6-stratified.md` |
| 4.7 | `±0.0130` | `reports/experiment_poc-v6-stratified.md` |
| 4.7 | `80%` | `reports/experiment_poc-v6-stratified.md` |
| 4.7 | `+0.0640` | `reports/experiment_poc-v6-stratified.md` |
| 4.7 | `±0.0206` | `reports/experiment_poc-v6-stratified.md` |
| 4.7 | `53%` | `reports/experiment_poc-v6-stratified.md` |
| 4.2 | `+1.9828` | `reports/stage1_evidence.md` |
| 4.2 | `+0.3707` | `reports/stage1_evidence.md` |
| 4.2 | `+0.2872` | `reports/stage1_evidence.md` |
| 4.8 | `+0.0549` | `reports/ablation_randomized_graph.md` |
| 4.8 | `+0.0552` | `reports/ablation_randomized_graph.md` |
| 4.8 | `+0.0347` | `reports/ablation_randomized_graph.md` |
| 4.2 | `0.255` | `reports/stage1_evidence.md` |
| 4.2 | `0.035` | `reports/stage1_evidence.md` |
