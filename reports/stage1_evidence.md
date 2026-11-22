# Stage-1 evidence — `poc-v6-stratified`

Regenerate with `python -m pipeline.stage1_evidence`. Plans replay from the LLM cache: **0 API calls**.
Scored on **200** learners, all five variants on the same learners with the same frozen reward.

## Where the return actually comes from

| variant | mean return |
|---|---|
| best of 40 random 5-course plans, whole catalogue (1,899) | -1.1869 |
| best of 40 random 5-course plans, 25-course shortlist | +0.7959 |
| best of 6 random 5-course plans, 25-course shortlist | +0.4365 |
| the LLM's first plan | +0.5200 |
| best of the LLM's 6 (oracle pick) | +0.8072 |

## Paired contrasts

| stage | what it compares | difference | 95% CI | win rate |
|---|---|---|---|---|
| retrieval | shortlist vs whole catalogue, 40 draws each | +1.9828 | ±0.0660 | 100% |
| generation | the LLM's 6 vs 6 random draws from its own shortlist | +0.3707 | ±0.0605 | 84% |
| ordering | best of the 6 vs the one the LLM put first | +0.2872 | ±0.0435 | 74% |

The 40-draw row is kept because it is the control a sceptic asks for, but it is not a fair test of the generator: a maximum over 40 draws beats a maximum over 6 of anything. The generation row is the like-for-like comparison.

The ordering row is the headroom the reward model is trying to capture. It is the reason a selection step exists at all.

## Are the plans actually different?

Jaccard overlap on course sets. Cosine on the feature vectors reads ~0.99 and means nothing: those vectors are mostly positive and share a constant bias term, so near-identical cosine is guaranteed by construction.

| comparison | mean Jaccard | |
|---|---|---|
| within one learner, across their 6 plans (3,000 pairs) | 0.255 | 1.4% identical, 90.5% share under half |
| across two learners, best-matching plan (2,991 pairs) | 0.035 | 85.6% share no course at all |

Two learners with different goals get genuinely different plans. Within one learner the six candidates overlap enough to be plausible alternatives and differ enough for the choice between them to matter.
