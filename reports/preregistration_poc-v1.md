# Pre-registration

Bundle **`poc-v1`** — `12217afa0ff7a081`  
Frozen 2026-10-01T00:37:29+00:00

Frozen BEFORE any bandit run. The reward is computed over a specific
taxonomy and prerequisite graph, so those are hashed with it: changing
either afterwards would change what the reward means while leaving the
reward's own hash untouched.

## Reward weights

| weight | value | why |
|---|---|---|
| `w_topic` | 1.0 | The learner's stated goal is the primary signal. Scaled so topic contributes ~29% of total reward spread across the pool. |
| `w_too_hard` | 0.45 | A one-level overshoot costs 0.45 — as much as the ENTIRE topic range. Being pushed past your depth should outweigh a better topic match, which is the sequencing behaviour under study. |
| `w_too_easy` | 0.15 | Boredom is real but milder than frustration. Set to one third of w_too_hard (3:1), keeping the term asymmetric; a symmetric penalty would make the optimal policy level-neutral and erase sequencing. |
| `w_time` | 0.15 | Over-budget is penalised but capped at 2x the learner's budget. At the earlier 3x clip time became the single largest term, ahead of both topic and difficulty. |
| `w_quality` | 0.1 | Ratings are a selection-biased popularity signal, not per-user utility, so the contribution is clipped to [-0.15, +0.10]. Uncapped, 1-star outliers spanned as much reward as the whole topic range. |
| `w_unknown_penalty` | 0.05 | A small cost per missing field so a course with no level and no duration cannot win on silence. Kept to ~6% of spread. |
| `w_goal` | 0.0 | Overlap with skills the learner still LACKS from their stated goal. Absent in poc-v1, which scored only present-tense fit. The bandit learned +0.006 on that feature -- correct, since nothing credited it -- and so ignored goal progress, ending at 0.25 against the baseline's 0.45 while earning MORE reward. Added because the reward omitted the stated objective; this is a specification gap, not weight-tuning to favour a policy. |

Asymmetry: too-hard / too-easy = **3.0:1**

Term shares of total reward spread over the real arm pool: level 29%,
topic 29%, time 19%, quality 16%, missing-metadata 7%.

## Provenance

**Author-specified, not feedback-grounded.** Zero thumbs-down
judgments were collected, so the weights come from term-balance
analysis over the actual arm pool, not from observed complaint
counts. That is a weaker derivation and should be stated as such.

## Hashed dependencies

| artefact | sha256 |
|---|---|
| `skills` | `faad020a17bdbbd6` |
| `skill_prereqs` | `8dc424963a821b25` |
| `study_plans` | `2416048f2df77d8b` |
| `study_plan_items` | `f1fa4d6d2e6b416a` |
| `users` | `22f946af3bdc1424` |
| `user_latent_state` | `68992bb0a42b6faf` |
| `arm_features` | `1ca5b2c5c5eb665b` |
| `arm_feature_spec` | `3e9acd4bec927761` |

If any of these change, this bundle no longer describes the
experiment and a new one must be registered under a new name.
