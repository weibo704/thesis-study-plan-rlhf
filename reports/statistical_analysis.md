# Consolidated statistical analysis — `poc-v6-stratified`

`python -m pipeline.statistical_analysis` · computed from the parquet under `data/agent/` · no API calls.

Every contrast is paired within learner. Cohen's *d* is the paired form — mean difference over the standard deviation of the differences — because the pairing is what removes between-learner variance. A 95% interval spanning zero is *not established* regardless of sign.

**Why effect size is reported alongside the interval.** With 2,500 learner-runs almost any non-zero effect clears significance. The interval says an effect is real; *d* says whether it matters.

## 1. Best-of-N selection (the headline experiment)

80 held-out learners, six candidate plans each, identical candidates across policies.

### Mean return by policy

| policy | mean return | mean goal progress |
|---|---|---|
| the LLM's first plan | 0.5985 | 0.5250 |
| a random one of its six | 0.4045 | 0.4375 |
| **chosen by the feedback model** | 0.7121 | 0.5500 |
| chosen by an oracle-regressed model | 0.6819 | 0.5500 |
| the best of the six (ceiling) | 0.8438 | 0.5250 |

### Paired contrasts on return

| contrast | n | difference | 95% CI | Cohen's d | win rate | ties | verdict |
|---|---|---|---|---|---|---|---|
| feedback RL over raw LLM | 80 | +0.1136 | ±0.0745 | +0.334 (small) | 51% | 19 | **established** |
| scoring over sampling | 80 | +0.3075 | ±0.0995 | +0.677 (medium) | 70% | 15 | **established** |
| more candidates alone | 80 | -0.1940 | ±0.1001 | -0.425 (small) | 25% | 15 | **established** |
| feedback vs oracle-regressed | 80 | +0.0301 | ±0.0321 | +0.206 (small) | 21% | 55 | not established |
| headroom remaining | 80 | +0.1317 | ±0.0465 | +0.621 (medium) | 48% | 42 | **established** |

### The headline contrast, decomposed

A selector that picks the plan the language model already placed first produces a difference of exactly zero, counted as a non-win. Those ties are the model *agreeing*, not failing, and they depress the headline win rate.

| quantity | value |
|---|---|
| learners | 80 |
| ties (model picked the LLM's own first plan) | 19 (24%) |
| win rate, all learners | 51% |
| **win rate where it overrode the LLM** | **67%** |
| mean gain when it overrode | +0.1489 |
| mean when it won | +0.3459 |
| mean when it lost | -0.2549 |

## 2. Baseline policy family

500 learners × 5 seeds ([7, 11, 23, 37, 53]) = 2,500 learner-runs per policy.

### Paired contrasts on total reward

> **These differ from `reports/experiment_poc-v6-stratified.md`, and deliberately.** That report pairs at the (learner, seed) level, giving n = 2,500 and treating the same learner under a different seed as an independent observation. Here each learner's reward is averaged over seeds first, giving n = 500 — one row per learner, which is what a learner actually is. Averaging removes seed noise, so the intervals widen (fewer rows) while the win rates rise (less noise to flip a sign). The means are identical either way. Cite whichever you prefer, but cite one of them consistently; the thesis currently quotes the per-seed figures.

| contrast | n | difference | 95% CI | Cohen's d | win rate | ties | verdict |
|---|---|---|---|---|---|---|---|
| learning (bandit over fixed scorer) | 500 | +0.2042 | ±0.0192 | +0.932 (large) | 95% | — | **established** |
| planning (lookahead over bandit) | 500 | +0.0640 | ±0.0247 | +0.227 (small) | 60% | — | **established** |
| learning + planning | 500 | +0.2683 | ±0.0232 | +1.015 (large) | 86% | — | **established** |

### The same contrasts on goal progress

Reported because it is a metric the policies do **not** optimise, which is what makes it informative: every policy that earns more reward reaches less of the learner's goal.

| contrast | n | difference | 95% CI | Cohen's d | win rate | ties | verdict |
|---|---|---|---|---|---|---|---|
| learning (bandit over fixed scorer) | 500 | -0.0364 | ±0.0089 | -0.359 (small) | 2% | 407 | **established** |
| planning (lookahead over bandit) | 500 | -0.0484 | ±0.0225 | -0.188 (negligible) | 22% | 225 | **established** |
| learning + planning | 500 | -0.0848 | ±0.0222 | -0.335 (small) | 16% | 226 | **established** |

### Per-seed means, to show the result is not one lucky draw

| seed | baseline | linucb | planner |
|---|---|---|---|
| 7 | 0.1038 | 0.3037 | 0.3721 |
| 11 | 0.1038 | 0.3136 | 0.3759 |
| 23 | 0.0798 | 0.2961 | 0.3638 |
| 37 | 0.0824 | 0.2864 | 0.3450 |
| 53 | 0.1038 | 0.2948 | 0.3581 |

## 3. How to read a 'not established' row

It means the interval spans zero, so the sign is not reliable at this sample size. It does **not** mean the effect is zero. §4.6 of the thesis shows the converse case: several selector variants differ by amounts the experiment cannot resolve, and the honest conclusion there is a ceiling on the evaluation rather than on the method.

## 4. Contrasts computed elsewhere

| result | where |
|---|---|
| retrieval / generation / ordering decomposition | `reports/stage1_evidence.md` |
| selector ceiling, label-noise and feature ablations | `reports/stage1_diagnosis.md` |
| randomized-graph ablation (runs in a scratch root) | `reports/ablation_randomized_graph.md` |
| real collected preferences, once any exist | `reports/real_feedback.md` |
