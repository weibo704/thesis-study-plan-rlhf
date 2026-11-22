# Content-based baseline

- policy: `content_baseline_v1` (fixed weights, no learning)
- users scored: **500**
- recommendations: **5,000** (top-10 each)
- distinct arms used at rank 1: **118**

## Weights (the baseline's own design, NOT the reward)

| term | weight |
|---|---|
| topic match | 1.0 |
| needed-skill overlap | 0.6 |
| level proximity | 0.25 |
| time fit | 0.15 |

## What the policy can and cannot see

Observable: goal labels, start level, weekly hours, self-reported
mastery — all things a user states or a real system could ask.

Hidden: `skill_prereqs`, study plans, `reward_spec`, and the latent
user fields (readiness, persistence, novelty_pref, price_sensitivity,
prereq_gap).

**Prerequisite coverage is deliberately NOT a feature.** The reward
credits prerequisite satisfaction, so a policy that could see
coverage would win by construction. It has to stay latent.

## Rank-1 family spread

| family | times at rank 1 |
|---|---|
| `programming-foundations` | 118 |
| `deep-learning` | 52 |
| `mlops-deployment` | 47 |
| `data-analysis-viz` | 45 |
| `mobile-development` | 33 |
| `software-engineering` | 31 |
| `security` | 26 |
| `classical-ml` | 25 |
| `web-frontend` | 24 |
| `math-for-ml` | 23 |
| `computer-vision` | 21 |
| `nlp-and-llm` | 19 |
| `web-backend-apis` | 17 |
| `databases` | 10 |
| `computer-systems` | 9 |
