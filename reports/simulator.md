# Synthetic learner population

> **world/ oracle.** This is the ground truth the reward is computed
> against. It must never be read when building `agent/` features.

- users: **500**
- latent mastery rows: **5,724**
- mean mastered skills per user: **2.8**

## Level mix (stratified, NOT catalog-matched)

| start level | users | share |
|---|---|---|
| 0 | 200 | 40% |
| 1 | 175 | 35% |
| 2 | 125 | 25% |

The catalog is 49% beginner arms. A population matching that
would leave the difficulty term inert for most users: measured
earlier, two reward specs differing 2.7x on the too-advanced weight
produced *identical* top-10s for a beginner, diverging only for an
advanced learner. Stratifying keeps difficulty a live signal.

## Goal depth

| target milestone | users |
|---|---|
| 2 | 75 |
| 3 | 55 |
| 4 | 40 |
| 5 | 65 |
| 6 | 30 |
| 7 | 20 |
| 8 | 15 |
| 9 | 20 |
| 10 | 35 |
| 11 | 65 |
| 12 | 30 |
| 13 | 25 |
| 14 | 10 |
| 15 | 5 |
| 17 | 10 |

- mean unmet prerequisite fraction at start: **0.61**

## Weekly hours

| hours | users |
|---|---|
| 3 | 25 |
| 4 | 72 |
| 5 | 55 |
| 6 | 91 |
| 7 | 32 |
| 8 | 93 |
| 10 | 80 |
| 12 | 45 |
| 15 | 6 |
| 20 | 1 |

## Goal families

| family | users |
|---|---|
| `classical-ml` | 35 |
| `computer-systems` | 35 |
| `computer-vision` | 35 |
| `data-analysis-viz` | 35 |
| `databases` | 35 |
| `deep-learning` | 35 |
| `math-for-ml` | 35 |
| `mlops-deployment` | 35 |
| `mobile-development` | 35 |
| `nlp-and-llm` | 35 |
| `programming-foundations` | 30 |
| `security` | 30 |
| `software-engineering` | 30 |
| `web-backend-apis` | 30 |
| `web-frontend` | 30 |
