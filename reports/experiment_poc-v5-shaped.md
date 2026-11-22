# Experiment — `poc-v5-shaped`

- learners: **200** | horizon **8** | seeds **[7, 11, 23]** | learner-runs **600** per policy

## Means

| policy | mean total reward | mean final goal progress |
|---|---|---|
| `baseline` | -0.6071 | 0.7850 |
| `linucb` | -0.5058 | 0.7933 |
| `planner` | -0.4112 | 0.7667 |

## Paired per-learner contrasts

Every learner runs under every policy, so the within-learner
difference removes variance from learners differing in goal
depth, readiness and hours. A 95% interval that spans zero
means the effect is not established, regardless of the sign.

### reward

| contrast | difference | 95% CI | win rate | verdict |
|---|---|---|---|---|
| learning | +0.1013 | ±0.0517 | 70% | **established** |
| planning | +0.0946 | ±0.1041 | 57% | not established |
| learning + planning | +0.1959 | ±0.1072 | 64% | **established** |

### progress

| contrast | difference | 95% CI | win rate | verdict |
|---|---|---|---|---|
| learning | +0.0083 | ±0.0176 | 3% | not established |
| planning | -0.0267 | ±0.0394 | 11% | not established |
| learning + planning | -0.0183 | ±0.0401 | 12% | not established |

## Reward mean by seed

| seed | `baseline` | `linucb` | `planner` |
|---|---|---|---|
| 7 | -0.5872 | -0.4285 | -0.5092 |
| 11 | -0.6939 | -0.6311 | -0.3608 |
| 23 | -0.5401 | -0.4577 | -0.3637 |

Per-seed means are shown because a single seed is not
evidence: an earlier 150-learner run reported the planner
ahead on seed 7 (+0.108) while seed 23 gave −0.032.
