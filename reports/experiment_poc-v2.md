# Experiment — `poc-v2`

- learners: **500** | horizon **8** | seeds **[7, 11, 23, 37, 53]** | learner-runs **2,500** per policy

## Means

| policy | mean total reward | mean final goal progress |
|---|---|---|
| `baseline` | 4.3366 | 0.7728 |
| `linucb` | 4.5543 | 0.7684 |
| `planner` | 4.6044 | 0.7432 |

## Paired per-learner contrasts

Every learner runs under every policy, so the within-learner
difference removes variance from learners differing in goal
depth, readiness and hours. A 95% interval that spans zero
means the effect is not established, regardless of the sign.

### reward

| contrast | difference | 95% CI | win rate | verdict |
|---|---|---|---|---|
| learning | +0.2177 | ±0.0147 | 80% | **established** |
| planning | +0.0500 | ±0.0260 | 53% | **established** |
| learning + planning | +0.2678 | ±0.0276 | 65% | **established** |

### progress

| contrast | difference | 95% CI | win rate | verdict |
|---|---|---|---|---|
| learning | -0.0044 | ±0.0073 | 2% | not established |
| planning | -0.0252 | ±0.0203 | 12% | **established** |
| learning + planning | -0.0296 | ±0.0201 | 12% | **established** |

## Reward mean by seed

| seed | `baseline` | `linucb` | `planner` |
|---|---|---|---|
| 7 | 4.3393 | 4.5470 | 4.6049 |
| 11 | 4.3120 | 4.5485 | 4.6314 |
| 23 | 4.3690 | 4.5830 | 4.5932 |
| 37 | 4.3503 | 4.5673 | 4.6068 |
| 53 | 4.3123 | 4.5256 | 4.5855 |

Per-seed means are shown because a single seed is not
evidence: an earlier 150-learner run reported the planner
ahead on seed 7 (+0.108) while seed 23 gave −0.032.
