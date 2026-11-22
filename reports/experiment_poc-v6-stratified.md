# Experiment — `poc-v6-stratified`

- learners: **500** | horizon **8** | seeds **[7, 11, 23, 37, 53]** | learner-runs **2,500** per policy

## Means

| policy | mean total reward | mean final goal progress |
|---|---|---|
| `baseline` | 0.0947 | 0.7428 |
| `linucb` | 0.2989 | 0.7064 |
| `planner` | 0.3630 | 0.6580 |

## Paired per-learner contrasts

Every learner runs under every policy, so the within-learner
difference removes variance from learners differing in goal
depth, readiness and hours. A 95% interval that spans zero
means the effect is not established, regardless of the sign.

### reward

| contrast | difference | 95% CI | win rate | verdict |
|---|---|---|---|---|
| learning | +0.2042 | ±0.0130 | 80% | **established** |
| planning | +0.0640 | ±0.0206 | 53% | **established** |
| learning + planning | +0.2683 | ±0.0218 | 67% | **established** |

### progress

| contrast | difference | 95% CI | win rate | verdict |
|---|---|---|---|---|
| learning | -0.0364 | ±0.0084 | 1% | **established** |
| planning | -0.0484 | ±0.0214 | 13% | **established** |
| learning + planning | -0.0848 | ±0.0209 | 10% | **established** |

## Reward mean by seed

| seed | `baseline` | `linucb` | `planner` |
|---|---|---|---|
| 7 | 0.1038 | 0.3037 | 0.3721 |
| 11 | 0.1038 | 0.3136 | 0.3759 |
| 23 | 0.0798 | 0.2961 | 0.3638 |
| 37 | 0.0824 | 0.2864 | 0.3450 |
| 53 | 0.1038 | 0.2948 | 0.3581 |

Per-seed means are shown because a single seed is not
evidence: an earlier 150-learner run reported the planner
ahead on seed 7 (+0.108) while seed 23 gave −0.032.
