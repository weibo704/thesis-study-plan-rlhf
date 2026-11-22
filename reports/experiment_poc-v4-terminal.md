# Experiment — `poc-v4-terminal`

- learners: **200** | horizon **8** | seeds **[7, 11, 23]** | learner-runs **600** per policy

## Means

| policy | mean total reward | mean final goal progress |
|---|---|---|
| `baseline` | -0.2146 | 0.7850 |
| `linucb` | -1.2301 | 0.3433 |
| `planner` | -1.3089 | 0.2650 |

## Paired per-learner contrasts

Every learner runs under every policy, so the within-learner
difference removes variance from learners differing in goal
depth, readiness and hours. A 95% interval that spans zero
means the effect is not established, regardless of the sign.

### reward

| contrast | difference | 95% CI | win rate | verdict |
|---|---|---|---|---|
| learning | -1.0155 | ±0.1197 | 45% | **established** |
| planning | -0.0789 | ±0.1323 | 59% | not established |
| learning + planning | -1.0944 | ±0.1361 | 41% | **established** |

### progress

| contrast | difference | 95% CI | win rate | verdict |
|---|---|---|---|---|
| learning | -0.4417 | ±0.0400 | 0% | **established** |
| planning | -0.0783 | ±0.0418 | 10% | **established** |
| learning + planning | -0.5200 | ±0.0441 | 3% | **established** |

## Reward mean by seed

| seed | `baseline` | `linucb` | `planner` |
|---|---|---|---|
| 7 | -0.1872 | -1.2143 | -1.2547 |
| 11 | -0.3114 | -1.2316 | -1.2685 |
| 23 | -0.1451 | -1.2443 | -1.4036 |

Per-seed means are shown because a single seed is not
evidence: an earlier 150-learner run reported the planner
ahead on seed 7 (+0.108) while seed 23 gave −0.032.
