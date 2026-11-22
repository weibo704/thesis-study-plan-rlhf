# Policy comparison — `poc-v2`

- bundle: `8f02f51d805deba5`
- learners: **500** | horizon **8** | candidates **60** | gamma **0.85**

| policy | learns | plans | mean total reward | mean final goal progress |
|---|---|---|---|---|
| `baseline` | no | no | 4.312 | 0.7700 |
| `linucb` | yes | no | 4.526 | 0.7640 |
| `planner` | yes | yes | 4.585 | 0.7580 |

## Reading the contrasts

- `linucb` − `baseline` isolates the value of **learning** the reward.
- `planner` − `linucb` isolates the value of **planning**: identical learned model, lookahead added.

The planner's transition model is its own and uses only observable data (which skills a course teaches). It never reads the prerequisite DAG — whether ordering pays off is something it must discover.
