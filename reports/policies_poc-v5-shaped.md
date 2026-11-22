# Policy comparison — `poc-v5-shaped`

- bundle: `a7be07763f05885a`
- learners: **200** | horizon **8** | candidates **60** | gamma **0.85**

| policy | learns | plans | mean total reward | mean final goal progress |
|---|---|---|---|---|
| `baseline` | no | no | -0.540 | 0.7900 |
| `linucb` | yes | no | -0.458 | 0.7650 |
| `planner` | yes | yes | -0.364 | 0.7700 |

## Reading the contrasts

- `linucb` − `baseline` isolates the value of **learning** the reward.
- `planner` − `linucb` isolates the value of **planning**: identical learned model, lookahead added.

The planner's transition model is its own and uses only observable data (which skills a course teaches). It never reads the prerequisite DAG — whether ordering pays off is something it must discover.
