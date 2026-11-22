# Policy comparison — `poc-v4-terminal`

- bundle: `206262b55cc2c795`
- learners: **200** | horizon **8** | candidates **60** | gamma **0.85**

| policy | learns | plans | mean total reward | mean final goal progress |
|---|---|---|---|---|
| `baseline` | no | no | -0.145 | 0.7900 |
| `linucb` | yes | no | -1.244 | 0.3150 |
| `planner` | yes | yes | -1.404 | 0.2300 |

## Reading the contrasts

- `linucb` − `baseline` isolates the value of **learning** the reward.
- `planner` − `linucb` isolates the value of **planning**: identical learned model, lookahead added.

The planner's transition model is its own and uses only observable data (which skills a course teaches). It never reads the prerequisite DAG — whether ordering pays off is something it must discover.
