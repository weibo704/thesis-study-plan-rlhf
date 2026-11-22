# Policy comparison — `poc-v1`

- bundle: `12217afa0ff7a081`
- learners: **120** | horizon **5** | candidates **60** | gamma **0.85**

| policy | learns | plans | mean total reward | mean final goal progress |
|---|---|---|---|---|
| `baseline` | no | no | 2.148 | 0.4688 |
| `linucb` | yes | no | 2.442 | 0.2639 |
| `planner` | yes | yes | 2.432 | 0.1801 |

## Reading the contrasts

- `linucb` − `baseline` isolates the value of **learning** the reward.
- `planner` − `linucb` isolates the value of **planning**: identical learned model, lookahead added.

The planner's transition model is its own and uses only observable data (which skills a course teaches). It never reads the prerequisite DAG — whether ordering pays off is something it must discover.
