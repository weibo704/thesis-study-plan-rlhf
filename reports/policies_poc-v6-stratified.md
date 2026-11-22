# Policy comparison — `poc-v6-stratified`

- bundle: `54539899ab24e4b4`
- learners: **500** | horizon **8** | candidates **60** | gamma **0.85**

| policy | learns | plans | mean total reward | mean final goal progress |
|---|---|---|---|---|
| `baseline` | no | no | 0.104 | 0.7500 |
| `linucb` | yes | no | 0.295 | 0.7160 |
| `planner` | yes | yes | 0.358 | 0.6600 |

## Reading the contrasts

- `linucb` − `baseline` isolates the value of **learning** the reward.
- `planner` − `linucb` isolates the value of **planning**: identical learned model, lookahead added.

The planner's transition model is its own and uses only observable data (which skills a course teaches). It never reads the prerequisite DAG — whether ordering pays off is something it must discover.
