# Contextual bandit vs content baseline

- pre-registered bundle: `af5001b0e627e125`
- learners: **150** | horizon: **5** | candidate set: **60** | alpha: **0.6**

## Outcome

| policy | mean reward | total reward | mean final goal progress |
|---|---|---|---|
| `baseline` | 0.4635 | 347.6 | 0.4517 |
| `linucb` | 0.5198 | 389.9 | 0.2500 |

Reward lift: **+0.0563** (+12.1%)

## Learned weights

| feature | theta |
|---|---|
| `topic` | +0.884 |
| `need` | +0.006 |
| `too_hard` | -0.415 |
| `too_easy` | -0.139 |
| `time_over` | -0.134 |
| `rating_c` | +0.088 |
| `has_level` | +0.052 |
| `has_hours` | +0.048 |
| `bias` | -0.049 |

Sign check: `too_hard` should be more negative than `too_easy`
if the agent recovered the reward's asymmetry from experience alone —
it never saw the spec.

## What this does and does not show

The bandit models the problem as stateless, but the environment
transitions: taking a course raises mastery, so step *t+1* faces a
different learner. A bandit cannot represent 'take this now so that
becomes viable later'. The MDP planner stage is the same setup with
lookahead added, which isolates the value of planning from the value
of learning.
