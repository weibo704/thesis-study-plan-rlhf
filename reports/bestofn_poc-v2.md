# Does RL improve LLM-generated study plans? — `poc-v2`

- learners: **120** (72 train / 48 test)
- plans generated: **720** (6 per learner, 5 courses each)
- preference model (feedback only): held-out pair accuracy **0.615**, correlation with true return **r = 0.820**
- regression model (oracle scalar): **r = 0.899**
- simulated feedback label noise: learners picked the genuinely better plan **64%** of the time

## Selection policies (identical candidates)

| policy | mean return | mean goal progress |
|---|---|---|
| `sample_1` | 0.6050 | 0.4583 |
| `random_of_n` | 0.4865 | 0.4375 |
| `best_of_n_preference` | 0.8499 | 0.4375 |
| `best_of_n_regression` | 0.8744 | 0.4583 |
| `best_of_n_oracle` | 0.9898 | 0.5208 |

## Paired contrasts (return)

| contrast | difference | 95% CI | win rate | verdict |
|---|---|---|---|---|
| feedback RL over raw LLM | +0.2448 | ±0.1483 | 54% | **established** |
| scoring over sampling | +0.3634 | ±0.1601 | 67% | **established** |
| more candidates alone (no scoring) | -0.1186 | ±0.1771 | 29% | not established |
| feedback vs oracle-regression model | -0.0245 | ±0.0177 | 4% | **established** |
| headroom left (oracle - feedback) | +0.1399 | ±0.0599 | 62% | **established** |

`random_of_n` is the control that separates *scoring helped*
from *having more candidates helped*. `best_of_n_oracle` is not a
competitor — it is the ceiling, showing how much of the available
gain the learned model captured.

## Learned reward-model weights

| feature | theta |
|---|---|
| `mean_topic` | +3.120 |
| `mean_too_hard` | -1.532 |
| `mean_too_easy` | -0.332 |
| `hours_over` | -0.383 |
| `goal_coverage` | +0.553 |
| `n_skills` | +0.041 |
| `level_monotonic` | +0.121 |
| `mean_rating` | +0.024 |
| `plan_len` | -0.866 |
| `bias` | -0.866 |
