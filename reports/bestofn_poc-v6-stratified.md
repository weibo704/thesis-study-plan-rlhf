# Does RL improve LLM-generated study plans? — `poc-v6-stratified`

- learners: **200** (120 train / 80 test)
- plans generated: **1,200** (6 per learner, 5 courses each)
- features: **7** observable · standardised on train · all 15 pairs per learner
- preference model (feedback only): held-out pair accuracy **0.602**, correlation with true return **r = 0.857**
- regression model (oracle scalar): **r = 0.862**
- simulated feedback label noise: learners picked the genuinely better plan **64%** of the time

## Selection policies (identical candidates)

| policy | mean return | mean goal progress |
|---|---|---|
| `sample_1` | 0.5985 | 0.5250 |
| `random_of_n` | 0.4045 | 0.4375 |
| `best_of_n_preference` | 0.7121 | 0.5500 |
| `best_of_n_regression` | 0.6819 | 0.5500 |
| `best_of_n_oracle` | 0.8438 | 0.5250 |

## Paired contrasts (return)

| contrast | difference | 95% CI | win rate | verdict |
|---|---|---|---|---|
| feedback RL over raw LLM | +0.1136 | ±0.0745 | 51% | **established** |
| scoring over sampling | +0.3075 | ±0.0995 | 70% | **established** |
| more candidates alone (no scoring) | -0.1940 | ±0.1001 | 25% | **established** |
| feedback vs oracle-regression model | +0.0301 | ±0.0321 | 21% | not established |
| headroom left (oracle - feedback) | +0.1317 | ±0.0465 | 48% | **established** |

`random_of_n` is the control that separates *scoring helped*
from *having more candidates helped*. `best_of_n_oracle` is not a
competitor — it is the ceiling, showing how much of the available
gain the learned model captured.

## Learned reward-model weights

| feature | theta |
|---|---|
| `mean_topic` | +0.313 |
| `mean_too_hard` | -0.838 |
| `mean_too_easy` | -0.184 |
| `goal_coverage` | +0.366 |
| `mean_rating` | +0.182 |
| `pos_weighted_difficulty` | +0.368 |
| `bias` | +0.000 |
