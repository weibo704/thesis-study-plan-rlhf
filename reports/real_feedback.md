# Real study feedback

- participants: **6**
- plan sets generated: **13**
- pairwise choices: **52**
- reasons given: **28**
- usable preference pairs: **52**

## Position bias

The left-hand plan was chosen **65%** of the time. At 50% position is not driving choices; far from it and the labels are partly about layout rather than content.

Median time on screen **10.7s**; **6%** of choices came in under 3 seconds. A high share of very fast choices is worth a look — two five-course plans take longer than that to read.

## Fitted on real preferences

Bradley-Terry on 36 training choices, held-out accuracy **0.750** on 16.

| feature | weight |
|---|---|
| `mean_topic` | -0.020 |
| `mean_too_hard` | -0.260 |
| `mean_too_easy` | +1.091 |
| `goal_coverage` | -1.520 |
| `mean_rating` | +0.137 |
| `pos_weighted_difficulty` | +0.105 |
| `bias` | +0.000 |

Compare these against the simulated fit in `reports/bestofn_poc-v6-stratified.md`. Agreement on the sign and rough magnitude of `mean_too_hard` is the single most informative check: it is what the simulated learners were most sure about.
