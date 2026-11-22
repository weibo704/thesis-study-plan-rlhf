# Randomized-graph ablation

The go/no-go gate from the ingestion plan, run at last. Three populations identical in learner id, start level, weekly hours, persistence and budget; only the graph their goals and starting mastery derive from differs.

`python -m pipeline.stage16_ablation` · bundle `poc-v6-stratified` weights · 500 learners × 5 seeds · no API calls

Degree-preserving swaps keep every skill's in- and out-degree but lengthen paths, so each learner is also matched to the random plan closest in depth to the real plan they would have received. Without that the random arm would differ in plan depth as well as ordering and neither could be blamed for a difference.

## Mean total reward

| graph | baseline | linucb | planner |
|---|---|---|---|
| `real` | +0.0349 | +0.2346 | +0.2817 |
| `random` | +0.0001 | +0.1797 | +0.2265 |
| `none` | -0.1178 | +0.1571 | +0.2238 |

## Paired contrasts — real graph vs each control

Every learner appears under all three graphs, so the within-learner difference removes variance from learners differing in level, hours and budget. A 95% interval spanning zero means the graph made no established difference to that policy.

| policy | contrast | difference | 95% CI | win rate | verdict |
|---|---|---|---|---|---|
| `baseline` | real − random | +0.0347 | ±0.0478 | 53% | not established |
| `baseline` | real − none | +0.1526 | ±0.0541 | 59% | **established** |
| `linucb` | real − random | +0.0549 | ±0.0451 | 56% | **established** |
| `linucb` | real − none | +0.0775 | ±0.0508 | 57% | **established** |
| `planner` | real − random | +0.0552 | ±0.0442 | 56% | **established** |
| `planner` | real − none | +0.0579 | ±0.0490 | 55% | **established** |

## Reading

A real-vs-random difference that is not established means the prerequisite structure adds no signal to the simulation beyond its shape — which is a result worth reporting, not a failure to hide. The `none` column is the weaker control: it removes the chain entirely, so learners start with almost no seeded mastery and the comparison confounds ordering with preparation.

A gap does **not** establish that the main comparison is uncontaminated. Only the `world/` → `agent/` firewall does that, and it is an implementation invariant rather than a measurement.
