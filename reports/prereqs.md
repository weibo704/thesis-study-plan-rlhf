# Prerequisite DAG and study_plans

> **Firewall.** Everything here lives in `data/world/` and must never
> be read by anything building `agent/` features. Co-occurrence edges
> are derived from `level_canonical` and skill overlap -- the same
> signals the recommender scores on -- so if they reached arm features
> while the reward credited prerequisite satisfaction, the bandit
> would win by construction.

- skills: **188**
- raw edges proposed: **266** (agent 151, co-occurrence 115)
- after dedup: **259**
- dropped to enforce acyclicity: **5**
- final DAG edges: **254**

## Edge provenance

| derivation | edges |
|---|---|
| `agent_proposed` | 151 |
| `co_occurrence_inferred` | 103 |

## Agreement between the two sources

- both propose the same direction: **7** edges
- they propose OPPOSITE directions: **3** edges

Where the two disagree is where the DAG is weakest. Corroborated
edges are floored at confidence 0.8; contradictions are resolved by
confidence during the acyclic pass and are worth a sentence in the
write-up rather than silent resolution.

## Study Plans

| family | milestones | skills |
|---|---|---|
| `classical-ml` | 11 | 17 |
| `classical-ml` | 13 | 19 |
| `classical-ml` | 11 | 17 |
| `classical-ml` | 13 | 19 |
| `classical-ml` | 11 | 17 |
| `classical-ml` | 11 | 17 |
| `classical-ml` | 11 | 17 |
| `classical-ml` | 12 | 18 |
| `classical-ml` | 16 | 32 |
| `classical-ml` | 11 | 18 |
| `classical-ml` | 11 | 17 |
| `classical-ml` | 10 | 16 |
| `classical-ml` | 15 | 31 |
| `classical-ml` | 11 | 17 |
| `computer-systems` | 6 | 9 |
| `computer-systems` | 6 | 11 |
| `computer-systems` | 12 | 21 |
| `computer-systems` | 6 | 14 |
| `computer-systems` | 3 | 3 |
| `computer-systems` | 6 | 9 |
| `computer-systems` | 9 | 12 |
| `computer-systems` | 5 | 8 |
| `computer-systems` | 11 | 16 |
| `computer-vision` | 13 | 22 |
| `computer-vision` | 5 | 5 |
| `computer-vision` | 3 | 3 |
| `computer-vision` | 5 | 5 |
| `computer-vision` | 4 | 4 |
| `computer-vision` | 5 | 5 |
| `computer-vision` | 5 | 6 |
| `data-analysis-viz` | 17 | 36 |
| `data-analysis-viz` | 3 | 3 |
| `data-analysis-viz` | 14 | 21 |
| `data-analysis-viz` | 13 | 19 |
| `data-analysis-viz` | 13 | 19 |
| `data-analysis-viz` | 18 | 37 |
| `data-analysis-viz` | 11 | 17 |
| `data-analysis-viz` | 12 | 18 |
| `data-analysis-viz` | 13 | 19 |
| `databases` | 11 | 16 |
| `databases` | 13 | 18 |
| `databases` | 11 | 24 |
| `databases` | 11 | 16 |
| `databases` | 13 | 18 |
| `databases` | 12 | 17 |
| `databases` | 10 | 15 |
| `databases` | 9 | 14 |
| `databases` | 14 | 20 |
| `databases` | 14 | 21 |
| `databases` | 13 | 19 |
| `databases` | 12 | 28 |
| `deep-learning` | 3 | 4 |
| `deep-learning` | 4 | 5 |
| `deep-learning` | 4 | 5 |
| `deep-learning` | 8 | 11 |
| `deep-learning` | 4 | 5 |
| `math-for-ml` | 3 | 3 |
| `math-for-ml` | 14 | 23 |
| `math-for-ml` | 3 | 3 |
| `math-for-ml` | 3 | 3 |
| `math-for-ml` | 4 | 5 |
| `math-for-ml` | 15 | 25 |
| `math-for-ml` | 13 | 21 |
| `math-for-ml` | 12 | 20 |
| `math-for-ml` | 4 | 4 |
| `math-for-ml` | 11 | 19 |
| `mlops-deployment` | 6 | 6 |
| `mlops-deployment` | 5 | 5 |
| `mlops-deployment` | 4 | 4 |
| `mlops-deployment` | 5 | 5 |
| `mlops-deployment` | 14 | 25 |
| `mlops-deployment` | 6 | 6 |
| `mlops-deployment` | 6 | 6 |
| `mlops-deployment` | 7 | 7 |
| `mlops-deployment` | 6 | 9 |
| `mobile-development` | 14 | 29 |
| `mobile-development` | 6 | 9 |
| `mobile-development` | 14 | 30 |
| `mobile-development` | 13 | 27 |
| `mobile-development` | 12 | 26 |
| `nlp-and-llm` | 3 | 3 |
| `nlp-and-llm` | 11 | 18 |
| `nlp-and-llm` | 12 | 19 |
| `nlp-and-llm` | 3 | 3 |
| `programming-foundations` | 7 | 10 |
| `programming-foundations` | 7 | 10 |
| `programming-foundations` | 8 | 11 |
| `programming-foundations` | 7 | 10 |
| `programming-foundations` | 7 | 10 |
| `programming-foundations` | 7 | 10 |
| `programming-foundations` | 7 | 10 |
| `programming-foundations` | 6 | 9 |
| `programming-foundations` | 7 | 10 |
| `programming-foundations` | 8 | 11 |
| `programming-foundations` | 12 | 21 |
| `programming-foundations` | 7 | 10 |
| `security` | 12 | 35 |
| `security` | 3 | 3 |
| `security` | 12 | 18 |
| `security` | 4 | 4 |
| `security` | 13 | 36 |
| `security` | 4 | 4 |
| `security` | 3 | 3 |
| `software-engineering` | 12 | 18 |
| `software-engineering` | 11 | 17 |
| `software-engineering` | 10 | 16 |
| `software-engineering` | 11 | 17 |
| `software-engineering` | 8 | 11 |
| `software-engineering` | 11 | 17 |
| `software-engineering` | 9 | 13 |
| `web-backend-apis` | 3 | 3 |
| `web-backend-apis` | 3 | 3 |
| `web-backend-apis` | 3 | 3 |
| `web-backend-apis` | 5 | 5 |
| `web-frontend` | 6 | 11 |
| `web-frontend` | 4 | 6 |
| `web-frontend` | 6 | 8 |
| `web-frontend` | 10 | 15 |
| `web-frontend` | 5 | 7 |
| `web-frontend` | 6 | 8 |
| `web-frontend` | 9 | 12 |
| `web-frontend` | 3 | 4 |

845 of 1662 study plan items sit at a position shared
with at least one other skill. Those ties are recorded rather than
broken: parallel skills at one depth are genuinely unordered, and an
arbitrary order would read downstream as a real sequencing claim.

## Validation status

Held-out labelled edges and inter-annotator agreement are **not**
done -- the plan drops them, since the DAG is scaffolding rather
than a contribution.

The **randomized-graph ablation** -- the plan's go/no-go gate -- HAS
now run: `python -m pipeline.stage16_ablation`, written up in
`reports/ablation_randomized_graph.md`. Against a degree-preserving
randomized graph, depth-matched per learner, the real edges are worth
+0.055 to both linucb and the planner (95% CI +/-0.045, established)
and +0.035 to the fixed-weight baseline (not established). The DAG
beats noise for the policies that learn.

That is all it means. A pass does NOT make the main comparison
uncontaminated -- only the world/ -> agent/ firewall does that, and
it is an implementation invariant rather than a measurement.
