# Skill taxonomy (frozen)

- leaves: **188** across **15** families
- assignments: **4,182** over **1,431** courses
- crosswalk rows: **1,828** (211 map, 1,617 drop)

## Leaves per family

| family | leaves |
|---|---|
| `classical-ml` | 14 |
| `computer-systems` | 14 |
| `data-analysis-viz` | 14 |
| `databases` | 14 |
| `math-for-ml` | 14 |
| `mlops-deployment` | 14 |
| `programming-foundations` | 14 |
| `security` | 14 |
| `software-engineering` | 14 |
| `web-frontend` | 14 |
| `computer-vision` | 12 |
| `nlp-and-llm` | 12 |
| `web-backend-apis` | 9 |
| `mobile-development` | 8 |
| `deep-learning` | 7 |

## Noisy-tag filter

- assignments before: **4,623**
- after: **4,204** (dropped 419, 9.1%)

An assignment survives only if the leaf label is semantically
close to the course's own title and description. This is what
removes tags like `Mergers & Acquisitions` on a Git course,
which are genuinely present in the upstream data.

## Audit status

Thresholds were calibrated against this catalog over three
passes and then deliberately frozen: further tuning against
individually inspected examples risks overfitting the filter
to the cases that happened to be looked at.

Known residual errors found by inspection:

- `FinTech` survives at family_fit 0.303, just over the 0.30
  floor, and is assigned to `web-frontend`. It is a domain,
  not a skill.
- `flask` was dropped as a false negative: a real framework
  whose short proper-noun name scores low against family prose
  and whose courses did not corroborate the family.

`reports/taxonomy_audit.csv` lists every leaf with an empty
`human_label` column. Precision is **not reported**: it needs
that column filled in. Until then this is an unvalidated
vocabulary, not a measured result.
