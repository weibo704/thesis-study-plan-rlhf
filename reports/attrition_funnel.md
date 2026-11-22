# Scope attrition funnel

| step | remaining | delta |
|---|---|---|
| spine rows | 3,522 | all normalized source rows |
| canonical after dedup | 3,362 | one row per dedup cluster |
| after dropping containers | 3,362 | -0 |
| after dropping non-English | 3,362 | -0 |
| after dropping text-less rows | 3,362 | -0 |
| after subject classification | 1,899 | -1463 |

**In-scope arm pool: 1,899**

## Band breakdown

| band | rows |
|---|---|
| out | 1,463 |
| in | 1,222 |
| ambiguous | 677 |

## Assigned family (in-scope only)

| family | rows |
|---|---|
| `programming-foundations` | 440 |
| `data-analysis-viz` | 285 |
| `software-engineering` | 173 |
| `classical-ml` | 157 |
| `mlops-deployment` | 143 |
| `math-for-ml` | 122 |
| `security` | 90 |
| `databases` | 84 |
| `web-frontend` | 84 |
| `deep-learning` | 83 |
| `computer-systems` | 66 |
| `web-backend-apis` | 58 |
| `computer-vision` | 49 |
| `mobile-development` | 39 |
| `nlp-and-llm` | 26 |

## Audit status

`reports/scope_audit_sample.csv` holds the full ambiguous band plus a
random 75 from each of the in and out bands, with an empty
`human_label` column. Precision and recall are **not reported yet**:
they require that column to be filled in. Until then these counts are
an unvalidated classifier's output, not a measured result.

Nothing is deleted. Out-of-scope rows remain in
`data/catalog/courses.parquet` with `arm_status='out_of_scope'`.
