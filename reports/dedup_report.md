# Dedup report

- rows in: **3522**
- clusters out: **3362**  (collapsed 160 duplicate rows)
- decisive merges (identical url_slug): **98**

## Candidate pair bands

| band | pairs | share |
|---|---|---|
| auto_merge | 210 | 5.0% |
| review | 306 | 7.3% |
| auto_reject | 3660 | 87.6% |

Review band is **7.3%** of scored pairs.
Plan section 3d targets 10-15%: outside that range, recalibrate
`config/thresholds.yaml` against hand-labelled pairs before trusting
the auto bands.

Review-band pairs are written to `reports/dedup_review.csv` for
manual adjudication; they are NOT merged automatically.
