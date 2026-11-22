# Donor merge report

- spine rows: **3,522** (coursera2021=3,522, edx=0)

| donor | join | spine rows matched | share of coursera2021 |
|---|---|---|---|
| `coursera_hours` | url_slug | 2,048 | 58.1% |
| `coursera2025` | title_institution | 718 | 20.4% |

## Fields donated into a gap

| field | count |
|---|---|
| `rating_count` | 2,274 |
| `duration_lower_bound` | 2,006 |
| `duration_upper_bound` | 2,006 |
| `effort_hours_total` | 1,996 |
| `raw_skills` | 1,416 |
| `subject_raw` | 718 |
| `calendar_weeks_lo` | 258 |
| `calendar_weeks_hi` | 258 |
| `level_canonical` | 3 |

A donor never creates a course. A donated value only fills a genuine
null; the spine is never overwritten. The slug-joined donor is applied
first, so the higher-precision source wins any contested field.
