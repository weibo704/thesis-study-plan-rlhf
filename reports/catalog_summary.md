# Catalog coverage report

Dataset version: `v1-2026-09-20`

## Sources

| source | licence | redistributable | records |
|---|---|---|---|
| coursera2021 | Unknown | False | 3,522 |
| coursera2025 | Apache-2.0 | True | 3,404 |
| coursera_hours | CC0-1.0 | True | 9,595 |

## Arm pool (pre-scope-filter)

- spine rows: **3,522**
- canonical courses after dedup: **3,362**
- collapsed as duplicates: **160**

By source:

- `coursera2021`: 3,362

## Field coverage (canonical rows)

| field | coverage |
|---|---|
| description | 100.0% |
| raw skill text | 100.0% |
| syllabus | 0.0% |
| difficulty level | 98.9% |
| rating | 97.7% |
| effort hours (total) | 56.2% |
| any duration signal | 63.7% |

## Difficulty provenance

| level_source | rows |
|---|---|
| `declared` | 3,165 |
| `declared_vocab_mapped` | 153 |
| `unknown` | 36 |
| `inferred_text` | 8 |

Distribution: 0=1,549, 1=805, 2=972  (0=beginner, 1=intermediate, 2=advanced)

## Known data damage

- `title_normalized`: **50** rows (1.5%) contain U+FFFD
- `description`: **1,765** rows (52.5%) contain U+FFFD
- `institution`: **94** rows (2.8%) contain U+FFFD

These replacement characters are baked into the Coursera 2021 export
at scrape time. The original bytes are gone, so they are unrecoverable
here. They matter because they sit inside description text that feeds
embeddings; treat as a documented limitation.

## Language

| language | rows |
|---|---|
| English | 3,362 |
