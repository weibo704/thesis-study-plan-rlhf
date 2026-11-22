# Datasheet — course catalog

Dataset version: `v1-2026-09-20`  
Feature spec: `arm-spec-v1`  Projection: `svd64-v1`

## Composition

- source rows: 16,521
- canonical courses: 3,362
- in scope: 1,899
- arms (frozen index): 1,899
- skills: 188 across 15 families
- skill assignments: 4,182
- prerequisite edges: 254

## Sources

| source | licence | redistributable | rows |
|---|---|---|---|
| coursera2021 | Unknown | False | 3,522 |
| coursera2025 | Apache-2.0 | True | 3,404 |
| coursera_hours | CC0-1.0 | True | 9,595 |

## Known limitations

1. Both the skill taxonomy and the prerequisite DAG are **model-generated**, not validated against a curriculum or expert.
2. **52.5% of descriptions contain U+FFFD** replacement characters, baked in upstream at scrape time and unrecoverable.
3. **Provider leakage resolved by construction.** The arm pool is a single platform (Coursera); edX was dropped because its disjoint field coverage made the provider perfectly predictable from the arm features (AUC 1.000), which would have made any reward term over duration or rating partly a provider term.
4. **468 of 1,899 arms carry no skill** through the frozen taxonomy; the arm-by-skill matrix is 98.8% sparse. (Stage 7 extracted at least one raw skill phrase for 97.9% of arms; the gap is what the taxonomy crosswalk kept.)
5. The catalog predates the GenAI wave, so the thinnest families are `deep-learning` (7 leaves), `mobile-development` (8 leaves).
6. Duration coverage: **46.4%** of in-scope arms have an effort-hours total, after donor merges.
7. Scope precision/recall and taxonomy precision are **not measured**; audit samples with empty `human_label` columns are emitted instead.

## Intended use

Item space for a contextual-bandit vs content-based-baseline comparison. The taxonomy and DAG are scaffolding, not contributions.

## Firewall

`data/world/` (prerequisite DAG, study_plans) must never be read when building `data/agent/` features. Asserted in the build.
