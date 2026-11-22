# Skill extraction report

- in-scope courses: **1,899**
- courses with >=1 extracted skill: **1,860** (97.9%)
- skill assignments: **9,696**
- distinct skill phrases in use: **1,977**
- donor vocabulary size: **1,729** phrases

## How each course was handled

| path | courses |
|---|---|
| clean delimited skills (2025 donor) | 797 |
| segmented via donor vocabulary | 1,063 |
| skill text present, no dictionary hit | 39 |
| no skill text at all | 0 |

## Residue (unmatched by the dictionary)

- courses with residue: **1,102**
- mean unmatched tokens per course: **10.8**

Residue is preserved verbatim in `reports/skill_residue.parquet`.
It is the undelimited 2021 tag text the 2025 vocabulary does not
cover, and it is the input for the LLM extraction pass.

## Not done in this stage

- **LLM candidate generation** (plan 3b): the residue above needs it.
- **Adjudication band** of the merge cascade (plan 3d): the
  deterministic auto-merge and auto-reject bands can run without a
  model, but the middle band cannot.
- **Freezing the taxonomy** (plan 3f): premature until both land.

Both require an API key, which is not configured. What this stage
emits is a deterministic floor, not the frozen taxonomy.
