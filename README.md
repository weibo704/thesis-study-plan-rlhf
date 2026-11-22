# Learning to Rank LLM-Generated Study Plans from Human Feedback

Master's thesis project, Northeastern University. Large language models write
plausible study plans but rank their own output poorly. This asks whether a
reward model trained on pairwise human preferences can select better plans than
the model's own ordering — in a recommendation setting rather than a
conversational one.

**Thesis draft:** [`docs/thesis-draft.md`](docs/thesis-draft.md) ·
**architecture figure:** [`docs/stage1-flow.svg`](docs/stage1-flow.svg)

## The result

A generator writes six candidate plans from a 25-course retrieved shortlist; a
Bradley–Terry reward model fitted on noisy pairwise preferences selects among
them. On held-out learners, against a reward function frozen and content-hashed
before any run:

| Contrast | Difference | 95% CI | Verdict |
|---|---|---|---|
| Feedback RL over the raw LLM | **+0.1136** | ±0.0745 | established |
| Scoring over sampling | +0.3075 | ±0.0995 | established |
| More candidates alone, no scoring | −0.1940 | ±0.1001 | established |

That third row is the control that gives the first its meaning: generating more
candidates and picking randomly is *worse* than taking the first plan, so the
gain comes from the scoring rather than from having more options.

Decomposing the pipeline — retrieval contributes +1.9828, generation +0.3707,
and the language model's own ranking leaves **+0.2872 unclaimed**. That gap is
what the reward model recovers.

Two findings of independent interest. The preference-trained model slightly
*outperformed* one regressed directly on the true reward (0.7121 vs 0.6819), so
learning from noisy comparisons cost nothing against learning from the answer
key. And it recovered a **4.6:1** asymmetry between over- and under-challenging
a learner without ever observing a reward value, against 3:1 in the frozen spec.

## What is here

```
pipeline/   11 build stages: catalog -> 188-skill taxonomy -> 254-edge
            prerequisite graph -> 500-learner simulator -> frozen reward ->
            bandit/planner baselines -> best-of-N experiment
app/        the study participants use (pairwise plan comparison), a LangGraph
            feedback agent, online reward updating, and an inspector that shows
            how each recorded preference reshapes the recommendation
infra/      AWS CDK: App Runner + DynamoDB + Secrets Manager
docs/       thesis draft, figures, and the builders that produce them
reports/    every number in the thesis, regenerable from the pipeline
tests/      194 tests
```

## Two disciplines the results depend on

**Pre-registration.** The reward function, taxonomy and prerequisite graph are
frozen and content-hashed as one bundle before any policy runs. Tuning the
ground truth after seeing a result would invalidate the study, and the hash
makes that impossible to do quietly.

**The world/agent firewall.** No component that produces a recommendation may
read the prerequisite graph, the reward specification, or a learner's hidden
state — enforced by a build assertion that tracks real file reads. Its cost was
measured rather than assumed: a feature exposing the oracle graph learned a
weight of −0.011, indistinguishable from zero.

## Not in this repository

Excluded deliberately, and the pipeline rebuilds them:

| Path | Why |
|---|---|
| `.env` | API key. Copy `.env.example` and fill it in. |
| `data/catalog/llm_cache/` | 10,329 files, 249 MB. Embeds course descriptions from a source whose licence is "Unknown" and which `config/sources.yaml` marks `redistributable: false`. |
| `data/catalog/*.parquet`, `data/raw/`, `data/staging/` | Same source text. Rebuild with stages 1–11. |
| `data/agent/feedback/` | Participant data. |

Derived artifacts — audit CSVs with course titles and scores, skill
assignments, hashes — are included, which is what the ingestion plan specified
shipping in place of raw scraped text.

Because the cache is excluded, a fresh clone needs API access to rebuild. With
the cache present the whole pipeline replays offline with zero API calls.

## Setup

```bash
python3 -m venv .venv && ./.venv/bin/python -m pip install -r requirements.txt
```

Python 3.9 is the baseline (Homebrew's `python@3.12` on this machine has a
broken `pyexpat`, so `venv` creation fails there).

## Getting the data

Three Kaggle datasets. Stage 1 accepts either auth style:

- **`~/.kaggle/access_token`** — a `KGAT_...` bearer token (Kaggle's newer
  format). The `kaggle` python client does **not** understand these, so
  stage 1 talks to the REST API directly with `Authorization: Bearer`.
- **`~/.kaggle/kaggle.json`** — the classic username/key pair, used via the
  `kaggle` client.

`KAGGLE_API_TOKEN` is also read from the environment. Or download by hand
and place each CSV at the path stage 1 prints:

| source | file | licence |
|---|---|---|
| [Coursera 2021](https://www.kaggle.com/datasets/khusheekapoor/coursera-courses-dataset-2021) | `data/raw/coursera2021/Coursera.csv` | Unknown |
| [Coursera 2025](https://www.kaggle.com/datasets/yosefxx590/coursera-courses-and-skills-dataset-2025) | `data/raw/coursera2025/Coursera.csv` | Apache-2.0 |
| [edX](https://www.kaggle.com/datasets/imuhammad/edx-courses) | `data/raw/edx/edx_courses.csv` | GPL-2 |

Udemy is registered but **disabled** (`config/sources.yaml`): it has no
descriptions, skill tags or difficulty levels, so including it would leave
~85% of arms title-only.

## Running

```bash
./.venv/bin/python -m pipeline.cli status        # stages and artefacts
./.venv/bin/python -m pipeline.cli run           # everything implemented
./.venv/bin/python -m pipeline.cli run --from 3  # resume
./.venv/bin/python -m pipeline.cli run --only 5
```

| # | stage | status | output |
|---|---|---|---|
| 1 | `fetch` | done | `catalog/sources.parquet` |
| 2 | `parse` | done | `staging/<source>.parquet`, `reports/encoding_report.csv` |
| 3 | `normalize` | done | `staging/normalized.parquet` |
| 4 | `merge` | done | `catalog/field_provenance.parquet`, `reports/merge_report.md` |
| 5 | `dedup` | done | `staging/deduped.parquet`, `reports/dedup_review.csv` |
| – | `report_catalog` | done | `reports/catalog_summary.md` (coverage; folds into stage 11) |
| 6 | `taxonomy` | not started | 12 families / ~150 leaves, then freeze |
| 7 | `skills_assign` | done | `catalog/course_skills.parquet`, `catalog/skill_candidates.parquet` |
| 8 | `scope` | done | `catalog/courses.parquet`, `reports/attrition_funnel.md` |
| 9 | `prereqs` | done | `world/skill_prereqs.parquet`, `world/study_plans.parquet` |
| 10 | `features` | done | `agent/arm_index.parquet`, `agent/arm_features.parquet` |
| 11 | `validate` | done | `reports/DATASHEET.md`, `reports/smoke_test_recommendations.csv` |

## Decisions encoded in the code

Each of these is a reward-modelling hazard if done the obvious way; the
tests in `tests/test_parsers.py` are what keep them from regressing.

- **Sentinels never become imputed modes.** `"Not Calibrated"` and `"Mixed"`
  become nulls plus a flag. Imputing the mode manufactures a correlation
  between difficulty and popularity that flows into any level-match term.
- **Durations keep their interval.** Collapsing a bucket to a midpoint and
  log-transforming spikes the distribution at a few values, so a reward term
  monotone in duration degenerates into a step function over buckets.
- **Effort hours and calendar weeks never merge.** A month bucket is
  calendar span; converting it to effort needs an hours-per-week assumption,
  which is a property of the *user*. Baking it into a course row
  double-counts the learner's time budget downstream.
- **Every imputed value carries `*_is_imputed` and a `*_source`.**
  Missingness is provider-correlated, so without these the reward model
  becomes partly a provider classifier.
- **The donor never creates a row.** Coursera 2025 only fills genuine gaps
  in the 2021 spine, and every donated value is recorded in
  `field_provenance`.
- **Dedup is order-independent.** Union-find with a deterministic
  lexicographic root, so pairwise decisions in any order give one grouping.

## Current catalog

First real ingest (`v1-2026-09-20`): **4,312 canonical courses** before scope
filtering — 3,349 Coursera + 963 edX. Full numbers in
`reports/catalog_summary.md`. Three findings worth carrying forward:

- **Difficulty coverage is 99.1%**, far better than the plan assumed. Only 39
  rows are unknown, and the 0/1/2 split is 2,153 / 1,060 / 1,060 — no
  degenerate skew to beginner.
- **Duration is the weak field**: 22.3% have an effort total, 37.7% have any
  duration signal. The Coursera 2021↔2025 donor match is only 20.4%, because
  just ~22% of 2021 titles still exist in the 2024 snapshot.
- **40.8% of descriptions contain U+FFFD**, baked into the source at scrape
  time and unrecoverable. This feeds embeddings, so it is a documented
  limitation, not a bug to fix.

## Scope filter (stage 8)

Two independent signals, kept separable so a reviewer can see which carried
a decision: weighted family-keyword evidence over title / skills /
description, and TF-IDF cosine against a per-family profile document.

Thresholds in `config/thresholds.yaml` are **calibrated against this
catalog**, not guessed. Keyword score is bimodal (0 for ~half the rows, ≥4
for a quarter); similarity turned out weak — median 0.021 — so it can
promote or rescue but never veto keyword evidence.

Nothing is deleted. Out-of-scope rows stay in `courses.parquet` with
`arm_status='out_of_scope'`, so the funnel stays reconstructable.

## Skill extraction (stage 7, deterministic pass)

The two Coursera exports disagree about format in a useful way. The 2025
export lists skills **comma-delimited and clean** (1,729 usable phrases,
mean 2.18 words) — a ready-made canonical vocabulary. The 2021 export
concatenates tags into an undelimited blob with no reliable casing rule, so
splitting it on whitespace would fabricate tokens and is never done.

So: rows matched to a 2025 donor take its delimited skills directly; the
rest are segmented by longest-match against that vocabulary. Residue the
dictionary cannot cover is preserved verbatim in
`reports/skill_residue.parquet` rather than dropped.

Result: **11,510 assignments over 1,848 of 2,188 in-scope courses (84.5%)**,
mean 6.2 skills per course.

**Still needed, and blocked on an API key:** LLM candidate generation over
the residue, the adjudication band of the merge cascade, and the noisy-tag
filter. Source tags are genuinely noisy — a Git course carries a real
`Mergers & Acquisitions` tag in the upstream data — so the taxonomy is
**not frozen** and this is a deterministic floor, not the final vocabulary.

## Frontend

A read-only Flask explorer over the frozen artefacts:

```bash
./.venv/bin/python -m app.server
```

Then open http://127.0.0.1:5001. Four views: **Recommend** (free-text
profile), **Browse** (filter by family/level), **Skills** (difficulty and
time), **Study Plans** (topological milestones per family).

The recommender in the UI is deliberately the *same* trivial nearest-neighbour
scorer as the stage 11 smoke test — text similarity over the frozen SVD
embeddings. It ignores level, duration and the prerequisite graph. It is a
catalog explorer, **not** the thesis baseline. Profile text is projected
through the same frozen SVD basis as the arms; a fresh projection would put
the two in different spaces.

## Feedback capture (RL foundation, step 1)

The UI logs every recommendation turn and every thumbs up/down under
`data/agent/feedback/` as append-only JSONL.

**Recommendations are sampled, not ranked.** A deterministic top-k would log
propensities of 1.0, which makes inverse-propensity scoring degenerate and
rules out off-policy evaluation *permanently* — propensities cannot be
recovered after the fact. So the policy scores a 25-arm candidate set,
samples a slate from a temperature-0.3 softmax, and logs the **full action
probability vector** alongside the chosen items.

```
impressions.jsonl   session_id, turn_index, policy_id, temperature, seed,
                    candidate_set[25], action_prob_vector[25], shown[{arm,
                    slot, propensity}], refinements, dataset_version
feedback.jsonl      impression_event_id, arm_index, slot_position,
                    propensity, verdict, reason_text, reason_category
```

A thumbs-down opens a free-text reason. The reason is categorised by LLM
into a fixed vocabulary (`too_advanced`, `too_basic`, `too_long`,
`too_short`, `wrong_topic`, `low_quality`, `already_known`, `other`), the
rejected arm is excluded, and the reason refines the next turn.

**Why fixed categories:** these become the terms of `reward_spec`. Grounding
the weights in observed reason counts turns "we invented the weights" into a
derivation — which matters because the plan requires freezing `reward_spec`
before the first bandit run.

Refinement makes turn *n+1* conditional on turn *n*, so `session_id` and
`turn_index` are logged; the turns are **not** independent draws and must
not be analysed as if they were.

## Skill difficulty and time

`skills.parquet` carries `difficulty_mean` / `difficulty_median` (mean level
of the courses teaching the skill, 0=beginner … 2=advanced) at **99%
coverage**, plus `typical_weeks_median` at **73%** coverage.

Effort *hours* per skill became partly derivable once the CC0 slug-joined
donor was added (see below), but `typical_weeks_median` remains a
calendar-span proxy rather than learner effort and is labelled as such.

### Where duration comes from

`elvinrustam/coursera-dataset` (CC0) carries both `Course Url` and a real
`Duration` ("20 hours (approximately)"). Because it has a URL it joins on
`url_slug` **exactly** — 57.4% slug overlap with the spine, against 20.4%
for the title-matched 2025 donor. That took effort-hours coverage from
**22.3% to 53.8%**, and Coursera specifically from **0% to 46.5%**.

### Why edX is disabled

edX contributed 298 arms (13.6%) with **zero skills** and **zero ratings** —
it has no skills column at source. That made `has_any_skill` a perfect
provider classifier and confounded any duration- or rating-based reward
term. Dropping it removes the confound structurally and *raises* rating
coverage (83.5% -> 96.7%) and skill coverage (65.2% -> 75.4%). Cost:
effort-hours coverage falls 53.8% -> 46.4%. Re-enable in
`config/sources.yaml` if a cross-platform comparison is wanted later.

## Testing

```bash
./.venv/bin/python -m pytest tests/ -q
```

85 tests. `tests/make_fixtures.py` generates synthetic CSVs that mirror the
real schemas *and their defects* (cp1252 mojibake, `"Not Calibrated"`,
undelimited skill blobs, bucketed durations, non-English rows), so the
pipeline is exercisable before credentials are configured. They are not the
real data.

Integration tests assert the load-bearing invariants: `course_id`
uniqueness, exactly one canonical row per cluster, donor discipline, no
null-lookalike strings, effort/calendar separation, byte-identical rebuild,
and order-independent clustering.

## Layout

```
config/     sources.yaml, families.yaml, thresholds.yaml
pipeline/   paths, hashing, textnorm, parsers, stage01..stage05, cli
tests/      unit + integration, fixture generator
docs/       ingestion-plan.md
```

`THESIS_DATA_ROOT` redirects all generated artefacts to a scratch directory;
config is always read from the repo.
