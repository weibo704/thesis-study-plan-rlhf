# Material Ingestion Pipeline — Course Catalog for Recommender + Bandit Thesis

## Context

The thesis compares a **content-based baseline recommender** against a **contextual bandit layer** stacked on it, for self-paced AI/software-development learners. Everything downstream — user simulation, reward modeling, bandit training, offline evaluation — consumes this catalog as its item space. The catalog is therefore not preprocessing: it fixes the arm set, the item feature vectors, the difficulty signal the reward model depends on, and the study plan structure that makes multi-step recommendation meaningful at all.

This phase is **planning only — no implementation code**. Build is from scratch on public course metadata.

Decisions locked in this phase:
- Sources: **Coursera 2021 + Coursera 2025 + edX** (Udemy excluded from the arm pool — §1)
- Skill taxonomy: **two-tier, ~15 families / ~150 leaf skills**, agent-built by the method in §3
- Prerequisites: **agent-derived from the catalog itself** (§2 `world/`), not from external curricula
- Recency: **accept the 2020–2024 snapshot, document as a threat to validity**

**Scope revision (2026-09-27).** The item space is **data science + computer
science / software development**, not AI specifically. The original framing
("AI/software development") produced 12 families with no home for security,
mobile or systems courses, leaving ~55 legitimate CS courses classified
out-of-scope. Three families were added — `security`, `mobile-development`,
`computer-systems` — taking the count to 15. Stage 8 measured the effect:
in-scope pool 2,089 → 2,189.

**Scope stance.** The thesis contribution is the RL/bandit evaluation. The taxonomy and the prerequisite DAG are *scaffolding* — built as cheaply as possible while staying auditable, and explicitly not defended as contributions in their own right. §8 states the resulting limitation plainly.

**Lineage note.** The taxonomy method in §3 is informed by a prior public-sector learning platform the author contributed to (GovLab/InnovateUS "Learning Study Plans"). That project is organization-owned, carries no license, and its taxonomy is public-sector content (4 domains / 24 subdomains / 913 skills, including an `OPEN_JUSTICE` family, and a normalizer that discards "artificial intelligence" as too generic). **No code, prompts, taxonomy content, or data from it are reused here** — only the general approach, which is independently reimplemented below. Acknowledge the intellectual lineage in the thesis.

---

## 1. Source selection

All fields and licenses verified against the live dataset/API pages on 2026-09-20.

### Candidate matrix

| Source | Rows | Description | Skill tags | Level | Duration | Study Plans | Ratings | License |
|---|---|---|---|---|---|---|---|---|
| **Coursera 2021** (`khusheekapoor`) | 3,424 | ✅ full (some missing) | ✅ NLP-extracted | ✅ B/I/A + `Not Calibrated` | ❌ | ❌ | ✅ rating only | Unknown |
| **Coursera 2025** (`yosefxx590`) | ~3,000 | ❌ | ✅ "Gained Skills" | ✅ + `Mixed` | ✅ bucketed strings | ⚠️ `Learning Product` flag only | ✅ rating + count | Apache-2.0 |
| **edX** (`imuhammad`) | 976 | ✅ + **syllabus** | ❌ subject only | ✅ Intro/Inter/Adv | ✅ effort + length | ⚠️ MicroMasters flag only | ⚠️ `n_enrolled` | GPL-2 |
| **Udemy IT&SW** (`jilkothari`) | 22k | ❌ | ❌ | ❌ | ⚠️ lecture counts | ❌ | ✅ full | CC0 |
| **MIT Learn API** | 3,005 + 38 programs | ✅ | ⚠️ topics | ⚠️ sparse | ✅ rich | ✅ **ordered** `children[].position` | ❌ | ⛔ ruled out |

### Decisions and rationale

**Included, with distinct roles:**
- **Coursera 2021 — the spine.** Only source pairing full descriptions with skill tags. Has a real course URL (`/learn/<slug>`), giving a stable natural key.
- **Coursera 2025 — field donor, not new rows.** Joined on `normalize(title) + institution` to recover `duration`, `learning_product`, and review counts. Expect partial match (different snapshot); report the match rate and never let a donor row create a course.
- **edX — the second platform.** Description **+ syllabus** text (the main source of extractable prerequisite statements), `course_effort` × `course_length` (the only defensible effort total), `language`, and self-paced vs instructor-led.

**Excluded: Udemy (22k, CC0).** No description, no skill tags, no level. Including it would make ~85% of arms title-only, degenerating the content-based baseline and making "bandit beats baseline" partly a measure of which provider has metadata. Keep the loader behind a flag; document in the attrition funnel.

**Ruled out: MIT Learn API.** The only source with true ordered study_plans, open and unauthenticated. But its ToS bans bulk download *and* bans use "for developing, training, or improving artificial intelligence systems or large language models." Not usable without written permission. Its response shape remains the best available **schema reference**.

**Not re-scraping.** `coursera.org/robots.txt` disallows `/api/`, caps catalog pagination, and names ClaudeBot; Coursera's ToS prohibits scraping.

**Licensing stance (per direction).** Not treated as a blocker. One factual note: a completed thesis is deposited in the university repository, so "unpublished" does not fully hold. This changes nothing in the build — only the appendix: cite each source with license and retrieval date, and ship **derived** artifacts (embeddings, canonical skill assignments, hashes) rather than raw scraped text. A `sources.redistributable` flag gates the export step.

**Accepted limitation.** Snapshots run ~2020 (edX) to ~2024 (Coursera 2025); the catalog largely predates the modern LLM/GenAI course wave. Stated in threats-to-validity.

---

## 2. Schema design

Three namespaces with a **one-way dependency**, the central anti-circularity device:

```
catalog/   frozen facts from the sources          (read by everything)
world/     oracle: skill DAG, user state, reward  (never read by the agent)
agent/     arm index, feature matrices            (must not derive from world/)
```

A build-time assertion enforces that no `agent/` artifact derives from `world/`. Prerequisite satisfaction is then a **latent property the bandit must discover from features**, not a feature handed to it.

### catalog/ — frozen, immutable

**`sources`** — `source_id, name, url, license, terms_url, redistributable, retrieved_at, file_sha256, row_count, notes`

**`courses`** — one row per course, one frozen `dataset_version` (no per-row snapshot dates; mutating ratings in place would desynchronize logged rewards from the features that produced them)
- Identity: `course_id` (= `{source}:{url_slug}`), `source_id`, `url`, `url_slug`, `id_strategy`
- Text: `title_raw`, `title_normalized` (NFC + mojibake repair), `description`, `syllabus`
- Provider: `institution`, `platform`
- Structure: `content_granularity ∈ {course, project, program}` — **programs/Specializations are containers, excluded from the arm pool**
- Difficulty: `level_raw`, `level_canonical ∈ {0,1,2}`, `level_source ∈ {declared, inferred_title, inferred_text, unknown}`, `level_is_imputed`
- Duration: `effort_hours_total` (learner cost), `calendar_duration_weeks` (schedule constraint — kept **separate**), `duration_lower_bound`, `duration_upper_bound`, `duration_unit_original`, `duration_source`, `duration_is_imputed`
- Economics: `access_model ∈ {free, audit_free_cert_paid, paid_one_time, subscription}`, `price_usd`, `certification`
- Popularity: `rating_mean`, `rating_count`, `rating_is_missing`, `enrollment_count`
- Other: `language`, `pace`, `subject_raw`, `domain_family`
- Pipeline: `in_scope`, `scope_reason`, `dedup_cluster_id`, `is_canonical`, `arm_status ∈ {active, retired, out_of_scope}`

**`field_provenance`** — `course_id, field_name, source_id, transform`. Required because canonical rows merge the 2021 spine and the 2025 donor; without it you cannot answer "where did this level come from."

**`skills`** — flat table with a fixed-depth family column: `skill_id, skill_slug, label, family, tier ∈ {family, skill}, aliases, support_count`. Deliberately **not** a self-referencing `parent_skill_id` — fixed depth gives family rollup without recursive ancestor-propagation bugs.

The 15 families: `programming-foundations`, `software-engineering`,
`web-frontend`, `web-backend-apis`, `databases`, `data-analysis-viz`,
`math-for-ml`, `classical-ml`, `deep-learning`, `nlp-and-llm`,
`computer-vision`, `mlops-deployment`, `security`, `mobile-development`,
`computer-systems`.

**`course_skills`** — `course_id, skill_id, relation ∈ {teaches, requires}, tag_source ∈ {source_field, text_extracted, inferred}, tag_confidence, evidence_text`. The `requires` rows store extracted prerequisite statements — which are almost always *skills* ("basic Python"), not named courses.

**`tag_crosswalk`** — `raw_tag, skill_id, action ∈ {map, drop, review}, decision_source ∈ {auto_merge, adjudicated, auto_reject, human}, score_bundle, reason, reviewer`. The frozen output of §3 and the audit trail for noisy-tag filtering.

### world/ — oracle (never read by the agent)

No hand-transcribed curricula, no citation apparatus, no external program framing. The DAG is derived from the catalog and the frozen taxonomy.

**`skill_prereqs`** — the oracle DAG: `prereq_skill_id, skill_id, support_evidence, confidence, derivation ∈ {agent_proposed, co_occurrence_inferred}, model_version, prompt_version`

Two candidate-generation paths, usable together:
- **Agent-proposed** — prompt an LLM with the frozen skill labels and descriptions from `skills` and ask which skills presuppose which. Methodologically consistent with the taxonomy step. **Generate per family**, not over all pairs: ~150 skills is ~11k ordered pairs, but 12 families × 8–20 skills each is a tractable partial-order request per family, plus a short explicit pass for cross-family edges (e.g. `math-for-ml` → `classical-ml`). Same determinism rules as §3 — temperature 0, responses cached by prompt hash.
- **Co-occurrence-inferred** — a cheap statistical signal from the catalog itself: a skill that consistently appears on courses at higher `level_canonical` than skills it co-occurs with is a candidate downstream skill. Gives a second signal to cross-check agent proposals against.

Record `derivation` per edge and report agreement between the two paths — where they disagree is where the DAG is weakest, and that is worth a sentence in the write-up.

Acyclicity: drop lowest-confidence edges in any cycle; **document the pruning count**.

⚠️ **Firewall consequence.** The co-occurrence path derives edges from `level_canonical` and skill overlap — the same features the recommender scores on. With no external grounding left, the `world/` → `agent/` firewall is the only thing preventing circularity: if prerequisite edges reached `agent/` features while the reward credits prerequisite satisfaction, the bandit would win by construction. The build assertion in §4 must therefore hold absolutely for `skill_prereqs` and anything derived from it — see the note under randomized-graph ablation for why the ablation alone does **not** catch this.

**`study_plans` / `study_plan_items`** — **ordered skill milestones**, generated by topologically sorting the pruned DAG within each `domain_family`. `study_plans`: `study_plan_id, name, domain_family, target_level`; `study_plan_items`: `study_plan_id, position, skill_id, is_optional`. MOOC courses are *candidates for* a milestone, not fixed members — the datasets contain no ordered MOOC sequences, and inventing `position` over unordered membership would be fabrication dressed as data. Where the topological order is under-constrained (parallel skills at the same depth), record ties rather than breaking them arbitrarily.

**Reserved for next phase:** `users` + `user_latent_state`, and `reward_spec` (frozen and hashed before any bandit run).

### agent/ — experiment artifacts

**`arm_index`** — `arm_index, dedup_cluster_id, arm_status, first_seen_dataset_version`. Keyed on **`dedup_cluster_id`, not `course_id`**: keying on a source row means re-running dedup silently re-points trained arm parameters. Append-only with explicit retirement.

**`arm_feature_spec`** — `feature_name, index, dtype, source_field, transform, imputation_rule, spec_version`. The feature vector must be an auditable artifact, not implicit in code.

**`arm_features`** / `embeddings` — matrices carrying `arm_index` **as a column** plus a row-order hash; the loader asserts alignment rather than trusting row order. Text embeddings projected to k=32–64 by an SVD **fit once on training arms and frozen**, with `projection_version` recorded.

**Reserved for next phase:** `candidate_sets` and `interactions` (logging the full action-probability vector, not just the chosen propensity).

---

## 3. Skill taxonomy construction

Goal: ~12 families / ~150 leaf skills over the in-scope catalog, produced semi-automatically then **frozen** as `skills` + `tag_crosswalk`.

### Why not the incremental design

The prior public-sector pipeline classifies each course against an **already-approved vocabulary**, scoring existing families first and gating new ones behind a high threshold. That is a *maintenance* design and it is wrong here for two reasons: there is no seed vocabulary to start from, and its ID resolution searches the current store, so the taxonomy you get depends on the order courses were processed. Pointed at an empty vocabulary, the first course would define the taxonomy and everything after would be pulled toward it.

**Therefore: batch, not incremental.** Generate candidates over the whole catalog first, then run a single global merge pass. Order-independence is a build requirement, asserted in §4.

### Stage 3a — seed the families by hand

Author the family layer manually (the list in §2). The family layer is a decision the thesis should own and be able to defend, not an emergent artifact. Each family gets a one-paragraph scope note and 3–5 in/out boundary examples, used later as classification guidance and printed in the appendix.

### Stage 3b — candidate leaf generation

Input: for each in-scope course, `title + description + raw skill tags + (edX) syllabus`. Ask an LLM for verb-first or noun-phrase skill labels constrained to 2–4 words, no abbreviations, each with the supporting evidence span and a proposed family.

Guard rails from the outset:
- **Reject any label that is just a family name or broader** — the failure mode that produces a "Generative AI" leaf under a "Generative AI" family.
- **No stoplist that drops core domain terms.** The prior project discarded "ai"/"artificial intelligence" as too generic; here those are the subject matter. Handle over-broad labels by the breadth test below, not by a keyword blocklist.
- Cap candidates per course (~8) so long descriptions don't dominate the vocabulary.

Expect on the order of 10–20k raw candidates from ~2k courses. That is the input to merging, not the taxonomy.

### Stage 3c — normalize and block

Deterministic merge key: casefold → Unicode NFKD → strip punctuation → drop stopwords/articles → **sort tokens** → join. Token sorting makes "testing automated" and "automated testing" collide, which is the single highest-yield normalization.

Block candidates to avoid an O(n²) comparison: shared-token inverted index with high-document-frequency tokens suppressed (a token appearing in most candidates carries no signal), plus embedding k-NN (small hosted or local sentence-transformer model, top-k ≈ 5) to catch synonym pairs with no shared tokens ("neural networks" / "deep learning").

### Stage 3d — three-band merge cascade

For each candidate pair compute token Jaccard, containment, character sequence ratio, and embedding cosine. Then:

| Band | Action |
|---|---|
| **High agreement** across lexical + embedding scores | **auto-merge**, no LLM call |
| **Middle** — conflicting or borderline scores | **LLM adjudication**: same-concept? one subsumes the other? canonical label? with a one-line reason |
| **Low** across the board | **auto-reject**, keep separate |

Resolve cascading merges with **union-find** so transitive groups collapse consistently, then pick the canonical label per group (highest support, shortest compliant name as tiebreak).

Calibrate the band boundaries on **your own** data by hand-labeling ~200 sampled pairs and choosing cut-points that put roughly 10–15% of pairs in the adjudication band — enough to catch the hard cases, small enough to review. Do not import thresholds from a different corpus; short scraped skill tags behave differently from generated skill phrases.

### Stage 3e — prune to ~150 leaves

- **Support threshold**: drop leaves attached to fewer than N courses (start N≈5). A leaf with 1–2 courses cannot support learning in either the recommender or the bandit and mostly reflects one course's phrasing.
- **Breadth test**: drop or promote a leaf broad enough to be a family.
- **Per-family caps**: target 8–20 leaves per family; families that overflow get a second review pass, families that underfill signal either a catalog gap or a mis-specified family.
- Record every drop in `tag_crosswalk` with `action = drop` and a reason. The attrition from 10–20k candidates to ~150 leaves is itself a reportable table.

### Stage 3f — validate, then freeze

- **Multi-model agreement** on a sample: have 2–3 different models independently score a sampled batch of leaf labels (distinct knowledge body? correct granularity? duplicate of an existing leaf?), take the median, and flag disagreement above a threshold for human review. Cheap, and it surfaces exactly the labels where one model is idiosyncratic.
- **Human audit** of (a) the whole adjudication band, (b) a random sample of auto-merge and auto-reject decisions, (c) all dropped-by-support leaves near the cutoff. Report precision/recall of the automated decisions against this audit, plus inter-annotator agreement on a 50-pair subset with a second reader.
- **Assign the noisy raw tags.** The verified false positives ("lambda calculus" on a solar-cell course) are caught here: a raw tag maps to a leaf only if its embedding similarity to the course text clears a threshold and its PMI with the course's family is positive. Report the false-positive rate before vs after.
- **Freeze.** Write `skills` and `tag_crosswalk` as immutable Parquet with a content hash in the manifest. Every downstream run reads the frozen files; nothing re-derives the taxonomy. This is what keeps arm features stable across runs.

### Determinism requirements

The prior pipeline is instructive as a list of things to avoid: it sets temperature 0 and a seed but retries failures at *higher* temperature, resolves IDs against mutable store state, and versions the taxonomy as the literal string `"current"`. For this build:

- Temperature 0, and **no retry-at-higher-temperature** — a failed extraction is logged and reviewed, not resampled until it complies.
- **Cache every LLM response keyed by a hash of (prompt, model, input)**, committed alongside the build. A rebuild replays the cache and needs no API access; this is what makes the taxonomy reproducible by a reader.
- `skill_id` derived from the frozen canonical slug, never from a mutable lookup.
- LLM calls are advisory in the *middle* band only. The auto-merge and auto-reject bands are pure deterministic arithmetic, so most decisions are reproducible by construction.
- Assert order-independence: shuffle the candidate input and confirm the merged output is identical.

---

## 4. Cleaning and validation plan

Issues below are **observed in the actual data**, not hypothetical.

| # | Issue | Evidence | Handling |
|---|---|---|---|
| 1 | **False-positive skill tags** | "lambda calculus" on *Silicon Thin Film Solar Cells*; "wait (system call)" on a business capstone | Handled in §3f: similarity threshold + family PMI, recorded in `tag_crosswalk` |
| 2 | **Encoding mojibake** | "École Polytechnique" → "�cole"; "you'll" → "you�ll" | Detect source encoding (cp1252/latin-1 vs utf-8), repair, NFC-normalize into `title_normalized`. Corrupts dedup and embeddings **non-uniformly** — worst for non-English institutions |
| 3 | **Sentinel strings, not nulls** | `Not Calibrated` in Difficulty Level and Course Rating; `Mixed` in the 2025 level column; `NA` in edX | Map to null + set `*_is_missing`. Treat `Mixed` as `content_granularity = program`, not a difficulty |
| 4 | **Missing-not-at-random difficulty** | Coursera 2021: Beginner 41%, Advanced 29%, remaining 30% includes `Not Calibrated` | Do **not** impute the mode — that manufactures a correlation between level and popularity/recency that flows into any level-match reward term. Impute only from title/description with `level_source` recorded, else leave null |
| 5 | **Missing descriptions** | Documented in the 2021 column description | Require description **or** skill tags for arm-pool admission; count exclusions |
| 6 | **Incompatible duration units** | 2025: buckets ("1 – 3 Months"); edX: effort ("5 hours per week") × length (weeks) | Keep the **interval**, don't collapse to a point estimate — midpoints plus a log transform spike at a few values, so any monotone reward term becomes a step function and the bandit learns bucket rank, not time fit. Compute `effort_hours_total` only where two of {weeks, hours/week, total} are known. **Never bake an assumed hours/week into a course record** — that is a *user* parameter, and doing so double-counts the time budget |
| 7 | **Effort vs calendar conflation** | edX weeks are partly cohort calendar constraints | Separate fields (§2); for self-paced learners calendar weeks are not a cost |
| 8 | **Rating incomparable; edX has none** | Coursera ~4.7 mean; edX has only `n_enrolled` | Empirical-Bayes shrinkage toward the **provider** mean weighted by `rating_count`; always carry `rating_is_missing` |
| 9 | **Duplicates** | 2021 and 2025 are both Coursera snapshots, heavy overlap, no shared id | Cascade: exact `url_slug` → `normalize(title) + institution` → fuzzy + embedding, with a manual review band. Duplicate arms split reward mass — a correctness issue |
| 10 | **Non-English courses** | edX ~20% non-English | Filter to English for the arm pool; record in the funnel |
| 11 | **Containers in the item pool** | Coursera 2025: 17% `Specialization` | Excluded via `content_granularity`; retained as reference rows |
| 12 | **Provider leakage** | Field coverage differs systematically by source | Train a classifier to predict `provider` from the arm feature vector. If AUC ≈ 1, the reward model is partly a provider detector — stratify, drop provider-identifying features, or narrow the pool |

### Scope filtering

Family keyword rules over title + description + mapped skills, then an embedding classifier for the ambiguous band, with a **manually audited sample and reported precision/recall**. Expect ~4,400 raw rows to yield roughly **1,200–2,500 in-scope arms**. Publish the **attrition funnel** — it pre-empts the obvious question.

### Validation assertions (hard-coded, fail the build)

- `arm_index` is a bijection onto canonical in-scope courses; no two arms share a `dedup_cluster_id`
- every `course_skills.skill_id` resolves in `skills`; every `tag_crosswalk` mapping resolves
- **`skill_prereqs` is acyclic**
- feature-matrix row count equals arm count; hash matches the manifest; no NaN
- `*_is_imputed = 1` wherever a value was filled
- no `agent/` artifact has a `world/` input (namespace firewall)
- taxonomy build is **order-independent** (shuffled input → identical output)
- `redistributable = false` sources contribute no raw text to exported artifacts

### Prerequisite-graph validation

Held-out hand-labeled edges and inter-annotator agreement are **dropped** — that rigor was calibrated to defending the DAG as a contribution, which is no longer the framing. What remains:

- **Randomized-graph ablation — a go/no-go gate, not a post-hoc check.** Run the bandit/baseline comparison with (i) agent-derived edges, (ii) degree-preserving randomized edges, (iii) no edges. Run this **before** building further on the DAG. If (i) ≈ (ii), the prerequisite structure adds no signal to the simulation — report that as a result rather than hiding it.

  **What this gate cannot tell you.** It tests whether the DAG beats noise, not whether it is independent of the recommender's own features. Because co-occurrence edges encode level ordering, (i) > (ii) is close to guaranteed and is consistent with the bandit having learned nothing beyond "recommend easier things first." Only the `world/`→`agent/` firewall rules that out. Interpret a passing ablation as "the DAG is not noise," never as "the comparison is uncontaminated," and say so in the write-up.

- **Acyclicity assertion** — unchanged, hard-coded in the build gate.
- **Face-validity spot-check** — hand-review ~20–30 agent-proposed edges for obvious errors (a leaf skill claimed as prerequisite of its own family root, reversed direction on an obvious pair). Roughly an hour; catches egregious failures before they propagate. Not full labeling, and not reported as precision/recall.

---

## 5. Output format (the deliverable)

```
catalog/
  sources.parquet
  courses.parquet              # canonical, one row per course
  field_provenance.parquet
  skills.parquet               # 12 families / ~150 leaves, FROZEN
  course_skills.parquet        # relation = teaches | requires
  tag_crosswalk.parquet        # FROZEN, with decision_source + reason
  llm_cache/                   # hashed prompt→response, enables offline rebuild
world/
  skill_prereqs.parquet        # oracle DAG, agent-derived + co-occurrence
  study_plans.parquet  study_plan_items.parquet   # topological sort per domain_family
agent/
  arm_index.parquet            # keyed on dedup_cluster_id
  arm_feature_spec.yaml
  arm_features.parquet         # arm_index column + row-order hash
  embeddings.parquet           # frozen SVD, projection_version
reports/
  attrition_funnel.md  taxonomy_attrition.md  validation_report.json
  tag_audit.md  dedup_review.csv  prereq_eval.md
  manifest.json                # per-file sha256, row counts, dataset_version
  DATASHEET.md                 # thesis appendix
```

**Parquet** as canonical (typed, list-valued skill columns, stable), optional **DuckDB** build for ad-hoc queries, CSV mirrors only for the small hand-curated tables that benefit from being diffable in git.

**Consumer contract.** The baseline reads `arm_features` + `embeddings` + `course_skills(teaches)`; the bandit additionally reads `arm_index` for stable integer arm identity. Neither reads `world/`. The simulator and reward model read `world/`. A single `load_catalog(dataset_version)` entry point asserts manifest hashes and the feature-spec version on every load, so schema drift fails loudly.

---

## 6. Pipeline stages

1. `fetch` — download the three datasets, record sha256 + retrieval date
2. `parse` — encoding detection and mojibake repair; raw → typed staging
3. `normalize` — sentinels → null + flags; level, duration intervals, rating shrinkage, `access_model`, `content_granularity`
4. `merge` — 2025 donor onto the 2021 spine; write `field_provenance`; report match rate
5. `dedup` — cascade matcher; assign `dedup_cluster_id`, `is_canonical`; emit review CSV
6. `taxonomy` — §3: seed families → generate candidates → normalize/block → three-band merge → prune → validate → **freeze**
7. `skills_assign` — map raw tags through the frozen crosswalk; extract `requires` from edX syllabus text
8. `scope` — classify in/out of scope; emit attrition funnel
9. `prereqs` — agent-propose skill prerequisite edges from the frozen taxonomy (per family, plus a cross-family pass); cross-check against the co-occurrence signal; prune to acyclic; build study_plans by topological sort within each `domain_family`
10. `features` — freeze `arm_index`, fit and freeze the SVD projection, materialize matrices per `arm_feature_spec`
11. `validate` — run all assertions; emit `validation_report.json` + `manifest.json` + datasheet

The manual bottleneck is gone. Stage 6 keeps the only real hand-curation cost (the taxonomy audit: ~200 labeled pairs plus the adjudication band); stage 9 is agent-assisted generation plus the ~1-hour face-validity spot-check — on the order of a day or two, not weeks. Everything else is mechanical.

---

## 7. Verification

- **Build gate**: `validate` runs every assertion in §4 and exits non-zero on failure.
- **Order-independence**: shuffle taxonomy candidate input, rebuild, assert identical `skills` + `tag_crosswalk` hashes.
- **Offline rebuild**: clear API credentials, rebuild from `llm_cache/`, confirm all manifest hashes unchanged. This is the reproducibility claim the thesis rests on.
- **Sanity spot-checks**: 20 random canonical rows reviewed by hand against their live course URLs.
- **Dedup quality**: label the review band; report matcher precision/recall.
- **Taxonomy quality**: precision/recall of auto-merge/auto-reject against the human audit; false-positive rate of raw-tag assignment before vs after pruning; inter-annotator agreement.
- **DAG check**: assert acyclicity; print longest path and tie-depth per family as a readability check on the generated DAG.
- **Firewall check**: assert no `agent/` artifact has any input path reaching `skill_prereqs` or the co-occurrence statistics. With external grounding gone this is the load-bearing correctness assertion, not a formality.
- **Prereq go/no-go**: randomized-graph ablation, run before building further on the DAG.
- **Leakage check**: provider-prediction AUC from arm features, reported as a number.
- **Downstream smoke test**: trivial nearest-neighbor recommender over `embeddings` for 5 hand-written profiles; eyeball the top-10. Fastest way to catch a corrupted text pipeline.

---

## 8. Flagged for the next phase (not decided here)

Downstream choices the schema leaves room for, surfaced because they constrain it:

1. **Pre-register and hash `skills`, `skill_prereqs`, and `reward_spec` together — before the first bandit run.** In a simulator study the reward function *is* the ground truth, so tuning its weights after seeing the bandit lose invalidates the result. With the taxonomy and DAG now agent-derived rather than externally grounded, there is no independent oracle left in the system, and retroactively adjusting taxonomy boundaries or prerequisite edges is the *same* violation as retuning the reward. Freeze all three as one versioned bundle, not the reward spec alone.
2. **Asymmetric difficulty penalty.** A symmetric `|level_gap|` penalty makes the optimal policy level-neutral and erases the sequencing behavior being studied — too-hard must cost more than too-easy.
3. **Never log a deterministic policy.** The content-based baseline is deterministic, so propensities are 0/1 and IPS is degenerate. Log with ε-mixing (ε ≥ 0.05) or softmax-over-scores; report effective sample size and clipping.
4. **Layered circularity.** Simulator → reward model → bandit → evaluated by simulator. If the simulator already emits reward, the learned reward model should be a *realism study*, not the bandit's training signal. Include at least one metric the bandit does not optimize (prerequisite-violation rate, skill-goal attainment, time-to-goal).
5. **Slate/study plan OPE.** Sequence propensities multiply and IPS variance explodes; restrict off-policy claims to slot 1 (or position-level PBM-IPS) and evaluate multi-step study_plans in the simulator only.
6. **Optional**: request MIT Open Learning permission if true ordered study_plans become necessary; keep Udemy behind a flag if the arm pool proves too small.

---

## 9. Threats to validity (state plainly in the thesis)

Write these out rather than letting a reviewer find them:

1. **Both the skill taxonomy and the prerequisite structure are model-generated**, not validated against a real curriculum or a domain expert. They are scaffolding for the RL evaluation, not claims about how people actually learn.
2. **The randomized-graph ablation is the sole safeguard** against the prerequisite structure being uninformative noise — and it only establishes "better than noise." It does not establish independence from the recommender's features; the `world/`→`agent/` firewall does, and that is an implementation invariant, not an empirical finding.
3. **No equivalent safeguard exists for the taxonomy boundaries** beyond the tag-crosswalk audit in stage 6. The 15 families are an authored choice, revised once already when the scope broadened from AI to CS/DS generally.
6. **The AI-specific families are the thinnest.** Because the catalog predates the GenAI wave, `nlp-and-llm` holds 31 courses and `computer-vision` 55, against 512 for `programming-foundations`. Any claim about AI-learner study_plans specifically rests on a small slice of the arm pool.
4. **The catalog predates the modern LLM/GenAI course wave** (snapshots ~2020–2024), so the item space under-represents exactly the topics today's AI learners most want.
5. **Difficulty and duration are substantially imputed** and missingness is provider-correlated; the provider-leakage AUC check quantifies how much.

This is a reasonable trade given the thesis's focus on RL evaluation methodology rather than taxonomy construction — but it is a trade, and saying so is stronger than having it surfaced at a defense.
