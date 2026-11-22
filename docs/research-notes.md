# Research notes

The record that does not belong in the thesis: what was tried, what broke,
what a measurement overturned. Kept because the corrections are often more
informative than the results, and because a reader who wants to know whether
to trust the numbers deserves to see how they were arrived at.

Ordered roughly chronologically within each section.

---

## 1. Measurements that overturned a conclusion

These are the ones that matter. In each case a conclusion had already been
drawn and acted on before measurement contradicted it.

**Plan similarity was reported as 0.989 and the generator blamed.** Cosine
similarity on the plan feature vectors reads near 1.0 by construction: the
vectors are mostly positive and share a constant bias term. Several turns of
work went into "fixing" a generator that was not broken. Measured properly, as
Jaccard overlap on course sets, within-learner similarity is **0.255** and
across-learner **0.035**, with 85.6% of learner pairs sharing no course at all.
The generator was never the binding constraint; the selector was.

*Lesson carried forward: a similarity metric on feature vectors that share a
bias term measures the bias term.*

**The selector gap was described as slack and turned out to be a ceiling.** The
report recommended adding sequence-describing features. Four candidate causes
were then separated by measurement: noiseless labels buy +0.003, more aggregate
features make it worse, interaction and squared terms move nothing, and the
training curve flattens. Every variant falls inside every other's confidence
interval. ~49% of headroom is what the design yields; the rest is an
evaluation-power problem.

**The domain gate was tuned on too small a probe.** A 6-vs-6 query probe
suggested in-domain and out-of-domain separated by 0.141, and the threshold was
set at 0.52. A 14-vs-8 probe put the real gap at **0.029** — in-domain floor
0.506 ("become a data engineer"), out-of-domain ceiling 0.477 ("how to cook
Italian food"). The original threshold rejected a legitimate query on its first
live test.

**Course-level similarity was the obvious gate and was wrong.** "How to trade
stocks" scores **0.722** against the nearest course, *above* "become a backend
web developer" at **0.702**, because the scope classifier admitted finance
courses. Skill-level similarity discriminates where course-level cannot.

**The randomized-graph ablation was invalid on first run.** Degree-preserving
edge swaps preserve each node's in- and out-degree but **lengthen paths** — mean
plan depth went 8.7 → 12.8, and seeded mastery rows 6,606 → 10,034. The random
arm therefore differed in depth as well as ordering, and the uncorrected run
showed the real graph *losing* on some policies. Depth-matching each learner
fixed it.

**A single-seed result was reported as a win.** The planner appeared ahead on
seed 7 (+0.108) and behind on seed 23 (−0.032). Everything is multi-seed and
paired now.

**The +40% best-of-N result did not replicate.** On the corrected population it
became +0.044 and not established. Diagnosis: partly underpowered, partly
heterogeneity. The current +0.1136 is what survives.

---

## 2. Bugs with consequences

**Encoding detection fell through to cp1252.** UTF-8 was rejected because the
source contains baked-in replacement characters; the fallback mis-decoded the
corpus and a text-repair library then masked it convincingly. Fixing detection
revealed **52.5%** of descriptions carry unrecoverable damage. Had this not been
caught, the embeddings would have been built on mojibake.

**`abs(hash(...))` for the propensity seed.** Python salts string hashing per
process, so the logged propensity was not reproducible across restarts and would
have differed per container once deployed. Replaced with sha256. This one is
quiet and would have invalidated any off-policy analysis.

**A test overwrote live data.** A reproducibility test called the population
builder directly and replaced the 500-learner file with its own 40. Now runs in
a scratch root via subprocess and asserts the live population survived.

**`--force` destroyed a pre-registration.** Overwrote the single frozen bundle
file, losing the earlier record — precisely what pre-registration exists to
prevent. Bundles are now per-name files that refuse to overwrite.

**The planner was broken in two ways that cancelled.** Independent successor
draws injected noise at 0.8× the signal, and the learner-state model never
changed level, so the value function was exactly flat.

**`nlp-and-llm` vanished from the population.** Four of 122 plans, none drawn.
Fixed with family round-robin.

**The datasheet's limitations were hardcoded prose and went stale.** It claimed
40.8% damaged text (actually 52.5%) and 2,188 arms (actually 1,899) long after
both had moved, and a hand-edit would have been silently reverted by the next
run. Every figure now derives from the data.

---

## 3. What the real feedback says, and why it is uncomfortable

52 comparisons from 6 participants were collected through the deployed service.
This is far too little to fit seven weights — 36 training pairs — and nothing
below is a result. It is, however, the only evidence in the project about
whether simulated preferences resemble real ones, and it does not fully agree.

| feature | simulated fit | fitted on 52 real comparisons |
|---|---|---|
| `mean_too_hard` | **−0.838** | −0.260 |
| `mean_too_easy` | −0.184 | **+1.091** |
| `goal_coverage` | +0.366 | **−1.520** |
| `mean_topic` | +0.313 | −0.020 |
| `mean_rating` | +0.182 | +0.137 |
| `pos_weighted_difficulty` | +0.368 | +0.105 |

Three things stand out.

**Real participants preferred *easier* plans, strongly.** `mean_too_easy` flips
from a penalty to the largest positive weight. The simulated learner was built
to find under-challenge mildly bad; real testers appear to find it good.

**Goal coverage flips sign.** The frozen reward pays for covering the skills a
learner still lacks; real participants chose *against* it. One reading: a plan
covering more goal skills is necessarily broader and harder, and the people
testing preferred narrow and tractable. If so, `goal_coverage` and
`mean_too_easy` are measuring the same preference twice.

**Held-out accuracy on real preferences is 0.750, against 0.602 on simulated.**
Real people were *more* self-consistent than the 64%-accurate simulated learner.
The noise model may be pessimistic.

**The serious caveat.** The left-hand plan was chosen **65%** of the time.
Position bias of that size means a meaningful share of this signal is layout,
not judgement. Randomising left/right assignment per comparison should be the
next change to the study app — it is a small fix and without it these numbers
cannot be cleanly interpreted.

This belongs in the thesis as a threat-to-validity observation on the
simulator, not as a result. It is direct evidence bearing on Limitation 1.

---

## 4. Things deliberately not done

**The scope classifier audit.** The sample is prepared with an empty
`human_label` column; roughly 100 rows of hand-labelling would close the one
unmeasured step in the data chapter. Not done for time.

**Ethics review.** A student study was planned and abandoned because the IRB
timeline exceeded the project's. Testing is limited to the author and
supervisor, which is why §3.9's online update rule exists at all — it is built
for a handful of comparisons, not a cohort.

**Prerequisite extraction from syllabus text.** Planned when edX was in scope;
edX was dropped and the 123 syllabus texts went with it.

**Off-policy evaluation.** Propensities are logged correctly and the machinery
is in place, but no IPS or SNIPS estimate is reported. With a deterministic
selector and few logged impressions the estimators would be degenerate.

---

## 5. Open questions worth an experiment

1. **Why did preference learning match direct regression?** Two explanations —
   the ranking objective suiting a ranking task, or noise acting as
   regularisation — and the data does not separate them. A sample-size sweep at
   matched label noise would.
2. **Does the real-vs-simulated weight disagreement survive position-bias
   correction and a larger sample?** If `goal_coverage` is genuinely negative to
   real learners, the frozen reward is wrong in a way that matters more than any
   result in the thesis.
3. **Does the planning contrast become meaningful under a graded goal?** §5.3
   predicts it should. Cheap to test once a `poc-v7` bundle exists.
4. **How much of the retrieval contribution is the shortlist size rather than
   the ranking?** 25 was chosen without tuning.
