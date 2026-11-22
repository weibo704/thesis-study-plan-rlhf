# Stage 1 Report

**What has been built, what it shows, and what it does not**

Bundle `poc-v6-stratified` · 157 tests passing · 2026-10-03

---

## 1. The question

The thesis asks a narrow question: **does reinforcement learning from human
feedback make an LLM's output better, when the output is a study plan?**

Not "can an LLM write a study plan" — it obviously can. The question is
whether feedback collected from learners, processed into a learned reward
model, picks a better plan than the LLM's own first instinct. That is the
headline experiment, and everything else in the project exists to make it
measurable.

To measure it at all you need four things that did not exist:

1. A **catalogue** of real courses with difficulty, effort and quality
   metadata, so a plan can be scored on something other than vibes.
2. A **skill taxonomy and prerequisite graph**, so "this course comes too
   early" has a meaning.
3. A **population of learners** with goals and starting levels, so the same
   plan can be good for one person and wrong for another.
4. A **reward function**, frozen in advance, so improvements are measured
   against a fixed target rather than a moving one.

Stage 1 built all four, plus the agents and the experiment that uses them.
The accompanying flowchart ([`stage1-flow.drawio`](stage1-flow.drawio)) shows
how they fit together at runtime.

---

## 2. What was built

### 2.1 The catalogue — three sources, one spine

Three public datasets, each playing a different role:

| source | rows | role | licence |
|---|---|---|---|
| Coursera 2021 | 3,522 | the spine — the only source pairing full descriptions with skill tags | unknown |
| Coursera 2025 | 3,404 | donor — contributes duration and review counts, never creates a row | Apache-2.0 |
| Coursera hours | 9,595 | donor — joins on URL slug, contributes real effort hours | CC0 |

A "donor" never adds a course. It only fills a gap in a spine row that
already exists. The slug-joinable hours dataset matched **58.1%** of the
spine; the title-matched 2025 snapshot matched **20.4%**. Together they took
effort-hours coverage from 22% to **56.2%**. Every donated field records
where it came from, because a merged row otherwise cannot answer "where did
this difficulty level come from?"

Two sources were considered and dropped. **edX** contributed no skills that
survived filtering, and keeping it introduced a provider-leakage problem —
field coverage differed so sharply by platform that a model could partly
identify the provider from the features alone. Dropping it removed the
problem outright. **Udemy** has 22,000 rows but no descriptions, no skill
tags and no difficulty levels; including it would have made 85% of the item
space title-only and turned any result into a measurement of which platform
has better metadata.

From 3,522 spine rows: deduplication collapsed 160 into their twins, leaving
3,362 canonical courses; subject classification kept the data-science,
computer-science and software-development ones. **Final item pool: 1,899
courses.** Nothing is deleted — out-of-scope rows stay in the catalogue
marked as such, so the attrition is auditable rather than asserted.

### 2.2 The taxonomy — 188 skills across 15 families

The 15 families were authored by hand, because a taxonomy's top level is a
decision the thesis should be able to defend rather than an artifact that
emerged from whatever course happened to be processed first. Each family
carries a scope note and in/out examples.

The leaves were generated: an LLM read each course's title, description and
raw skill tags and proposed short skill labels; those were normalised
(token-sorted, so "automated testing" and "testing automated" collide),
blocked to avoid comparing everything to everything, and merged in three
bands — high agreement merges automatically with no LLM call, the borderline
middle band is adjudicated, the rest stays separate. **188 leaves survived,
covering 4,182 assignments over 1,431 courses.**

A final filter removed 419 assignments (9.1%) whose skill label was not
semantically close to the course's own text. This is what strips tags like
"Mergers & Acquisitions" on a Git course — genuinely present in the upstream
data, and genuinely wrong.

Skill extraction then ran over the in-scope pool: **1,860 of 1,899 courses
(97.9%)** carry at least one skill, for 9,696 assignments.

### 2.3 The prerequisite graph and study plans

Two independent methods proposed prerequisite edges. An **agent** read the
frozen skill labels family by family and proposed a partial order (151
edges). A **statistical** pass looked for skills that consistently appear on
harder courses than the skills they co-occur with (115 edges). After
deduplication, 259 candidates; 5 were dropped to break cycles; **254 edges
remain.**

Where the two methods agree is the strongest part of the graph (7 edges
corroborated); where they point in opposite directions (3 edges) is the
weakest, and that is reported rather than silently resolved.

Topologically sorting this graph produced **122 study plans**, averaging 13.6
milestones each. An earlier version built one plan per family and produced
15 shallow plans — too short for sequencing to matter. Building plans as
goal-ancestor closures instead (everything you need to reach *this* skill)
produced the depth the experiment needs.

### 2.4 The learners

**500 simulated learners**, each with a goal, a starting level, a weekly
hour budget and a set of already-mastered skills (2.8 on average).

Two deliberate choices. The level mix is **stratified, not catalogue-matched
(200 beginner / 175 intermediate / 125 advanced)**: the catalogue is 49%
beginner, and a population matching it would leave the difficulty term inert
for most learners — measured directly, two reward specifications differing
2.7× on the too-advanced penalty produced *identical* top-10 lists for a
beginner. Second, goals are round-robined across families, after one run
produced a population in which an entire family never appeared as anyone's
goal.

### 2.5 The reward — frozen before anything was run

The reward function is the ground truth in a simulator study. Tuning its
weights after seeing a policy lose would invalidate the result entirely, so
the weights are **frozen and content-hashed before any experiment runs** —
together with the taxonomy and the prerequisite graph, because changing
either of those would change what the reward *means* while leaving the
reward's own hash untouched.

The current bundle (`poc-v6-stratified`, hash `54539899ab24e4b4`) weights
topic match, difficulty fit, time fit, course quality, missing metadata and
goal progress. Every weight carries a written rationale. The important one:
**overshooting a learner's level costs three times more than undershooting
it.** A symmetric penalty makes the optimal policy level-neutral, which
erases the sequencing behaviour the thesis is about.

### 2.6 The firewall

The single most load-bearing invariant in the codebase:

> **Nothing that builds a recommendation may read the prerequisite graph,
> the reward specification, or the learners' hidden state.**

This is enforced by a build assertion that tracks actual file reads, not by
convention. The reason is circularity: the prerequisite edges are partly
derived from difficulty levels, which are also what the recommender scores
on. If those edges reached the recommender's features while the reward
credited prerequisite satisfaction, the system would win by construction and
the result would mean nothing.

The cost of this discipline was measured rather than assumed. A feature
exposing the oracle graph to the reward model was tried, and its learned
weight came out at **−0.011** — statistically indistinguishable from zero.
**The firewall costs nothing here.** That is a finding, not a hope.

---

## 3. How a recommendation is actually made

This is the part the flowchart covers. The path from a query to a plan has
five steps and two LLM agents.

**1 — The query.** A learner says what they want, their level, and how many
hours a week they have.

**2 — Retrieval.** The query is embedded and the 25 closest courses are
pulled from the 1,899. This is the step that matters most, and section 4
shows by how much.

**3 — The plan-generation agent** (temperature 0.4) sees only those 25
courses and the learner's profile, and writes **six distinct five-course
plans**, each ordered, each with a named strategy ("Foundations first",
"Direct immersion"). It may not reach outside the shortlist; course IDs it
invents are dropped rather than repaired.

**4 — The reward model scores all six.** This is the learned component — six
features describing each plan (topic match, how much of it is too hard, how
much too easy, goal coverage, average rating, and whether the difficulty
rises across the sequence), weighted by coefficients fitted from feedback.

**5 — Best-of-N.** The highest-scoring plan is shown. In the interface the
slate is sampled from a softmax over the scores rather than taken
deterministically, so every impression is logged with the probability it was
chosen with — a deterministic policy logs propensities of 0 and 1, which
makes any later off-policy analysis degenerate.

### Where the feedback goes

The learner reacts: thumbs up or down, and if down, a reason in their own
words. That reason goes to the **feedback agent** (temperature 0.7, built as
a LangGraph state machine), which is not a classifier with a fixed branch
list — it decides for itself what to do:

- **Act** — the feedback is credible and actionable: interpret it, update the
  learner's model (max level, max weekly hours, topics to avoid).
- **Clarify** — ambiguous or contradicts something they said earlier: ask one
  question and change nothing yet.
- **Record** — note it, deliberately change no policy.

Only feedback that reaches the **act** branch becomes training signal. It is
turned into **pairwise comparisons** — this plan was preferred over that one
— and fitted with a Bradley-Terry model. The next section explains what that
means and why it is done that way.

**This is the loop the flowchart emphasises.** The two agents are not in
series. The plan-generation agent keeps writing candidates the same way it
always did; what feedback changes is *the scorer that picks between them*.
The generator's temperature stays low-ish and the feedback agent's stays
high, deliberately — if both drifted, you could not tell whether a better
outcome came from better feedback interpretation or just from a different
roll of the dice in generation.

The learner model updated by the **act** branch closes a second, softer loop:
it constrains the *next* query, so a learner who said "too advanced" twice
stops being shown advanced courses at retrieval time, not just at scoring
time.

---

## 3a. How feedback becomes a reward model

This is the mechanical heart of the thesis, so it is worth spelling out.

### The problem with asking for a score

A learner cannot give you a number. Nobody looks at a study plan and says
"that was worth 0.74". And if you ask them to rate it out of five, the number
you get is not comparable between people — one learner's 3 is another's 5,
and a model trained on those ratings partly learns who is generous.

What people *can* do reliably is **compare**. Shown two plans, a learner can
say which they would rather take. That judgement is anchored, because both
plans were shown to the same person at the same moment, under the same mood
and the same standards. This is the same reason RLHF trains language models
on comparisons rather than on scores.

### The Bradley-Terry model

Bradley-Terry is the standard way to turn comparisons into a scoring
function. It assumes each item has a hidden quality, and that the
probability one beats another follows a logistic curve in the gap between
them:

```
P(plan a is preferred over plan b)  =  σ( θ · (φₐ − φᵦ) )
```

- **φ** is the plan's feature vector — the six numbers describing it
  (topic match, how much is too hard, how much too easy, goal coverage,
  average rating, whether difficulty rises across the sequence).
- **θ** is what we are learning: how much each feature matters.
- **σ** is the logistic function, squashing any real number into a
  probability.

Read it plainly: *the bigger the advantage plan a has on the features the
model thinks matter, the more likely a learner prefers it.* A tiny advantage
gives a near coin flip; a large one gives near certainty.

Fitting this is ordinary logistic regression, with one twist: **each training
row is the difference between two plans' feature vectors**, and the label is
a single bit — which one won. The features are z-scored on the training
learners first, so a feature measured in hours cannot dominate one measured
as a fraction purely because its numbers are larger.

### Why the subtraction is the whole trick

Three consequences follow from training on `φₐ − φᵦ` rather than on φ:

**1. The learner cancels out.** Someone hard to please drags *both* plans
down by roughly the same amount, and that common term disappears in the
subtraction. What survives is only what distinguishes the two plans. This is
also why pairs are built **within one learner and never across two** — a
cross-learner pair would reintroduce exactly the term the subtraction was
supposed to remove, and the model would quietly start learning who is easy
to please instead of what makes a plan good.

**2. A constant carries no information.** Anything identical across the two
plans appears on both sides of the subtraction and cancels, so it cannot
express "a is better than b". The fit therefore has no intercept. There is a
visible demonstration of this in the learned weights table in §4.4: the
feature vector does carry a constant `bias` term, and the fit assigned it
**+0.000** — exactly as the algebra requires, which is a small but genuine
check that the implementation does what the model says.

**3. Only the ordering is learnable, and that is fine.** Doubling every
weight in θ gives exactly the same ranking of plans. The scale is arbitrary.
The model is not learning a reward in any unit — it is learning a *ranking
rule*, which is all best-of-N needs.

### What the model is deliberately denied

The environment computes a real-valued return for every plan. That is how
the experiment is scored. **The preference model never sees it.** Each
training row it receives is a vector of feature differences and one bit.
Everything it knows about what makes a study plan good, it inferred from
comparisons alone.

This is the point. A model fitted directly on the environment's number is an
*oracle*: it is handed the exact quantity it is being asked to predict, and
beating the LLM with it would demonstrate nothing about feedback. Both are
run, precisely so the comparison can be made — and §4.3 shows the
preference-trained model at 0.7121 against the oracle-fitted regression's
0.6819. **Learning from comparisons cost nothing measurable against learning
from the answer key.**

### Where the comparisons come from today

They are simulated, and this is the honest limitation. A simulated learner
comparing two plans picks the genuinely better one with probability

```
σ( 1.5 × (true_return_a − true_return_b) )
```

— so near-identical plans are close to a coin flip, and clearly different
ones are ranked reliably. Across the run, simulated learners picked the
better plan **64% of the time**. That noise is deliberate: a model trained
on perfect preferences would look far better than one trained on anything a
real person produces, and the result would not transfer. Each learner
contributes all 15 pairs from their 6 plans.

The reward model's held-out pair accuracy is **0.602**, which should be read
against that 64% ceiling rather than against 100%. Its correlation with true
return is **r = 0.857** — it is learning the shape of the reward well, from a
signal that is wrong a third of the time.

### How real feedback would enter

The interface already logs what is needed: every impression records the full
candidate set, which plan was shown, and the probability it was chosen with;
every thumbs-up or thumbs-down links back to that impression. What does not
exist yet is the step that turns those logs into pairs.

The intended path: a thumbs-down on the shown plan, paired with the plan the
learner accepted after regeneration, is a comparison. So is a plan kept
against one that was skipped.

This is where the **act / clarify / record** split earns its keep. Only an
**act** verdict produces a pair. A feedback that the agent judged ambiguous
goes to **clarify** and contributes nothing, and one it judged
non-actionable goes to **record** and contributes nothing. The agent is, in
effect, the gatekeeper deciding which human reactions are clean enough to
become training labels — which is a meaningful piece of the thesis, because
a naive system that treats every thumbs-down as a preference label learns
from complaints that were really about something else.

---

## 4. What the numbers say

### 4.1 Each stage earns its keep — except in one surprising way

Measured on 200 learners, every variant scored by the same frozen reward,
all plans replayed from cache with zero API calls
(`reports/stage1_evidence.md`, regenerate with
`python -m pipeline.stage1_evidence`):

| variant | mean return |
|---|---|
| best of 40 random 5-course plans, whole catalogue (1,899) | **−1.1869** |
| best of 40 random 5-course plans, from the 25-course shortlist | +0.7959 |
| best of 6 random 5-course plans, from the 25-course shortlist | +0.4365 |
| the LLM's first plan | +0.5200 |
| best of the LLM's six | **+0.8072** |

| stage | difference | 95% CI | win rate |
|---|---|---|---|
| **retrieval** (shortlist vs whole catalogue) | +1.9828 | ±0.0660 | 100% |
| **generation** (the LLM's 6 vs 6 random draws from its own shortlist) | +0.3707 | ±0.0605 | 84% |
| **ordering** (best of the 6 vs the one it put first) | +0.2872 | ±0.0435 | 74% |

Read that top row carefully. **Retrieval is doing most of the work.** Random
plans drawn from the whole catalogue score deeply negative; the same random
procedure restricted to 25 relevant courses scores strongly positive. The
single largest effect in the entire system is narrowing the candidate pool
before the LLM ever sees it.

The generator then adds a further +0.37 over random draws from that same
shortlist — a real contribution, and the comparison has to be like-for-like
to see it. Against *forty* random draws the LLM's six look like a wash
(+0.80 vs +0.81), but that is not a fair test: a maximum over forty draws
beats a maximum over six of almost anything. Both rows are reported.

The **ordering** row is the one that justifies the whole thesis. The LLM
writes six plans and puts one first. The best of the six is worth +0.287 more
than the one it chose. **The LLM cannot reliably rank its own output.** That
gap is the headroom a learned reward model is trying to capture.

### 4.2 The plans are genuinely different

An earlier draft of this analysis reported plan similarity at 0.989 and
concluded the generator was producing near-identical plans. **That was
wrong** — it was cosine similarity on feature vectors that are mostly
positive and share a constant bias term, where a near-1.0 reading is
guaranteed by construction and carries no information.

Measured properly, as overlap in the actual course sets:

| comparison | mean Jaccard | |
|---|---|---|
| within one learner, across their six plans | **0.255** | 1.4% identical, 90.5% share under half |
| across two learners, best-matching plan | **0.035** | 85.6% share no course at all |

Two learners with different goals get genuinely different plans. Within one
learner, the six candidates overlap enough to be plausible alternatives and
differ enough that choosing between them matters.

### 4.3 The headline result: feedback helps

On 200 learners (120 train / 80 held out), 1,200 generated plans
(`reports/bestofn_poc-v6-stratified.md`):

| selection policy | mean return |
|---|---|
| the LLM's first plan | 0.5985 |
| a random one of its six | 0.4045 |
| **chosen by the feedback-trained reward model** | **0.7121** |
| chosen by a model fitted on the true reward | 0.6819 |
| the best one (the ceiling) | 0.8438 |

| contrast | difference | 95% CI | win rate | verdict |
|---|---|---|---|---|
| feedback RL over raw LLM | +0.1136 | ±0.0745 | 51% | **established** |
| scoring over sampling | +0.3075 | ±0.0995 | 70% | **established** |
| more candidates alone, no scoring | −0.1940 | ±0.1001 | 25% | **established** |
| headroom remaining | +0.1317 | ±0.0465 | 48% | **established** |

Three things to notice.

**The feedback model beats the raw LLM, and the interval excludes zero.**
That is the thesis claim, measured.

**The 51% win rate understates it, and the reason matters.** A selector that
picks the LLM's own first plan produces a difference of exactly zero, which
counts as a non-win. That happens for **24%** of test learners — the reward
model agreeing with the LLM rather than failing to beat it. Among the 76%
where it actually overrides the LLM:

| | |
|---|---|
| win rate when it overrides | **67%** (41 of 61) |
| mean gain when it overrides | **+0.1489** |
| mean when it wins | +0.3459 |
| mean when it loses | −0.2549 |

So the model intervenes on three learners in four, is right two times in
three when it does, and its wins are larger than its losses. Both framings
belong in the thesis: 51% is the honest headline over all learners, and the
decomposition is what the number actually means.

**The third row is the control that makes the first one mean something.**
Generating more candidates and picking randomly is *worse* than taking the
first plan — meaningfully worse. So the gain is not "more options help"; it
is specifically the scoring that helps.

**The feedback model slightly beat the model fitted on the true reward**
(0.7121 vs 0.6819), though that difference is not established. Learning from
noisy pairwise preferences did not cost anything measurable against learning
from the oracle's actual numbers. This is the most interesting single number
in the report, and it deserves more investigation than it has had.

And a caveat the write-up should keep: **the simulated learners picked the
genuinely better plan only 64% of the time.** The reward model is learning
from deliberately noisy preferences, which is realistic, and its held-out
pair accuracy of 0.602 should be read against that ceiling rather than
against 1.0. Its correlation with true return is r = 0.857.

### 4.4 What the model learned

| feature | weight |
|---|---|
| `mean_too_hard` | **−0.838** |
| `pos_weighted_difficulty` | +0.368 |
| `goal_coverage` | +0.366 |
| `mean_topic` | +0.313 |
| `mean_too_easy` | −0.184 |
| `mean_rating` | +0.182 |
| `bias` | +0.000 |

From noisy pairwise preferences alone, with no access to the reward
specification, the model learned that **being pushed past your level is 4.6
times worse than being bored** — and the frozen reward encodes a 3:1
asymmetry. It recovered the right shape, and slightly overshot the
magnitude. It also learned that difficulty should *rise* across a plan
(`pos_weighted_difficulty`, positive), which is sequencing behaviour emerging
from preferences rather than being specified.

Adding these ordering-sensitive features is what moved the headline result
from "not established" to established. Before them, the reward model saw a
plan as a bag of courses and could not tell a well-ordered sequence from a
scrambled one.

### 4.5 Does the prerequisite graph carry signal?

The ingestion plan called for a randomized-graph ablation as a **go/no-go
gate**, to be run before building further on the DAG. It had never run. It
has now (`reports/ablation_randomized_graph.md`).

Three populations, identical in learner id, start level, weekly hours,
persistence and budget. Only the graph their goals and starting mastery
derive from differs: the real 254 edges; a degree-preserving randomization
where every skill keeps its exact in- and out-degree but *which* skill
precedes which is destroyed; and no edges at all.

One trap had to be closed first. Degree-preserving swaps preserve local
degree but **lengthen paths** — mean plan depth went 8.7 → 12.8. Left
uncorrected, the random arm would have differed in plan depth as well as
ordering and neither could be blamed for a difference. Each learner is
therefore matched to the random plan closest in depth to the real plan they
would have received, which brings assigned depth to 7.7 in both arms.

| policy | contrast | difference | 95% CI | win rate | verdict |
|---|---|---|---|---|---|
| `baseline` | real − random | +0.0347 | ±0.0478 | 53% | not established |
| `linucb` | real − random | +0.0549 | ±0.0451 | 56% | **established** |
| `planner` | real − random | +0.0552 | ±0.0442 | 56% | **established** |
| `baseline` | real − none | +0.1526 | ±0.0541 | 59% | **established** |
| `linucb` | real − none | +0.0775 | ±0.0508 | 57% | **established** |
| `planner` | real − none | +0.0579 | ±0.0490 | 55% | **established** |

**The gate passes, narrowly, and only for the policies that learn.** Real
edges are worth +0.055 over a shape-matched random graph to both the bandit
and the planner. For the fixed-weight baseline the difference is not
established — which is coherent: a policy with frozen weights has no
mechanism to exploit prerequisite structure, so it should not care what the
graph says.

Two cautions belong with this result. The effect is small — around a 56% win
rate — so the DAG beats noise without being the dominant force in the
simulation. And passing this gate says **nothing** about contamination: only
the `world/` → `agent/` firewall establishes that the recommender is not
reading the answers, and that is an implementation invariant, not a
measurement.

### 4.6 The baseline family

The course-by-course policies (`reports/experiment_poc-v6-stratified.md`),
500 learners × 5 seeds:

| contrast | difference | 95% CI | win rate |
|---|---|---|---|
| learning (bandit over fixed scorer) | +0.2042 | ±0.0130 | 80% |
| planning (lookahead over bandit) | +0.0640 | ±0.0206 | 53% |

Learning clearly helps. **Planning is statistically established and
practically negligible** — a 53% win rate and an effect size around 0.075
means the lookahead wins barely more often than a coin flip. With 2,500
learner-runs almost anything clears significance; this is a good example of
why the report states win rate and effect size alongside the interval.

One honest negative: **every policy that earns more reward makes less goal
progress** (−0.036 to −0.085, all established). The policies are optimising
what the reward actually pays for — course-by-course fit — and the goal term
is not strong enough to make them finish what the learner came for. That is a
reward specification problem, and it is visible precisely because goal
progress is tracked as a metric the policies do not optimise.

---

## 5. What went wrong

Worth recording, because several of these shaped the design.

**Encoding.** Detection rejected UTF-8 because the source file contains
baked-in replacement characters, fell through to cp1252, and mis-decoded
everything. The text-repair library then papered over the damage
convincingly enough that it went unnoticed. Fixing the detection revealed
that **52.5% of descriptions contain unrecoverable damage** from the original
scrape. The bytes are gone; this is a documented limitation, not a bug that
can be fixed.

**A test clobbered live data.** A reproducibility test called the population
builder directly and overwrote the real 500-learner file with its own 40.
Now it runs in a scratch directory via subprocess and asserts the live
population survived.

**The planner was broken in two ways that cancelled out.** It drew
independent successor samples (injecting noise at 0.8× the signal) and its
learner-state model never changed level, so the value function was exactly
flat. Both fixed.

**A single-seed result was reported as a win.** It was not; paired analysis
across five seeds put it at a 53% win rate. Everything is multi-seed and
paired now.

**`--force` overwrote the preregistration file**, destroying the record of an
earlier frozen bundle — which is precisely what pre-registration exists to
prevent. Bundles are now per-name files that refuse to overwrite.

**The +40% best-of-N result did not replicate** when re-run on the corrected
population. The honest diagnosis was that it had been underpowered. The
current +0.1136 is what survives.

**And the similarity error in §4.2**, which reversed advice given over
several turns: the generator was never the binding constraint. The selector
is.

---

## 6. Limitations

State these before a reader finds them.

1. **The taxonomy and the prerequisite graph are model-generated.** Neither
   has been validated against a real curriculum or a domain expert. They are
   scaffolding for the RL evaluation, not claims about how people learn.
2. **The learners are simulated.** Every number here is measured against a
   reward function this project wrote. The reward model learning from noisy
   preferences is a realistic setup; the preferences themselves are not real
   people's.
3. **The scope classifier is unvalidated.** The 3,362 → 1,899 filter has an
   audit sample prepared but the human labels are not filled in, so its
   precision and recall are unreported. The counts are a classifier's output,
   not a measured result.
4. **The catalogue predates the LLM course wave** (snapshots 2020–2024), so
   the item space under-represents exactly the topics a current learner most
   wants. `nlp-and-llm` has only 26 courses.
5. **Difficulty and duration are substantially imputed**, and missingness
   correlates with provider.
6. **The selector captures about 49% of the available headroom.** §7 shows
   this is a ceiling for the design rather than slack: label quality, feature
   count, nonlinearity and training size were each tested and none moved the
   result outside its confidence interval.
7. **Goal progress is weak across every policy**, because every learner has
   exactly one goal skill and 99% of courses do not touch it. The objective
   is barely present in the reward; §7 has the fix.

---

## 7. Where it goes next

Both open problems were diagnosed before deciding what to do about them
(`reports/stage1_diagnosis.md`, `python -m pipeline.stage1_diagnosis`, zero
API calls). The diagnosis changed the answer in both cases.

### The selector gap is a ceiling, not a to-do

An earlier draft of this report recommended adding more sequence-describing
features. **That was measured and it is wrong.** Four candidate causes were
separated, all selecting from the same six candidate plans on the same
held-out learners:

| variant | gain over LLM | 95% CI | headroom captured |
|---|---|---|---|
| linear, 6 features — current | +0.1194 | ±0.0754 | 48.7% |
| noiseless preference labels | +0.1220 | ±0.0760 | 49.7% |
| all 15 aggregate features | +0.0984 | ±0.0769 | 40.1% |
| + per-position difficulty and topic | +0.1323 | ±0.0733 | 54.0% |

Perfect labels buy +0.003, so **feedback quality is not the constraint**.
More aggregate features make it *worse*. Interaction and squared terms were
also tried and moved nothing. The training curve (30 → 60 → 120 learners:
29% → 43% → 49% captured) is flattening rather than climbing.

Most importantly: **every variant lands inside every other variant's
confidence interval**, and that interval is about half the size of the
effect being measured. With 80 held-out learners the experiment cannot tell
these options apart. Resolving a +0.013 difference would need roughly two
orders of magnitude more learners.

So ~49% is what this design yields, and the remaining +0.13 is an
evaluation-power problem before it is a modelling one. **The right action is
to report the ceiling, not to keep tuning against it.** This is a POC; the
residual is documented rather than chased.

### The goal term is under-specified, not under-weighted

The diagnosis here found something more basic than a weight problem:

| quantity | value |
|---|---|
| goal skills per learner | **1.0** — every learner has exactly one |
| so `goal_progress` is | **binary**, 0 or 1, no partial credit |
| courses touching the goal set | **1.3%** |
| mean goal term | **+0.0070** |
| mean too-hard term | **−0.1401**, i.e. **20× larger** |
| `w_goal_terminal` | **0.0** — finishing pays nothing |

Each learner's goal is the single *terminal* milestone of their study plan.
The term is therefore inactive on 99% of courses, while difficulty, topic
and time are live on every one. **Policies trading goal progress for reward
are responding correctly to this reward.** It is not a policy failure and it
is not a tuning error — the objective was specified in a way that barely
registers.

The fix has two parts, and needs both:

1. **Define the goal as the path, not the endpoint.** Set `goal_skill_ids`
   to the skills on the learner's plan up to their target milestone. This
   data already exists — it is how the 122 plans were built, and each user
   row already carries `earlier_labels`. It takes the goal from 1 skill to a
   mean of 12.5, makes the term fire **12× more often** (1.3% → 15.3% of
   courses), and makes progress graded over ~11 steps instead of binary.

2. **Turn on `w_goal_terminal`.** Part 1 alone does not rescale the term,
   because `need_overlap` divides by the number of skills still needed — a
   bigger goal set shrinks each course's share proportionally. A terminal
   payout for attainment is what makes finishing worth the detour, and it
   only becomes a meaningful signal once progress is graded rather than a
   coin flip.

Two things to be careful about. First, this requires freezing a new bundle
(`poc-v7`) and re-running stages 16 and 17; the `poc-v6` results stay valid
for `poc-v6` and both get reported. Second, and more important: **the
terminal weight must be derived from a stated principle and frozen before
the run** — for instance, "full attainment should outweigh the difficulty
cost of the hardest viable path." Tuning it until the planner wins is
exactly the pre-registration violation the whole design exists to prevent.

There is a real upside. Planning currently beats the bandit by +0.0640 at a
53% win rate — established but negligible — and the likely reason is that
nothing in the present reward requires lookahead. A sparse terminal payout
creates a credit-assignment problem a bandit structurally cannot solve and a
planner can. This change is the most plausible route to making the planning
contrast mean something.

### Get real feedback through the interface

The UI, the feedback agent and
the propensity logging all work and are running. The preference pairs are
currently simulated. Replacing even a few hundred of them with real human
judgements would convert the central claim from "RL from simulated feedback"
to "RL from human feedback", which is a materially stronger thesis.

---

## Appendix — reproducing the numbers

| claim | source |
|---|---|
| catalogue construction, coverage, damage | `reports/catalog_summary.md`, `reports/merge_report.md` |
| item-pool attrition 3,522 → 1,899 | `reports/attrition_funnel.md` |
| taxonomy, 188 leaves | `reports/taxonomy.md` |
| prerequisite graph, 254 edges, 122 plans | `reports/prereqs.md` |
| learner population | `reports/simulator.md` |
| frozen reward weights and rationale | `reports/preregistration_poc-v6-stratified.md` |
| retrieval / generation / ordering, plan diversity | `reports/stage1_evidence.md` — `python -m pipeline.stage1_evidence` |
| best-of-N headline result | `reports/bestofn_poc-v6-stratified.md` |
| selector ceiling, goal-term diagnosis | `reports/stage1_diagnosis.md` — `python -m pipeline.stage1_diagnosis` |
| bandit and planner baselines | `reports/experiment_poc-v6-stratified.md` |
| randomized-graph ablation | `reports/ablation_randomized_graph.md` — `python -m pipeline.stage16_ablation` |

Every LLM response is cached by a hash of model, prompt and temperature, so
the whole pipeline replays offline with no API access. The evidence script in
this report makes **zero** API calls, and produces byte-identical output
under two different Python interpreters.
