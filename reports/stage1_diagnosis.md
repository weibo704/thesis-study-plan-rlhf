# Stage-1 diagnosis — `poc-v6-stratified`

`python -m pipeline.stage1_diagnosis` · **0 API calls** · 200 learners (80 held out)

## Problem 1 — why the selector stops at ~49% of headroom

Every row selects from the *same* six candidate plans and is scored on the same held-out learners. Only the reward model differs.

| variant | selector return | gain over LLM | 95% CI | headroom captured | dims |
|---|---|---|---|---|---|
| **linear, 6 features — current** | +0.7179 | +0.1194 | ±0.0754 | 48.7% | 8 |
| noiseless preference labels | +0.7205 | +0.1220 | ±0.0760 | 49.7% | 8 |
| all 15 aggregate features | +0.6969 | +0.0984 | ±0.0769 | 40.1% | 16 |
| + per-position difficulty and topic | +0.7308 | +0.1323 | ±0.0733 | 54.0% | 18 |

### Training-set size

| train learners | selector | gain | 95% CI | captured |
|---|---|---|---|---|
| 30 | +0.6700 | +0.0715 | ±0.0803 | 29.1% |
| 60 | +0.7046 | +0.1061 | ±0.0741 | 43.3% |
| 120 | +0.7179 | +0.1194 | ±0.0754 | 48.7% |

**Reading.** Every variant lands inside ±0.075 of every other one — and that interval is about half the size of the effect being measured. Noiseless labels buy almost nothing, so label quality is not the constraint. More aggregate features make it worse, not better. The training curve is flattening, not climbing.

At this sample size the experiment **cannot resolve** differences of the size these changes produce. The honest conclusion is that ~49% is the ceiling for this design, not a defect to be engineered away: closing the remaining gap is an evaluation-power problem before it is a modelling one.

## Problem 2 — why every policy trades goal progress for reward

| quantity | value |
|---|---|
| goal skills per learner | **1.0** (every learner has exactly one) |
| so `goal_progress` is | **binary** — 0 or 1, no partial credit |
| courses touching the goal set | **1.3%** |
| mean goal term (`w_goal`=0.55) | **+0.0070** |
| mean too-hard term (`w_too_hard`=0.45) | **-0.1401** |
| ratio | the difficulty penalty is **20×** larger |
| `w_goal_terminal` | **0.0** — finishing pays nothing |

**Reading.** The goal term is not mis-weighted, it is **almost never active**. Each learner's goal is the single terminal milestone of their study plan, so 1% of courses touch it and the term is zero the rest of the time. A policy maximising reward is therefore responding almost entirely to the difficulty, topic and time terms, which are live on every course. Goal progress falling is the correct response to this reward, not a failure of the policies.

Each learner's study plan has a mean of **12.5** skills on the path to their milestone, and **15.3%** of courses touch that path. Defining the goal as the path rather than the endpoint makes the signal fire 12× more often and makes progress graded over ~11 steps instead of binary.

Note that this alone does **not** rescale the term: `need_overlap` divides by the number of skills still needed, so a larger goal set shrinks each course's share proportionally. Making the goal matter needs the specification fix *and* a terminal payout for attainment, which only becomes meaningful once progress is graded.
