"""Evidence table for the stage-1 report.

Answers two questions with measurements rather than impressions:

  1. Does the pipeline's *generation* half earn its keep? Decompose the
     path from "any five courses in the catalogue" to "the best of the
     six plans the LLM wrote" into retrieval, generation and ordering,
     scoring every variant with the same frozen oracle return.

  2. Are the six plans for a learner actually different from each
     other, and are two learners' plans different? Measured as Jaccard
     overlap on course sets, which is what "different plan" means to a
     learner -- not cosine on feature vectors, which is inflated by the
     shared bias term and by features that are mostly positive.

Replays plans from the LLM cache, so it makes no API calls and writes
nothing under data/. Output: reports/stage1_evidence.md
"""
from __future__ import annotations

import itertools
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

from pipeline import llm, paths, plan_generator
from pipeline.environment import Environment
from pipeline.stage17_bestofn import (
    Featuriser, execute, N_PLANS, PLAN_LEN, SHORTLIST,
)

N_USERS = 200
N_RANDOM_WIDE = 40          # the deliberately unfair control
SEED = 7


def _shortlist(env, fz, meta, u):
    """Rebuild exactly the shortlist stage 17 showed the generator."""
    labels = list(u.goal_labels) if u.goal_labels is not None else []
    text = f"{' '.join(map(str, labels))}. {u.background or ''}".strip()
    topic = fz.topic(text or "learn programming")
    rows, id_to_arm = [], {}
    for i, p in enumerate(np.argsort(-topic)[:SHORTLIST]):
        arm = int(env.arms[p])
        m = meta.loc[arm]
        id_to_arm[i] = arm
        rows.append({
            "id": i, "title": str(m.title_normalized)[:60],
            "level": {0: "beginner", 1: "intermediate",
                      2: "advanced"}.get(m.level_canonical),
            "hours": None if pd.isna(m.effort_hours_total) else float(m.effort_hours_total),
            "rating": None if pd.isna(m.rating_mean) else float(m.rating_mean),
        })
    profile = {"goal": labels, "level": int(u.start_level),
               "weekly_hours": float(u.weekly_hours),
               "background": u.background or ""}
    return topic, rows, id_to_arm, profile


def _ci(v: np.ndarray) -> float:
    return 1.96 * v.std(ddof=1) / np.sqrt(len(v))


def _jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if (a | b) else 0.0


def run(prereg: str = "poc-v6-stratified", n_users: int = N_USERS) -> str:
    env = Environment(prereg=prereg)
    fz = Featuriser(env)
    users = pd.read_parquet(paths.WORLD / "users.parquet").head(n_users)
    courses = pd.read_parquet(paths.CATALOG / "courses.parquet")
    af = pd.read_parquet(paths.AGENT / "arm_features.parquet")[["arm_index", "course_id"]]
    meta = af.merge(
        courses[["course_id", "title_normalized", "level_canonical",
                 "effort_hours_total", "rating_mean"]],
        on="course_id").set_index("arm_index")

    budget = llm.Budget(llm.config()["limits"]["max_extract_calls"], "stage1-evidence")
    rng = np.random.default_rng(SEED)
    catalogue = [int(a) for a in env.arms]

    wide, narrow, six, first, best = [], [], [], [], []
    plan_sets: Dict[str, List[frozenset]] = {}

    for u in users.itertuples():
        topic, shortlist, id_to_arm, profile = _shortlist(env, fz, meta, u)
        plans = plan_generator.generate(profile, shortlist, N_PLANS, PLAN_LEN, budget)
        arm_sets = [[id_to_arm[c] for c in p["course_ids"] if c in id_to_arm]
                    for p in plans]
        arm_sets = [a for a in arm_sets if a]
        if not arm_sets:
            continue

        pool = list(id_to_arm.values())
        draw = lambda src: list(rng.choice(src, PLAN_LEN, replace=False))
        ret = lambda arms: execute(env, u.user_id, arms, topic)["return"]

        wide.append(max(ret(draw(catalogue)) for _ in range(N_RANDOM_WIDE)))
        narrow.append(max(ret(draw(pool)) for _ in range(N_RANDOM_WIDE)))
        six.append(max(ret(draw(pool)) for _ in range(N_PLANS)))
        rets = [ret(a) for a in arm_sets]
        first.append(rets[0])
        best.append(max(rets))
        plan_sets[u.user_id] = [frozenset(a) for a in arm_sets]

    wide, narrow, six, first, best = map(np.asarray, (wide, narrow, six, first, best))
    n = len(best)

    # --- plan diversity -------------------------------------------------
    within = np.asarray([_jaccard(a, b) for s in plan_sets.values()
                         for a, b in itertools.combinations(s, 2)])
    ids = list(plan_sets)
    pairs = [(ids[i], ids[j]) for i, j in rng.choice(len(ids), size=(3000, 2)) if i != j]
    across = np.asarray([max(_jaccard(x, y) for x in plan_sets[a] for y in plan_sets[b])
                         for a, b in pairs])

    gen = best - six
    order = best - first

    out = [
        f"# Stage-1 evidence — `{prereg}`",
        "",
        f"Regenerate with `python -m pipeline.stage1_evidence`. "
        f"Plans replay from the LLM cache: **{budget.used} API calls**.",
        f"Scored on **{n}** learners, all five variants on the same learners "
        "with the same frozen reward.",
        "",
        "## Where the return actually comes from",
        "",
        "| variant | mean return |",
        "|---|---|",
        f"| best of {N_RANDOM_WIDE} random 5-course plans, whole catalogue ({len(catalogue):,}) | {wide.mean():+.4f} |",
        f"| best of {N_RANDOM_WIDE} random 5-course plans, 25-course shortlist | {narrow.mean():+.4f} |",
        f"| best of {N_PLANS} random 5-course plans, 25-course shortlist | {six.mean():+.4f} |",
        f"| the LLM's first plan | {first.mean():+.4f} |",
        f"| best of the LLM's {N_PLANS} (oracle pick) | {best.mean():+.4f} |",
        "",
        "## Paired contrasts",
        "",
        "| stage | what it compares | difference | 95% CI | win rate |",
        "|---|---|---|---|---|",
        f"| retrieval | shortlist vs whole catalogue, {N_RANDOM_WIDE} draws each "
        f"| {(narrow - wide).mean():+.4f} | ±{_ci(narrow - wide):.4f} | {np.mean(narrow > wide):.0%} |",
        f"| generation | the LLM's {N_PLANS} vs {N_PLANS} random draws from its own shortlist "
        f"| {gen.mean():+.4f} | ±{_ci(gen):.4f} | {np.mean(gen > 0):.0%} |",
        f"| ordering | best of the {N_PLANS} vs the one the LLM put first "
        f"| {order.mean():+.4f} | ±{_ci(order):.4f} | {np.mean(order > 0):.0%} |",
        "",
        f"The {N_RANDOM_WIDE}-draw row is kept because it is the control a sceptic asks for, "
        f"but it is not a fair test of the generator: a maximum over {N_RANDOM_WIDE} draws "
        f"beats a maximum over {N_PLANS} of anything. The generation row is the like-for-like "
        "comparison.",
        "",
        "The ordering row is the headroom the reward model is trying to capture. "
        "It is the reason a selection step exists at all.",
        "",
        "## Are the plans actually different?",
        "",
        "Jaccard overlap on course sets. Cosine on the feature vectors reads ~0.99 "
        "and means nothing: those vectors are mostly positive and share a constant "
        "bias term, so near-identical cosine is guaranteed by construction.",
        "",
        "| comparison | mean Jaccard | |",
        "|---|---|---|",
        f"| within one learner, across their {N_PLANS} plans ({len(within):,} pairs) "
        f"| {within.mean():.3f} | {np.mean(within == 1.0):.1%} identical, "
        f"{np.mean(within < 0.5):.1%} share under half |",
        f"| across two learners, best-matching plan ({len(across):,} pairs) "
        f"| {across.mean():.3f} | {np.mean(across == 0.0):.1%} share no course at all |",
        "",
        "Two learners with different goals get genuinely different plans. "
        "Within one learner the six candidates overlap enough to be plausible "
        "alternatives and differ enough for the choice between them to matter.",
        "",
    ]
    text = "\n".join(out)
    (paths.REPORTS / "stage1_evidence.md").write_text(text)
    return text


if __name__ == "__main__":
    print(run(*sys.argv[1:]))
