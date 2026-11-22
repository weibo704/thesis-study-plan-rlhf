"""Stage 12 -- synthetic learner population (world/ oracle).

Generates `users` and `user_latent_state`. This is the ground truth the
reward is computed against, so it lives in world/ and must never be read
when building agent/ features.

Two design choices that matter more than they look:

* **Levels are stratified, not sampled from the catalog.** 49% of arms are
  beginner, so a population drawn to match the catalog would be mostly
  beginners — and for a beginner the difficulty term is exactly zero,
  because the top candidates are already beginner-level. Measured earlier:
  two reward specs with a 2.7x difference in the too-advanced weight gave
  *identical* top-10s for a beginner, and only diverged (82% overlap) for
  an advanced learner. A beginner-heavy population would make difficulty
  matching inert and the bandit would be doing topic matching alone.

* **Goals come from the study plans.** A user's target is a milestone in a
  family's topological ordering, and their initial mastery covers the
  earlier milestones. That gives a real prerequisite gap to close, which
  is what makes multi-step recommendation meaningful rather than decorative.
"""
from __future__ import annotations

import argparse
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from . import hashing, llm, paths, persona

# Stratified so difficulty is a live signal for most of the population.
LEVEL_MIX = {0: 0.40, 1: 0.35, 2: 0.25}
# Weekly study hours, lognormal-ish: most people have little time.
HOURS_CHOICES = np.array([2, 3, 5, 8, 12, 20])
HOURS_WEIGHTS = np.array([0.22, 0.24, 0.24, 0.16, 0.10, 0.04])


def _mastery_for(level: int, plan: pd.DataFrame, target_pos: int,
                 readiness: float, rng: np.random.Generator) -> Dict[str, float]:
    """Initial mastery over a plan's skills, scaled by the learner's readiness.

    `readiness` is drawn per user and controls how much of the earlier
    milestones they actually hold. It exists because the first version
    hard-coded N(0.75, 0.22) for earlier milestones, which put ~86% of
    prerequisites above the 0.5 threshold: 39% of the population ended up
    with every prerequisite already satisfied, so sequencing was not a live
    problem for them. Measured, that was not plan sparsity (0% of goals
    lacked prereq edges) and not goal shallowness (deeper goals had LESS
    gap) -- it was this generator being too generous.

    Making readiness explicit turns the prerequisite gap into a designed,
    tunable property of the population rather than an accident of constants.
    """
    mastery: Dict[str, float] = {}
    # Low readiness -> earlier milestones only partly held -> real gap.
    earlier_mean = 0.30 + 0.60 * readiness
    for r in plan.itertuples():
        if r.position < target_pos:
            m = float(np.clip(rng.normal(earlier_mean, 0.25), 0.0, 1.0))
        elif r.position == target_pos:
            m = float(np.clip(rng.normal(0.20 * readiness, 0.15), 0.0, 1.0))
        else:
            m = float(np.clip(rng.normal(0.05, 0.08), 0.0, 1.0))
        # A higher-level learner carries more incidental knowledge.
        m = float(np.clip(m + 0.08 * level * rng.random(), 0.0, 1.0))
        mastery[r.skill_id] = round(m, 3)
    return mastery


def _agent_population(usable, by_plan, plan_family, label_of, id_of,
                      n_users, rng) -> List[Dict[str, Any]]:
    """Generate learners with an agent instead of independent draws.

    Independent sampling cannot correlate background, available hours and
    already-held skills. An agent produces a coherent person, so a
    full-time worker with no tech background gets few hours AND no prior
    skills, rather than those being rolled separately.
    """
    budget = llm.Budget(llm.config()["limits"]["max_extract_calls"], "persona")
    out: List[Dict[str, Any]] = []
    variant = 0
    # Round-robin the target level so the stratification survives.
    level_cycle = [lv for lv, share in LEVEL_MIX.items()
                   for _ in range(int(round(share * 20)))]
    # Round-robin the FAMILY too. Sampling plans uniformly let a thin family
    # vanish from the population entirely -- nlp-and-llm has only 4 plans of
    # 122, and one run drew none of them. That family is both the thinnest
    # and the most AI-relevant, so losing it silently is the worst case.
    by_family: Dict[str, List[str]] = {}
    for pid_ in usable:
        fam = plan_family.get(pid_)
        by_family.setdefault(fam, []).append(pid_)
    family_cycle = sorted(by_family)
    while len(out) < n_users:
        target_level = level_cycle[variant % len(level_cycle)]
        fam = family_cycle[variant % len(family_cycle)]
        pid = str(rng.choice(by_family[fam]))
        plan = by_plan[pid].sort_values("position")
        max_pos = int(plan["position"].max())
        # The plan's goal IS its terminal milestone; the learner targets it.
        target_pos = max_pos
        goal_labels = plan[plan["is_goal"]]["label"].tolist() or \
            plan[plan["position"] == max_pos]["label"].tolist()
        earlier = plan[plan["position"] < target_pos]["label"].tolist()
        if not earlier:
            earlier = plan[plan["position"] == 0]["label"].tolist()

        people = persona.generate(plan_family.get(pid, ""), goal_labels,
                                  earlier, persona.BATCH, variant,
                                  target_level=target_level, budget=budget)
        variant += 1
        if not people:
            continue
        for person in people:
            if len(out) >= n_users:
                break
            person.update({
                "goal_plan_id": pid, "goal_position": target_pos,
                "goal_labels": goal_labels, "earlier_labels": earlier,
            })
            out.append(person)
        if variant % 20 == 0:
            print(f"[simulator] agent personas {len(out)}/{n_users} "
                  f"(api calls {budget.used})")
    return out


def run(n_users: int = 500, seed: int = 20260930,
        mode: str = "agent") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    skills = pd.read_parquet(paths.CATALOG / "skills.parquet")
    items = pd.read_parquet(paths.WORLD / "study_plan_items.parquet")
    plans = pd.read_parquet(paths.WORLD / "study_plans.parquet")

    label_of = dict(zip(skills["skill_id"], skills["label"]))
    fam_of = dict(zip(skills["skill_id"], skills["family"]))
    by_plan = {pid: g for pid, g in items.groupby("study_plan_id")}
    plan_family = dict(zip(plans["study_plan_id"], plans["domain_family"]))
    # Every plan is a goal-ancestor closure with depth >= 3 by construction.
    usable = list(by_plan)
    goal_of = dict(zip(plans["study_plan_id"], plans["goal_skill_id"]))

    id_of = {v: k for k, v in label_of.items()}
    users: List[Dict[str, Any]] = []
    latent: List[Dict[str, Any]] = []

    if mode == "agent":
        people = _agent_population(usable, by_plan, plan_family, label_of,
                                   id_of, n_users, rng)
        for i, person in enumerate(people):
            uid = f"U{i:04d}"
            pid = person["goal_plan_id"]
            plan = by_plan[pid].sort_values("position")
            target_pos = person["goal_position"]
            known = {id_of[l] for l in person["known_labels"] if l in id_of}
            earlier_ids = set(plan[plan["position"] < target_pos]["skill_id"])
            # Readiness is now OBSERVED from the persona rather than drawn:
            # the fraction of the earlier chain this person actually holds.
            readiness = (len(known & earlier_ids) / len(earlier_ids)
                         if earlier_ids else 0.0)
            for r in plan.itertuples():
                if r.skill_id in known:
                    m = float(np.clip(rng.normal(0.85, 0.10), 0.5, 1.0))
                elif r.position < target_pos:
                    m = float(np.clip(rng.normal(0.20, 0.15), 0.0, 0.49))
                else:
                    m = float(np.clip(rng.normal(0.05, 0.07), 0.0, 0.49))
                if m > 0.01:
                    latent.append({"user_id": uid, "skill_id": r.skill_id,
                                   "family": fam_of.get(r.skill_id),
                                   "mastery": round(m, 3)})
            users.append({
                "user_id": uid, "start_level": person["start_level"],
                "readiness": round(readiness, 3),
                "goal_plan_id": pid, "goal_family": plan_family.get(pid),
                "goal_position": target_pos,
                "goal_skill_ids": [id_of[l] for l in person["goal_labels"]
                                   if l in id_of],
                "goal_labels": person["goal_labels"],
                "weekly_hours": person["weekly_hours"],
                "persistence": person["persistence"],
                "novelty_pref": person["novelty_pref"],
                "price_sensitivity": person["price_sensitivity"],
                "background": person["background"],
                "motivation": person["motivation"],
                "source": "agent",
                "seed": int(rng.integers(0, 2**31)),
            })
        n_users = len(users)

    levels = rng.choice(list(LEVEL_MIX), size=max(n_users, 1),
                        p=list(LEVEL_MIX.values()))
    for i in range(0 if mode == "agent" else n_users):
        uid = f"U{i:04d}"
        level = int(levels[i])
        pid = str(rng.choice(usable))
        plan = by_plan[pid].sort_values("position")
        max_pos = int(plan["position"].max())
        target_pos = max_pos
        target_skills = plan[plan["is_goal"]]["skill_id"].tolist() or \
            plan[plan["position"] == max_pos]["skill_id"].tolist()

        # Readiness spans under-prepared to well-prepared, so the population
        # contains both learners who need the prerequisite chain and
        # learners who can go straight at the goal.
        readiness = float(np.clip(rng.beta(2.2, 2.2), 0.02, 0.98))
        mastery = _mastery_for(level, plan, target_pos, readiness, rng)
        hours = float(rng.choice(HOURS_CHOICES, p=HOURS_WEIGHTS))

        users.append({
            "user_id": uid,
            "start_level": level,
            "readiness": round(readiness, 3),
            "goal_plan_id": pid,
            "goal_family": plan_family.get(pid),
            "goal_position": target_pos,
            "goal_skill_ids": target_skills,
            "goal_labels": [label_of.get(s) for s in target_skills],
            "weekly_hours": hours,
            # Tolerance for a bad recommendation before disengaging.
            "persistence": round(float(np.clip(rng.beta(5, 2), 0.05, 0.99)), 3),
            # Weight on a course being new rather than already-mastered.
            "novelty_pref": round(float(np.clip(rng.normal(0.55, 0.2), 0, 1)), 3),
            "price_sensitivity": round(float(np.clip(rng.beta(2, 3), 0, 1)), 3),
            "background": "", "motivation": "", "source": "statistical",
            "seed": int(rng.integers(0, 2**31)),
        })
        for sid, m in mastery.items():
            if m > 0.01:
                latent.append({"user_id": uid, "skill_id": sid,
                               "family": fam_of.get(sid), "mastery": m})

    udf = pd.DataFrame(users)
    ldf = pd.DataFrame(latent)
    hashing.write_parquet(udf, paths.WORLD / "users.parquet", stage="simulator")
    hashing.write_parquet(ldf, paths.WORLD / "user_latent_state.parquet", stage="simulator")

    # prerequisite gap: how much of the goal's prerequisites are unmet
    dag = pd.read_parquet(paths.WORLD / "skill_prereqs.parquet")
    preds: Dict[str, set] = {}
    for r in dag.itertuples():
        preds.setdefault(r.skill_id, set()).add(r.prereq_skill_id)
    mast = {(r.user_id, r.skill_id): r.mastery for r in ldf.itertuples()}
    gaps = []
    for u in users:
        need = set()
        for s in u["goal_skill_ids"]:
            need |= preds.get(s, set())
        if not need:
            gaps.append(np.nan); continue
        unmet = sum(1 for s in need if mast.get((u["user_id"], s), 0.0) < 0.5)
        gaps.append(unmet / len(need))
    udf["prereq_gap"] = gaps

    lines = [
        "# Synthetic learner population", "",
        "> **world/ oracle.** This is the ground truth the reward is computed",
        "> against. It must never be read when building `agent/` features.", "",
        f"- users: **{len(udf):,}**",
        f"- latent mastery rows: **{len(ldf):,}**",
        f"- mean mastered skills per user: "
        f"**{(ldf['mastery'] > 0.5).groupby(ldf['user_id']).sum().mean():.1f}**", "",
        "## Level mix (stratified, NOT catalog-matched)", "",
        "| start level | users | share |", "|---|---|---|",
    ]
    for lv, n in udf["start_level"].value_counts().sort_index().items():
        lines.append(f"| {int(lv)} | {n} | {n/len(udf):.0%} |")
    lines += [
        "", "The catalog is 49% beginner arms. A population matching that",
        "would leave the difficulty term inert for most users: measured",
        "earlier, two reward specs differing 2.7x on the too-advanced weight",
        "produced *identical* top-10s for a beginner, diverging only for an",
        "advanced learner. Stratifying keeps difficulty a live signal.", "",
        "## Goal depth", "", "| target milestone | users |", "|---|---|",
    ]
    for pos, n in udf["goal_position"].value_counts().sort_index().items():
        lines.append(f"| {int(pos)} | {n} |")
    lines += [
        "", f"- mean unmet prerequisite fraction at start: "
        f"**{np.nanmean(gaps):.2f}**", "",
        "## Weekly hours", "", "| hours | users |", "|---|---|",
    ]
    for h, n in udf["weekly_hours"].value_counts().sort_index().items():
        lines.append(f"| {h:g} | {n} |")
    lines += ["", "## Goal families", "", "| family | users |", "|---|---|"]
    for fam, n in udf["goal_family"].value_counts().items():
        lines.append(f"| `{fam}` | {n} |")
    (paths.REPORTS / "simulator.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    hashing.write_parquet(udf, paths.WORLD / "users.parquet", stage="simulator")
    hashing.record_stage("simulator", users=len(udf), latent_rows=len(ldf),
                         seed=seed, mode=mode)
    print(f"[simulator] mode={mode} users={len(udf):,} "
          f"latent_rows={len(ldf):,} seed={seed}")
    print(f"[simulator] level mix {dict(udf.start_level.value_counts().sort_index())}")
    print(f"[simulator] mean unmet prereq fraction {np.nanmean(gaps):.2f}")
    return udf


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 12: synthetic learners")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--mode", choices=["agent", "statistical"], default="agent",
                    help="agent: coherent LLM-generated learners (cached); "
                         "statistical: independent draws from fixed distributions")
    a = ap.parse_args()
    run(a.n, a.seed, a.mode)


if __name__ == "__main__":
    main()
